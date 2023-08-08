#include <vector>

#include "convolution.h"
#include "explicit.h"
#include "direct.h"
#include "utils.h"

using namespace std;
using namespace utils;
using namespace fftwpp;

// Constants used for initialization and testing.
const Complex I(0.0,1.0);
const double E=exp(1.0);
const Complex iF(sqrt(3.0),sqrt(7.0));
const Complex iG(sqrt(5.0),sqrt(11.0));

bool Test=false;

size_t A=2; // number of inputs
size_t B=1; // number of outputs

inline void init(Complex **F, size_t m, size_t A)
{
  if(A % 2 == 0) {
    size_t M=A/2;
    double factor=1.0/sqrt((double) M);
    for(size_t s=0; s < M; ++s) {
      double ffactor=(1.0+s)*factor;
      double gfactor=1.0/(1.0+s)*factor;
      Complex *fs=F[s];
      Complex *gs=F[s+M];
      if(Test) {
        for(size_t k=0; k < m; k++) {
          fs[k]=factor*iF*pow(E,k*I);
          gs[k]=factor*iG*pow(E,k*I);
        }
      } else {
        for(size_t k=0; k < m; k++) {
          fs[k]=ffactor*Complex(k,k+1);
          gs[k]=gfactor*Complex(k,2*k+1);
        }
      }
    }
  } else {
    for(size_t a=0; a < A; ++a) {
      for(size_t k=0; k < m; ++k) {
        F[a][k]=(a+1)*Complex(k,k+1);
      }
    }
  }
}

// Pair-wise binary multiply for A=2 or A=4.
// NB: example function, not optimised or threaded.
void multA(Complex **F, size_t m,
           const size_t indexsize,
           const size_t *index,
           size_t r, size_t threads)
{
  switch(A) {
    case 2: multbinary(F,m,indexsize,index,r,threads); break;
    case 4: multbinary2(F,m,indexsize,index,r,threads); break;
    default:
      cerr << "A=" << A << " is not yet implemented" << endl;
      exit(1);
  }

  for(size_t b=1; b < B; ++b) {
    double factor=1.0+b;
    for(size_t i=0; i < m; ++i) {
      F[b][i]=factor*F[0][i];
    }
  }
}

int main(int argc, char *argv[])
{
  fftw::maxthreads=parallel::get_max_threads();

  bool Direct=false;
  bool Implicit=true;
  bool Explicit=false;
  bool Output=false;
  bool Normalized=true;
  bool Inplace=true;

  double K=1.0; // Time limit (seconds)
  size_t minCount=20;
  size_t m=11; // Problem size

  int stats=MEDIAN; // Type of statistics used in timing test.

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

#ifdef __GNUC__
  optind=0;
#endif
  for (;;) {
    int c = getopt(argc,argv,"hdeiptA:B:I:K:Om:n:uS:T:");
    if (c == -1) break;

    switch (c) {
      case 0:
        break;
      case 'd':
        Direct=true;
        break;
      case 'e':
        Explicit=true;
        Implicit=false;
        break;
      case 'i':
        Implicit=true;
        Explicit=false;
        break;
      case 'I':
        Inplace=atoi(optarg) > 0;
        break;
      case 'p':
        break;
      case 'A':
        A=atoi(optarg);
        break;
      case 'B':
        B=atoi(optarg);
        break;
      case 'K':
        K=atof(optarg);
        break;
      case 'O':
        Output=true;
        break;
      case 't':
        Test=true;
        break;
      case 'u':
        Normalized=false;
        break;
      case 'm':
        m=atoi(optarg);
        break;
      case 'T':
        fftw::maxthreads=max(atoi(optarg),1);
        break;
      case 'S':
        stats=atoi(optarg);
        break;
      case 'h':
      default:
        usage(1);
        usageExplicit(1);
        usageTest();
        exit(1);
    }
  }

  size_t n=cpadding(m);

  cout << "n=" << n << endl;
  cout << "m=" << m << endl;

  if(K == 0) minCount=1;
  cout << "K=" << K << endl;
  K *= 1.0e9;

  // Explicit and direct methods are only implemented for binary convolutions.
  if(!Implicit)
    A=2;

  if(B < 1)
    B=1;

  if(!Explicit)
    Inplace=false;

  size_t np=Explicit ? n : m;
  size_t C=max(A,B);
  Complex *f=ComplexAlign(C*np);
  Complex *g=Inplace ? f : ComplexAlign(C*np);

  Complex **F=new Complex *[C];
  Complex **G=Inplace ? F : new Complex *[C];

  for(size_t s=0; s < C; ++s)
    F[s]=f+s*np;

  if(!Inplace)
    for(size_t s=0; s < C; ++s)
      G[s]=g+s*np;

  Complex *h0=NULL;
  if(Test || Direct) {
    h0=ComplexAlign(m*B);
    if(!Normalized) {
      cerr << "-u option is incompatible with -d and -t." << endl;
      exit(-1);
    }
  }

  vector<double> T;

  if(Implicit) {
    ImplicitConvolution C(m,A,B);
    cout << "threads=" << C.Threads() << endl << endl;

    multiplier *mult=NULL;
    switch(B) {
      case 1:
        switch(A) {
          case 1: mult=multautoconvolution; break;
          case 2: mult=multbinary; break;
          case 4: mult=multbinary2; break;
          case 6: mult=multbinary3; break;
          case 8: mult=multbinary4; break;
          case 16: mult=multbinary8; break;
          default:
            cerr << "A=" << A << ", B=" << B << " is not yet implemented"
                 << endl;
            exit(1);
        }
        break;
      default:
        mult=multA;
        break;
    }
    if(!mult) {
      cerr << "A=" << A << ", B=" << B << " is not yet implemented"
           << endl;
      exit(1);
    }

    double sum=0.0;
    while(sum <= K || T.size() < minCount) {
      init(F,m,A);
      cpuTimer c;
      C.convolve(F,mult);
//    C.convolve(F[0],F[1]);
      double t=c.nanoseconds();
      T.push_back(t);
      sum += t;
    }

    timings("Implicit",m,T.data(),T.size(),stats);
    T.clear();

    if(Normalized) {
      double norm=0.5/m;
      for(size_t b=0; b < B; ++b)
        for(size_t i=0; i < m; i++)
          F[b][i] *= norm;
    }

    if(Output) {
      for(size_t b=0; b < B; ++b) {
        for(size_t i=0; i < m; i++)
          cout << F[b][i] << endl;
        cout << endl;
      }
    }

    if(Test || Direct) {
      for(size_t b=0; b < B; ++b) {
        for(size_t i=0; i < m; i++) {
          h0[i+b*m]=F[b][i];
        }
      }
    }
  }

  if(Explicit) {
    if(A != 2) {
      cerr << "Explicit convolutions for A=" << A << " are not yet implemented" << endl;
      exit(1);
    }

    Multiplier *mult;
    if(Normalized) mult=multbinary;
    else mult=multbinaryUnNormalized;
;
    ExplicitConvolution C(n,m,F[0],G[0]);
    double sum=0.0;
    while(sum <= K || T.size() < minCount) {
      init(F,m,A);
      cpuTimer c;
      C.convolve(F,mult,G);
//      C.convolve(F[0],F[1]);
      double t=c.nanoseconds();
      T.push_back(t);
      sum += t;
    }

    cout << endl;
    timings("Explicit",m,T.data(),T.size(),stats);
    T.clear();

    if(Output)
      for(size_t i=0; i < m; i++)
        cout << F[0][i] << endl;

    cout << endl;

    if(Test || Direct)
      for(size_t i=0; i < m; i++)
        h0[i]=F[0][i];
  }

  if(Direct) {
    directconv C(m);
    if(A % 2 == 0 && A > 2)
      A=2;
    init(F,m,A);
    Complex *h=ComplexAlign(m);
    cpuTimer c;
    if(A == 2)
      C.convolve(h,F[0],F[1]);
    if(A == 1)
      C.autoconvolve(h,F[0]);
    T[0]=c.nanoseconds();

    cout << endl;
    timings("Direct",m,T.data(),1);

    if(Output)
      for(size_t i=0; i < m; i++)
        cout << h[i] << endl;

    { // compare implicit or explicit version with direct verion:
      double error=0.0;
      double norm=0.0;
      for(size_t b=0; b < B; ++b) {
        double factor=1.0+b;
        for(size_t k=0; k < m; k++) {
          error += abs2(h0[k+b*m]-factor*h[k]);
          norm += abs2(h[k]);
        }
      }
      if(norm > 0) error=sqrt(error/norm);
      cout << "error=" << error << endl;
      if (error > 1e-12)
        cerr << "Caution! error=" << error << endl;
    }

    if(Test)
      for(size_t i=0; i < m; i++)
        h0[i]=h[i];
    deleteAlign(h);
  }

  if(Test) {
    Complex *h=ComplexAlign(n*B);
    // test accuracy of convolution methods:
    double error=0.0;
    cout << endl;
    double norm=0.0;

    bool testok=false;

    // Exact solutions for test case.
    if(A % 2 == 0) {
      testok=true;
      for(size_t k=0; k < m; k++) {
        h[k]=iF*iG*(k+1)*pow(E,k*I);
        //  h[k]=iF*iG*(k*(k+1)/2.0*(k-(2*k+1)/3.0));
      }
    }

    // autoconvolution of f[k]=k
    if(A == 1) {
      testok=true;
      for(size_t k=0; k < m; k++)
        h[k]=k*(0.5*k*(k+1)) - k*(k+1)*(2*k+1)/6.0;
    }

    if(!testok) {
      cout << "ERROR: no test case for A="<<A<<endl;
      exit(1);
    }

    for(size_t k=0; k < m; k++) {
      error += abs2(h0[k]-h[k]);
      norm += abs2(h[k]);
    }
    if(norm > 0) error=sqrt(error/norm);
    cout << "error=" << error << endl;
    if (error > 1e-12)
      cerr << "Caution! error=" << error << endl;
    deleteAlign(h);
  }

  delete [] F;
  deleteAlign(f);

  return 0;
}
