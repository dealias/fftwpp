#include <vector>

#include "convolution.h"
#include "explicit.h"
#include "direct.h"
#include "utils.h"
#include "options.h"
#include "options.h"

using namespace std;
using namespace utils;
using namespace fftwpp;

size_t A=2; // Number of inputs
size_t B=1; // Number of outputs
bool compact=false;

bool Test=false;

// Pair-wise binary multiply for A=2 or A=4.
// NB: example function, not optimised or threaded.
void multA(double **F, size_t m,
           const size_t indexsize,
           const size_t* index,
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

inline void init(Complex **F, size_t m,  size_t A)
{
  if(!compact)
    for(size_t i=0; i < A; ++i)
      F[i][m]=0.0;

  const Complex I(0.0,1.0);
  const double E=exp(1.0);
  const double iF=sqrt(3.0);
  const double iG=sqrt(5.0);

  if(A % 2 != 0 && A != 1) {
    cerr << "A=" << A << " is not yet implemented" << endl;
    exit(1);
  }
  if(A == 1) {
    Complex *f=F[0];
    for(size_t k=0; k < m; ++k) {
      f[k]=iF*pow(E,k*I);
    }
  }
  if(A % 2 == 0) {
    size_t M=A/2;
    double factor=1.0/sqrt((double) M);
    for(size_t s=0; s < M; s++) {
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
        fs[0]=1.0*ffactor;
        for(size_t k=1; k < m; k++) fs[k]=ffactor*Complex(k,k+1);
        gs[0]=2.0*gfactor;
        for(size_t k=1; k < m; k++) gs[k]=gfactor*Complex(k,2*k+1);
      }
    }
  }
}

void test(size_t m, Complex *h0)
{

  Complex *h=ComplexAlign(m);
  double error=0.0;
  cout << endl;
  double norm=0.0;
  size_t mm=m;

  const Complex I(0.0,1.0);
  const double E=exp(1.0);
  const double F=sqrt(3.0);
  const double G=sqrt(5.0);

  for(size_t k=0; k < mm; k++) {
    h[k]=F*G*(2*mm-1-k)*pow(E,k*I);
    //      h[k]=F*G*(4*m*m*m-6*(k+1)*m*m+(6*k+2)*m+3*k*k*k-3*k)/6.0;
    error += abs2(h0[k]-h[k]);
    norm += abs2(h[k]);
  }
  if(norm > 0) error=sqrt(error/norm);
  cout << "error=" << error << endl;
  if (error > 1e-12) {
    cerr << "Caution! error=" << error << endl;
  }
  deleteAlign(h);
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
    int c = getopt(argc,argv,"hdeiptA:B:I:K:Om:n:uS:T:X:");
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
      case 'X':
        compact=atoi(optarg) == 0;
        break;
      case 'h':
      default:
        usage(1);
        usageExplicit(1);
        usageCompact(1);
        usageTest();
        usageb();
        exit(1);
    }
  }

  size_t n=hpadding(m);

  cout << "n=" << n << endl;
  cout << "m=" << m << endl;

  if(K == 0) minCount=1;
  cout << "K=" << K << endl;
  K *= 1.0e9;

  size_t np=Explicit ? n/2+1 : m+!compact;

  // explicit and direct convolutions are only implemented for binary
  // convolutions.
  if(!Implicit)
    A=2;

  if(B < 1)
    B=1;

  if(!Explicit)
    Inplace=false;

  size_t C=max(A,B);
  Complex *f=ComplexAlign(C*np);
  double *g=Inplace ? (double *) f : doubleAlign(C*n);
  Complex **F=new Complex *[C];
  double **G=Inplace ? NULL : new double *[C*n];

  for(size_t s=0; s < C; ++s)
    F[s]=f+s*np;

  if(!Inplace)
    for(size_t s=0; s < C; ++s)
      G[s]=g+s*n;

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
    ImplicitHConvolution C(m,compact,A,B);
    cout << "threads=" << C.Threads() << endl << endl;

    if (A % 2 != 0) {
      cerr << "A=" << A << " is not yet implemented" << endl;
      exit(1);
    }

    realmultiplier *mult=0;
    if(B == 1) {
      switch(A) {
        case 2: mult=multbinary; break;
        case 4: mult=multbinary2; break;
        default: mult=multA;
      }
    } else
      mult=multA;

    double sum=0.0;
    while(sum <= K || T.size() < minCount) {
      init(F,m,A);
      cpuTimer c;
      C.convolve(F,mult);
//      C.convolve(F[0],F[1]);
      double t=c.nanoseconds();
      T.push_back(t);
      sum += t;
    }

    timings("Implicit",2*m-1,T.data(),T.size(),stats);
    T.clear();

    if(Normalized) {
      double norm=1.0/(3.0*m);
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
      for(size_t b=0; b<B; ++b) {
        for(size_t i=0; i < m; i++) {
          h0[i+b*m]=F[b][i];
        }
      }
    }
  }

  if(Explicit) {
    ExplicitHConvolution C(n,m,f,g);
    Realmultiplier *mult;
    if(Normalized) mult=multbinary;
    else mult=multbinaryUnNormalized;
;
    double sum=0.0;
    while(sum <= K || T.size() < minCount) {
      init(F,m,2);
      cpuTimer c;
      C.convolve(F,mult,G);
//      C.convolve(F[0],F[1]);
      double t=c.nanoseconds();
      T.push_back(t);
      sum += t;
    }

    cout << endl;
    timings("Explicit",2*m-1,T.data(),T.size(),stats);
    T.clear();

    if(Output)
      for(size_t i=0; i < m; i++)
        cout << f[i] << endl;
    cout << endl;

    if(Test || Direct)
      for(size_t i=0; i < m; i++)
        h0[i]=f[i];
  }

  if(Direct) {
    directconvh C(m);
    init(F,m,2);
    Complex *h=ComplexAlign(m);
    cpuTimer c;
    C.convolve(h,F[0],F[1]);
    T[0]=c.nanoseconds();

    cout << endl;
    timings("Direct",2*m-1,T.data(),1);
    T.clear();

    if(Output)
      for(size_t i=0; i < m; i++)
        cout << h[i] << endl;

    { // compare implicit or explicit version with direct verion:
      double error=0.0;
      cout << endl;
      double norm=0.0;
      for(size_t b=0; b < B; ++b) {
        double factor=1.0+b;
        for(size_t k=0; k < m; k++) {
          error += abs2(h0[k+b*m]-factor*h[k]);
          norm += abs2(h[k]);
        }
      }
      if(norm > 0)
        error=sqrt(error/norm);
      cout << "error=" << error << endl;
      if (error > 1e-12)
        cerr << "Caution! error=" << error << endl;
    }

    if(Test)
      for(size_t i=0; i < m; i++)
        h0[i]=h[i];
    deleteAlign(h);
  }

  if(Test)
    test(m,h0);

  deleteAlign(f);
  delete [] F;

  return 0;
}
