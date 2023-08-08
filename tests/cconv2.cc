#include <vector>

#include "convolution.h"
#include "explicit.h"
#include "direct.h"
#include "utils.h"
#include "Array.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

// Number of iterations.
size_t N0=10000000;
size_t N=0;
size_t nx=0;
size_t ny=0;
size_t mx=4;
size_t my=0;

inline void init(Complex **F, size_t nxp, size_t nyp, size_t A)
{
  if(A%2 == 0) {
    size_t M=A/2;
    double factor=1.0/sqrt((double) M);
    for(size_t s=0; s < M; ++s) {
      double S=sqrt(1.0+s);
      double ffactor=S*factor;
      double gfactor=1.0/S*factor;
      array2<Complex> f(nxp,nyp,F[s]);
      array2<Complex> g(nxp,nyp,F[M+s]);
#pragma omp parallel for
      for(size_t i=0; i < mx; ++i) {
        for(size_t j=0; j < my; j++) {
          f[i][j]=ffactor*Complex(i,j);
          g[i][j]=gfactor*Complex(2*i,j+1);
        }
      }
    }
  } else {
    cerr << "Init not implemented for A=" << A << endl;
    exit(1);
  }
}

void add(Complex *f, Complex *F)
{
  for(size_t i=0; i < mx; ++i) {
    size_t imy=i*my;
    Complex *fi=f+imy;
    Complex *Fi=F+imy;
    for(size_t j=0; j < my; ++j) {
      fi[j] += Fi[j];
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
  bool Pruned=false;

  double K=1.0; // Time limit (seconds)
  size_t minCount=20;

  size_t A=2;
  size_t B=1;

  int stats=0; // Type of statistics used in timing test.

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

#ifdef __GNUC__
  optind=0;
#endif
  for (;;) {
    int c = getopt(argc,argv,"hdeiptA:B:K:Om:x:y:n:T:uS:");
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
        Pruned=false;
        break;
      case 'i':
        Implicit=true;
        Explicit=false;
        break;
      case 'p':
        Explicit=true;
        Implicit=false;
        Pruned=true;
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
      case 'u':
        Normalized=false;
        break;
      case 'm':
        mx=my=atoi(optarg);
        break;
      case 'x':
        mx=atoi(optarg);
        break;
      case 'y':
        my=atoi(optarg);
        break;
      case 'T':
        fftw::maxthreads=max(atoi(optarg),1);
        break;
      case 'S':
        stats=atoi(optarg);
        break;
      case 'h':
      default:
        usage(2);
        usageExplicit(2);
        exit(1);
    }
  }

  if(my == 0) my=mx;

  nx=cpadding(mx);
  ny=cpadding(my);

  cout << "nx=" << nx << ", ny=" << ny << endl;
  cout << "mx=" << mx << ", my=" << my << endl;

  if(K == 0) minCount=1;
  cout << "K=" << K << endl;
  K *= 1.0e9;

  size_t align=ALIGNMENT;
  array2<Complex> h0;
  if(Direct) {
    if(!Normalized) {
      cerr << "-u option is incompatible with -d." << endl;
      exit(-1);
    }

    h0.Allocate(mx,my,align);
  }
  int nxp=Explicit ? nx : mx;
  int nyp=Explicit ? ny : my;

  if(!Implicit)
    A=2;

  if(B < 1) B=1;
  if(B > A) {
    cerr << "B=" << B << " is not yet implemented for A=" << A << endl;
    exit(1);
  }

  // Allocate input/ouput memory and set up pointers
  Complex **F=new Complex *[A];
  for(size_t a=0; a < A; ++a)
    F[a]=ComplexAlign(nxp*nyp);

  // For easy access of first element
  array2<Complex> f(mx,my,F[0]);

  vector<double> T;

  if(Implicit) {
    multiplier *mult;

    switch(A) {
      case 2: mult=multbinary; break;
      case 4: mult=multbinary2; break;
      case 6: mult=multbinary3; break;
      case 8: mult=multbinary4; break;
      case 16: mult=multbinary8; break;
      default: cout << "Multiplication for A=" << A
                    << " is not yet implemented" << endl; exit(1);
    }

    ImplicitConvolution2 C(mx,my,A,B);
    cout << "threads=" << C.Threads() << endl << endl;;

    double sum=0.0;
    while(sum <= K || T.size() < minCount) {
      init(F,mx,my,A);
      cpuTimer c;
      C.convolve(F,mult);
//      C.convolve(F[0],F[1]);
      double t=c.nanoseconds();
      T.push_back(t);
      sum += t;
    }

    timings("Implicit",mx*my,T.data(),T.size(),stats);
    T.clear();

    if(Normalized) {
      double norm=0.25/(mx*my);
      for(size_t i=0; i < mx; i++)
        for(size_t j=0; j < my; j++)
          f[i][j] *= norm;
    }

    if(Direct) {
      for(size_t i=0; i < mx; i++)
        for(size_t j=0; j < my; j++)
          h0[i][j]=f[i][j];
    }

    if(Output) {
      for(size_t i=0; i < mx; i++) {
        for(size_t j=0; j < my; j++)
          cout << f[i][j] << "\t";
        cout << endl;
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

    ExplicitConvolution2 C(nx,ny,mx,my,F[0],Pruned);
    cout << "threads=" << C.Threads() << endl << endl;;

    double sum=0.0;
    while(sum <= K || T.size() < minCount) {
      init(F,nxp,nyp,A);
      cpuTimer c;
      C.convolve(F,mult);
      double t=c.nanoseconds();
      T.push_back(t);
      sum += t;
    }

    cout << endl;
    timings(Pruned ? "Pruned" : "Explicit",mx*my,T.data(),T.size(),stats);
    T.clear();

    if(Direct) {
      for(size_t i=0; i < mx; i++)
        for(size_t j=0; j < my; j++)
          h0[i][j]=F[0][nyp*i+j];
    }

    if(Output) {
      for(size_t i=0; i < mx; i++) {
        for(size_t j=0; j < my; j++) {
          cout << F[0][nyp*i+j] << "\t";
        }
        cout << endl;
      }
    }
    else
      cout << F[0][0] << endl;
  }

  if(Direct) {
    array2<Complex> h(mx,my,align);
    directconv2<Complex> C(mx,my);
    init(F,mx,my,2);
    cpuTimer c;
    C.convolve(h,F[0],F[1]);
    T[0]=c.nanoseconds();

    cout << endl;
    timings("Direct",mx*my,T.data(),1);
    T.clear();

    if(Output)
      for(size_t i=0; i < mx; i++) {
        for(size_t j=0; j < my; j++)
          cout << h[i][j] << "\t";
        cout << endl;
      } else cout << h[0][0] << endl;

    { // compare implicit or explicit version with direct verion:
      double error=0.0;
      cout << endl;
      double norm=0.0;
      for(size_t i=0; i < mx; i++) {
        for(size_t j=0; j < my; j++) {
          error += abs2(h0[i][j]-h[i][j]);
          norm += abs2(h[i][j]);
        }
      }

      if(norm > 0) error=sqrt(error/norm);
      cout << "error=" << error << endl;
      if (error > 1e-12) cerr << "Caution! error=" << error << endl;
    }
  }

  for(size_t a=0; a < A; ++a)
    deleteAlign(F[a]);
  delete [] F;

  return 0;
}
