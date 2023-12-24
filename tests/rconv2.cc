#include <vector>

#include "convolution.h"
#include "explicit.h"
#include "direct.h"
#include "utils.h"
#include "options.h"
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

inline void init(Complex **F, size_t ny, size_t mx, size_t my, size_t A)
{
  for(size_t a=0; a < A; ++a) {
    double *fa=(double *) (F[a]);
    for(size_t i=0; i < mx; ++i) {
      for(size_t j=0; j < my; ++j)
        fa[ny*i+j]=i+j+1+a;
    }
  }
}

int main(int argc, char *argv[])
{
  fftw::maxthreads=parallel::get_max_threads();

  bool Output=false;
  bool Normalized=true;

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

  A=2;
  B=1;

  size_t C=max(A,B);
  size_t nyp=ny/2+1;
  // Allocate input/ouput memory and set up pointers
  Complex **F=new Complex *[A];
  for(size_t a=0; a < A; ++a)
    F[a]=ComplexAlign(C*nx*nyp);

  vector<double> T;

  Multiplier *mult;
  if(Normalized) mult=multbinary;
  else mult=multbinaryUnNormalized;

  ExplicitRConvolution2 Convolve(nx,ny,mx,my,F[0]);
  cout << "threads=" << Convolve.Threads() << endl << endl;;

  double sum=0.0;
  while(sum <= K || T.size() < minCount) {
    init(F,2*nyp,mx,my,A);
    cpuTimer c;
    Convolve.convolve(F,mult);
//    Convolve.convolve(F[0],F[1]);
    double t=c.nanoseconds();
    T.push_back(t);
    sum += t;
  }

  cout << endl;
  timings("Explicit",mx*my,T.data(),T.size(),stats);
  T.clear();

  double *f=(double *) (F[0]);

  cout << endl;
  if(Output) {
    cout << endl;
    for(size_t i=0; i < mx; i++) {
      for(size_t j=0; j < my; j++) {
        cout << f[2*nyp*i+j] << "\t";
      }
      cout << endl;
    }
  } else
    cout << f[0] << endl;

  for(size_t a=0; a < A; ++a)
    deleteAlign(F[a]);
  delete [] F;

  return 0;
}
