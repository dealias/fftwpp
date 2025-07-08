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
size_t nx=0;
size_t ny=0;
size_t nz=0;
size_t mx=4;
size_t my=0;
size_t mz=0;

inline void init(double **F, size_t mx, size_t my, size_t mz, size_t A)
{
  for(size_t a=0; a < A; ++a) {
    double *fa=(double *) (F[a]);
    for(size_t i=0; i < mx; ++i) {
      for(size_t j=0; j < my; ++j) {
        for(size_t k=0; k < mz; ++k) {
          fa[mz*(my*i+j)+k]=i+(a+1)*j+a*k+1;
        }
      }
    }
  }
}

int main(int argc, char *argv[])
{
  fftw::maxthreads=parallel::get_max_threads();

  bool Output=false;
  bool Normalized=true;

  double s=1.0; // Time limit (seconds)
  size_t N=20;

  size_t A=2; // Number of independent inputs
  size_t B=1; // Number of independent outputs

  size_t stats=0; // Type of statistics used in timing test.

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

#ifdef __GNUC__
  optind=0;
#endif
  for (;;) {
    int c = getopt(argc,argv,"hdeiptA:B:s:Om:x:y:z:n:T:uS:");
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
      case 's':
        s=atof(optarg);
        break;
      case 'O':
        Output=true;
        break;
      case 'u':
        Normalized=false;
        break;
      case 'm':
        mx=my=mz=atoi(optarg);
        break;
      case 'x':
        mx=atoi(optarg);
        break;
      case 'y':
        my=atoi(optarg);
        break;
      case 'z':
        mz=atoi(optarg);
        break;
      case 'T':
        fftw::maxthreads=max(atoi(optarg),1);
        break;
      case 'S':
        stats=atoi(optarg);
        break;
      case 'h':
      default:
        usage(3);
        exit(1);
    }
  }

  if(my == 0) my=mx;
  if(mz == 0) mz=mx;

  nx=cpadding(mx);
  ny=cpadding(my);
  nz=cpadding(mz);

  cout << "nx=" << nx << ", ny=" << ny << ", nz=" << ny << endl;
  cout << "mx=" << mx << ", my=" << my << ", mz=" << mz << endl;

  if(s == 0) N=1;
  cout << "s=" << s << endl;
  s *= 1.0e9;

  A=2;
  B=1;

  size_t C=max(A,B);
  size_t nzp=nz/2+1;

  // Allocate input/ouput memory and set up pointers
  size_t size=mx*my*mz;
  size_t Size=nx*ny*nzp;
  double *f=doubleAlign(C*size);
  Complex *f0=ComplexAlign(C*Size);

  double **F=new double *[C];
  Complex **F0=new Complex *[C];

  for(size_t c=0; c < C; ++c) {
    F[c]=f+size*c;
    F0[c]=f0+Size*c;
  }

  vector<double> T;

  Multiplier *mult;
  if(Normalized) mult=multBinary;
  else mult=multBinaryUnNormalized;

  ExplicitRConvolution3 Convolve(nx,ny,nz,mx,my,mz,F0[0]);
  cout << "threads=" << Convolve.Threads() << endl << endl;;

  double sum=0.0;
  while(sum < s || T.size() < N) {
    init(F,mx,my,mz,A);
    cpuTimer c;
    for(size_t a=0; a < A; ++a) {
      double *f=F[a];
      double *f0=(double *) (F0[a]);
      for(size_t i=0; i < mx; ++i)
        for(size_t j=0; j < my; ++j)
          for(size_t k=0; k < mz; ++k)
            f0[2*nzp*(ny*i+j)+k]=f[mz*(my*i+j)+k];
    }
    Convolve.convolve(F0,mult);
    for(size_t b=0; b < B; ++b) {
      double *f=F[b];
      double *f0=(double *) (F0[b]);
      for(size_t i=0; i < mx; ++i)
        for(size_t j=0; j < my; ++j)
          for(size_t k=0; k < mz; ++k)
            f[mz*(my*i+j)+k]=f0[2*nzp*(ny*i+j)+k];
    }
    double t=c.nanoseconds();
    T.push_back(t);
    sum += t;
  }

  cout << endl;
  timings("Explicit",mx*my*mz,T.data(),T.size(),stats);
  T.clear();

  cout << endl;
  if(Output) {
    for(size_t i=0; i < mx; ++i) {
      for(size_t j=0; j < my; ++j) {
        for(size_t k=0; k < mz; ++k)
          cout << f[mz*(my*i+j)+k] << " ";
        cout << endl;
      }
      cout << endl;
    }
  } else
    cout << f[0] << endl;

  delete [] F0;
  deleteAlign(f0);

  delete [] F;
  deleteAlign(f);

  return 0;
}
