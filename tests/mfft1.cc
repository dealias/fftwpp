#include <getopt.h>

#include "Complex.h"
#include "Array.h"
#include "fftw++.h"
#include "utils.h"
#include "options.h"
#include "timing.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

// Number of iterations.
size_t N=10000000;

bool Direct=false, Implicit=true, Explicit=false, Pruned=false;

inline void init(array2<Complex>& f, size_t mx, size_t my)
{
  for(size_t i=0; i < mx; ++i)
    for(size_t j=0; j < my; j++)
      f[i][j]=Complex(i,j);
}

size_t outlimit=100;

int main(int argc, char *argv[])
{
  fftw::maxthreads=parallel::get_max_threads();

  size_t mx=4;
  size_t my=4;

  int stats=0; // Type of statistics used in timing test.

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

#ifdef __GNUC__
  optind=0;
#endif
  for (;;) {
    int c = getopt(argc,argv,"h:m:x:y:N:T:S:");
    if (c == -1) break;

    switch (c) {
      case 0:
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
      case 'N':
        N=atoi(optarg);
        break;
      case 'T':
        fftw::maxthreads=max(atoi(optarg),1);
        break;
      case 'S':
        stats=atoi(optarg);
        break;
      case 'h':
      default:
        usageCommon(2);
        exit(0);
    }
  }

  if(my == 0) my=mx;

  cout << "mx=" << mx << ", my=" << my << endl;
  cout << "N=" << N << endl;

  size_t align=ALIGNMENT;

  array2<Complex> f(mx,my,align);
  array2<Complex> g(mx,my,align);

  mfft1d Forward(my,-1,mx,1,my);
  mfft1d Backward(my,1,mx,1,my);

  cout << "\nInput:" << endl;
  init(f,mx,my);
  if(mx*my < outlimit) {
    for(size_t i=0; i < mx; i++) {
      for(size_t j=0; j < my; j++)
        cout << f[i][j] << "\t";
      cout << endl;
    }
  } else {
    cout << f[0][0] << endl;
  }

  cout << "\nOutput:" << endl;
  Forward.fft(f);
  if(mx*my < outlimit) {
    for(size_t i=0; i < mx; i++) {
      for(size_t j=0; j < my; j++)
        cout << f[i][j] << "\t";
      cout << endl;
    }
  } else {
    cout << f[0][0] << endl;
  }

  cout << "\nBack to input:" << endl;
  Backward.fftNormalized(f);
  if(mx*my < outlimit) {
    for(size_t i=0; i < mx; i++) {
      for(size_t j=0; j < my; j++)
        cout << f[i][j] << "\t";
      cout << endl;
    }
  } else {
    cout << f[0][0] << endl;
  }


  cout << endl;
  double *T=new double[N];
  for(size_t i=0; i < N; ++i) {
    init(f,mx,my);
    cpuTimer c;
    Forward.fft(f);
    Backward.fft(f);
    T[i]=0.5*c.nanoseconds();
    Backward.Normalize(f);
  }
  timings("mfft1, in-place",mx,T,N,stats);
  delete [] T;

  return 0;
}
