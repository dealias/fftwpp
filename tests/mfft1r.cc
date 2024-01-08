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


inline void init(array2<double>& f, size_t mx, size_t my)
{
  for(size_t i=0; i < mx; ++i)
    for(size_t j=0; j < my; j++)
      f[i][j] = 10 * i + j;
}

size_t outlimit=100;

int main(int argc, char *argv[])
{
  fftw::maxthreads=parallel::get_max_threads();

  // Number of iterations.
  size_t N=10000000;

  size_t mx=4;
  size_t my=4;

  size_t stats=0; // Type of statistics used in timing test.

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

#ifdef __GNUC__
  optind=0;
#endif
  for (;;) {
    int c = getopt(argc,argv,"hm:x:y:N:T:S:");
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
  size_t np=mx/2+1;
  cout << "np=" << np << endl;
  size_t M=my;

  cout << "N=" << N << endl;

  size_t align=ALIGNMENT;

  array2<double> f(mx,my,align);
  array2<Complex> g(np,my,align);

  size_t rstride=1;
  size_t cstride=1;
  size_t rdist=mx;
  size_t cdist=np;

  mrcfft1d Forward(mx, // length of transform
                   M,  // number of transforms
                   rstride,
                   cstride,
                   rdist,
                   cdist,
                   f,  // input array
                   g); // output array
  mcrfft1d Backward(mx, // length of transform
                    M,  // number of transforms
                    cstride,
                    rstride,
                    cdist,
                    rdist,
                    g,  // input array
                    f); // output array

  cout << "\nInput:" << endl;
  init(f,mx,my);
  if(mx*my < outlimit)
    cout << f << endl;
  else
    cout << f[0][0] << endl;

  cout << "\nOutput:" << endl;
  Forward.fft(f,g);
  if(mx*my < outlimit)
    cout << g << endl;
  else
    cout << g[0][0] << endl;

  cout << "\nBack to input:" << endl;
  Backward.fftNormalized(g,f);
  if(mx*my < outlimit)
    cout << f << endl;
  else
    cout << f[0][0] << endl;


  cout << endl;
  if(N > 0) {
    double *T=new double[N];
    for(size_t i=0; i < N; ++i) {
      init(f,mx,my);
      cpuTimer c;
      Forward.fft(f,g);
      Backward.fft(g,f);
      T[i]=0.5*c.nanoseconds();
      Backward.Normalize(f);
    }
    timings("mfft1r, in-place",mx,T,N,stats);
    delete [] T;
  }

  return 0;
}
