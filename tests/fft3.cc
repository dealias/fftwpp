#include "Complex.h"
#include "Array.h"
#include "fftw++.h"
#include "utils.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

// Number of iterations.
unsigned int N0 = 10000000;
unsigned int N = 0;
unsigned int mx = 4;
unsigned int my = 4;
unsigned int mz = 4;

bool Direct=false, Implicit=true, Explicit=false, Pruned=false;

inline void init(array3<Complex>& f) 
{
  for(unsigned int i = 0; i < mx; ++i)
    for(unsigned int j = 0; j < my; j++)
      for(unsigned int k = 0; k < mz; k++) 
        f(i, j, k)=Complex(10 * k + i, j);
}
  
unsigned int outlimit = 100;

int main(int argc, char* argv[])
{
  fftw::maxthreads = get_max_threads();
  int r = -1; // which of the 8 options do we do?  r=-1 does all of them.

  int stats=0; // Type of statistics used in timing test.

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  
  
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"hN:m:x:y:z:n:T:S:r:");
    if (c == -1) break;
                
    switch (c) {
      case 0:
        break;
      case 'N':
        N=atoi(optarg);
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
      case 'n':
        N0=atoi(optarg);
        break;
      case 'T':
        fftw::maxthreads=max(atoi(optarg),1);
        break;
      case 'S':
        stats=atoi(optarg);
        break;
      case 'r':
        r=atoi(optarg);
        break;
      case 'h':
      default:
        usageFFT(2);
        exit(0);
    }
  }

  if(my == 0)
    my = mx;

  cout << "mx=" << mx << ", my=" << my << ", mz=" << mz << endl;
  
  if(N == 0) {
    N = N0 / mx / my;
    N = max(N, 20);
  }
  cout << "N=" << N << endl;
  
  size_t align = sizeof(Complex);

  array3<Complex> f(mx, my, mz, align);
  array3<Complex> g(mx, my, mz, align);

  double *T = new double[N];

  if(r == -1 || r == 0) { // conventional FFT, in-place
    fft3d Forward3(-1, f);
    fft3d Backward3(1, f);
    
    for(unsigned int i = 0; i < N; ++i) {
      init(f);
      seconds();
      Forward3.fft(f);
      Backward3.fft(f);
      T[i]=0.5*seconds();
      Backward3.Normalize(f);
    }
    timings("fft3d, in-place", mx, T, N, stats);
  }
  
  if(r == -1 || r == 1) { // conventional FFT, out-of-place
    fft3d Forward3(-1, f, g);
    fft3d Backward3(1, f, g);
    
    for(unsigned int i = 0; i < N; ++i) {
      init(f);
      seconds();
      Forward3.fft(f,g);
      Backward3.fft(g,f);
      T[i]=0.5*seconds();
      Backward3.Normalize(f);
    }
    timings("fft3d, out-of-place", mx, T, N, stats);
  }

  delete [] T;
  
  return 0;
}

