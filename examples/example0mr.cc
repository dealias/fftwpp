#define OUTPUT 0

int K=10;

#include "Array.h"
#include "fftw++.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;
using namespace parallel;

int main()
{
  cout << "Multiple 1D real-to-complex and complex-to-real FFTs" << endl;

  fftw::maxthreads=get_max_threads();

  if(OUTPUT) K=1;

  size_t nx=8192;
#if OUTPUT
  nx=8;
#endif
  size_t ny=nx;
  size_t align=sizeof(Complex);

#define FFTX 1
#define INPLACE 1

#if FFTX
  size_t N=nx;
  size_t M=ny;
  size_t nxp=nx/2+1;
  size_t rstride=ny+2; // Add 1 complex of padding for speed
  size_t cstride=ny+2;
  size_t rdist=1;
  size_t cdist=1;
#else
  size_t N=ny;
  size_t M=nx;
  size_t nyp=ny/2+1;
  size_t rstride=1;
  size_t cstride=1;
  size_t rdist=INPLACE ? 2*nyp+2 : ny+2;
  size_t cdist=nyp;
#endif

#if INPLACE

  cout << "In-place transform" << endl;
#if FFTX
  array2<double> F(2*nxp,rstride,align);
  array2<Complex> g(nxp,cstride,(Complex *)F());
#else
  array2<double> F(nx,rdist,align);
  array2<Complex> g(nx,cdist,(Complex *)F());
#endif

#else

  cout << "Out-of-place transform" << endl;
#if FFTX
  array2<double> F(nx,rstride,align);
  array2<Complex> g(nxp,cstride,align);
#else
  array2<double> F(nx,rdist,align);
  array2<Complex> g(nx,cdist,align);
#endif

#endif

  double *f=F();

  mrcfft1d Forward(N, // length of transform
                   M,  // number of transforms
                   rstride,
                   cstride,
                   rdist,
                   cdist,
                   F,  // input array
                   g); // output array
  mcrfft1d Backward(N, // length of transform
                    M,  // number of transforms
                    cstride,
                    rstride,
                    cdist,
                    rdist,
                    g,  // input array
                    F); // output array

  // Initialize data:
  for(size_t i=0; i < nx; i++)
    for(size_t j=0; j < ny; j++)
      f[rstride*i+rdist*j]=i+j;

#if OUTPUT
  cout << endl << "input:" << endl;

  for(size_t j=0; j < ny; j++) {
    for(size_t i=0; i < nx; i++) {
      cout << f[rstride*i+rdist*j] << " ";
    }
    cout << endl;
  }
#endif

  cpuTimer c;

  for(int i=0; i < K; ++i)
    Forward.fft(f,g);

  double time=c.seconds();

#if OUTPUT
  cout << endl << "output:" << endl;

#if FFTX
  for(size_t j=0; j < ny; j++) {
    for(size_t i=0; i < nxp; i++) {
      cout << g(cstride*i+j) << " ";
    }
    cout << endl;
  }
#else
  for(size_t j=0; j < nyp; j++) {
    for(size_t i=0; i < nx; i++) {
      cout << g(cdist*i+j) << " ";
    }
    cout << endl;
  }
#endif

#endif

  cpuTimer d;
  for(int i=0; i < K; ++i)
    Backward.fftNormalized(g,f);
  time += d.seconds();

#if OUTPUT
  cout << endl << "back to input:" << endl;

  for(size_t j=0; j < ny; j++) {
    for(size_t i=0; i < nx; i++) {
      cout << f[rstride*i+rdist*j] << " ";
    }
    cout << endl;
  }
#endif

  cout << endl << "average seconds: " << time/K << endl;
}
