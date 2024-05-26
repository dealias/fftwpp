#define OUTPUT 0

int K=100;

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

  size_t nx=2048;
  size_t ny=nx;
  size_t nxp=nx/2+1;
  size_t align=sizeof(Complex);

  size_t cstride=ny+1;
  size_t rstride=2*cstride;
  size_t cdist=1;
  size_t rdist=1;
  size_t M=ny;

#define INPLACE 0

#if INPLACE
  array2<double> F(2*nxp,rstride,align);
  array2<Complex> g(nxp,cstride,(Complex *)F());
//  double *F=(double *) g();
  cout << "In-place transform" << endl;
#else
  array2<double> F(nx,rstride,align);
  array2<Complex> g(nxp,cstride);
  cout << "Out-of-place transform" << endl;
#endif
  double *f=(double *) F();

  mrcfft1d Forward(nx, // length of transform
                   M,  // number of transforms
                   rstride,
                   cstride,
                   rdist,
                   cdist,
                   F,  // input array
                   g); // output array
  mcrfft1d Backward(nx, // length of transform
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

  for(int i=0; i < K; ++i)
    Forward.fft(f,g);

#if OUTPUT
  cout << endl << "output:" << endl;

  for(size_t j=0; j < ny; j++) {
    for(size_t i=0; i < nxp; i++) {
      cout << g(cstride*i+cdist*j) << " ";
    }
    cout << endl;
  }
#endif

  for(int i=0; i < K; ++i)
    Backward.fftNormalized(g,f);

#if OUTPUT
  cout << endl << "back to input:" << endl;

  for(size_t j=0; j < ny; j++) {
    for(size_t i=0; i < nx; i++) {
      cout << f[rstride*i+rdist*j] << " ";
    }
    cout << endl;
  }
#endif

}
