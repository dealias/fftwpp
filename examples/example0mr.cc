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

  size_t nx=4, ny=4;
  size_t nyp=ny/2+1;
  size_t align=sizeof(Complex);

  cout << "Out-of-place transforms:" << endl;

  array2<Complex> g(nx,nyp,align);

#define INPLACE 1

#if INPLACE
  double *F=(double *) g();
  double *f=F;
#else
  array2<double> F(nx,ny,align);
  double *f=(double *) F();
#endif

  size_t rstride=1;
  size_t cstride=1;
  size_t rdist=ny;
  size_t cdist=nyp;
  size_t M=nx;

  mrcfft1d Forward(ny, // length of transform
                   M,  // number of transforms
                   rstride,
                   cstride,
                   rdist,
                   cdist,
                   F,  // input array
                   g); // output array
  mcrfft1d Backward(ny, // length of transform
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
      f[rdist*i+j]=i+j;

  cout << endl << "input:" << endl;

  for(size_t i=0; i < nx; i++) {
    for(size_t j=0; j < ny; j++)
      cout << f[rdist*i+j] << " ";
    cout << endl;
  }

  Forward.fft(f,g);

  cout << endl << "output:" << endl << g;

  Backward.fftNormalized(g,f);

  cout << endl << "back to input:" << endl;

  for(size_t i=0; i < nx; i++) {
    for(size_t j=0; j < ny; j++)
      cout << f[rdist*i+j] << " ";
    cout << endl;
  }

}
