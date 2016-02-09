#include "Array.h"
#include "fftw++.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

int main()
{  
  cout << "Multiple 1D real-to-complex and complex-to-real FFTs" << endl;

  fftw::maxthreads=get_max_threads();
  
  unsigned int nx=4, ny=4;
  unsigned int nyp=ny/2+1;
  size_t align=sizeof(Complex);
  
  cout << "Out-of-place transforms:" << endl;
    
  array2<double> f(nx,ny,align);
  array2<Complex> g(nx,nyp,align);
  size_t rstride=1;
  size_t cstride=1;
  size_t rdist=ny;
  size_t cdist=nyp;
  unsigned int M=nx;
  
  mrcfft1d Forward(ny, // length of transform
                   M,  // number of transforms
                   rstride,
                   cstride,
                   rdist,
                   cdist,
                   f,  // input array
                   g); // output array
  mcrfft1d Backward(ny, // length of transform
                    M,  // number of transforms
                    cstride,
                    rstride,
                    cdist,
                    rdist,
                    g,  // input array
                    f); // output array

  // Initialize data:
  for(unsigned int i=0; i < nx; i++) 
    for(unsigned int j=0; j < ny; j++) 
      f(i,j)=i+j;
        
  cout << endl << "input:" << endl << f;

  Forward.fft(f,g);
  
  cout << endl << "output:" << endl << g;
  
  Backward.fftNormalized(g,f);
  
  cout << endl << "back to input:" << endl << f;

}
