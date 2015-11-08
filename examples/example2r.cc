#include "Array.h"
#include "fftw++.h"

// Compile with
// g++ -I .. -fopenmp example2r.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

int main()
{
  cout << "2D real to complex out-of-place FFT" << endl;

  fftw::maxthreads=get_max_threads();
  
  unsigned int nx=4, ny=5;
  unsigned int nyp=ny/2+1;
  size_t align=sizeof(Complex);
  
  array2<Complex> F(nx,nyp,align);
  array2<double> f(nx,ny,align);               // For out-of-place transforms
//  array2<double> f(nx,2*nyp,(double *) F()); // For in-place transforms
  
  rcfft2d Forward(nx,ny,f,F);
  crfft2d Backward(nx,ny,F,f);
  
  for(unsigned int i=0; i < nx; i++) 
    for(unsigned int j=0; j < ny; j++) 
      f(i,j)=i+j;
        
  cout << endl << "input:" << endl << f;

  Forward.fft0(f,F);
  
  cout << endl << "output:" << endl << F;
  
  Backward.fft0Normalized(F,f);
  
  cout << endl << "back to input:" << endl << f;
}
