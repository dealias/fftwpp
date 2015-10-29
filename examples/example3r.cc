#include "Array.h"
#include "fftw++.h"

// Compile with
// g++ -I .. -fopenmp example3r.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

int main()
{
  cout << "3D real to complex out-of-place FFT" << endl;

  fftw::maxthreads=get_max_threads();
  
  unsigned int nx=4, ny=5, nz=6;
  unsigned int nzp=nz/2+1;
  size_t align=sizeof(Complex);
  
  array3<double> f(nx,ny,nz,align);
  array3<Complex> g(nx,ny,nzp,align);
//  array3<double> f(nx,ny,2*nzp,(double *) g()); // For in-place transforms

  rcfft3d Forward(nx,ny,nz,f,g);
  crfft3d Backward(nx,ny,nz,g,f);
  
  for(unsigned int i=0; i < nx; i++) 
    for(unsigned int j=0; j < ny; j++) 
      for(unsigned int k=0; k < nz; k++) 
      f(i,j,k)=i+j+k;
	
  cout << "\ninput:\n" << f;
  
  Forward.fft(f,g);
  
//  cout << endl << "output:" << endl << g;
//  Forward.deNyquist(g);
  
  cout << "\noutput:\n" << g;
  
  Backward.fftNormalized(g,f);
  
  cout << "\nback to input:\n" << f;
}
