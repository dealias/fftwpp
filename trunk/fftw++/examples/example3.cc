#include "Array.h"
#include "fftw++.h"

// Compile with
// g++ -I .. -fopenmp example3.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace Array;
using namespace fftwpp;

int main() 
{
  cout << "3D complex to complex in-place FFT" << endl;

  fftw::maxthreads=get_max_threads();
  
  unsigned int nx=4, ny=5, nz=6;
  size_t align=sizeof(Complex);
  
  array3<Complex> f(nx,ny,nz,align);
  
  fft3d Forward3(-1,f);
  fft3d Backward3(1,f);
  
  for(unsigned int i=0; i < nx; i++) 
    for(unsigned int j=0; j < ny; j++) 
      for(unsigned int k=0; k < nz; k++) 
      f(i,j,k)=i+j+k;
	
  cout << "\ninput:\n" << f;
  
  Forward3.fft(f);
  
  cout << "\noutput:\n" << f;
  
  Backward3.fftNormalized(f);
  
  cout << "\nback to input:\n" << f;
}
