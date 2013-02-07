#include "Array.h"
#include "fftw++.h"

// Compile with
// g++ -I .. -fopenmp example2r.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace Array;
using namespace fftwpp;

int main()
{
  fftw::maxthreads=get_max_threads();
  
  unsigned int nx=4, ny=5;
  unsigned int nyp=ny/2+1;
  size_t align=sizeof(Complex);
  
  array2<double> f(nx,ny,align);
  array2<Complex> g(nx,nyp,align);
  
  rcfft2d Forward(ny,f,g);
  crfft2d Backward(ny,g,f);
  
  for(unsigned int i=0; i < nx; i++) 
    for(unsigned int j=0; j < ny; j++) 
      f(i,j)=i+j;
	
  cout << f << endl;

  Forward.fft(f,g);
  
  cout << g << endl;
  
  Backward.fftNormalized(g,f);
  
  cout << f << endl;
}
