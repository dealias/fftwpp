#include "Array.h"
#include "fftw++.h"

// Compile with
// g++ -I .. -fopenmp example3.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace Array;
using namespace fftwpp;

int main() 
{
  unsigned int nx=4, ny=5, nz=6;
  size_t align=sizeof(Complex);
  
  array3<Complex> f(nx,ny,nz,align);
  
  fft3d Forward3(-1,f);
  fft3d Backward3(1,f);
  
  for(unsigned int i=0; i < nx; i++) 
    for(unsigned int j=0; j < ny; j++) 
      for(unsigned int k=0; k < nz; k++) 
      f(i,j,k)=i+j+k;
	
  cout << f << endl;
  
  Forward3.fft(f);
  
  cout << f << endl;
  
  Backward3.fftNormalized(f);
  
  cout << f << endl;
}
