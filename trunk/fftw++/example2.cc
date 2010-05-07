#include "Array.h"
#include "fftw++.h"

// Compile with g++ example2.cc fftw++.cc -lfftw3

using namespace std;
using namespace Array;
using namespace fftwpp;

int main()
{
  unsigned int nx=4, ny=5;
  size_t align=sizeof(Complex);
  
  array2<Complex> f(nx,ny,align);
  
  fft2d Forward2(-1,f);
  fft2d Backward2(1,f);
  
  for(unsigned int i=0; i < nx; i++) 
    for(unsigned int j=0; j < ny; j++) 
      f(i,j)=i+j;
	
  cout << f << endl;
  
  Forward2.fft(f);
  
  cout << f << endl;
  
  Backward2.fftNormalized(f);
  
  cout << f << endl;
}
