#include "Array.h"
#include "fftw++.h"

// Compile with
// g++ -I .. -fopenmp example2.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

int main()
{
  cout << "Tranposition of complex variables using the Array class" << endl;
  fftw::maxthreads=get_max_threads();

  unsigned int nx=4, ny=4;
  size_t align=sizeof(Complex);
  
  array2<Complex> f(nx,ny,align);
  array2<Complex> ft(ny,nx,align);
  
  Transpose T(nx,ny,1,f(),ft(),fftw::maxthreads);
  Transpose Tinv(ny,nx,1,ft(),f(),fftw::maxthreads);
  
  for(unsigned int i=0; i < nx; i++) 
    for(unsigned int j=0; j < ny; j++) 
      f(i,j)=Complex(i,j);

  cout << "\ninput:\n" << f;
  
  T.transpose(f(),ft());
    
  cout << "\noutput:\n" << ft;
  
  Tinv.transpose(ft(),f());

  cout << "\nback to input:\n" << f;
}
