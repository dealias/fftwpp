#include "Array.h"
#include "fftw++.h"

// Compile with
// g++ -I .. -fopenmp example2.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;
using namespace parallel;

int main()
{
  cout << "2D complex to complex in-place FFT" << endl;
  fftw::maxthreads=get_max_threads();

  size_t nx=4, ny=4;
  size_t align=sizeof(Complex);

  array2<Complex> f(nx,ny,align);

  fft2d Forward(-1,f);
  fft2d Backward(1,f);

  for(size_t i=0; i < nx; i++)
    for(size_t j=0; j < ny; j++)
      f(i,j)=Complex(i,j);

  cout << "\ninput:\n" << f;

  Forward.fft(f);

  cout << "\noutput:\n" << f;

  Backward.fftNormalized(f);

  cout << "\nback to input:\n" << f;
}
