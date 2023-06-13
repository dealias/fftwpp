#include "Array.h"
#include "fftw++.h"

// Compile with
// g++ -I .. -fopenmp example3.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;
using namespace parallel;

int main()
{
  cout << "3D complex to complex in-place FFT" << endl;

  fftw::maxthreads=get_max_threads();

  size_t nx=4, ny=4, nz=4;
  size_t align=sizeof(Complex);

  array3<Complex> f(nx,ny,nz,align);

  fft3d Forward(-1,f);
  fft3d Backward(1,f);

  for(size_t i=0; i < nx; i++)
    for(size_t j=0; j < ny; j++)
      for(size_t k=0; k < nz; k++)
        f(i,j,k)=Complex(10*k+i,j);

  cout << "\ninput:\n" << f;

  Forward.fft(f);

  cout << "\noutput:\n" << f;

  Backward.fftNormalized(f);

  cout << "\nback to input:\n" << f;
}
