#include "Array.h"
#include "fftw++.h"

// Compile with
// g++ -I .. -fopenmp example1.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

int main()
{
  cout << "1D Complex to complex in-place FFT" << endl;

  fftw::maxthreads=get_max_threads();

  unsigned int n=4;
  size_t align=sizeof(Complex);
  
  array1<Complex> f(n,align);
  
  fft1d Forward(-1,f);
  fft1d Backward(1,f);
  
  for(unsigned int i=0; i < n; i++) f[i]=i;

  cout << "\ninput:\n" << f << endl;

  Forward.fft(f);

  cout << "\noutput:\n" << f << endl;
  
  Backward.fftNormalized(f);

  cout << "\nback to input:\n" << f << endl;
}
