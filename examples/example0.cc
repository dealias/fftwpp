#include "fftw++.h"

// Compile with:
// g++ -I .. -fopenmp example0.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace fftwpp;
using namespace parallel;

int main()
{
  fftw::maxthreads=get_max_threads();

  cout << "1D complex to complex in-place FFT, not using the Array class"
       << endl;

//  cout << endl << "FFTW requires memory alignment of " << fftw_alignment()
//       << "."  << endl;

  size_t n=4;
  Complex *f=ComplexAlign(n);

  fft1d Forward(n,-1);
  fft1d Backward(n,1);

  for(size_t i=0; i < n; i++) f[i]=i;

  cout << "\ninput:" << endl;
  for(size_t i=0; i < n; i++) cout << f[i] << endl;

  Forward.fft(f);

  cout << "\noutput:" << endl;
  for(size_t i=0; i < n; i++) cout << f[i] << endl;

  Backward.fftNormalized(f);

  cout << "\ntransformed back:" << endl;
  for(size_t i=0; i < n; i++) cout << f[i] << endl;

  deleteAlign(f);
}
