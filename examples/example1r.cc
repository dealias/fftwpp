#include "Array.h"
#include "fftw++.h"

// Compile with:
// g++ -I .. -fopenmp example1r.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

int main()
{
  cout << "1D real to complex out-of-place FFT" << endl;
  
  fftw::maxthreads=get_max_threads();
  
  unsigned int n=4;
  unsigned int np=n/2+1;
  size_t align=sizeof(Complex);
  
  array1<Complex> F(np,align);
  array1<double> f(n,align);               // For out-of-place transforms
//  array1<double> f(2*np,(double *) F()); // For in-place transforms
  
  rcfft1d Forward(n,f,F);
  crfft1d Backward(n,F,f);
  
  for(unsigned int i=0; i < n; i++) f[i]=i;

  cout << endl << "input:" << endl << f << endl;
        
  Forward.fft(f,F);
  
  cout << endl << "output:" << endl << F << endl;
  
  Backward.fftNormalized(F,f);
  
  cout << endl << "back to input:" << endl << f << endl;
}
