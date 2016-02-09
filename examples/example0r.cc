#include "fftw++.h"

// Compile with:
// g++ -I .. -fopenmp example0r.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace fftwpp;

int main()
{
  cout << "1D real to complex out-of-place FFTs not using the Array class" 
       << endl;
   
  fftw::maxthreads=get_max_threads();
  
  unsigned int n=4;
  unsigned int np=n/2+1;
  double *f=FFTWdouble(n);
  Complex *g=FFTWComplex(np);
  
  rcfft1d Forward(n,f,g);
  crfft1d Backward(n,g,f);
  
  for(unsigned int i=0; i < n; i++) f[i]=i;
        
  cout << "\ninput (" << n << " doubles):" << endl;
  for(unsigned int i=0; i < n; i++) cout << f[i] << (i!=n-1 ? " " : "\n");

  Forward.fft(f,g);
  
  cout << "\noutput (" << np << " complex):" << endl;
  for(unsigned int i=0; i < np; i++) cout << g[i] <<  (i!=np-1 ? " " : "\n");
  
  Backward.fftNormalized(g,f);
  
  cout << "\ntransformed back:" << endl;
  for(unsigned int i=0; i < n; i++) cout << f[i] <<  (i!=n-1 ? " " : "\n");
  
  deleteAlign(g);
  deleteAlign(f);
}
