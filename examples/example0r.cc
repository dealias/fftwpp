#include "fftw++.h"

// Compile with:
// g++ -I .. -fopenmp example0r.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace fftwpp;

int main()
{
  fftw::maxthreads=get_max_threads();
  
  unsigned int n=4;
  unsigned int np=n/2+1;
  double *f=FFTWdouble(n);
  Complex *g=FFTWComplex(np);
  
  rcfft1d Forward(n,f,g);
  crfft1d Backward(n,g,f);
  
  for(unsigned int i=0; i < n; i++) f[i]=i;
	
  Forward.fft(f,g);
  
  for(unsigned int i=0; i < np; i++) cout << g[i] << endl;
  
  Backward.fftNormalized(g,f);
	
  for(unsigned int i=0; i < n; i++) cout << f[i] << endl;
  
  FFTWdelete(g);
  FFTWdelete(f);
}
