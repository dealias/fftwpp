#include "fftw++.h"

// Compilation: g++ example0.cc fftw++.cc -lfftw3

using namespace std;
using namespace fftwpp;

int main()
{
  unsigned int n=4; 
  Complex *f=ComplexAlign(n);
  
  fft1d Forward(n,-1);
  fft1d Backward(n,1);
  
  for(unsigned int i=0; i < n; i++) f[i]=i;
	
  Forward.fft(f);
  Backward.fftNormalized(f);
	
  for(unsigned int i=0; i < n; i++) cout << f[i] << endl;
  
  FFTWdelete(f);
}
