#include "Array.h"
#include "fftw++.h"

// Compilation: g++ example1r.cc fftw++.cc -lfftw3

using namespace std;
using namespace Array;
using namespace fftwpp;

int main()
{
  unsigned int n=5;
  unsigned int np=n/2+1;
  size_t align=sizeof(Complex);
  
  array1<double> f(n,align);
  array1<Complex> g(np,align);
  
  rcfft1d Forward(n,f,g);
  crfft1d Backward(n,g,f);
  
  for(unsigned int i=0; i < n; i++) f[i]=i;
	
  Forward.fft(f,g);
  
  cout << g << endl;
  
  Backward.fftNormalized(g,f);
  
  cout << f << endl;
}
