#include "Array.h"
#include "fftw++.h"

// Compilation: g++ example1.cc fftw++.cc -lfftw3

using namespace std;
using namespace Array;
using namespace fftwpp;

int main()
{
  unsigned int n=4;
  size_t align=sizeof(Complex);
  
  array1<Complex> f(n,align);
  
  fft1d Forward(-1,f);
  
  for(unsigned int i=0; i < n; i++) f[i]=i;
	
  Forward.fft(f);
  
  cout << f << endl;
}
