#include "fftw++.h"
#include "seconds.h"

// Compile with:
// g++ -I .. -fopenmp example0m.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace fftwpp;

int main()
{
  fftw::maxthreads=1;
  unsigned int T=4;
  
  unsigned int M=4096;
  unsigned int n=4096; 
  Complex *f=ComplexAlign(n*M);
  unsigned int K=M/T;
    
  mfft1d Forward(n,-1,K,M,1);
  mfft1d Backward(n,1,K,M,1);
  
  for(unsigned int i=0; i < n*M; i++) f[i]=i;
	
  seconds();
  
  unsigned int N=100;
  
  for(int j=0; j < N; ++j) {
#pragma omp parallel for num_threads(T)
  for(int i=0; i < T; ++i)
    Forward.fft(f+i*K);
  
#pragma omp parallel for num_threads(T)
  for(int i=0; i < T; ++i)
    Backward.fftNormalized(f+i*K);
  }
	
//  for(unsigned int i=0; i < n; i++) cout << f[i] << endl;
  cout << seconds() << endl;
  
  FFTWdelete(f);
}
