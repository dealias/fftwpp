#include "fftw++.h"
#include "seconds.h"

// Compile with:
// g++ -I .. -fopenmp example0m.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace fftwpp;

int main()
{
  fftw::maxthreads=4;
  unsigned int T=1;
  
  unsigned int M=8192;
  unsigned int n=8192;
  unsigned int N=10;

  Complex *f=ComplexAlign(n*M);
  unsigned int K=M/T;
    
  mfft1d Forward(n,-1,K,M,1);
  mfft1d Backward(n,1,K,M,1);
  
  for(unsigned int i=0; i < n*M; i++) f[i]=i;
	
  seconds();
  
  for(unsigned int j=0; j < N; ++j) {
#pragma omp parallel for num_threads(T)
    for(unsigned int i=0; i < T; ++i)
      Forward.fft(f+K*i);
  
#pragma omp parallel for num_threads(T)
    for(unsigned int i=0; i < T; ++i)
      Backward.fftNormalized(f+K*i);
  }
	
  cout << seconds() << endl;
  
  Complex sum=0.0;
  for(unsigned int i=0; i < n*M; i++) 
    sum += f[i];
  
  cout << sum << endl;
  
  deleteAlign(f);
}
