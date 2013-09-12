#include "fftw++.h"
#include "seconds.h"

// Compile with:
// g++ -I .. -fopenmp example0m.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace fftwpp;

int main()
{  
  cout << "Multiple 1D complex to complex in-place FFTs" << endl;

  fftw::maxthreads=4;
  unsigned int T=1;
  
  unsigned int M=10;
  unsigned int n=8;
  unsigned int N=1;

  Complex *f=ComplexAlign(n*M);
  unsigned int K=M/T;
    
  mfft1d Forward(n,-1,K,M,1);
  mfft1d Backward(n,1,K,M,1);

  for(unsigned int i=0; i < n*M; i++) f[i]=i;

  unsigned int outlimit=100;
  
  if(n*M < outlimit) {
    cout << endl << "input:" << endl;
    for(unsigned int i=0; i < n; i++) {
      for(unsigned int j=0; j < M; j++) {
        Complex value=n*j+i;
        f[M*i+j]=value;
        cout << value << "\t";
      }
      cout << endl;
    }
  }

  // Timing test:
  seconds();
  
  for(unsigned int j=0; j < N; ++j) {
#pragma omp parallel for num_threads(T)
    for(unsigned int i=0; i < T; ++i)
      Forward.fft(f+K*i);
  
#pragma omp parallel for num_threads(T)
    for(unsigned int i=0; i < T; ++i)
      Backward.fftNormalized(f+K*i);
  }
  
  double time=seconds();
  
  if(n*M < outlimit) {
    cout << endl << "back to input:" << endl;
    for(unsigned int i=0; i < n; i++) {
      for(unsigned int j=0; j < M; j++) {
        cout << f[M*i+j] << "\t";
      }
      cout << endl;
    }
  }

  cout << endl << "average seconds: " << time/N << endl;

  Complex sum=0.0;
  for(unsigned int i=0; i < n*M; i++) 
    sum += f[i];

  cout << endl << "sum of outputs (used for error-checking): " << sum << endl;
  
  deleteAlign(f);
}
