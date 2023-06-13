
#include "fftw++.h"
#include "seconds.h"

// Compile with:
// g++ -I .. -fopenmp example0m.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace fftwpp;
using namespace parallel;

int main()
{
  cout << "Multiple 1D complex to complex in-place FFTs" << endl;

  fftw::maxthreads=4;

  size_t M=10; // Number of FFTs performed
  size_t n=8; // Length of FFT
  size_t N=10; // Number data points for timing test.

  Complex *f=ComplexAlign(n*M);

  mfft1d Forward(n,-1,M,M,1);
  mfft1d Backward(n,1,M,M,1);

  for(size_t i=0; i < n*M; i++) f[i]=i;

  size_t outlimit=100;

  if(n*M < outlimit) {
    cout << endl << "input:" << endl;
    for(size_t i=0; i < n; i++) {
      for(size_t j=0; j < M; j++) {
        Complex value=n*j+i;
        f[M*i+j]=value;
        cout << value << "\t";
      }
      cout << endl;
    }
  }

  // Timing test:
  cpuTimer c;

  for(size_t j=0; j < N; ++j) {
    Forward.fft(f);
    Backward.fftNormalized(f);
  }

  double time=c.seconds();

  if(n*M < outlimit) {
    cout << endl << "back to input:" << endl;
    for(size_t i=0; i < n; i++) {
      for(size_t j=0; j < M; j++) {
        cout << f[M*i+j] << "\t";
      }
      cout << endl;
    }
  }

  cout << endl << "average seconds: " << time/N << endl;

  Complex sum=0.0;
  for(size_t i=0; i < n*M; i++)
    sum += f[i];

  cout << endl << "sum of outputs (used for error-checking): " << sum << endl;

  deleteAlign(f);
}
