#include "Complex.h"
#include "convolve.h"

// Compile with:
// g++ -I .. -Ofast -fopenmp exampleconvr.cc ../convolve.cc ../fftw++.cc ../parallel.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace fftwpp;
using namespace parallel;

void init(double *f, double *g, size_t L)
{
  for(size_t i=0; i < L; ++i) {
    f[i]=i+1;
    g[i]=i+2;
  }
}

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  size_t A=2; // Two inputs
  size_t B=1; // One output

  size_t L=8; // Length of input arrays
  size_t M=15; // Minimal padded length for dealiasing via 1/2 padding

  // allocate arrays:
  double *f=doubleAlign(L);
  double *g=doubleAlign(L);
  double *F[]={f,g};

  cout << "1D real convolution:" << endl;
  init(f,g,L);

  cout << "\ninput:\nf\tg" << endl;
  for(size_t i=0; i < L; i++)
    cout << f[i] << "\t" << g[i] << endl;

  Application app(A,B,multBinary,fftw::maxthreads);
  fftPadReal fft(L,M,app);
  Convolution C(&fft);

  C.convolve((Complex **) F);

  cout << "\noutput:" << endl;
  for(size_t i=0; i < L; i++)
    cout << f[i] << endl;

  deleteAlign(g);
  deleteAlign(f);

  return 0;
}
