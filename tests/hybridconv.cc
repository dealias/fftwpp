#include "convolve.h"
#include "timing.h"

#define OUTPUT 0
#define CENTERED 0

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

unsigned int A=2; // number of inputs
unsigned int B=1; // number of outputs
unsigned int L=512; // input data length
unsigned int M=1024; // minimum padded length

int stats=MEAN; // Type of statistics used in timing test

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv);

  ForwardBackward FB(A,B);
#if CENTERED
  fftPadCentered fft(L,M,FB);
#else
  fftPad fft(L,M,FB);
#endif

  double *T=new double[K];

  Complex **f=new Complex *[max(A,B)];
  for(unsigned int a=0; a < A; ++a)
    f[a]=ComplexAlign(fft.inputSize());

  for(unsigned int a=0; a < A; ++a) {
    Complex *fa=f[a];
    for(unsigned int j=0; j < L; ++j) {
#if OUTPUT
        fa[j]=Complex(j,(1.0+a)*j+1);
#else
        fa[j]=0.0;
#endif
    }
  }

  Convolution Convolve(&fft,A,B);

#if OUTPUT
  K=1;
#endif
  for(unsigned int k=0; k < K; ++k) {
    seconds();
    Convolve.convolve(f,multbinary);
    T[k]=seconds();
  }

  cout << endl;
  timings("Hybrid",L,T,K,stats);

  cout << endl;
#if OUTPUT
  for(unsigned int b=0; b < B; ++b)
    for(unsigned int j=0; j < L; ++j)
        cout << f[b][j] << endl;
#endif

  delete [] T;

  return 0;
}
