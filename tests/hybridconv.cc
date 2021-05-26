#include "convolve.h"

#define OUTPUT 0

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

unsigned int A=2; // number of inputs
unsigned int B=1; // number of outputs
unsigned int C=1; // number of copies
unsigned int L=512; // input data length
unsigned int M=1024; // minimum padded length

int main(int argc, char* argv[])
{
  fftw::maxthreads=1;//get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv);

  ForwardBackward FB(A,B);
//  fftPad fft(L,M,FB,C);
  fftPadCentered fft(L,M,FB,C);

  Complex **f=new Complex *[max(A,B)];
  for(unsigned int a=0; a < A; ++a)
    f[a]=ComplexAlign(C*L);

  for(unsigned int a=0; a < A; ++a) {
    Complex *fa=f[a];
    for(unsigned int j=0; j < L; ++j) {
      for(unsigned int c=0; c < C; ++c) {
#if OUTPUT
        fa[C*j+c]=Complex(j,(1.0+a)*j+1);
#else
        fa[C*j+c]=0.0;
#endif
      }
    }
  }

  Convolution Convolve(fft,A,B);

#if OUTPUT
  unsigned int K=1;
#else
  unsigned int K=100000;
#endif
  double t0=totalseconds();

  for(unsigned int k=0; k < K; ++k)
    Convolve.convolve(f,multbinary);

  double t=totalseconds();
  cout << (t-t0)/K << endl;
  cout << endl;
#if OUTPUT
  for(unsigned int b=0; b < B; ++b)
    for(unsigned int j=0; j < L; ++j)
      for(unsigned int c=0; c < C; ++c)
        cout << f[b][C*j+c] << endl;
#endif

  return 0;
}