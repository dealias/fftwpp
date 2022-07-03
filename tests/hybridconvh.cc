#include "convolve.h"

#define OUTPUT 1

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

unsigned int A=2; // number of inputs
unsigned int B=1; // number of outputs
unsigned int C=1; // number of copies
unsigned int S=0; // strides not implemented for Hermitian convolutions
unsigned int L=512; // input data length
unsigned int M=768; // minimum padded length

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv);

  ForwardBackward FB(A,B);

  fftPadHermitian fft(L,M,FB,C,S);

  if(S == 0) S=C;

  unsigned int H=ceilquotient(L,2);

  Complex **f=new Complex *[max(A,B)];
  for(unsigned int a=0; a < A; ++a)
    f[a]=ComplexAlign(fft.inputSize());

  for(unsigned int a=0; a < A; ++a) {
    Complex *fa=f[a];
#if OUTPUT
    for(unsigned int c=0; c < C; ++c)
      fa[c]=1.0+a;
    for(unsigned int j=1; j < H; ++j) {
      for(unsigned int c=0; c < C; ++c)
        fa[S*j+c]=Complex(j,(1.0+a)*j+1);
    }
#else
    for(unsigned int j=0; j < H; ++j) {
      for(unsigned int c=0; c < C; ++c)
        fa[S*j+c]=0.0;
    }
#endif
  }

  ConvolutionHermitian Convolve(&fft,A,B);

#if OUTPUT
  K=1;
#endif
  double t0=totalseconds();

  for(unsigned int k=0; k < K; ++k)
    Convolve.convolve(f,realmultbinary);

  double t=totalseconds();
  cout << (t-t0)/K << endl;
  cout << endl;
#if OUTPUT
  for(unsigned int b=0; b < B; ++b)
    for(unsigned int j=0; j < H; ++j)
      for(unsigned int c=0; c < C; ++c)
        cout << f[b][S*j+c] << endl;
#endif

  return 0;
}
