#include "convolve.h"
#include "timing.h"
#include "direct.h"

#define CENTERED 1

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

unsigned int A=2; // number of inputs
unsigned int B=1; // number of outputs
unsigned int L=512; // input data length
unsigned int M=1024; // minimum padded length

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv);

  unsigned int K0=100000000;
  if(K == 0) K=max(K0/M,20);
  cout << "K=" << K << endl << endl;

  double *T=new double[K];

  ForwardBackward FB(A,B,multbinary);
#if CENTERED
  fftPadCentered fft(L,M,FB);
#else
  fftPad fft(L,M,FB);
#endif

  unsigned int N=max(A,B);
  Complex **f=new Complex *[N];
  unsigned int size=fft.embed() ? fft.outputSize() : fft.inputSize();
  Complex *F=ComplexAlign(N*size);
  for(unsigned int a=0; a < A; ++a)
    f[a]=F+a*size;

  for(unsigned int a=0; a < A; ++a) {
    Complex *fa=f[a];
    for(unsigned int j=0; j < L; ++j) {
        fa[j]=Output ? Complex(j,(1.0+a)*j+1) : 0.0;
    }
  }

  //DirectConvolution C(L);
  //Complex *h=ComplexAlign(L);
  //C.convolve(h,f[0],f[1]);

  Convolution Convolve(&fft,A,B,fft.embed() ? F : NULL);

  if(Output)
    K=1;
  for(unsigned int k=0; k < K; ++k) {
    seconds();
    Convolve.convolve(f);
    T[k]=seconds();
  }

  cout << endl;
  timings("Hybrid",L,T,K,stats);
  cout << endl;

  if(Output)
    for(unsigned int b=0; b < B; ++b)
      for(unsigned int j=0; j < L; ++j)
        cout << f[b][j] << endl;//" " << h[j] << endl; // Assumes B=2

  //delete h;
  delete [] T;

  return 0;
}
