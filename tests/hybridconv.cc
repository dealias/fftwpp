#include "convolve.h"

#define OUTPUT 0

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

int main(int argc, char* argv[])
{
  fftw::maxthreads=1;//get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  L=512;
  M=1024;

  optionsHybrid(argc,argv);

  ForwardBackward FB;
  Application *app=&FB;

  fftPad fft(L,M,*app,C);

  Complex *f=ComplexAlign(C*L);
  Complex *g=ComplexAlign(C*L);

  for(unsigned int j=0; j < L; ++j) {
    for(unsigned int c=0; c < C; ++c) {
#if OUTPUT
      f[C*j+c]=Complex(j,j+1);
      g[C*j+c]=Complex(j,2*j+1);
#else
      f[C*j+c]=0.0;
      g[C*j+c]=0.0;
#endif
    }
  }

  Convolution Convolve(fft);

  Complex *F[]={f,g};
//  Complex *h=ComplexAlign(L);
//  Complex *H[]={h};
#if OUTPUT
  unsigned int K=1;
#else
  unsigned int K=10000;
#endif
  double t0=totalseconds();

  for(unsigned int k=0; k < K; ++k)
    Convolve.convolve(F,F,multbinary);

  double t=totalseconds();
  cout << (t-t0)/K << endl;
  cout << endl;
#if OUTPUT
  for(unsigned int j=0; j < L; ++j)
    for(unsigned int c=0; c < C; ++c)
      cout << F[0][C*j+c] << endl;
#endif

  return 0;
}
