#include "convolve.h"

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

  fftPad fft(L,M,*app);

  unsigned int L0=fft.bufferLength();
  Complex *f=ComplexAlign(L0);
  Complex *g=ComplexAlign(L0);

  for(unsigned int j=0; j < L; ++j) {
#if OUTPUT
    f[j]=Complex(j,j+1);
    g[j]=Complex(j,2*j+1);
#else
    f[j]=0.0;
    g[j]=0.0;
#endif
  }

  Convolution Convolve(fft);

  Complex *F[]={f,g};
//  Complex *h=ComplexAlign(L0);
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
    cout << F[0][j] << endl;
#endif

  return 0;
}
