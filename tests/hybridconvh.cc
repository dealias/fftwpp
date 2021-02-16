#include "convolve.h"

#define OUTPUT 1

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
  M=768;

  optionsHybrid(argc,argv);

  ForwardBackward FB;
  Application *app=&FB;

  fftPadHermitian fft(L,M,*app);

  unsigned int H=ceilquotient(L,2);
  
  Complex *f=ComplexAlign(H);
  Complex *g=ComplexAlign(H);

#if OUTPUT
  f[0]=1.0;
  g[0]=2.0;
  for(unsigned int j=1; j < H; ++j) {
    f[j]=Complex(j,j+1);
    g[j]=Complex(j,2*j+1);
  }
#else
  for(unsigned int j=0; j < H; ++j) {
    f[j]=0.0;
    g[j]=0.0;
  }
#endif

  ConvolutionHermitian Convolve(fft);

  Complex *F[]={f,g};
//  Complex *h=ComplexAlign(H);
//  Complex *H[]={h};
#if OUTPUT
  unsigned int K=1;
#else
  unsigned int K=10000;
#endif
  double t0=totalseconds();

  for(unsigned int k=0; k < K; ++k)
    Convolve.convolve(F,F,realmultbinary);

  double t=totalseconds();
  cout << (t-t0)/K << endl;
  cout << endl;
#if OUTPUT
  for(unsigned int j=0; j < H; ++j)
    cout << F[0][j] << endl;
#endif

  return 0;
}
