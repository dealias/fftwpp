#include "convolve.h"

#define OUTPUT 0
#define CENTERED 0

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

unsigned int A=2; // number of inputs
unsigned int B=1; // number of outputs
unsigned int C=1; // number of copies
unsigned int S=0; // stride between copies (0 means C)
unsigned int L=512; // input data length
unsigned int M=1024; // minimum padded length

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv);

  if(S == 0) S=C;

  ForwardBackward FB(A,B);

  cout << "Explicit:" << endl;
  // Minimal explicit padding
#if CENTERED
  fftPadCentered fft0(L,M,FB,C,S,true,true);
#else
  fftPad fft0(L,M,FB,C,S,true,true);
#endif

  double mean0=fft0.report(FB);

  // Optimal explicit padding
#if CENTERED
  fftPadCentered fft1(L,M,FB,C,S,true,false);
#else
  fftPad fft1(L,M,FB,C,S,true,false);
#endif
  double mean1=min(mean0,fft1.report(FB));

  // Hybrid padding
#if CENTERED
  fftPadCentered fft(L,M,FB,C,S);
#else
  fftPad fft(L,M,FB,C,S);
#endif

  double mean=fft.report(FB);

  if(mean0 > 0)
    cout << "minimal ratio=" << mean/mean0 << endl;
  cout << endl;

  if(mean1 > 0)
    cout << "optimal ratio=" << mean/mean1 << endl;

  Complex *f=ComplexAlign(fft.inputSize());
  Complex *h=ComplexAlign(fft.inputSize());
  Complex *F=ComplexAlign(fft.outputSize());
  Complex *W0=ComplexAlign(fft.workSizeW());

  for(unsigned int j=0; j < L; ++j)
    for(unsigned int c=0; c < C; ++c)
      f[S*j+c]=Complex(j+1+c,j+2+c);

#if CENTERED
  fftPadCentered fft2(L,fft.M,C,S,fft.M,1,1,fftw::maxthreads,fft.q == 1);
#else
  fftPad fft2(L,fft.M,C,S,fft.M,1,1);
#endif

  Complex *F2=ComplexAlign(fft2.outputSize());

  fft2.forward(f,F2);

  fft.pad(W0);
  double error=0.0, error2=0.0;
  double norm=0.0, norm2=0.0;
  for(unsigned int r=0; r < fft.R; r += fft.increment(r)) {
    fft.forward(f,F,r,W0);
    for(unsigned int k=0; k < fft.noutputs(r); ++k) {
#if OUTPUT
      if(k%fft.m == 0) cout << endl;
#endif
      for(unsigned int c=0; c < C; ++c) {
        unsigned int K=S*k+c;
        unsigned int i=fft.index(r,K);
        error += abs2(F[K]-F2[i]);
        norm += abs2(F2[i]);
#if OUTPUT
        cout << i << ": " << F[K] << endl;
#endif
      }
    }
    fft.backward(F,h,r,W0);
  }

#if OUTPUT
  cout << endl;
  for(unsigned int j=0; j < fft2.noutputs(); ++j)
    for(unsigned int c=0; c < C; ++c) {
      unsigned int J=S*j+c;
      cout << J << ": " << F2[J] << endl;
    }
#endif

  double scale=1.0/fft.normalization();

  if(L < 30) {
    cout << endl;
    cout << "Inverse:" << endl;
    for(unsigned int j=0; j < L; ++j) {
      for(unsigned int c=0; c < C; ++c) {
        unsigned int J=S*j+c;
        cout << h[J]*scale << endl;
      }
    }
    cout << endl;
  }

  for(unsigned int r=0; r < fft.R; r += fft.increment(r)) {
    for(unsigned int k=0; k < fft.noutputs(r); ++k) {
      for(unsigned int c=0; c < C; ++c) {
        unsigned int K=S*k+c;
        F[K]=F2[fft.index(r,K)];
      }
    }
    fft.backward(F,h,r,W0);
  }

  for(unsigned int j=0; j < L; ++j) {
    for(unsigned int c=0; c < C; ++c) {
      unsigned int J=S*j+c;
      error2 += abs2(h[J]*scale-f[J]);
      norm2 += abs2(f[J]);
    }
  }

  if(norm > 0) error=sqrt(error/norm);
  if(norm2 > 0) error2=sqrt(error2/norm2);

  double eps=1e-12;
  if(error > eps || error2 > eps)
    cerr << endl << "WARNING: " << endl;
  cout << endl;
  cout << "forward error=" << error << endl;
  cout << "backward error=" << error2 << endl;

  return 0;
}
