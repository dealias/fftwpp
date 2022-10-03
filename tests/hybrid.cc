#include "convolve.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

unsigned int A=2; // number of inputs
unsigned int B=1; // number of outputs

int main(int argc, char* argv[])
{
  L=512; // input data length
  M=1024; // minimum padded length

  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv,true);

  cout << "L=" << L << endl;
  cout << "M=" << M << endl;

  if(S == 0) S=C;

  Application app(A,B,multbinary,fftw::maxthreads,0,mx,Dx,Ix);

  cout << "Explicit:" << endl;
  // Minimal explicit padding
#if Centered
  fftPadCentered fft0(L,M,C,S,M,1,1,1,app.mult);
#else
  fftPad fft0(L,M,C,S,M,1,1,1,app.mult);
#endif

  double median0=fft0.report(app);

  // Optimal explicit padding
#if Centered
  fftPadCentered fft1(L,M,app,C,S,true);
#else
  fftPad fft1(L,M,app,C,S,true);
#endif
  double median1=min(median0,fft1.report(app));

  cout << endl;
  cout << "Hybrid:" << endl;

  // Hybrid padding
#if Centered
  fftPadCentered fft(L,M,app,C,S);
#else
  fftPad fft(L,M,app,C,S);
#endif

  double median=fft.report(app);

  if(median0 > 0)
    cout << "minimal ratio=" << median/median0 << endl;
  cout << endl;

  if(median1 > 0)
    cout << "optimal ratio=" << median/median1 << endl;

  Complex *f=ComplexAlign(fft.inputSize());
  Complex *h=ComplexAlign(fft.inputSize());
  Complex *F=ComplexAlign(fft.outputSize());
  Complex *W0=ComplexAlign(fft.workSizeW());

  for(unsigned int j=0; j < L; ++j)
    for(unsigned int c=0; c < C; ++c)
      f[S*j+c]=Complex(j+1+c,j+2+c);

#if Centered
  fftPadCentered fft2(L,fft.M,C,S,fft.M,1,1,1,app.mult,
                      fftw::maxthreads,fft.q == 1);
#else
  fftPad fft2(L,fft.M,C,S,fft.M,1,1,1,app.mult);
#endif

  Complex *F2=ComplexAlign(fft2.outputSize());

  fft2.forward(f,F2);

  fft.pad(W0);
  double error=0.0, error2=0.0;
  double norm=0.0, norm2=0.0;
  for(unsigned int r=0; r < fft.R; r += fft.increment(r)) {
    fft.forward(f,F,r,W0);
    for(unsigned int k=0; k < fft.noutputs(r); ++k) {
      if(Output && k%fft.m == 0) cout << endl;

      for(unsigned int c=0; c < C; ++c) {
        unsigned int K=S*k+c;
        unsigned int i=fft.index(r,K);
        error += abs2(F[K]-F2[i]);
        norm += abs2(F2[i]);
        if(Output)
          cout << i << ": " << F[K] << endl;
      }
    }
    fft.backward(F,h,r,W0);
  }

  if(Output) {
    cout << endl;
    for(unsigned int j=0; j < fft2.noutputs(); ++j)
      for(unsigned int c=0; c < C; ++c) {
        unsigned int J=S*j+c;
        cout << J << ": " << F2[J] << endl;
      }
  }

  double scale=1.0/fft.normalization();

  if(Output) {
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
  cout << "Forward Error: " << error << endl;
  cout << "Backward Error: " << error2 << endl;

  return 0;
}
