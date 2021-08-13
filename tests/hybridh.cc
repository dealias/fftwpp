#include "convolve.h"

#define OUTPUT 1

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

unsigned int A=2; // number of inputs
unsigned int B=1; // number of outputs
unsigned int C=1; // number of copies
unsigned int S=0; // stride between copies (0 means C)
unsigned int L=512; // input data length
unsigned int M=768; // minimum padded length

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
  fftPadHermitian fft0(L,M,FB,C,S,true,true);

  double mean0=fft0.report(FB);

  // Optimal explicit padding
  fftPadHermitian fft1(L,M,FB,C,S,true,false);
  double mean1=min(mean0,fft1.report(FB));

  // Hybrid padding
  fftPadHermitian fft(L,M,FB,C,S);

  double mean=fft.report(FB);

  if(mean0 > 0)
    cout << "minimal ratio=" << mean/mean0 << endl;
  cout << endl;

  if(mean1 > 0)
    cout << "optimal ratio=" << mean/mean1 << endl;
  cout << endl;

  Complex *f=ComplexAlign(fft.inputSize());
  Complex *h=ComplexAlign(fft.inputSize());
  Complex *F=ComplexAlign(fft.outputSize());
  Complex *W0=ComplexAlign(fft.workSizeW());

  unsigned int H=ceilquotient(L,2);

  for(unsigned int c=0; c < C; ++c)
    f[c]=1+c;
  for(unsigned int j=1; j < H; ++j)
    for(unsigned int c=0; c < C; ++c)
      f[S*j+c]=Complex(j+1+c,j+2+c);

  fftPadHermitian fft2(L,fft.M,C,S,fft.M,1,1);

  Complex *F2=ComplexAlign(fft2.outputSize());
  double *F2r=(double *) F2;

  fft2.forward(f,F2);

  fft.pad(W0);
  double error=0.0, error2=0.0;
  double norm=0.0, norm2=0.0;
  unsigned int noutputs=fft.noutputs();
  unsigned int Snoutputs=S*noutputs;
  for(unsigned int r=0; r < fft.R; r += fft.increment(r)) {
    fft.forward(f,F,r,W0);
    unsigned int D1=r == 0 ? fft.D0 : fft.D;
    for(unsigned int d=0; d < D1; ++d) {
      double *Fr=(double *) (F+fft.b*d);
      unsigned int offset=Snoutputs*d;
      for(unsigned int k=0; k < noutputs; ++k) {
#if OUTPUT
        if(k%fft.m == 0) cout << endl;
#endif
        for(unsigned int c=0; c < C; ++c) {
          unsigned int K=S*k+c;
          unsigned int i=fft.index(r,K+offset);
          error += abs2(Fr[K]-F2r[i]);
          norm += abs2(F2r[i]);
#if OUTPUT
          cout << i << ": " << Fr[K] << endl;
#endif
        }
      }
    }
    fft.backward(F,h,r,W0);
  }

#if OUTPUT
  cout << endl;
  for(unsigned int j=0; j < Snoutputs; j += S)
    for(unsigned int c=0; c < C; ++c) {
      unsigned int J=j+c;
      cout << J << ": " << F2r[J] << endl;
    }
#endif

  double scale=1.0/fft.normalization();

  if(L < 30) {
    cout << endl;
    cout << "Inverse:" << endl;
    for(unsigned int j=0; j < H; ++j)
      for(unsigned int c=0; c < C; ++c) {
        unsigned int J=S*j+c;
        cout << h[J]*scale << endl;
      }
    cout << endl;
  }

  for(unsigned int r=0; r < fft.R; r += fft.increment(r)) {
    unsigned int D1=r == 0 ? fft.D0 : fft.D;
    for(unsigned int d=0; d < D1; ++d) {
      double *Fr=(double *) (F+fft.b*d);
      unsigned int offset=Snoutputs*d;
      for(unsigned int k=0; k < Snoutputs; k += S) {
        for(unsigned int c=0; c < C; ++c) {
          unsigned int K=k+c;
          Fr[K]=F2r[fft.index(r,K+offset)];
        }
      }
    }
    fft.backward(F,h,r,W0);
  }

    for(unsigned int j=0; j < H; ++j) {
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
