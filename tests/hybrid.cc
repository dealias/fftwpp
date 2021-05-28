#include "convolve.h"

#define OUTPUT 1
#define CENTERED 0

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

  cout << "Explicit:" << endl;
  // Minimal explicit padding
#if CENTERED
  fftPadCentered fft0(L,M,FB,C,true,true);
#else
  fftPad fft0(L,M,FB,C,true,true);
#endif

  double mean0=fft0.report(FB);

  // Optimal explicit padding
#if CENTERED
  fftPadCentered fft1(L,M,FB,C,true,false);
#else
  fftPad fft1(L,M,FB,C,true,false);
#endif
  double mean1=min(mean0,fft1.report(FB));

  // Hybrid padding
#if CENTERED
  fftPadCentered fft(L,M,FB,C);
#else
  fftPad fft(L,M,FB,C);
#endif

  double mean=fft.report(FB);

  if(mean0 > 0)
    cout << "minimal ratio=" << mean/mean0 << endl;
  cout << endl;

  if(mean1 > 0)
    cout << "optimal ratio=" << mean/mean1 << endl;

  Complex *f=ComplexAlign(C*L);
  Complex *h=ComplexAlign(C*L);
  Complex *F=ComplexAlign(fft.fullOutputSize());
  Complex *W0=ComplexAlign(fft.workSizeW());

  unsigned int Length=L;

  for(unsigned int j=0; j < Length; ++j)
    for(unsigned int c=0; c < C; ++c)
      f[C*j+c]=Complex(j+1+c,j+2+c);

#if CENTERED
  fftPadCentered fft2(L,fft.M,C,fft.M,1,1);
#else
  fftPad fft2(L,fft.M,C,fft.M,1,1);
#endif

  Complex *F2=ComplexAlign(fft2.noutputs());

  fft2.forward(f,F2);

  fft.pad(W0);
  double error=0.0, error2=0.0;
  double norm=0.0, norm2=0.0;
  for(unsigned int r=0; r < fft.R; r += fft.increment(r)) {
    fft.forward(f,F,r,W0);
#if OUTPUT
//    cout << endl << "r=" << r << endl;
    for(unsigned int k=0; k < fft.noutputs(); ++k) {
      if(k % fft.Cm == 0) cout << endl;
//      cout << "index=" << fft.index(r,k) << endl;
      unsigned int i=fft.index(r,k);
      error += abs2(F[k]-F2[i]);
      norm += abs2(F2[i]);
      cout << F[k] << endl;
    }
#endif
    fft.backward(F,h,r,W0);
  }

  cout << endl;
  for(unsigned int j=0; j < fft2.noutputs(); ++j)
    cout << F2[j] << endl;

  double scale=1.0/fft.normalization();

  if(L < 30) {
    cout << endl;
    cout << "Inverse:" << endl;
    for(unsigned int j=0; j < fft.ninputs(); ++j)
      cout << h[j]*scale << endl;
    cout << endl;
  }

  for(unsigned int j=0; j < fft2.noutputs(); ++j)
    F[j]=F2[j];

  for(unsigned int r=0; r < fft.R; r += fft.increment(r)) {
    for(unsigned int k=0; k < fft.noutputs(); ++k)
      F[k]=F2[fft.index(r,k)];
    fft.backward(F,h,r,W0);
  }

  for(unsigned int j=0; j < fft.ninputs(); ++j) {
    error2 += abs2(h[j]*scale-f[j]);
    norm2 += abs2(f[j]);
  }

  if(norm > 0) error=sqrt(error/norm);
  if(norm2 > 0) error2=sqrt(error2/norm2);

  double eps=1e-12;
  if(error > eps || error2 > eps)
    cerr << endl << "WARNING: " << endl;
  cout << "forward error=" << error << endl;
  cout << "backward error=" << error2 << endl;

#if 0
  unsigned int m=fft.m;
  unsigned int p=fft.b/(C*m); // effective value
  unsigned int n=fft.n;
  unsigned int q=fft.q;

  for(unsigned int s=0; s < m; ++s) {
    for(unsigned int t=0; t < p; ++t) {
      for(unsigned int r=0; r < n; ++r) {
        unsigned int R=r,S=s; // FIXME for inner loop.
        if(fft.D > 1 && fft.p <= 2) {
          R=fft.residue(r,q);
          if(fft.p == 1 ? R > q/2 : R >= ceilquotient(q,2))
            S=s > 0 ? s-1 : m-1;
        }

        for(unsigned int c=0; c < C; ++c) {
          unsigned int i=C*(n*(p*S+t)+R)+c;
          error += abs2(F[C*(m*(p*r+t)+s)+c]-F2[i]);
          norm += abs2(F2[i]);
        }
      }
    }
#endif

  return 0;
}
