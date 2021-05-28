#include "convolve.h"

#define OUTPUT 1
#define CENTERED 1

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
  cout << endl;

  Complex *f=ComplexAlign(C*L);
  Complex *F=ComplexAlign(fft.fullOutputSize());
  fft.W0=ComplexAlign(fft.workSizeW());

  unsigned int Length=L;

  for(unsigned int j=0; j < Length; ++j)
    for(unsigned int c=0; c < C; ++c)
      f[C*j+c]=Complex(j+1,j+2);

  fft.forward(f,F);

#if OUTPUT
  for(unsigned int j=0; j < fft.fullOutputSize(); ++j) {
    if(j % fft.Cm == 0) cout << endl;
    cout << F[j] << endl;
  }
#endif

  Complex *f0=ComplexAlign(C*L);
  Complex *F0=ComplexAlign(fft.fullOutputSize());

  for(unsigned int j=0; j < fft.fullOutputSize(); ++j)
    F0[j]=F[j];

#if 0
  cout << endl;
  for(unsigned int j=0; j < fft.fullOutputSize(); ++j)
    cout << F0[j] << endl;
  cout << endl;
#endif

  fft.backward(F0,f0);

  double scale=1.0/fft.normalization();

  if(L < 30) {
    cout << endl;
    cout << "Inverse:" << endl;
    for(unsigned int j=0; j < C*L; ++j)
      cout << f0[j]*scale << endl;
    cout << endl;
  }

#if CENTERED
  fftPadCentered fft2(L,fft.M,C,fft.M,1,1);
#else
  fftPad fft2(L,fft.M,C,fft.M,1,1);
#endif

  Complex *F2=ComplexAlign(fft2.fullOutputSize());

  for(unsigned int j=0; j < L; ++j)
    for(unsigned int c=0; c < C; ++c)
      f[C*j+c]=Complex(j+1,j+2);
  fft2.forward(f,F2);

#if 0
  cout << endl;
  for(unsigned int j=0; j < fft.fullOutputSize(); ++j)
    cout << F2[j] << endl;
  cout << endl;
#endif

  double error=0.0, norm=0.0;
  double error2=0.0, norm2=0.0;

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
  }

  for(unsigned int j=0; j < C*L; ++j) {
    error2 += abs2(f0[j]*scale-f[j]);
    norm2 += abs2(f[j]);
  }

  if(norm > 0) error=sqrt(error/norm);
  double eps=1e-12;
  if(error > eps || error2 > eps)
    cerr << endl << "WARNING: " << endl;
  cout << "forward error=" << error << endl;
  cout << "backward error=" << error2 << endl;

  return 0;
}
