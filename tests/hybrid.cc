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

  cout << "Explicit:" << endl;
  // Minimal explicit padding
  fftPad fft0(L,M,*app,C,true,true);

  double mean0=fft0.report(*app);

  // Optimal explicit padding
  fftPad fft1(L,M,*app,C,true,false);
  double mean1=min(mean0,fft1.report(*app));

  // Hybrid padding
  fftPad fft(L,M,*app,C);
//  fftPadCentered fft(L,M,*app,C);

  double mean=fft.report(*app);

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

#if 0
  for(unsigned int j=0; j < fft.outputs(); ++j)
    cout << F[j] << endl;
#endif

  Complex *f0=ComplexAlign(C*L);
  Complex *F0=ComplexAlign(fft.fullOutputSize());

  for(unsigned int j=0; j < fft.outputs(); ++j)
    F0[j]=F[j];

  fft.backward(F0,f0);

  double scale=1.0/fft.normalization();

  if(L < 30) {
    cout << endl;
    cout << "Inverse:" << endl;
    for(unsigned int j=0; j < C*L; ++j)
      cout << f0[j]*scale << endl;
    cout << endl;
  }

  fftPad fft2(L,fft.M,C,fft.M,1,1);
  Complex *F2=ComplexAlign(fft2.fullOutputSize());

  for(unsigned int j=0; j < L; ++j)
    for(unsigned int c=0; c < C; ++c)
      f[C*j+c]=Complex(j+1,j+2);
  fft2.forward(f,F2);

#if 0
  cout << endl;
  for(unsigned int j=0; j < fft.outputs(); ++j)
    cout << F2[j] << endl;
  cout << endl;
#endif

  double error=0.0, norm=0.0;
  double error2=0.0, norm2=0.0;

  unsigned int m=fft.m;
  unsigned int p=fft.b/(C*m);
  unsigned int n=fft.n;

  for(unsigned int s=0; s < m; ++s) {
    for(unsigned int t=0; t < p; ++t) {
      for(unsigned int r=0; r < n; ++r) {
        for(unsigned int c=0; c < C; ++c) {
          unsigned int i=C*(n*(p*s+t)+r)+c;
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
