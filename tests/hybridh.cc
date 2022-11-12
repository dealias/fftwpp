#include "convolve.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

unsigned int A=2; // number of inputs
unsigned int B=1; // number of outputs

int main(int argc, char* argv[])
{
  L=511; // input data length
  M=766; // minimum padded length

  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv,true);

  cout << "L=" << L << endl;
  cout << "M=" << M << endl;

  Application app(A,B,realmultbinary,fftw::maxthreads,0,mx,Dx,Ix);

  cout << "Explicit:" << endl;
  // Minimal explicit padding
  fftPadHermitian fft0(L,M,C,M,1,1,1,app.mult);

  double median0=fft0.report(app);

  // Optimal explicit padding
  fftPadHermitian fft1(L,M,app,C,true);
  double median1=min(median0,fft1.report(app));

  cout << endl;
  cout << "Hybrid:" << endl;

  // Hybrid padding
  fftPadHermitian fft(L,M,app,C);

  double median=fft.report(app);

  if(median0 > 0)
    cout << "minimal ratio=" << median/median0 << endl;
  cout << endl;

  if(median1 > 0)
    cout << "optimal ratio=" << median/median1 << endl;
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
      f[C*j+c]=Complex(j+1+c,j+2+c);

  fftPadHermitian fft2(L,fft.M,C,fft.M,1,1,1,app.mult);

  Complex *F2=ComplexAlign(fft2.outputSize());
  double *F2r=(double *) F2;

  fft2.forward(f,F2);

  fft.pad(W0);
  double error=0.0, error2=0.0;
  double norm=0.0, norm2=0.0;
  unsigned int noutputs=C*fft.noutputs();
  for(unsigned int r=0; r < fft.R; r += fft.increment(r)) {
    fft.forward(f,F,r,W0);
    unsigned int D1=r == 0 ? fft.D0 : fft.D;
    for(unsigned int d=0; d < D1; ++d) {
      double *Fr=(double *) (F+fft.b*d);
      unsigned int offset=noutputs*d;
      for(unsigned int k=0; k < noutputs; ++k) {
        unsigned int i=fft.Index(r,k+offset);
        error += abs2(Fr[k]-F2r[i]);
        norm += abs2(F2r[i]);
        if(Output) {
          if(k%fft.Cm == 0) cout << endl;
          cout << i << ": " << Fr[k] << endl;
        }
      }
    }
    fft.backward(F,h,r,W0);
  }

  if(Output) {
    cout << endl;
    for(unsigned int j=0; j < C*fft2.noutputs(); ++j)
      cout << j << ": " << F2r[j] << endl;
  }

  double scale=1.0/fft.normalization();

  if(Output) {
    cout << endl;
    cout << "Inverse:" << endl;
    for(unsigned int j=0; j < fft.inputSize(); ++j)
      cout << h[j]*scale << endl;
    cout << endl;
  }

  for(unsigned int r=0; r < fft.R; r += fft.increment(r)) {
    unsigned int D1=r == 0 ? fft.D0 : fft.D;
    for(unsigned int d=0; d < D1; ++d) {
      double *Fr=(double *) (F+fft.b*d);
      unsigned int offset=noutputs*d;
      for(unsigned int k=0; k < noutputs; ++k) {
        unsigned int K=k+offset;
        Fr[k]=F2r[fft.Index(r,K)];
      }
    }
    fft.backward(F,h,r,W0);
  }

  for(unsigned int j=0; j < fft.inputSize(); ++j) {
    error2 += abs2(h[j]*scale-f[j]);
    norm2 += abs2(f[j]);
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
