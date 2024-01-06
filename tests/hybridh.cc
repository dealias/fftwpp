#include "convolve.h"
#include "options.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

size_t A=2; // number of inputs
size_t B=1; // number of outputs

int main(int argc, char *argv[])
{
  L=511; // input data length
  M=766; // minimum padded length

  fftw::maxthreads=parallel::get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv,true);

  cout << "L=" << L << endl;
  cout << "M=" << M << endl;

// Disable overwrite optimization to allow indexing transformed values.
  Application app(1,1,multNone,fftw::maxthreads,false,true,mx,Dx,Ix);

  cout << endl << "Minimal Explicit:" << endl;
  // Minimal explicit padding
  fftPadHermitian fft0(L,M,app,C,M,1,1);

  double median0=fft0.report();

  // Optimal explicit padding
  cout << endl << "Optimal Explicit:" << endl;
  Application appE(app);
  appE.D=1;
  fftPadHermitian fft1(L,M,appE,C,true);
  double median1=min(median0,fft1.report());

  cout << endl;
  cout << "Hybrid:" << endl;

  // Hybrid padding
  fftPadHermitian fft(L,M,app,C);

  double median=fft.report();

  if(median0 > 0)
    cout << endl << "minimal ratio=" << median/median0 << endl;

  if(median1 > 0)
    cout << "optimal ratio=" << median/median1 << endl;

  size_t H=ceilquotient(L,2);

  Complex *f=ComplexAlign(C*H);
  Complex *h=ComplexAlign(C*H);
  Complex *F=ComplexAlign(fft.outputSize());
  Complex *W0=ComplexAlign(fft.workSizeW());

  for(size_t c=0; c < C; ++c)
    f[c]=1+c;
  for(size_t j=1; j < H; ++j)
    for(size_t c=0; c < C; ++c)
      f[C*j+c]=Complex(C*j+c+1,C*j+c+2);

  fftPadHermitian fft2(L,fft.M,app,C,fft.M,1,1);

  Complex *F2=ComplexAlign(fft2.outputSize());
  double *F2r=(double *) F2;

  fft2.forward(f,F2);

  fft.pad(W0);
  double error=0.0, error2=0.0;
  double norm=0.0, norm2=0.0;
  size_t stride=C*fft.noutputs(0);
  for(size_t r=0; r < fft.R; r += fft.increment(r)) {
    fft.forward(f,F,r,W0);
    size_t D1=r == 0 ? fft.D0 : fft.D;
    for(size_t d=0; d < D1; ++d) {
      double *Fr=(double *) (F+fft.b*d);
      size_t offset=stride*d;
      for(size_t k=0; k < stride; ++k) {
        size_t i=fft.Index(r,k+offset);
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
    cout << "Explicit output:" << endl;
    size_t stride=C*fft2.noutputs(0);
    for(size_t j=0; j < stride; ++j)
      cout << j << ": " << F2r[j] << endl;
  }

  double scale=1.0/fft.normalization();

  if(Output) {
    cout << endl;
    cout << "Inverse:" << endl;
    for(size_t j=0; j < H; ++j)
      for(size_t c=0; c < C; ++c)
        cout << h[C*j+c]*scale << endl;
    cout << endl;
  }

  for(size_t r=0; r < fft.R; r += fft.increment(r)) {
    size_t D1=r == 0 ? fft.D0 : fft.D;
    for(size_t d=0; d < D1; ++d) {
      double *Fr=(double *) (F+fft.b*d);
      size_t offset=stride*d;
      for(size_t k=0; k < stride; ++k) {
        size_t s=k+offset;
        Fr[k]=F2r[fft.Index(r,s)];
      }
    }
    fft.backward(F,h,r,W0);
  }

  for(size_t j=0; j < H; ++j)
    for(size_t c=0; c < C; ++c) {
      size_t J=C*j+c;
      error2 += abs2(h[J]*scale-f[J]);
      norm2 += abs2(f[J]);
    }

  if(norm > 0) error=sqrt(error/norm);
  if(norm2 > 0) error2=sqrt(error2/norm2);

  cout << endl;
  cout << "Forward Error: " << error << endl;
  cout << "Backward Error: " << error2 << endl;

  return 0;
}
