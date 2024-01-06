#include "convolve.h"
#include "options.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

int main(int argc, char *argv[])
{
  L=512; // input data length
  M=1024; // minimum padded length

  fftw::maxthreads=parallel::get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv,true);

  cout << "L=" << L << endl;
  cout << "M=" << M << endl;

  if(S == 0) S=C;

// Disable overwrite optimization to allow indexing transformed values.
  Application app(1,1,multNone,fftw::maxthreads,false,true,mx,Dx,Ix);

  cout << endl << "Minimal Explicit:" << endl;
  // Minimal explicit padding
  fftPad *fft0=Centered ? new fftPadCentered(L,M,app,C,S,M,1,1) :
    new fftPad(L,M,app,C,S,M,1,1);

  double median0=fft0->report();

  // Optimal explicit padding
  cout << endl << "Optimal Explicit:" << endl;
  Application appE(app);
  appE.D=1;
  fftPad *fft1=Centered ? new fftPadCentered(L,M,appE,C,S,true) :
    new fftPad(L,M,appE,C,S,true);

  double median1=min(median0,fft1->report());

  cout << endl;
  cout << "Hybrid:" << endl;

  // Hybrid padding
  fftPad *fft=Centered ? new fftPadCentered(L,M,app,C,S) :
    new fftPad(L,M,app,C,S);

  double median=fft->report();

  if(median0 > 0)
    cout << endl << "minimal ratio=" << median/median0 << endl;

  if(median1 > 0)
    cout << "optimal ratio=" << median/median1 << endl;

  Complex *f=ComplexAlign(S*L);
  Complex *h=ComplexAlign(S*L);
  Complex *F=ComplexAlign(fft->outputSize());
  Complex *W0=ComplexAlign(fft->workSizeW());

  for(size_t j=0; j < L; ++j)
    for(size_t c=0; c < C; ++c)
      f[S*j+c]=Complex(C*j+c+1,C*j+c+2);

  fftPad* fft2=Centered ? new fftPadCentered(L,fft->M,app,C,S,fft->M,1,1) :
    new fftPad(L,fft->M,app,C,S,fft->M,1,1);

  Complex *F2=ComplexAlign(fft2->outputSize());

  fft2->forward(f,F2);

  fft->pad(W0);
  double error=0.0, error2=0.0;
  double norm=0.0, norm2=0.0;
  for(size_t r=0; r < fft->R; r += fft->increment(r)) {
    fft->forward(f,F,r,W0);
    for(size_t k=0; k < fft->noutputs(r); ++k) {
      if(Output && k%fft->m == 0) cout << endl;

      for(size_t c=0; c < C; ++c) {
        size_t s=S*k+c;
        size_t i=fft->Index(r,s);
        error += abs2(F[s]-F2[i]);
        norm += abs2(F2[i]);
        if(Output)
          cout << i << ": " << F[s] << endl;
      }
    }
    fft->backward(F,h,r,W0);
  }

  if(Output) {
    cout << endl;
    cout << "Explicit output:" << endl;
    for(size_t j=0; j < fft2->noutputs(0); ++j)
      for(size_t c=0; c < C; ++c) {
        size_t J=S*j+c;
        cout << J << ": " << F2[J] << endl;
      }
  }

  double scale=1.0/fft->normalization();

  if(Output) {
    cout << endl;
    cout << "Inverse:" << endl;
    for(size_t j=0; j < L; ++j) {
      for(size_t c=0; c < C; ++c) {
        size_t J=S*j+c;
        cout << h[J]*scale << endl;
      }
    }
    cout << endl;
  }

  for(size_t r=0; r < fft->R; r += fft->increment(r)) {
    for(size_t k=0; k < fft->noutputs(r); ++k) {
      for(size_t c=0; c < C; ++c) {
        size_t s=S*k+c;
        F[s]=F2[fft->Index(r,s)];
      }
    }
    fft->backward(F,h,r,W0);
  }

  for(size_t j=0; j < L; ++j) {
    for(size_t c=0; c < C; ++c) {
      size_t J=S*j+c;
      error2 += abs2(h[J]*scale-f[J]);
      norm2 += abs2(f[J]);
    }
  }

  if(norm > 0) error=sqrt(error/norm);
  if(norm2 > 0) error2=sqrt(error2/norm2);

  cout << endl;
  cout << "Forward Error: " << error << endl;
  cout << "Backward Error: " << error2 << endl;

  return 0;
}
