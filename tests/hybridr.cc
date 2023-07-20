#include "convolve.h"

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

// Disable overwrite optimization for these tests.
  Application app(1,2,multNone,fftw::maxthreads,mx,Dx,Ix);

  cout << endl << "Minimal Explicit:" << endl;
  // Minimal explicit padding
  fftPadReal fft0(L,M,C,S,M,1,1,app);

  double median0=fft0.report();

  // Optimal explicit padding
  cout << endl << "Optimal Explicit:" << endl;
  Application appE(app);
  appE.D=1;
  fftPadReal fft1(L,M,appE,C,S,true);

  double median1=min(median0,fft1.report());

  cout << endl;
  cout << "Hybrid:" << endl;

  // Hybrid padding
  fftPadReal fft(L,M,app,C,S);

  double median=fft.report();

  if(median0 > 0)
    cout << endl << "minimal ratio=" << median/median0 << endl;

  if(median1 > 0)
    cout << "optimal ratio=" << median/median1 << endl;

  double *f=doubleAlign(2*fft.inputSize());
  double *h=doubleAlign(2*fft.inputSize());
  Complex *F=ComplexAlign(fft.outputSize());
  Complex *W0=ComplexAlign(fft.workSizeW());

  for(size_t j=0; j < L; ++j)
    for(size_t c=0; c < C; ++c)
      f[S*j+c]=C*j+c+1;

  fftPadReal fft2(L,fft.M,C,S,fft.M,1,1,app);

  Complex *F2=ComplexAlign(fft2.outputSize());

  fft2.forward((Complex *) f,F2);

  fft.pad(W0);
//  double error=0.0;
  double error2=0.0;
//  double norm=0.0;
  double norm2=0.0;
  for(size_t r=0; r < fft.R; r += fft.increment(r)) {

    fft.forward((Complex *) f,F,r,W0);
#if 0
    for(size_t k=0; k < fft.noutputs(r); ++k) {
      if(Output && k%fft.m == 0) cout << endl;

      for(size_t c=0; c < C; ++c) {
        size_t K=S*k+c;
        size_t i=fft.Index(r,K);
        error += abs2(F[K]-F2[i]);
        norm += abs2(F2[i]);
        if(Output)
          cout << i << ": " << F[K] << endl;
      }
    }
#endif
    fft.backward(F,(Complex *) h,r,W0);
  }

  if(Output) {
    cout << endl;
    for(size_t j=0; j < fft2.noutputs(0); ++j)
      for(size_t c=0; c < C; ++c) {
        size_t J=S*j+c;
        cout << J << ": " << F2[J] << endl;
      }
  }

  double scale=1.0/fft.normalization();

  if(Output) {
    cout << endl;
    cout << "Inverse:" << endl;
    for(size_t j=0; j < L; ++j) {
      for(size_t c=0; c < C; ++c)
        cout << h[S*j+c]*scale << endl;
    }
    cout << endl;
  }

  /*
  for(size_t r=0; r < fft.R; r += fft.increment(r)) {
    for(size_t k=0; k < fft.noutputs(r); ++k) {
      for(size_t c=0; c < C; ++c) {
        size_t K=S*k+c;
        F[K]=F2[fft.Index(r,K)];
      }
    }
    fft.backward(F,(Complex *) h,r,W0);
  }
  */

  for(size_t j=0; j < L; ++j) {
    for(size_t c=0; c < C; ++c) {
      size_t J=S*j+c;
      error2 += abs2(h[J]*scale-f[J]);
      norm2 += abs2(f[J]);
    }
  }

//  if(norm > 0) error=sqrt(error/norm);
  if(norm2 > 0) error2=sqrt(error2/norm2);

  //double eps=1e-12;
  //if(error > eps || error2 > eps)
  //  cerr << endl << "WARNING: " << endl;
  cout << endl;
//  cout << "Forward Error: " << error << endl;
  cout << "Backward Error: " << error2 << endl;

  return 0;
}
