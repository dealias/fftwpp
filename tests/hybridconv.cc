#include <vector>

#include "convolve.h"
#include "timing.h"
#include "direct.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

size_t A=2; // number of inputs
size_t B=1; // number of outputs

int main(int argc, char *argv[])
{
  L=8;  // input data length
  M=16; // minimum padded length

  fftw::maxthreads=parallel::get_max_threads();
#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv);

  cout << "L=" << L << endl;
  cout << "M=" << M << endl;

  if(Output || testError)
    K=0;
  if(K == 0) minCount=1;
  cout << "K=" << K << endl << endl;
  K *= 1.0e9;

  vector<double> T;

  Application app(A,B,multbinary,fftw::maxthreads,mx,Dx,Ix);
  fftPad *fft=Centered ? new fftPadCentered(L,M,app) : new fftPad(L,M,app);
  Convolution Convolve(fft);

  Complex **f=ComplexAlign(max(A,B),L);
  for(size_t a=0; a < A; ++a) {
    Complex *fa=f[a];
    for(size_t j=0; j < L; ++j)
      fa[j]=Output || testError ? Complex(j,(1.0+a)*j+1) : 0.0;
  }

  Complex *h=NULL;
  if(testError) {
    h=ComplexAlign(L);
    DirectConvolution C(L);
    if(Centered)
      C.Cconvolve(h,f[0],f[1]);
    else
      C.convolve(h,f[0],f[1]);
  }

  double sum=0.0;
  while(sum <= K || T.size() < minCount) {
    double t;
    if(normalized || testError) {
      cpuTimer c;
      Convolve.convolve(f);
      t=c.nanoseconds();
    } else {
      cpuTimer c;
      Convolve.convolveRaw(f);
      t=c.nanoseconds();
    }
    T.push_back(t);
    sum += t;
  }

  cout << endl;
  timings("Hybrid",L,T.data(),T.size(),stats);
  cout << endl;

  if(Output) {
    if(testError)
      cout << "Hybrid:" << endl;
    for(size_t b=0; b < B; ++b)
      for(size_t j=0; j < L; ++j)
        cout << f[b][j] << endl;
  }

  if(testError) {
    if(Output) {
      cout << endl;
      cout << "Direct:" << endl;
      for(size_t j=0; j < L; ++j)
        cout << h[j] << endl;
      cout << endl;
    }
    double err=0.0;
    double norm=0.0;

    // Assumes B=1
    Complex* f0=f[0];
    for(size_t j=0; j < L; ++j) {
      Complex hj=h[j];
      err += abs2(f0[j]-hj);
      norm += abs2(hj);
    }
    double relError=sqrt(err/norm);
    cout << "Error: "<< relError << endl;
    deleteAlign(h);
  }

  return 0;
}
