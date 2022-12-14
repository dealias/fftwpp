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
  L=7;  // input data length
  M=10; // minimum padded length

  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv);

  cout << "L=" << L << endl;
  cout << "M=" << M << endl;

  if(Output||testError)
    K=0;
  if(K == 0) minCount=1;
  cout << "K=" << K << endl << endl;
  K *= 1.0e9;

  size_t H=ceilquotient(L,2);

  vector<double> T;

  Application app(A,B,realmultbinary,fftw::maxthreads,0,mx,Dx,Ix);
  fftPadHermitian fft(L,M,app);
  Complex **f=ComplexAlign(max(A,B),fft.inputSize());
  Convolution Convolve(&fft,f);

  for(size_t a=0; a < A; ++a) {
    Complex *fa=f[a];
    if(Output || testError) {
      fa[0]=1.0+a;
      for(size_t j=1; j < H; ++j)
        fa[j]=Complex(j,(1.0+a)*j+1);
    } else
      for(size_t j=0; j < H; ++j)
        fa[j]=0.0;
  }

  Complex *h=NULL;
  if(testError) {
    h=ComplexAlign(H);
    DirectHConvolution C(H);
    C.convolve(h,f[0],f[1]);
  }

  double sum=0.0;
  while(sum <= K || T.size() < minCount) {
    double t;
    if(normalized || testError) {
      double t0=nanoseconds();
      Convolve.convolve(f);
      t=nanoseconds()-t0;
    } else {
      double t0=nanoseconds();
      Convolve.convolveRaw(f);
      t=nanoseconds()-t0;
    }
    T.push_back(t);
    sum += t;
  }

  cout << endl;
  timings("Hermitian Hybrid",L,T.data(),T.size(),stats);
  cout << endl;

  if(Output) {
    if(testError)
      cout << "Hermitian Hybrid:" << endl;
    for(size_t b=0; b < B; ++b)
      for(size_t j=0; j < H; ++j)
        cout << f[b][j] << endl;
  }

  if(testError) {
    if(Output) {
      cout << endl;
      cout << "Direct:" << endl;
      for(size_t j=0; j < H; ++j)
        cout << h[j] << endl;
      cout << endl;
    }
    double err=0.0;
    double norm=0.0;
    Complex hj;

    // Assumes B=1
    Complex* f0=f[0];
    for(size_t j=0; j < H; ++j) {
      hj=h[j];
      err += abs2(f0[j]-hj);
      norm += abs2(hj);
    }
    double relError=sqrt(err/norm);
    cout << "Error: "<< relError << endl;
    deleteAlign(h);
  }

  return 0;
}
