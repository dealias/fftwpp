#include "convolve.h"
#include "timing.h"
#include "direct.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

unsigned int A=2; // number of inputs
unsigned int B=1; // number of outputs
unsigned int L=512; // input data length
unsigned int M=768; // minimum padded length

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv);

  unsigned int K0=200000000;
  if(K == 0) K=max(K0/M,20);
  cout << "K=" << K << endl << endl;

  double *T=new double[K];

  ForwardBackward FB(A,B,realmultbinary);
  fftPadHermitian fft(L,M,FB);

  unsigned int H=ceilquotient(L,2);

  unsigned int N=max(A,B);
  Complex **f=new Complex *[N];
  unsigned int size=fft.embed() ? fft.outputSize() : fft.inputSize();
  Complex *F=ComplexAlign(N*size);
  for(unsigned int a=0; a < A; ++a)
    f[a]=F+a*size;

  for(unsigned int a=0; a < A; ++a) {
    Complex *fa=f[a];
    if(Output || testError) {
      fa[0]=1.0+a;
      for(unsigned int j=1; j < H; ++j)
        fa[j]=Complex(j,(1.0+a)*j+1);
    } else
      for(unsigned int j=0; j < H; ++j)
        fa[j]=0.0;
  }

  Complex *h;
  if(testError) {
    h=ComplexAlign(H);
    DirectHConvolution C(H);
    C.convolve(h,f[0],f[1]);
  }
  ConvolutionHermitian Convolve(&fft,A,B,fft.embed() ? F : NULL);

  if(Output||testError)
    K=1;

  for(unsigned int k=0; k < K; ++k) {
    seconds();
    Convolve.convolve(f);
    T[k]=seconds();
  }

  cout << endl;
  timings("Hermitian Hybrid",L,T,K,stats);
  cout << endl;

  if(Output){
    if(testError) {
      cout << "Hermitian Hybrid:" << endl;
    }
    for(unsigned int b=0; b < B; ++b)
      for(unsigned int j=0; j < H; ++j)
        cout << f[b][j] << endl;
  }

  if(testError) {
    double err=0.0;
    if(Output) {
      cout<<endl;
      cout << "testError:" << endl;
      for(unsigned int j=0; j < H; ++j)
        cout << h[j] << endl;
      cout << endl;
    }
    for(unsigned int j=0; j < H; ++j) {
      Complex hj=h[j];
      for(unsigned int b=0; b < B; ++b)
            err += abs2(f[b][j]-hj);
    }
    cout << "Error: "<< sqrt(err) << endl;
    deleteAlign(h);
  }

  delete [] T;

  return 0;
}
