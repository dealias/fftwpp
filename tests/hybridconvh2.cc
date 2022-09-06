#include "convolve.h"
#include "timing.h"
#include "direct.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

unsigned int A=2; // number of inputs
unsigned int B=1; // number of outputs
unsigned int L=7; // input data length
unsigned int M=10; // minimum padded length

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv);

  unsigned int Lx=L;
  unsigned int Ly=L;
  unsigned int Mx=M;
  unsigned int My=M;

  unsigned int Sx=0; // x stride (0 means ceilquotient(Ly,2))

  unsigned int K0=10000000;
  if(K == 0) K=max(K0/(Mx*My),20);
  if(Output || testError)
    K=1;
  cout << "K=" << K << endl << endl;

  cout << "Lx=" << Lx << endl;
  cout << "Ly=" << Ly << endl;
  cout << "Mx=" << Mx << endl;
  cout << "My=" << My << endl;
  cout << endl;

  unsigned int Hx=ceilquotient(Lx,2);
  unsigned int Hy=ceilquotient(Ly,2);

  if(Sx == 0) Sx=Hy;

  double *T=new double[K];

  Application appx(A,B);
  fftPadCentered fftx(Lx,Mx,appx,Hy,Sx);
  Application appy(A,B,realmultbinary,appx.Threads(),fftx.l);
  ConvolutionHermitian convolvey(Ly,My,appy);
  ConvolutionHermitian2 Convolve2(&fftx,&convolvey);

//  ConvolutionHermitian2 Convolve2(Lx,Mx,Ly,My,A,B);

  unsigned int N=max(A,B);
  Complex **f=new Complex *[N];
  unsigned int size=fftx.inputSize();
  Complex *f0=ComplexAlign(N*size);
  for(unsigned int a=0; a < A; ++a)
    f[a]=f0+a*size;

  for(unsigned int a=0; a < A; ++a) {
    Complex *fa=f[a];
    for(unsigned int i=0; i < Lx; ++i) {
      for(unsigned int j=0; j < Hy; ++j) {
        int I=Lx % 2 ? i : -1+i;
        fa[Sx*i+j]=Output || testError ? Complex((a+1)*I,j+a) : 0.0;
      }
    }
    HermitianSymmetrizeX(Hx,Hy,Lx/2,fa,Sx);
  }

  Complex *h=NULL;
  if(testError) {
    h=ComplexAlign(Lx*Hy);
    DirectHConvolution2 C(Hx,Hy);
    C.convolve(h,f[0],f[1],false,Lx%2,Ly%2);
  }

  if(normalized || testError) {
    for(unsigned int k=0; k < K; ++k) {
      seconds();
      Convolve2.convolve(f);
      T[k]=seconds();
    }
  } else {
    for(unsigned int k=0; k < K; ++k) {
      seconds();
      Convolve2.convolveRaw(f);
      T[k]=seconds();
    }
  }

  cout << endl;
  timings("Hybrid",L,T,K,stats);
  cout << endl;

  if(Output) {
    if(testError)
      cout << "Hybrid:" << endl;
    for(unsigned int b=0; b < B; ++b) {
      for(unsigned int i=0; i < Lx; ++i) {
        for(unsigned int j=0; j < Hy; ++j) {
          cout << f[b][Sx*i+j] << " ";
        }
        cout << endl;
      }
    }
  }

  if(testError) {
    if(Output) {
      cout << endl;
      cout << "Direct:" << endl;
      for(unsigned int i=0; i < Lx; ++i) {
        for(unsigned int j=0; j < Hy; ++j) {
          cout << h[Sx*i+j] << " ";
        }
        cout << endl;
      }
      cout << endl;
    }
    double err=0.0;
    double norm=0.0;
    // Assumes B=1
    for(unsigned int i=0; i < Lx; ++i) {
      for(unsigned int j=0; j < Hy; ++j) {
        Complex hij=h[Sx*i+j];
        err += abs2(f[0][Sx*i+j]-hij);
        norm += abs2(hij);
      }
    }
    double relError=sqrt(err/norm);
    cout << "Error: "<< relError << endl;
    deleteAlign(h);
  }
  return 0;
}
