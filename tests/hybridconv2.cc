#include "convolve.h"
#include "timing.h"
#include "direct.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

unsigned int A=2; // number of inputs
unsigned int B=1; // number of outputs

int main(int argc, char* argv[])
{
  Lx=Ly=8;  // input data length
  Mx=My=16; // minimum padded length

  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv);

  cout << "Lx=" << Lx << endl;
  cout << "Ly=" << Ly << endl;
  cout << "Mx=" << Mx << endl;
  cout << "My=" << My << endl;

  unsigned int Sx=0; // x stride (0 means Ly)

  unsigned int K0=10000000;
  if(K == 0) K=max(K0/((unsigned long long) Mx*My),20);
  if(Output || testError)
    K=1;
  cout << "K=" << K << endl << endl;

  if(Sx == 0) Sx=Ly;

  double *T=new double[K];

  Application appx(A,B,multNone,fftw::maxthreads,0,mx,Dx,Ix);
  fftPad fftx(Lx,Mx,appx,Ly,Sx);
  Application appy(A,B,multbinary,appx.Threads(),fftx.l,my,Dy,Iy);

  Convolution convolvey(Ly,My,appy);
  Convolution2 Convolve2(&fftx,&convolvey);

//  Convolution2 Convolve2(Lx,Mx,Ly,My,A,B);

  unsigned int N=max(A,B);
  Complex **f=new Complex *[N];
  unsigned int size=fftx.inputSize();
  Complex *f0=ComplexAlign(N*size);
  for(unsigned int a=0; a < A; ++a)
    f[a]=f0+a*size;

  for(unsigned int a=0; a < A; ++a) {
    Complex *fa=f[a];
    for(unsigned int i=0; i < Lx; ++i) {
      for(unsigned int j=0; j < Ly; ++j) {
        fa[Sx*i+j]=Output || testError ? Complex((1.0+a)*i,j+a) : 0.0;
      }
    }
  }

  Complex *h=NULL;
  if(testError) {
    h=ComplexAlign(Lx*Ly);
    DirectConvolution2 C(Lx,Ly);
    C.convolve(h,f[0],f[1]);
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
  timings("Hybrid",Lx*Ly,T,K,stats);
  cout << endl;

  if(Output) {
    if(testError)
      cout << "Hybrid:" << endl;
    for(unsigned int b=0; b < B; ++b) {
      for(unsigned int i=0; i < Lx; ++i) {
        for(unsigned int j=0; j < Ly; ++j) {
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
        for(unsigned int j=0; j < Ly; ++j) {
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
      for(unsigned int j=0; j < Ly; ++j) {
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
