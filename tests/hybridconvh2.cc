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
  Lx=Ly=7;  // input data length
  Mx=My=10; // minimum padded length

  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv);

  cout << "Lx=" << Lx << endl;
  cout << "Ly=" << Ly << endl;
  cout << "Mx=" << Mx << endl;
  cout << "My=" << My << endl;

  unsigned int K0=10000000;
  if(K == 0) K=max(K0/((unsigned long long) Mx*My),20);
  if(Output || testError)
    K=1;
  cout << "K=" << K << endl << endl;

  unsigned int Hx=ceilquotient(Lx,2);
  unsigned int Hy=ceilquotient(Ly,2);

  if(Sx == 0) Sx=Hy;

  double *T=new double[K];
  unsigned int N=max(A,B);

  Application appx(A,B,multNone,fftw::maxthreads,0,mx,Dx,Ix);
  fftPadCentered fftx(Lx,Mx,appx,Hy,Sx);
  bool embed=fftx.embed();
  unsigned int size=embed ? fftx.outputSize() : fftx.inputSize();
  Complex **f=ComplexAlign(N,size);
  Application appy(A,B,realmultbinary,appx.Threads(),fftx.l,my,Dy,Iy);
  ConvolutionHermitian convolvey(Ly,My,appy);
  ConvolutionHermitian2 Convolve2(&fftx,&convolvey,embed ? f : NULL);

//  ConvolutionHermitian2 Convolve2(Lx,Mx,Ly,My,A,B);

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
    DirectHConvolution2 C(Hx,Hy,Lx%2,Sx);
    C.convolve(h,f[0],f[1],false);
  }

  if(normalized || testError) {
    for(unsigned int k=0; k < K; ++k) {
      double t0=nanoseconds();
      Convolve2.convolve(f);
      T[k]=nanoseconds()-t0;
    }
  } else {
    for(unsigned int k=0; k < K; ++k) {
      double t0=nanoseconds();
      Convolve2.convolveRaw(f);
      T[k]=nanoseconds()-t0;
    }
  }

  cout << endl;
  timings("Hybrid",Lx*Hy,T,K,stats);
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
          cout << h[Hy*i+j] << " ";
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
        Complex hij=h[Hy*i+j];
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
