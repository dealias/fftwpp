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
  Lx=Ly=Lz=8;  // input data length
  Mx=My=Mz=16; // minimum padded length

  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv);

  cout << "Lx=" << Lx << endl;
  cout << "Ly=" << Ly << endl;
  cout << "Lz=" << Lz << endl;
  cout << "Mx=" << Mx << endl;
  cout << "My=" << My << endl;
  cout << "Mz=" << Mz << endl;

  unsigned int K0=10000000;
  if(K == 0) K=max(K0/((unsigned long long) Mx*My*Mz),20);
  if(Output || testError)
    K=1;
  cout << "K=" << K << endl << endl;

  if(Sy == 0) Sy=Lz;
  if(Sx == 0) Sx=Ly*Sy;

  double *T=new double[K];
  unsigned int N=max(A,B);
  Complex **f=new Complex *[N];

  Application appx(A,B,multNone,fftw::maxthreads,0,mx,Dx,Ix);
  fftPad fftx(Lx,Mx,appx,Sy == Lz? Ly*Lz : Lz,Sx);
  bool embed=fftx.embed();
  unsigned int size=embed ? fftx.outputSize() : fftx.inputSize();
  Complex *F=ComplexAlign(N*size);
  Application appy(A,B,multNone,appx.Threads(),fftx.l,my,Dy,Iy);
  fftPad ffty(Ly,My,appy,Lz,Sy);
  Application appz(A,B,multbinary,appy.Threads(),ffty.l,mz,Dz,Iz);
  Convolution convolvez(Lz,Mz,appz);
  Convolution2 convolveyz(&ffty,&convolvez);
  Convolution3 Convolve3(&fftx,&convolveyz,embed ? F : NULL);

//  Convolution3 Convolve3(Lx,Mx,Ly,My,Lz,Mz,A,B);

  for(unsigned int a=0; a < A; ++a) {
    Complex *fa=F+a*size;
    f[a]=fa;
    for(unsigned int i=0; i < Lx; ++i) {
      for(unsigned int j=0; j < Ly; ++j) {
        for(unsigned int k=0; k < Lz; ++k) {
          fa[Sx*i+Sy*j+k]=Output || testError ?
            Complex((1.0+a)*i+k,j+k+a) : 0.0;
        }
      }
    }
  }

  Complex *h=NULL;
  if(testError) {
    h=ComplexAlign(Lx*Ly*Lz);
    DirectConvolution3 C(Lx,Ly,Lz,Sx,Sy);
    C.convolve(h,f[0],f[1]);
  }

  if(normalized || testError) {
    for(unsigned int k=0; k < K; ++k) {
      seconds();
      Convolve3.convolve(f);
      T[k]=seconds();
    }
  } else {
    for(unsigned int k=0; k < K; ++k) {
      seconds();
      Convolve3.convolveRaw(f);
      T[k]=seconds();
    }
  }

  cout << endl;
  timings("Hybrid",Lx*Ly*Lz,T,K,stats);
  cout << endl;

  if(Output) {
    if(testError)
      cout << "Hybrid:" << endl;
    for(unsigned int b=0; b < B; ++b) {
      Complex *fb=f[b];
      for(unsigned int i=0; i < Lx; ++i) {
        for(unsigned int j=0; j < Ly; ++j) {
          for(unsigned int k=0; k < Lz; ++k) {
            cout << fb[Sx*i+Sy*j+k] << " ";
          }
          cout << endl;
        }
        cout << endl;
      }
      cout << endl;
    }
  }
  if(testError) {
    if(Output) {
      cout << endl;
      cout << "Direct:" << endl;
      for(unsigned int i=0; i < Lx; ++i) {
        for(unsigned int j=0; j < Ly; ++j) {
          for(unsigned int k=0; k < Lz; ++k) {
            cout << h[Lz*(Ly*i+j)+k] << " ";
          }
          cout << endl;
        }
        cout << endl;
      }
      cout << endl;
    }
    double err=0.0;
    double norm=0.0;

    // Assumes B=1
    for(unsigned int i=0; i < Lx; ++i)
        for(unsigned int j=0; j < Ly; ++j)
          for(unsigned int k=0; k < Lz; ++k){
            Complex hijk=h[Lz*(Ly*i+j)+k];
            err += abs2(f[0][Sx*i+Sy*j+k]-hijk);
            norm += abs2(hijk);
          }
    double relError=sqrt(err/norm);
    cout << "Error: "<< relError << endl;
    deleteAlign(h);
  }
  return 0;
}
