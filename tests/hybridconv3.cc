#include "convolve.h"
#include "direct.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

unsigned int A=2; // number of inputs
unsigned int B=1; // number of outputs
unsigned int L=512; // input data length
unsigned int M=1024; // minimum padded length

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv);

  unsigned int Lx=L; // TODO: x
  unsigned int Ly=L;
  unsigned int Lz=L;
  unsigned int Mx=M; // TODO: X
  unsigned int My=M;
  unsigned int Mz=M;

  unsigned int K0=10000000;
  if(K == 0) K=max(K0/(Mx*My*Mz),20);
  if(Output || testError)
    K=1;
  cout << "K=" << K << endl << endl;

  unsigned int Sy=0; // y-stride (0 means Lz)
  unsigned int Sx=0; // x-stride (0 means Ly*Sy)

  cout << "Lx=" << Lx << endl;
  cout << "Ly=" << Ly << endl;
  cout << "Lz=" << Lz << endl;
  cout << "Mx=" << Mx << endl;
  cout << "My=" << My << endl;
  cout << "Mz=" << Mz << endl;
  cout << endl;

  if(Sy == 0) Sy=Lz;
  if(Sx == 0) Sx=Ly*Sy;

  Complex **f=new Complex *[max(A,B)];
  for(unsigned int a=0; a < A; ++a)
    f[a]=ComplexAlign(Lx*Sx);

  ForwardBackward FBx(A,B,multbinary);
  fftPad fftx(Lx,Mx,FBx,Sx == Ly*Lz ? Sx : Lz,Sx);
  ForwardBackward FBy(A,B,multbinary,FBx.Threads(),fftx.l);
  fftPad ffty(Ly,My,FBy,Lz,Sy);
  ForwardBackward FBz(A,B,multbinary,FBy.Threads(),ffty.l);
  Convolution convolvez(Lz,Mz,FBz);
  Convolution2 convolveyz(&ffty,&convolvez);
  Convolution3 Convolve3(&fftx,&convolveyz);

//  Convolution3 Convolve3(Lx,Mx,Ly,My,Lz,Mz,A,B);

  double T=0;
  Complex *h=NULL;
  for(unsigned int c=0; c < K; ++c) {

    for(unsigned int a=0; a < A; ++a) {
      Complex *fa=f[a];
      for(unsigned int i=0; i < Lx; ++i) {
        for(unsigned int j=0; j < Ly; ++j) {
          for(unsigned int k=0; k < Lz; ++k) {
            fa[Sx*i+Sy*j+k]=Complex((1.0+a)*i+k,j+k+a);
          }
        }
      }
    }

    if(Lx*Ly*Lz < 200 && c == 0) {
      for(unsigned int a=0; a < A; ++a) {
        for(unsigned int i=0; i < Lx; ++i) {
          for(unsigned int j=0; j < Ly; ++j) {
            for(unsigned int k=0; k < Lz; ++k) {
              cout << f[a][Sx*i+Sy*j+k] << " ";
            }
            cout << endl;
          }
          cout << endl;
        }
        cout << endl;
      }
    }
    if(testError) {
      h=ComplexAlign(Lx*Ly*Lz);
      DirectConvolution3 C(Lx,Ly,Lz);
      C.convolve(h,f[0],f[1]);
    }
    seconds();
    Convolve3.convolve(f);
    T += seconds();
  }

  cout << "median=" << T/K << endl;

  Complex sum=0.0;
  for(unsigned int b=0; b < B; ++b) {
    Complex *fb=f[b];
    for(unsigned int i=0; i < Lx; ++i) {
      for(unsigned int j=0; j < Ly; ++j) {
        for(unsigned int k=0; k < Lz; ++k)
          sum += fb[Sx*i+Sy*j+k];
      }
    }
  }

  cout << "sum=" << sum << endl;
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
            cout << h[Sx*i+Sy*j+k] << " ";
          }
          cout << endl;
        }
        cout << endl;
      }
      cout << endl;
    }
    double err=0.0;
    double norm=0.0;
    for(unsigned int i=0; i < Lx; ++i)
        for(unsigned int j=0; j < Ly; ++j)
          for(unsigned int k=0; k < Lz; ++k){
            err += abs2(f[0][Sx*i+Sy*j+k]-h[Sx*i+Sy*j+k]);
            norm += abs2(h[Sx*i+Sy*j+k]);
          }
    double relError=sqrt(err/norm);
    cout << "Error: "<< relError << endl;
    deleteAlign(h);
  }
  return 0;
}
