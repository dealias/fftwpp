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

  unsigned int K0=10000000;

  unsigned int Lx=L;
  unsigned int Ly=L;
  unsigned int Mx=M;
  unsigned int My=M;

  unsigned int Sx=0; // x stride (0 means Ly)

  if(K == 0) K=max(K0/(Mx*My),20);

  if(Output || testError)
    K=1;

  cout << "K=" << K << endl << endl;

  cout << "Lx=" << Lx << endl;
  cout << "Ly=" << Ly << endl;
  cout << "Mx=" << Mx << endl;
  cout << "My=" << My << endl;
  cout << endl;

  if(Sx == 0) Sx=Ly;

  Complex **f=new Complex *[max(A,B)];
  for(unsigned int a=0; a < A; ++a)
    f[a]=ComplexAlign(Lx*Sx);

  Application appx(A,B,multbinary);
  fftPad fftx(Lx,Mx,appx,Ly,Sx);
  Application appy(A,B,multbinary,appx.Threads(),fftx.l);
  Convolution convolvey(Ly,My,appy);
  Convolution2 Convolve2(&fftx,&convolvey);

//  Convolution2 Convolve2(Lx,Mx,Ly,My,A,B);

  double T=0;
  Complex *h=NULL;
  for(unsigned int c=0; c < K; ++c) {

    for(unsigned int a=0; a < A; ++a) {
      Complex *fa=f[a];
      for(unsigned int i=0; i < Lx; ++i) {
        for(unsigned int j=0; j < Ly; ++j) {
          fa[Sx*i+j]=Complex((1.0+a)*i,j+a);
        }
      }
    }

    if(Lx*Ly < 200 && c == 0) {
      cout << endl << "Inputs:" << endl;
      for(unsigned int a=0; a < A; ++a) {
        for(unsigned int i=0; i < Lx; ++i) {
          for(unsigned int j=0; j < Ly; ++j) {
            cout << f[a][Sx*i+j] << " ";
          }
          cout << endl;
        }
        cout << endl;
      }
    }

    if(testError) {
      h=ComplexAlign(Lx*Ly);
      DirectConvolution2 C(Lx,Ly);
      C.convolve(h,f[0],f[1]);
    }

    seconds();
    Convolve2.convolve(f);
    T += seconds();
  }

  cout << "median=" << T/K << endl;

  Complex sum=0.0;
  for(unsigned int b=0; b < B; ++b) {
    Complex *fb=f[b];
    for(unsigned int i=0; i < Lx; ++i) {
      for(unsigned int j=0; j < Ly; ++j) {
        sum += fb[Sx*i+j];
      }
    }
  }

  cout << "sum=" << sum << endl;
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
    for(unsigned int i=0; i < Lx; ++i) {
      for(unsigned int j=0; j < Ly; ++j) {
        err += abs2(f[0][Sx*i+j]-h[Sx*i+j]);
        norm += abs2(h[Sx*i+j]);
      }
    }
    double relError=sqrt(err/norm);
    cout << "Error: "<< relError << endl;
    deleteAlign(h);
  }
  return 0;
}
