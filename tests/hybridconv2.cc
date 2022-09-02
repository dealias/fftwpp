#include "convolve.h"

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

  ForwardBackward FBx(A,B,multbinary);
  fftPad fftx(Lx,Mx,FBx,Ly,Sx);
  ForwardBackward FBy(A,B,multbinary,FBx.Threads(),fftx.l);
  Convolution convolvey(Ly,My,FBy);
  Convolution2 Convolve2(&fftx,&convolvey);

//  Convolution2 Convolve2(Lx,Mx,Ly,My,A,B);

  double T=0;

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
    for(unsigned int b=0; b < B; ++b) {
      for(unsigned int i=0; i < Lx; ++i) {
        for(unsigned int j=0; j < Ly; ++j) {
          cout << f[b][Sx*i+j] << " ";
        }
        cout << endl;
      }
    }
  }
  return 0;
}
