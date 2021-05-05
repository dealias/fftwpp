#include "convolve.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

unsigned int A=2; // number of inputs
unsigned int B=1; // number of outputs
unsigned int C=1; // number of copies
unsigned int L=512; // input data length
unsigned int M=1024; // minimum padded length

int main(int argc, char* argv[])
{
  fftw::maxthreads=1;//get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv);

  ForwardBackward FB(A,B);

  unsigned int Lx=L;
  unsigned int Ly=L;
  unsigned int Mx=M;
  unsigned int My=M;

  cout << "Lx=" << Lx << endl;
  cout << "Mx=" << Mx << endl;
  cout << endl;

  Complex **f=new Complex *[max(A,B)];
  for(unsigned int a=0; a < A; ++a)
    f[a]=ComplexAlign(Lx*Ly);


//      fftPad fftx(Lx,Mx,Ly,Lx,2,1);
//      fftPad ffty(Ly,My,1,Ly,2,1);

//  fftPad fftx(Lx,Mx,FB,Ly);
//  fftPad ffty(Ly,My,FB,1);
//  Convolution convolvey(ffty,A,B);

//  Convolution2 Convolve2(fftx,convolvey);
  Convolution2 Convolve2(Lx,Ly,Mx,My,A,B);

  unsigned int K=10;
  double T=0;

  for(unsigned int k=0; k < K; ++k) {

    for(unsigned int a=0; a < A; ++a) {
      Complex *fa=f[a];
      for(unsigned int i=0; i < Lx; ++i) {
        for(unsigned int j=0; j < Ly; ++j) {
          fa[Ly*i+j]=Complex((1.0+a)*i,j+a);
        }
      }
    }

    if(Lx*Ly < 200 && k == 0) {
      for(unsigned int a=0; a < A; ++a) {
        for(unsigned int i=0; i < Lx; ++i) {
          for(unsigned int j=0; j < Ly; ++j) {
            cout << f[a][Ly*i+j] << " ";
          }
          cout << endl;
        }
        cout << endl;
      }
    }

    seconds();
    Convolve2.convolve(f,multbinary);
    T += seconds();
  }

  cout << "mean=" << T/K << endl;

  Complex sum=0.0;
  for(unsigned int b=0; b < B; ++b) {
    Complex *fb=f[b];
    for(unsigned int i=0; i < Lx; ++i) {
      for(unsigned int j=0; j < Ly; ++j) {
        sum += fb[Ly*i+j];
      }
    }
  }

  cout << "sum=" << sum << endl;
  cout << endl;

  if(Lx*Ly < 200) {
    for(unsigned int b=0; b < B; ++b) {
      for(unsigned int i=0; i < Lx; ++i) {
        for(unsigned int j=0; j < Ly; ++j) {
          cout << f[b][Ly*i+j] << " ";
        }
        cout << endl;
      }
    }
  }
  return 0;
}
