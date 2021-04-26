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

  Complex **f=new Complex *[A];
  Complex **h=new Complex *[B];
  for(unsigned int a=0; a < A; ++a)
    f[a]=ComplexAlign(Lx*Ly);
  for(unsigned int b=0; b < B; ++b)
    h[b]=ComplexAlign(Lx*Ly);

  array2<Complex> f0(Lx,Ly,f[0]);
  array2<Complex> f1(Lx,Ly,f[1]);

  for(unsigned int i=0; i < Lx; ++i) {
    for(unsigned int j=0; j < Ly; ++j) {
      f0[i][j]=Complex(i,j);
      f1[i][j]=Complex(2*i,j+1);
    }
  }

  if(Lx*Ly < 200) {
    for(unsigned int i=0; i < Lx; ++i) {
      for(unsigned int j=0; j < Ly; ++j) {
        cout << f0[i][j] << " ";
      }
      cout << endl;
    }
  }

//      fftPad fftx(Lx,Mx,Ly,Lx,2,1);
//      fftPad ffty(Ly,My,1,Ly,2,1);

//  fftPad fftx(Lx,Mx,FB,Ly);
//  fftPad ffty(Ly,My,FB,1);
//  Convolution convolvey(ffty,A,B);

//  Convolution2 Convolve2(fftx,convolvey);
  Convolution2 Convolve2(Lx,Ly,Mx,My,A,B);

  unsigned int K=10;
  double t0=totalseconds();

  for(unsigned int k=0; k < K; ++k)
    Convolve2.convolve(f,h,multbinary);

  double t=totalseconds();
  cout << (t-t0)/K << endl;
  cout << endl;

  array2<Complex> h0(Lx,Ly,h[0]);

  Complex sum=0.0;
  for(unsigned int i=0; i < Lx; ++i) {
    for(unsigned int j=0; j < Ly; ++j) {
      sum += h0[i][j];
    }
  }

  cout << "sum=" << sum << endl;
  cout << endl;

  if(Lx*Ly < 200) {
    for(unsigned int i=0; i < Lx; ++i) {
      for(unsigned int j=0; j < Ly; ++j) {
        cout << h0[i][j] << " ";
      }
      cout << endl;
    }
  }
  return 0;
}
