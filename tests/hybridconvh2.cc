#include "convolve.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

unsigned int A=2; // number of inputs
unsigned int B=1; // number of outputs
unsigned int C=1; // number of copies
unsigned int L=512; // input data length
unsigned int M=768; // minimum padded length

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

  unsigned int Hx=ceilquotient(Lx,2);
  unsigned int Hy=ceilquotient(Ly,2);

  Complex **f=new Complex *[A];
  Complex **h=f;//=new Complex *[B];

  for(unsigned int a=0; a < A; ++a)
    f[a]=ComplexAlign(Lx*Hy);
//  for(unsigned int b=0; b < B; ++b)
//    h[b]=ComplexAlign(Lx*Hy);


  array2<Complex> f0(Lx,Hy,f[0]);
  array2<Complex> f1(Lx,Hy,f[1]);

  array2<Complex> h0(Lx,Hy,h[0]);

//  fftPadCentered fftx(Lx,Mx,Ly,Lx,2,1);
//  fftPadHermitian ffty(Ly,My,FB,1);

//  fftPadCentered fftx(Lx,Mx,FB,Hy);
//  fftPadHermitian ffty(Ly,My,1,Hy,3,2);
//  ConvolutionHermitian convolvey(ffty,A,B);
//  ConvolutionHermitian2 Convolve2(fftx,convolvey);

  ConvolutionHermitian2 Convolve2(Lx,Ly,Mx,My,A,B);

  unsigned int K=500;
//  double t0=totalseconds();

  double T=0;

  for(unsigned int k=0; k < K; ++k) {
    for(unsigned int i=0; i < Lx; ++i) {
      for(unsigned int j=0; j < Hy; ++j) {
        int I=Lx % 2 ? i : -1+i;
        f0[i][j]=Complex(I,j);
        f1[i][j]=Complex(2*I,(j+1));
      }
    }

    HermitianSymmetrizeX(Hx,Hy,Lx/2,f0);
    HermitianSymmetrizeX(Hx,Hy,Lx/2,f1);

    if(Lx*Hy < 200 && k == 0) {
      for(unsigned int i=0; i < Lx; ++i) {
        for(unsigned int j=0; j < Hy; ++j) {
          cout << f0[i][j] << " ";
        }
        cout << endl;
      }
      cout << endl;
    }

    seconds();
    Convolve2.convolve(f,h,realmultbinary);
//    Convolve2.convolve(f,h,multadvection2);
    T += seconds();
  }

  cout << "mean=" << T/K << endl;

//  double t=totalseconds();
//  cout << (t-t0)/K << endl;
//  cout << endl;

  Complex sum=0.0;
  for(unsigned int i=0; i < Lx; ++i) {
    for(unsigned int j=0; j < Hy; ++j) {
      sum += h0[i][j];
    }
  }

  cout << "sum=" << sum << endl;
  cout << endl;

  if(Lx*Hy < 200) {
    for(unsigned int i=0; i < Lx; ++i) {
      for(unsigned int j=0; j < Hy; ++j) {
        cout << h0[i][j] << " ";
      }
      cout << endl;
    }
  }
  return 0;
}
