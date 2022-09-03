#include "convolve.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

unsigned int A=2; // number of inputs
unsigned int B=1; // number of outputs
unsigned int L=7; // input data length
unsigned int M=12; // minimum padded length

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

  unsigned int K0=10000000;
  if(K == 0) K=max(K0/(Mx*My),20);
  cout << "K=" << K << endl << endl;

  unsigned int Sx=0; // x stride (0 means ceilquotient(Ly,2))

  cout << "Lx=" << Lx << endl;
  cout << "Mx=" << Mx << endl;
  cout << endl;

  unsigned int Hx=ceilquotient(Lx,2);
  unsigned int Hy=ceilquotient(Ly,2);

  if(Sx == 0) Sx=Hy;

  Complex **f=new Complex *[A];

  for(unsigned int a=0; a < A; ++a)
    f[a]=ComplexAlign(Lx*Sx);

  array2<Complex> f0(Lx,Sx,f[0]);
  array2<Complex> f1(Lx,Sx,f[1]);

  array2<Complex> h0(Lx,Sx,f[0]);

  Application app(A,B,realmultbinary);
//  Application app(A,B,multadvection2);

  fftPadCentered fftx(Lx,Mx,app,Hy,Sx);
  ConvolutionHermitian convolvey(Ly,My,app);

  ConvolutionHermitian2 Convolve2(&fftx,&convolvey);

  double T=0;

  for(unsigned int c=0; c < K; ++c) {

    for(unsigned int i=0; i < Lx; ++i) {
      for(unsigned int j=0; j < Hy; ++j) {
        int I=Lx % 2 ? i : -1+i;
        f0[i][j]=Complex(I,j);
        f1[i][j]=Complex(2*I,(j+1));
      }
    }

    HermitianSymmetrizeX(Hx,Hy,Lx/2,f0,Sx);
    HermitianSymmetrizeX(Hx,Hy,Lx/2,f1,Sx);

    if(Output) {
      for(unsigned int i=0; i < Lx; ++i) {
        for(unsigned int j=0; j < Hy; ++j) {
          cout << f0[i][j] << " ";
        }
        cout << endl;
      }
      cout << endl;
    }

    seconds();
    Convolve2.convolve(f);
    T += seconds();
  }

  cout << "median=" << T/K << endl;

  Complex sum=0.0;
  for(unsigned int i=0; i < Lx; ++i) {
    for(unsigned int j=0; j < Hy; ++j) {
      sum += h0[i][j];
    }
  }

  cout << "sum=" << sum << endl;
  cout << endl;

  if(Output) {
    for(unsigned int i=0; i < Lx; ++i) {
      for(unsigned int j=0; j < Hy; ++j) {
        cout << h0[i][j] << " ";
      }
      cout << endl;
    }
  }
  return 0;
}
