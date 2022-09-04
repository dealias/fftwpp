#include "convolve.h"
#include "direct.h"

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
  if(Output || testError)
    K=1;
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
  //array2<Complex> h;


  cout << "sum=" << sum << endl;
  cout << endl;

  if(Output) {
    if(testError)
      cout << "Hybrid:" << endl;
    for(unsigned int i=0; i < Lx; ++i) {
      for(unsigned int j=0; j < Hy; ++j) {
        cout << h0[i][j] << " ";
      }
      cout << endl;
    }
  }

  if(testError) {

    array2<Complex> h(Lx,Hy,ComplexAlign(Lx*Sx));
    array2<Complex> g0(Lx,Sx,ComplexAlign(Lx*Sx));
    array2<Complex> g1(Lx,Sx,ComplexAlign(Lx*Sx));


    for(unsigned int i=0; i < Lx; ++i) {
      for(unsigned int j=0; j < Hy; ++j) {
        int I=Lx % 2 ? i : -1+i;
        g0[i][j]=Complex(I,j);
        g1[i][j]=Complex(2*I,(j+1));
      }
    }
    DirectHConvolution2 C(Hx,Hy);
    C.convolve(h,g0,g1,true);

    if(Output) {
      cout << endl;
      cout << "Direct:" << endl;
      for(unsigned int i=0; i < Lx; ++i) {
        for(unsigned int j=0; j < Hy; ++j) {
          cout << h[i][j] << " ";
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
        Complex hij=h[i][j];
        err += abs2(h0[i][j]-hij);
        norm += abs2(hij);
      }
    }
    double relError=sqrt(err/norm);
    cout << "Error: "<< relError << endl;

  }
  return 0;
}
