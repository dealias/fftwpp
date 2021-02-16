#include "convolve.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

int main(int argc, char* argv[])
{
  fftw::maxthreads=1;//get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  L=512;
  M=768;

  optionsHybrid(argc,argv);

  ForwardBackward FB;
  Application *app=&FB;

  unsigned int Lx=L;
  unsigned int Ly=L;
  unsigned int Mx=M;
  unsigned int My=M;

  cout << "Lx=" << Lx << endl;
  cout << "Mx=" << Mx << endl;
  cout << endl;

//      fftPadCentered fftx(Lx,Mx,Ly,Lx,2,1);
  fftPadCentered fftx(Lx,Mx,*app,Ly);

//      fftPadHermitian ffty(Ly,My,1,Ly,2,1);
  fftPadHermitian ffty(Ly,My,FB,1);

  ConvolutionHermitian convolvey(ffty);

  Complex **f=new Complex *[A];
  Complex **h=new Complex *[B];

  unsigned int Hy=ceilquotient(Ly,2);

  for(unsigned int a=0; a < A; ++a)
    f[a]=ComplexAlign(Lx*Hy);
  for(unsigned int b=0; b < B; ++b)
    h[b]=ComplexAlign(Lx*Hy);

  array2<Complex> f0(Lx,Hy,f[0]);
  array2<Complex> f1(Lx,Hy,f[1]);

  for(unsigned int i=0; i < Lx; ++i) {
    for(unsigned int j=0; j < Hy; ++j) {
      f0[i][j]=Complex(i,j*0);
      f1[i][j]=Complex(2*i,(j+1)*0);
    }
  }

  // TODO: HermitianSymmetrize

  if(Lx*Hy < 200) {
    for(unsigned int i=0; i < Lx; ++i) {
      for(unsigned int j=0; j < Hy; ++j) {
        cout << f0[i][j] << " ";
      }
      cout << endl;
    }
  }

  ConvolutionHermitian2 Convolve2(fftx,convolvey);

  unsigned int K=1000;
  double t0=totalseconds();

  for(unsigned int k=0; k < K; ++k)
    Convolve2.convolve(f,h,realmultbinary);

  double t=totalseconds();
  cout << (t-t0)/K << endl;
  cout << endl;

  array2<Complex> h0(Lx,Hy,h[0]);

  Complex sum=0.0;
  for(unsigned int i=0; i < Lx; ++i) {
    for(unsigned int j=0; j < Hy; ++j) {
      sum += h0[i][j];
    }
  }

  cout << "sum=" << sum << endl;
  cout << endl;

  if(Lx*Ly < 200) {
    for(unsigned int i=0; i < Lx; ++i) {
      for(unsigned int j=0; j < Hy; ++j) {
        cout << h0[i][j] << " ";
      }
      cout << endl;
    }
  }
  return 0;
}
