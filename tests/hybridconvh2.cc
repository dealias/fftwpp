#include "convolve.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

inline void HermitianSymmetrizeX(unsigned int mx, unsigned int my,
                                 unsigned int xorigin, Complex *f)
{
  unsigned int offset=xorigin*my;
  unsigned int stop=mx*my;
  f[offset].im=0.0;
  for(unsigned int i=my; i < stop; i += my)
    f[offset-i]=conj(f[offset+i]);
}

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

  unsigned int Hx=ceilquotient(Lx,2);
  unsigned int Hy=ceilquotient(Ly,2);

//      fftPadCentered fftx(Lx,Mx,Ly,Lx,2,1);
  fftPadCentered fftx(Lx,Mx,*app,Hy);

  fftPadHermitian ffty(Ly,My,1,Hy,3,1);
//  fftPadHermitian ffty(Ly,My,FB,1);

  ConvolutionHermitian convolvey(ffty);

  Complex **f=new Complex *[A];
  Complex **h=new Complex *[B];

  for(unsigned int a=0; a < A; ++a)
    f[a]=ComplexAlign(Lx*Hy);
  for(unsigned int b=0; b < B; ++b)
    h[b]=ComplexAlign(Lx*Hy);

  array2<Complex> f0(Lx,Hy,f[0]);
  array2<Complex> f1(Lx,Hy,f[1]);

  for(unsigned int i=0; i < Lx; ++i) {
    for(unsigned int j=0; j < Hy; ++j) {
      f0[i][j]=Complex(i,j);
      f1[i][j]=Complex(2*i,(j+1));
    }
  }

  HermitianSymmetrizeX(Hx,Hy,Lx/2,f0);
  HermitianSymmetrizeX(Hx,Hy,Lx/2,f1);

  if(Lx*Hy < 200) {
    for(unsigned int i=0; i < Lx; ++i) {
      for(unsigned int j=0; j < Hy; ++j) {
        cout << f0[i][j] << " ";
      }
      cout << endl;
    }
  }

  ConvolutionHermitian2 Convolve2(fftx,convolvey);

  unsigned int K=10;
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
