#include "Complex.h"
#include "cfftw++.h"
#include "convolve.h"
#include <complex.h>

using namespace std;
using namespace utils;
using namespace Array;

namespace fftwpp {

class HybridConvolution {
  Application *app;
  fftPad *fft;
public:
  Convolution *convolve;

  HybridConvolution(size_t L, multiplier mult=multBinary, size_t M=0, size_t A=2, size_t B=1, size_t threads=fftw::maxthreads) {
    if(M == 0) M=A*L-A+1;
    app=new Application(A,B,mult,threads);
    fft=new fftPad(L,M,*app);
    convolve=new Convolution(fft);
  }

  ~HybridConvolution() {
    delete convolve;
    delete fft;
    delete app;
  }
};

class HybridConvolutionHermitian {
  Application *app;
  fftPadHermitian *fft;
public:
  Convolution *convolve;

  HybridConvolutionHermitian(size_t L, multiplier mult=realMultBinary, size_t M=0, size_t A=2, size_t B=1, size_t threads=fftw::maxthreads) {
    if(M == 0) M=3*ceilquotient(L,2)-2*(L%2);
    app=new Application(A,B,mult,threads);
    fft=new fftPadHermitian(L,M,*app);
    convolve=new Convolution(fft);
  }

  ~HybridConvolutionHermitian() {
    delete convolve;
    delete fft;
    delete app;
  }
};

class HybridConvolution2 {
  Application *appx;
  fftPad *fftx;
  Application *appy;
  fftPad *ffty;
public:
  Convolution2 *convolve2;

  HybridConvolution2(size_t Lx, size_t Ly, multiplier mult=multBinary, size_t Mx=0, size_t My=0, size_t A=2, size_t B=1, size_t threads=fftw::maxthreads)
  {
    if(Mx == 0) Mx=A*Lx-A+1;
    if(My == 0) My=A*Ly-A+1;
    appx=new Application(A, B, multNone, threads);
    fftx=new fftPad(Lx, Mx, *appx, Ly);
    appy=new Application(A, B, mult, *appx);
    ffty=new fftPad(Ly, My, *appy);
    convolve2=new Convolution2(fftx, ffty);
  }

  ~HybridConvolution2() {
    delete convolve2;
    delete ffty;
    delete appy;
    delete fftx;
    delete appx;
  }
};

class HybridConvolutionHermitian2 {
  Application *appx;
  Application *appy;
  fftPadCentered *fftx;
  fftPadHermitian *ffty;

public:
  Convolution2 *convolve2;

  HybridConvolutionHermitian2(size_t Lx, size_t Ly,
                              multiplier mult=realMultBinary, size_t Mx=0,
                              size_t My=0, size_t A=2, size_t B=1,
                              size_t threads=fftw::maxthreads)
  {
    if(Mx == 0) Mx=3*ceilquotient(Lx,2)-2*(Lx%2);
    if(My == 0) My=3*ceilquotient(Ly,2)-2*(Ly%2);
    size_t Hy=ceilquotient(Ly,2);
    appx = new Application(A,B,multNone,threads);
    fftx = new fftPadCentered(Lx,Mx,*appx,Hy,Hy);
    appy = new Application(A,B,mult,*appx);
    ffty = new fftPadHermitian(Ly,My,*appy);
    convolve2 = new Convolution2(fftx,ffty);
  }

  ~HybridConvolutionHermitian2() {
    delete convolve2;
    delete ffty;
    delete appy;
    delete fftx;
    delete appx;
  }
};

class HybridConvolution3 {
  Application *appx;
  fftPad *fftx;
  Application *appy;
  fftPad *ffty;
  Application *appz;
  fftPad *fftz;
public:
  Convolution3 *convolve3;

  HybridConvolution3(size_t Lx, size_t Ly, size_t Lz,  multiplier mult=multBinary, size_t Mx=0, size_t My=0, size_t Mz=0, size_t A=2, size_t B=1, size_t threads=fftw::maxthreads)
  {
    if(Mx == 0) Mx=A*Lx-A+1;
    if(My == 0) My=A*Ly-A+1;
    if(Mz == 0) Mz=A*Lz-A+1;
    appx = new Application(A, B, multNone, threads);
    fftx = new fftPad(Lx, Mx, *appx, Ly*Lz);
    appy = new Application(A, B, multNone, *appx);
    ffty = new fftPad(Ly, My, *appy, Lz);
    appz = new Application(A, B, mult, *appy);
    fftz = new fftPad(Lz, Mz, *appz);
    convolve3 = new Convolution3(fftx, ffty, fftz);
  }

  ~HybridConvolution3() {
    delete convolve3;
    delete fftz;
    delete appz;
    delete ffty;
    delete appy;
    delete fftx;
    delete appx;
  }
};

class HybridConvolutionHermitian3 {
  Application *appx;
  fftPadCentered *fftx;
  Application *appy;
  fftPadCentered *ffty;
  Application *appz;
  fftPadHermitian *fftz;
public:
  Convolution3 *convolve3;

  HybridConvolutionHermitian3(size_t Lx, size_t Ly, size_t Lz, multiplier mult=realMultBinary, size_t Mx=0, size_t My=0, size_t Mz=0, size_t A=2, size_t B=1, size_t threads=fftw::maxthreads)
  {
    if(Mx == 0) Mx=3*ceilquotient(Lx,2)-2*(Lx%2);
    if(My == 0) My=3*ceilquotient(Ly,2)-2*(Ly%2);
    if(Mz == 0) Mz=3*ceilquotient(Lz,2)-2*(Lz%2);
    size_t Hz=ceilquotient(Lz,2);
    appx = new Application(A, B, multNone, threads);
    fftx = new fftPadCentered(Lx, Mx, *appx, Ly*Hz);
    appy = new Application(A, B, multNone, *appx);
    ffty = new fftPadCentered(Ly, My, *appy, Hz);
    appz = new Application(A, B, mult, *appy);
    fftz = new fftPadHermitian(Lz, Mz, *appz);
    convolve3 = new Convolution3(fftx, ffty, fftz);
  }

  ~HybridConvolutionHermitian3() {
    delete convolve3;
    delete fftz;
    delete appz;
    delete ffty;
    delete appy;
    delete fftx;
    delete appx;
  }
};

};
