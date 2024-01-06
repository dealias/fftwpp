#include "Complex.h"
#include "convolve.h"
#include "Array.h"

// Compile with:
// g++ -I .. -Ofast -fopenmp exampleconvh2.cc ../convolve.cc ../fftw++.cc ../parallel.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;
using namespace parallel;

inline void init(array2<Complex>& f, array2<Complex>& g)
{
  size_t Lx=f.Nx();
  size_t Hy=f.Ny();
  for(size_t i=0; i < Lx; ++i) {
    int I=Lx % 2 ? i : -1+i;
    for(size_t j=0; j < Hy; ++j) {
      f[i][j]=Complex(I+1,j+3);
      g[i][j]=Complex(I+2,2*j+3);
    }
  }
}

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  size_t A=2; // Two inputs
  size_t B=1; // One output

  size_t Hx=4;
  size_t Lx=2*Hx-1; // Length of input arrays in x direction
  size_t Hy=4;      // Length of input arrays in y direction
  size_t Ly=2*Hy-1;
  size_t Mx=10;     // Minimal padded length for dealiasing via 2/3 padding
  size_t My=10;     // Minimal padded length for dealiasing via 2/3 padding

  cout << "2D Hermitian-symmetric convolution:" << endl;

  // allocate arrays:
  size_t align=sizeof(Complex);
  array2<Complex> f(Lx,Hy,align);
  array2<Complex> g(Lx,Hy,align);
  Complex *F[]={f,g};

  init(f,g);
  size_t x0=Lx/2; // x center
  HermitianSymmetrizeX(Hx,Hy,x0,f);
  HermitianSymmetrizeX(Hx,Hy,x0,g);

  cout << "\ninput:" << endl;
  cout << "f:" << endl << f;
  cout << "g:" << endl << g;

  Application appx(A,B,multNone,fftw::maxthreads);
  fftPadCentered fftx(Lx,Mx,appx,Hy);
  Application appy(A,B,realMultBinary,appx);
  fftPadHermitian ffty(Ly,My,appy);
  Convolution2 C(&fftx,&ffty);

  C.convolve(F);

  cout << "\noutput:" << endl << f;

  return 0;
}
