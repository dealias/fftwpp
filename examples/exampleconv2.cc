#include "Complex.h"
#include "convolve.h"
#include "Array.h"

// Compile with:
// g++ -I .. -fopenmp exampleconv2.cc ../convolve.cc ../fftw++.cc ../parallel.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;
using namespace parallel;

inline void init(array2<Complex>& f, array2<Complex>& g)
{
  size_t Lx=f.Nx();
  size_t Ly=f.Nx();
  for(size_t i=0; i < Lx; ++i) {
    for(size_t j=0; j < Ly; j++) {
      f[i][j]=Complex(i,j);
      g[i][j]=Complex(2*i,j+1);
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

  size_t Lx=4; // Length of input arrays in x direction
  size_t Ly=4; // Length of input arrays in y direction
  size_t Mx=7; // Minimal padded length for dealiasing via 1/2 padding
  size_t My=7; // Minimal padded length for dealiasing via 1/2 padding

  cout << "2D non-centered complex convolution:" << endl;

  size_t align=sizeof(Complex);
  array2<Complex> f(Lx,Ly,align);
  array2<Complex> g(Lx,Ly,align);
  Complex *F[]={f,g};

  init(f,g);

  cout << "\ninput:" << endl;
  cout << "f:" << endl << f;
  cout << "g:" << endl << g;

  Application appx(A,B,multNone,fftw::maxthreads);
  fftPad fftx(Lx,Mx,appx,Ly);
  Application appy(A,B,multBinary,appx);
  fftPad ffty(Ly,My,appy);
  Convolution2 C(&fftx,&ffty);

  C.convolve(F);

  cout << "\noutput:" << endl << f;

  return 0;
}
