#include "Complex.h"
#include "convolve.h"
#include "Array.h"

// Compile with:
// g++ -I .. -Ofast -fopenmp exampleconv3.cc ../convolve.cc ../fftw++.cc ../parallel.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;
using namespace parallel;

inline void init(array3<Complex>& f, array3<Complex>& g)
{
  size_t Lx=f.Nx();
  size_t Ly=f.Ny();
  size_t Lz=f.Ny();
  for(size_t i=0; i < Lx; ++i) {
    for(size_t j=0; j < Ly; ++j) {
      for(size_t k=0; k < Lz; ++k) {
        f[i][j][k]=Complex(i+1,j+3+k);
        g[i][j][k]=Complex(i+k+1,2*j+3+k);
      }
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
  size_t Lz=4; // Length of input arrays in z direction
  size_t Mx=7; // Minimal padded length for dealiasing via 1/2 padding
  size_t My=7; // Minimal padded length for dealiasing via 1/2 padding
  size_t Mz=7; // Minimal padded length for dealiasing via 1/2 padding

  cout << "3D complex convolution:" << endl;

  size_t align=sizeof(Complex);
  array3<Complex> f(Lx,Ly,Lz,align);
  array3<Complex> g(Lx,Ly,Lz,align);
  Complex *F[]={f,g};

  init(f,g);

  cout << "\ninput:" << endl;
  cout << "f:" << endl << f;
  cout << "g:" << endl << g;

  Application appx(A,B,multNone,fftw::maxthreads);
  fftPad fftx(Lx,Mx,appx,Ly*Lz);
  Application appy(A,B,multNone,appx);
  fftPad ffty(Ly,My,appy,Lz);
  Application appz(A,B,multBinary,appy);
  fftPad fftz(Lz,Mz,appz);
  Convolution3 C(&fftx,&ffty,&fftz);

  C.convolve(F);

  cout << "\noutput:" << endl << f;

  return 0;
}
