#include "Complex.h"
#include "convolve.h"
#include "Array.h"

// Compile with:
// g++ -I .. -Ofast -fopenmp exampleconvh3.cc ../convolve.cc ../fftw++.cc ../parallel.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;
using namespace parallel;

inline void init(array3<Complex>& f, array3<Complex>& g)
{
  size_t Lx=f.Nx();
  size_t Ly=f.Ny();
  size_t Hz=f.Nz();
  for(size_t i=0; i < Lx; ++i) {
    for(size_t j=0; j < Ly; ++j) {
      for(size_t k=0; k < Hz; ++k) {
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

  size_t Hx=4;
  size_t Lx=2*Hx-1; // Length of input arrays in x direction
  size_t Hy=4;
  size_t Ly=2*Hy-1; // Length of input arrays in y direction
  size_t Hz=4;      // Length of input arrays in z direction
  size_t Lz=2*Hy-1;
  size_t Mx=10;     // Minimal padded length for dealiasing via 2/3 padding
  size_t My=10;     // Minimal padded length for dealiasing via 2/3 padding
  size_t Mz=10;     // Minimal padded length for dealiasing via 2/3 padding

  cout << "3D Hermitian-symmetric convolution:" << endl;

  size_t align=sizeof(Complex);
  array3<Complex> f(Lx,Ly,Hz,align);
  array3<Complex> g(Lx,Ly,Hz,align);
  Complex *F[]={f,g};

  init(f,g);
  size_t x0=Lx/2; // x center
  size_t y0=Ly/2; // y center
  HermitianSymmetrizeXY(Hx,Hy,Hz,x0,y0,f);
  HermitianSymmetrizeXY(Hx,Hy,Hz,x0,y0,g);

  cout << "\ninput:" << endl;
  cout << "f:" << endl << f;
  cout << "g:" << endl << g;

  Application appx(A,B,multNone,fftw::maxthreads);
  fftPadCentered fftx(Lx,Mx,appx,Ly*Hz);
  Application appy(A,B,multNone,appx);
  fftPadCentered ffty(Ly,My,appy,Hz);
  Application appz(A,B,realMultBinary,appy);
  fftPadHermitian fftz(Lz,Mz,appz);
  Convolution3 C(&fftx,&ffty,&fftz);

  C.convolve(F);

  cout << "\noutput:" << endl << f;

  return 0;
}
