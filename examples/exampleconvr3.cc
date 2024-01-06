#include "Complex.h"
#include "convolve.h"
#include "Array.h"

// Compile with:
// g++ -I .. -Ofast -fopenmp exampleconvr3.cc ../convolve.cc ../fftw++.cc ../parallel.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;
using namespace parallel;

inline void init(array3<double>& f, array3<double>& g)
{
  size_t Lx=f.Nx();
  size_t Ly=f.Ny();
  size_t Lz=f.Ny();
  for(size_t i=0; i < Lx; ++i) {
    for(size_t j=0; j < Ly; ++j) {
      for(size_t k=0; k < Lz; ++k) {
        f[i][j][k]=i+j+1;
        g[i][j][k]=i+2*j+k+1;
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

  cout << "3D real convolution:" << endl;

  size_t align=sizeof(double);
  array3<double> f(Lx,Ly,Lz,align);
  array3<double> g(Lx,Ly,Lz,align);
  double *F[]={f,g};

  init(f,g);

  cout << "\ninput:" << endl;
  cout << "f:" << endl << f;
  cout << "g:" << endl << g;

  Application appx(A,B,multNone,fftw::maxthreads);
  fftPadReal fftx(Lx,Mx,appx,Ly*Lz);
  Application appy(A,B,multNone,appx);
  fftPad ffty(Ly,My,appy,Lz);
  Application appz(A,B,multBinary,appy);
  fftPad fftz(Lz,Mz,appz);
  Convolution3 C(&fftx,&ffty,&fftz);

  C.convolve((Complex **) F);

  cout << "\noutput:" << endl << f;

  return 0;
}
