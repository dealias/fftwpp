#include "Complex.h"
#include "convolution.h"
#include "Array.h"

// Compile with:
// g++ -I .. -fopenmp examplecconv3.cc ../convolution.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

// size of problem
unsigned int mx=4;
unsigned int my=4;
unsigned int mz=4;

inline void init(array3<Complex>& f, array3<Complex>& g, unsigned int M=1) 
{
  double factor=1.0/sqrt((double) M);
  for(unsigned int s=0; s < M; ++s) {
    double S=sqrt(1.0+s);
    double ffactor=S*factor;
    double gfactor=1.0/S*factor;
    for(unsigned int i=0; i < mx; ++i) {
      unsigned int I=s*mx+i;
      for(unsigned int j=0; j < my; j++) {
        for(unsigned int k=0; k < mz; k++) {
          f[I][j][k]=ffactor*Complex(i+k,j+k);
          g[I][j][k]=gfactor*Complex(2*i+k,j+1+k);
        }
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

  cout << "3D non-centered complex convolution:" << endl;

  size_t align=sizeof(Complex);
  array3<Complex> f(mx,my,mz,align);
  array3<Complex> g(mx,my,mz,align);

  cout << "\ninput:" << endl;
  init(f,g);
  cout << "f:" << endl << f;
  cout << "g:" << endl << g;

  ImplicitConvolution3 C(mx,my,mz);
  C.convolve(f,g);

  cout << "\noutput:" << endl << f;

  return 0;
}
