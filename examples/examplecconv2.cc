#include "Complex.h"
#include "convolution.h"
#include "Array.h"

// Compile with:
// g++ -I .. -fopenmp examplecconv2.cc ../convolution.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

// size of problem
unsigned int mx=4;
unsigned int my=4;

inline void init(array2<Complex>& f, array2<Complex>& g) 
{
  for(unsigned int i=0; i < mx; ++i) {
    for(unsigned int j=0; j < my; j++) {
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

  cout << "2D non-centered complex convolution:" << endl;

  size_t align=sizeof(Complex);
  array2<Complex> f(mx,my,align);
  array2<Complex> g(mx,my,align);
  init(f,g);

  cout << "\ninput:" << endl;
  cout << "f:" << endl << f;
  cout << "g:" << endl << g;

  ImplicitConvolution2 C(mx,my);
  C.convolve(f,g);

  cout << "\noutput:" << endl << f;

  return 0;
}
