#include "Complex.h"
#include "convolution.h"
#include "Array.h"

// Compile with:
// g++ -I .. -fopenmp examplecconv2.cc ../convolution.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace Array;
using namespace fftwpp;

// size of problem
unsigned int m=8;

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
#ifndef FFTWPP_SINGLE_THREAD
  fftw::maxthreads=omp_get_max_threads();
#endif

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  

  // 2d non-centered complex convolution
  cout << endl << "2D non-centered complex convolution:" << endl;
  size_t align=sizeof(Complex);
  array2<Complex> f(mx,my,align);
  array2<Complex> g(mx,my,align);
  init(f,g);
  cout << "input:" << endl;
  cout << "f:" << endl << f << endl;
  cout << "g:" << endl << g << endl;
  ImplicitConvolution2 C(mx,my);
  C.convolve(f,g);
  cout << "output:" << endl;
  cout << f << endl;

  return 0;
}
