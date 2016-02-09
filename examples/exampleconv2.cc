#include "Complex.h"
#include "convolution.h"
#include "Array.h"

// Compile with:
// g++ -I .. -fopenmp exampleconv2.cc ../convolution.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

// size of problem
unsigned int m=8;

unsigned int mx=4;
unsigned int my=4;

inline void init(array2<Complex>& f, array2<Complex>& g, unsigned int M=1) 
{
  unsigned int stop=2*mx-1;
  unsigned int stopoffset=stop;
  double factor=1.0/sqrt((double) M);
  for(unsigned int s=0; s < M; ++s) {
    double S=sqrt(1.0+s);
    double ffactor=S*factor;
    double gfactor=1.0/S*factor;
    for(unsigned int i=0; i < stop; ++i) {
      unsigned int I=s*stopoffset+i;
      for(unsigned int j=0; j < my; j++) {
        f[I][j]=ffactor*Complex(i,j);
        g[I][j]=gfactor*Complex(2*i,j+1);
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
  
  cout << "2D centered Hermitian-symmetric convolution:" << endl;

  size_t align=sizeof(Complex);
  array2<Complex> f(2*mx-1,my,align);
  array2<Complex> g(2*mx-1,my,align);
  init(f,g);
  cout << "\ninput:" << endl;
  cout << "f:" << endl << f;
  cout << "g:" << endl << g;
  
  /*
    cout << "input after symmetrization (done automatically):" << endl;
    HermitianSymmetrizeX(mx,my,mx-1,f);
    HermitianSymmetrizeX(mx,my,mx-1,g);
    cout << "f:" << endl << f << endl;
    cout << "g:" << endl << g << endl;
  */
  
  bool symmetrize=true;
  ImplicitHConvolution2 C(mx,my);
  C.convolve(f,g,symmetrize);
  cout << "\noutput:" << endl << f;

  return 0;
}
