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

void pmult(Complex **A,
           unsigned int m, unsigned int M,
           unsigned int offset) {
  Complex* a0=A[0]+offset;
  Complex* a1=A[1]+offset;
  for(unsigned int i=0; i < m; ++i) {
    //cout << a0[i] << "\t" << a1[i] << endl;
    a0[i] *= a1[i];
  }
}

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

  // 2d non-centered complex convolution
  cout << endl << "2D non-centered complex convolution:" << endl;
  size_t align=sizeof(Complex);
  array2<Complex> f(mx,my,align);
  array2<Complex> g(mx,my,align);
  init(f,g);
  cout << "input:" << endl;
  cout << "f:" << endl << f << endl;
  cout << "g:" << endl << g << endl;

  unsigned int Min=2, Mout=2;
  
  pImplicitConvolution2 C(mx,my,Min,Mout);
  Complex **FF=new Complex *[Min];
  FF[0]=f;
  FF[1]=g;


  Complex **U2=new Complex *[Min];
  for(unsigned int i=0; i < Min; ++i) 
    U2[i]=ComplexAlign(mx*my);
  Complex ***U1=new Complex **[fftw::maxthreads];
  for(unsigned int t=0; t < fftw::maxthreads; ++t) {
    U1[t]=new Complex*[Min];
    for(unsigned int i=0; i < Min; ++i) {
      U1[t][i]=ComplexAlign(my);
    }
  }

  C.convolve(FF,U2,U1,pmult,0);

  cout << "output:" << endl;
  cout << f << endl;

  return 0;
}
