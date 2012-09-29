#include "Complex.h"
#include "convolution.h"
#include "Array.h"

// Compile with:
// g++ -I .. -fopenmp exampleconv.cc ../convolution.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace Array;
using namespace fftwpp;

// size of problem
unsigned int m=8;

unsigned int mx=4;
unsigned int my=4;

inline void init(Complex *f, Complex *g) 
{
  unsigned int m1=m+1;
  for(unsigned int i=0; i < m1; i += m1) {
    Complex *fi=f+i;
    Complex *gi=g+i;
    for(unsigned int k=0; k < m; k++) fi[k]=Complex(k,k+1);
    for(unsigned int k=0; k < m; k++) gi[k]=Complex(k,2*k+1);
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
  
  { // 1D convolutions:
    cout << "size m=" << m << endl;
    
    // allocate arrays:
    Complex *f=ComplexAlign(m);
    Complex *g=ComplexAlign(m);
    
    // initialize arrays:
    init(f,g);
    cout << endl << "input:" << endl;
    cout << "f\tg" << endl;
    for(unsigned int i=0; i < m; i++) 
      cout << f[i] << "\t" << g[i] << endl;
    
    { // 1d non-centered complex convolution
      cout << endl << "non-centered complex convolution:" << endl;
      ImplicitConvolution C(m);
      C.convolve(f,g);
      for(unsigned int i=0; i < m; i++) cout << f[i] << endl;
    }
    
    init(f,g);

    { // 1d centered Hermitian-symmetric complex convolution
      cout << endl << "centered Hermitian-symmetric convolution:" << endl;
      ImplicitHConvolution C(m);
      C.convolve(f,g);
      for(unsigned int i=0; i < m; i++) cout << f[i] << endl;
    }
    
    deleteAlign(g);
    deleteAlign(f);
  }

  { // 1d non-centered complex convolution
    cout << endl << "2D non-centered complex convolution:" << endl;
    size_t align=sizeof(Complex);
    array2<Complex> f(mx,my,align);
    array2<Complex> g(mx,my,align);
    init(f,g);
    cout << "f:" << endl << f << endl;
    cout << "g:" << endl << g << endl;
    ImplicitConvolution2 C(mx,my);
    C.convolve(f,g);
    cout << "non-centered complex convolution:" << endl << f << endl;
  }

  return 0;
}
