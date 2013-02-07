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

int main(int argc, char* argv[])
{
#ifndef FFTWPP_SINGLE_THREAD
  fftw::maxthreads=omp_get_max_threads();
#endif

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  
  
  // 1d centered Hermitian-symmetric complex convolution
  cout << endl << "centered Hermitian-symmetric convolution:" << endl;

  // allocate arrays:
  Complex *f=ComplexAlign(m);
  Complex *g=ComplexAlign(m);

  init(f,g);
  cout << "input:\nf\tg" << endl;
  for(unsigned int i=0; i < m; i++) 
    cout << f[i] << "\t" << g[i] << endl;
  
  ImplicitHConvolution C(m);
  C.convolve(f,g);
  
  cout << "\noutput:" << endl;
  for(unsigned int i=0; i < m; i++) cout << f[i] << endl;
    
  deleteAlign(g);
  deleteAlign(f);

  return 0;
}
