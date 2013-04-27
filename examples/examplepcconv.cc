#include "Complex.h"
#include "convolution.h"
#include "Array.h"

// Compile with:
// g++ -I .. -fopenmp examplecconv.cc ../convolution.cc ../fftw++.cc -lfftw3 -lfftw3_omp

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
    //A[0][i] *= A[1][i];
    a0[i] *= a1[i];
  }
}

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
  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  
  
   // 1d non-centered complex convolution
    
  // allocate arrays:
  Complex *f=ComplexAlign(m);
  Complex *g=ComplexAlign(m);
  
  cout << endl << "non-centered complex convolution:" << endl;
  init(f,g);
  cout << "input:\nf\tg" << endl;
  for(unsigned int i=0; i < m; i++) 
    cout << f[i] << "\t" << g[i] << endl;
  
  unsigned int Min=2;
  unsigned int Mout=1;

  pImplicitConvolution C(m,Min,Mout);
  Complex **FF=new Complex *[Min];
  FF[0]=f;
  FF[1]=g;
  Complex **U1=new Complex *[Min];
  for(unsigned int i=0; i < Min; ++i) 
    U1[i]=ComplexAlign(m);
  C.convolve(FF,U1,pmult);
  
  cout << "\noutput:" << endl;
  for(unsigned int i=0; i < m; i++) cout << f[i] << endl;
  //for(unsigned int i=0; i < m; i++) cout << g[i] << endl;
    
  deleteAlign(g);
  deleteAlign(f);

  return 0;
}
