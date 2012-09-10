#include "Complex.h"
#include "convolution.h"

// Compile with:
// g++ -fopenmp exampleconv.cc convolution.cc fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace fftwpp;

// size of problem
unsigned int m=8;

inline void init(Complex *f, Complex *g) 
{
  unsigned int m1=m+1;
  for(unsigned int i=0; i < m1; i += m1) {
    double s=sqrt(1.0+i);
    double ffactor=(1.0+i)*s;
    double gfactor=1.0/(1.0+i);
    Complex *fi=f+i;
    Complex *gi=g+i;
    fi[0]=1.0*ffactor;
    for(unsigned int k=1; k < m; k++) fi[k]=ffactor*Complex(k,k+1);
    gi[0]=2.0*gfactor;
    for(unsigned int k=1; k < m; k++) gi[k]=gfactor*Complex(k,2*k+1);
  }
}

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  
  
  cout << "size m=" << m << endl;
  
  // allocate arrays:
  Complex *f=ComplexAlign(m);
  Complex *g=ComplexAlign(m);

  // initialize arrays:
  init(f,g);
  cout << endl << "input:" << endl;
  for(unsigned int i=0; i < m; i++) 
    cout << f[i] << "\t" << g[i] << endl;
  
  {
    cout << endl << "non-centered complex convolution:" << endl;
    ImplicitConvolution C(m);
    C.convolve(f,g);
    for(unsigned int i=0; i < m; i++) cout << f[i] << endl;
  }

  init(f,g);
  {
    cout << endl << "centered Hermitian-symmetric convolution:" << endl;
    ImplicitHConvolution C(m);
    C.convolve(f,g);
    for(unsigned int i=0; i < m; i++) cout << f[i] << endl;
  }

  deleteAlign(g);
  deleteAlign(f);

  return 0;
}
