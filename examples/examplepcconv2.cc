#include "Complex.h"
#include "convolution.h"
#include "Array.h"

using namespace std;
using namespace Array;
using namespace fftwpp;

void pmult(Complex **f,
           unsigned int m, unsigned int M,
           unsigned int offset) {
  Complex* f0=f[0]+offset;
  Complex* f1=f[1]+offset;
  for(unsigned int j=0; j < m; ++j)
    f0[j] *= f1[j];
}

inline void init(array2<Complex> &f0, array2<Complex> &f1, 
		 unsigned int mx, unsigned int my) 
{
  for(unsigned int i=0; i < mx; ++i) {
    for(unsigned int j=0; j < my; ++j) {
      f0[i][j]=Complex(i,j);
      f1[i][j]=Complex(2*i,j+1);
    }
  }
}

int main(int argc, char* argv[])
{
  cout << "2D non-centered complex convolution:" << endl;
  
  // Set maximum number of threads to be used:
  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  

  // Problem Size:
  unsigned int mx=4;
  unsigned int my=4;
  
  unsigned int A=2; // Number of inputs
  unsigned int B=1; // Number of outputs

  // Allocate input arrays:
  Complex **f = new Complex *[A];
  for(unsigned int s=0; s < A; ++s)
    f[s]=ComplexAlign(mx*my);
  
  // Set up the input:
  array2<Complex> f0(mx,my,f[0]);
  array2<Complex> f1(mx,my,f[1]);
  init(f0,f1,mx,my);
  cout << "input:" << endl;
  cout << "f[0]:" << endl << f0 << endl;
  cout << "f[1]:" << endl << f1 << endl;

  // Creat a convolution object C:
  pImplicitConvolution2 C(mx,my,A,B);

  // Perform the convolution:
  C.convolve(f,pmult,0);

  // Display output:
  cout << "output:" << f0 << endl;

  // Free input arrays:
  for(unsigned int s=0; s < A; ++s) 
    deleteAlign(f[s]);
  delete[] f;

  return 0;
}
