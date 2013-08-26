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
  for(unsigned int i=0; i < m; ++i)
    f0[i] *= f1[i];
}

inline void init(array3<Complex>& f0, array3<Complex>& f1, 
		 unsigned int mx, unsigned int my, unsigned int mz) 
{
  for(unsigned int i=0; i < mx; ++i) {
    for(unsigned int j=0; j < my; j++) {
      for(unsigned int k=0; k < mz; k++) {
	f0[i][j][k]=Complex(i+k,j+k);
	f1[i][j][k]=Complex(2*i+k,j+1+k);
      }
    }
  }
}

int main(int argc, char* argv[])
{
  // 3d non-centered complex convolution
  cout << endl << "3D non-centered complex convolution:" << endl;

  // Set maximum number of threads to be used:
  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  

  // Problem size:
  unsigned int mx=4;
  unsigned int my=4;
  unsigned int mz=4;

  unsigned int A=2; // Number of inputs
  unsigned int B=1; // Number of outputs

  // Allocate input arrays:
  Complex **f = new Complex *[A];
  for(unsigned int s=0; s < A; ++s)
    f[s]=ComplexAlign(mx*my*mz);
  array3<Complex> f0(mx,my,mz,f[0]);
  array3<Complex> f1(mx,my,mz,f[1]);

  cout << "input:" << endl;
  init(f0,f1,mx,my,mz);
  cout << "f[0]:" << endl << f0 << endl;
  cout << "f[1]:" << endl << f1 << endl;

  // Create convolution object C:
  pImplicitConvolution3 C(mx,my,mz,A,B);
  
  // Set up work arrays:
  Complex **U3;
  Complex ***U2;
  Complex ****U1;
  
  U3=new Complex*[A];
  for(unsigned int i=0; i < A; ++i)
    U3[i]=ComplexAlign(mx*my*mz);
  
  U2=new Complex**[fftw::maxthreads];
  for(unsigned int t=0; t < fftw::maxthreads; ++t) {
    U2[t]=new Complex*[A];
    for(unsigned int i=0; i < A; ++i) {
      U2[t][i]=ComplexAlign(my*mz);
    }
  }

  U1=new Complex***[fftw::maxthreads];
  for(unsigned int t=0; t < fftw::maxthreads; ++t) {
    U1[t]=new Complex**[A];
    U1[t][0]=new Complex*[1];
    for(unsigned int i=0; i < A; ++i) {
      U1[t][0][i]=ComplexAlign(mz);
    }
  }
  
  // Perform the convolution:
  unsigned int offset=0;
  C.convolve(f,U3,U2,U1,pmult,offset);

  // Display output:
  cout << "output:" << f0 << endl;

  // Free 1D work arrays:
  for(unsigned int t=0; t < fftw::maxthreads; ++t) {
    for(unsigned int i=0; i < A; ++i)
      deleteAlign(U1[t][0][i]);
    delete[] U1[t][0];
    delete[] U1[t];
  }
  delete[] U1;

  // Free 2D work arrays:
  for(unsigned int t=0; t < fftw::maxthreads; ++t) {
    for(unsigned int i=0; i < A; ++i)
      deleteAlign(U2[t][i]);
    delete[] U2[t];
  }
  delete[] U2;

  // Free 3D work arrays:
  for(unsigned int i=0; i < A; ++i)
    deleteAlign(U3[i]);
  delete[] U3;

  // Free input arrays:
  for(unsigned int s=0; s < A; ++s) 
    deleteAlign(f[s]);
  delete[] f;

  return 0;
}
