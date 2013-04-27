#include "Complex.h"
#include "convolution.h"
#include "Array.h"

// Compile with:
// g++ -I .. -fopenmp examplecconv3.cc ../convolution.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace Array;
using namespace fftwpp;

// size of problem
unsigned int mx=4;
unsigned int my=4;
unsigned int mz=4;

void pmult(Complex **A,
           unsigned int m, unsigned int M,
           unsigned int offset) {
  Complex* a0=A[0]+offset;
  Complex* a1=A[1]+offset;
  for(unsigned int i=0; i < m; ++i) {
    a0[i] *= a1[i];
  }
}

inline void init(array3<Complex>& f, array3<Complex>& g, unsigned int M=1) 
{
  double factor=1.0/sqrt((double) M);
  for(unsigned int s=0; s < M; ++s) {
    double S=sqrt(1.0+s);
    double ffactor=S*factor;
    double gfactor=1.0/S*factor;
    for(unsigned int i=0; i < mx; ++i) {
      unsigned int I=s*mx+i;
      for(unsigned int j=0; j < my; j++) {
        for(unsigned int k=0; k < mz; k++) {
          f[I][j][k]=ffactor*Complex(i+k,j+k);
          g[I][j][k]=gfactor*Complex(2*i+k,j+1+k);
        }
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

  // 3d non-centered complex convolution
  cout << endl << "3D non-centered complex convolution:" << endl;
  size_t align=sizeof(Complex);
  array3<Complex> f(mx,my,mz,align);
  array3<Complex> g(mx,my,mz,align);

  cout << "input:" << endl;
  init(f,g);
  cout << "f:" << endl << f << endl;
  cout << "g:" << endl << g << endl;


  unsigned int Min=2, Mout=1;

  pImplicitConvolution3 C(mx,my,mz,Min,Mout);
  Complex **FF=new Complex *[2];
  FF[0]=f;
  FF[1]=g;

  
  Complex **U3;
  Complex ***U2;
  Complex ****U1;
  
  U3=new Complex*[Min];
  for(unsigned int i=0; i < Min; ++i)
    U3[i]=ComplexAlign(mx*my*mz);
  
  U2=new Complex**[fftw::maxthreads];
  for(unsigned int t=0; t < fftw::maxthreads; ++t) {
    U2[t]=new Complex*[Min];
    for(unsigned int i=0; i < Min; ++i) {
      U2[t][i]=ComplexAlign(my*mz);
    }
  }

  U1=new Complex***[fftw::maxthreads];
  for(unsigned int t=0; t < fftw::maxthreads; ++t) {
    U1[t]=new Complex**[Min];
    U1[t][0]=new Complex*[1];
    for(unsigned int i=0; i < Min; ++i) {
      U1[t][0][i]=ComplexAlign(mz);
    }
  }
  
  unsigned int offset=0;
  C.convolve(FF,U3,U2,U1,pmult,offset);

  cout << "output:" << endl;
  cout << f << endl;

  return 0;
}
