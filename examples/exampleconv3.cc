#include "Complex.h"
#include "convolution.h"
#include "Array.h"

// Compile with:
// g++ -I .. -fopenmp exampleconv3.cc ../convolution.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

// size of problem
unsigned int mx=4;
unsigned int my=4;
unsigned int mz=4;

inline void init(array3<Complex>& f, array3<Complex>& g, unsigned int M=1) 
{
  unsigned int xstop=2*mx-1;
  unsigned int ystop=2*my-1;
  double factor=1.0/sqrt((double) M);
  for(unsigned int s=0; s < M; ++s) {
    double S=sqrt(1.0+s);
    double ffactor=S*factor;
    double gfactor=1.0/S*factor;
    for(unsigned int i=0; i < xstop; ++i) {
      unsigned int I=s*xstop+i;
      for(unsigned int j=0; j < ystop; ++j) {
        for(unsigned int k=0; k < mz; ++k) {
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
  
  cout << "3D centered Hermitian-symmetric convolution:" << endl;

  size_t align=sizeof(Complex);
  array3<Complex> f(2*mx-1,2*my-1,mz,align);
  array3<Complex> g(2*mx-1,2*my-1,mz,align);

  init(f,g);
  cout << "\ninput:" << endl;
  cout << "f:" << endl << f;
  cout << "g:" << endl << g;
  
  /*
    cout << "input after symmetrization (done automatically):" << endl;
    HermitianSymmetrizeXY(mx,my,mz,mx-1,my-1,f);
    HermitianSymmetrizeXY(mx,my,mz,mx-1,my-1,g);
    cout << "f:" << endl << f << endl;
    cout << "g:" << endl << g << endl;
  */
  
  bool symmetrize=true;
  ImplicitHConvolution3 C(mx,my,mz);
  C.convolve(f,g,symmetrize);
  cout << "output:" << endl << f;

  return 0;
}
