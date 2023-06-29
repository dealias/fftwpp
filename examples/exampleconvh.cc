#include "Complex.h"
#include "convolve.h"
#include "Array.h"

// Compile with:
// g++ -I .. -fopenmp exampleconv.cc ../convolve.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace fftwpp;
using namespace parallel;

inline void init(Complex *f, Complex *g, unsigned int H)
{
  f[0]=Complex(1,0);
  g[0]=Complex(2,0);
  for(unsigned int k=1; k < H; k++) {
    f[k]=Complex(k,k+1);
    g[k]=Complex(k,2*k+1);
  }
}

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

  size_t A=2; // Two inputs
  size_t B=1; // One output

  size_t L=7; // Length of input arrays
  size_t M=10; // Minimal padded length for dealiasing via 1/2 padding

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  // 1d centered Hermitian-symmetric complex convolution
  cout << "1D centered Hermitian-symmetric convolution:" << endl;

  // allocate arrays:
  size_t H=ceilquotient(L,2);

  Complex *f=ComplexAlign(H);
  Complex *g=ComplexAlign(H);
  Complex *F[]={f,g};

  init(f,g,H);
  cout << "\ninput:\nf\tg" << endl;
  for(unsigned int i=0; i < H; i++)
    cout << f[i] << "\t" << g[i] << endl;

  Application app(A,B,realmultbinary,fftw::maxthreads);
  fftPadHermitian fft(L,M,app);
  Convolution C(&fft);

  C.convolve(F);

  cout << "\noutput:" << endl;
  for(unsigned int i=0; i < H; i++)
    cout << f[i] << endl;

  deleteAlign(g);
  deleteAlign(f);

  return 0;
}
