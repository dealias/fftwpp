#include "Complex.h"
#include "convolve.h"

// Compile with:
// g++ -I .. -Ofast -fopenmp exampleconvh.cc ../convolve.cc ../fftw++.cc ../parallel.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace fftwpp;
using namespace parallel;

inline void init(Complex *f, Complex *g, unsigned int H)
{
  for(size_t i=0; i < H; ++i) {
    f[i]=Complex(i+1,i+3);
    g[i]=Complex(i+2,2*i+3);
  }
}

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

  size_t A=2; // Two inputs
  size_t B=1; // One output

  size_t H=4; // Length of input arrays
  size_t M=10; // Minimal padded length for dealiasing via 2/3 padding

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  size_t L=2*H-1;

  cout << "1D Hermitian-symmetric convolution:" << endl;

  // allocate arrays:
  Complex *f=ComplexAlign(H);
  Complex *g=ComplexAlign(H);
  Complex *F[]={f,g};

  init(f,g,H);
  HermitianSymmetrize(f);
  HermitianSymmetrize(g);

  cout << "\ninput:\nf\tg" << endl;
  for(unsigned int i=0; i < H; i++)
    cout << f[i] << "\t" << g[i] << endl;

  Application app(A,B,realMultBinary,fftw::maxthreads);
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
