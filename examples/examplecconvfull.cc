#include "Complex.h"
#include "convolution.h"
#include "Array.h"

// Compile with:
// g++ -I .. -fopenmp examplecconv.cc ../convolution.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace fftwpp;
using namespace parallel;

double f0[]={0,1,2,3,4};

void init(Complex *f, Complex *g, size_t m)
{
  for(size_t k=0; k < m; k++) {
    f[k]=f0[k];
    g[k]=f0[k];
  }
}

void multBinaryfull(Complex **F, size_t m,
                    const size_t indexsize,
                    const size_t *index,
                    size_t r, size_t threads)
{
  Complex* F0=F[0];
  Complex* F1=F[1];

  double sign=r == 0 ? 1 : -1;

  PARALLEL(
    for(size_t j=0; j < m; ++j) {
      Complex *F0j=F0+j;
      Complex *F1j=F1+j;
      Vec h=ZMULT(LOAD(F0j),LOAD(F1j));
      STORE(F0j,h);
      STORE(F1j,sign*h);
    }
    );
}

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

  // size of problem
  size_t m=sizeof(f0)/sizeof(double);

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  // allocate arrays:
  Complex *F[2];
  Complex *f=F[0]=ComplexAlign(2*m);
  Complex *g=F[1]=f+m;

  ImplicitConvolution C(m,2,2);

  cout << "1d non-centered complex convolution:" << endl;
  init(f,g,m);
  cout << "\ninput:\nf\tg" << endl;
  for(size_t i=0; i < m; i++)
    cout << f[i] << "\t" << g[i] << endl;

  C.convolve(F,multBinaryfull);

  cout << "\noutput:" << endl;
  for(size_t i=0; i < m; i++) cout << f[i] << endl;
  for(size_t i=0; i < m-1; i++) cout << g[i] << endl;

  deleteAlign(f);

  return 0;
}
