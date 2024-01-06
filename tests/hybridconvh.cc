#include <vector>

#include "convolve.h"
#include "timing.h"
#include "direct.h"
#include "options.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

size_t A=2; // number of inputs
size_t B=1; // number of outputs

int main(int argc, char *argv[])
{
  L=7;  // input data length
  M=10; // minimum padded length

  fftw::maxthreads=parallel::get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv);

  cout << "L=" << L << endl;
  cout << "M=" << M << endl;

  if(Output || testError)
    s=0;
  if(s == 0) N=1;
  cout << "s=" << s << endl << endl;
  s *= 1.0e9;

  size_t H=ceilquotient(L,2);

  vector<double> T;

  Application app(A,B,realMultBinary,fftw::maxthreads,true,true,mx,Dx,Ix);
  fftPadHermitian fft(L,M,app);
  Convolution Convolve(&fft);

  Complex **f=ComplexAlign(max(A,B),H);
  for(size_t a=0; a < A; ++a) {
    Complex *fa=f[a];
    for(size_t j=0; j < H; ++j)
      fa[j]=Output || testError ? Complex(j+a+1,(a+1)*j+3) : 0.0;
    HermitianSymmetrize(fa);
  }

  Complex *h=NULL;
  if(testError) {
    h=ComplexAlign(H);
    directconvh C(H);
    C.convolve(h,f[0],f[1]);
  }

  if(!Output && !testError)
    Convolve.convolve(f);

  double sum=0.0;
  while(sum <= s || T.size() < N) {
    double t;
    if(normalized || testError) {
      cpuTimer c;
      Convolve.convolve(f);
      t=c.nanoseconds();
    } else {
      cpuTimer c;
      Convolve.convolveRaw(f);
      t=c.nanoseconds();
    }
    T.push_back(t);
    sum += t;
  }

  cout << endl;
  timings("Hermitian Hybrid",L,T.data(),T.size(),stats);
  cout << endl;

  if(Output) {
    if(testError)
      cout << "Hermitian Hybrid:" << endl;
    for(size_t b=0; b < B; ++b)
      for(size_t j=0; j < H; ++j)
        cout << f[b][j] << endl;
  }

  if(testError) {
    if(Output) {
      cout << endl;
      cout << "Direct:" << endl;
      for(size_t j=0; j < H; ++j)
        cout << h[j] << endl;
      cout << endl;
    }
    double err=0.0;
    double norm=0.0;
    Complex hj;

    // Assumes B=1
    Complex* f0=f[0];
    for(size_t j=0; j < H; ++j) {
      hj=h[j];
      err += abs2(f0[j]-hj);
      norm += abs2(hj);
    }
    double relError=sqrt(err/norm);
    cout << "Error: "<< relError << endl;
    deleteAlign(h);
  }

  deleteAlign(f[0]); delete [] f;

  return 0;
}
