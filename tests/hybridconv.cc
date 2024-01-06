#include <vector>

#include "convolve.h"
#include "timing.h"
#include "direct.h"
#include "options.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

// Constants used for initialization and testing.
const Complex iF(sqrt(3.0),sqrt(7.0));
const Complex iG(sqrt(5.0),sqrt(11.0));

size_t A=2; // number of inputs
size_t B=1; // number of outputs

int main(int argc, char *argv[])
{
  L=8;  // input data length
  M=16; // minimum padded length

  fftw::maxthreads=parallel::get_max_threads();
#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv);

  cout << "L=" << L << endl;
  cout << "M=" << M << endl;

  bool single=Output || testError || accuracy;

  if(single)
    s=0;
  if(s == 0) N=1;
  cout << "s=" << s << endl << endl;
  s *= 1.0e9;

  vector<double> T;

  Application app(A,B,multBinary,fftw::maxthreads,true,true,mx,Dx,Ix);
  fftPad *fft=Centered ? new fftPadCentered(L,M,app) : new fftPad(L,M,app);
  Convolution Convolve(fft);

  Complex **f=ComplexAlign(max(A,B),L);

  if(accuracy) {
    for(size_t j=0; j < L; ++j) {
      Complex factor=expi(j);
      f[0][j]=iF*factor;
      f[1][j]=iG*factor;
    }
  } else {
    for(size_t a=0; a < A; ++a) {
      Complex *fa=f[a];
      for(size_t j=0; j < L; ++j)
        fa[j]=Output || testError ? Complex(j+a+1,(a+1)*j+3) : 0.0;
    }
  }

  Complex *h=NULL;
  if(accuracy) {
    h=ComplexAlign(L);
    // Exact solution for test case with two inputs
    for(size_t j=0; j < L; ++j)
      h[j]=iF*iG*(j+1)*expi(j);
  } else if(testError) {
    h=ComplexAlign(L);
    directconv<Complex> C(L);
    if(Centered)
      C.convolveC(h,f[0],f[1]);
    else
      C.convolve(h,f[0],f[1]);
  }

  if(!single)
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
  timings("Hybrid",L,T.data(),T.size(),stats);
  cout << endl;

  if(Output) {
    if(testError || accuracy)
      cout << "Hybrid:" << endl;
    for(size_t b=0; b < B; ++b)
      for(size_t j=0; j < L; ++j)
        cout << f[b][j] << endl;
  }

  if(testError || accuracy) {
    if(Output) {
      cout << endl;
      cout << (accuracy ? "Exact" : "Direct:") << endl;
      for(size_t j=0; j < L; ++j)
        cout << h[j] << endl;
      cout << endl;
    }
    double err=0.0;
    double norm=0.0;

    // Assumes B=1
    Complex* f0=f[0];
    for(size_t j=0; j < L; ++j) {
      Complex hj=h[j];
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
