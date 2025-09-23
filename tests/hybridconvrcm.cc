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

  Application app(A,B,multBinaryRCM,fftw::maxthreads,true,mx,Dx,Ix);
  fftPad *fft=new fftPad(L/2,M/2,app);
  Convolution Convolve(fft);

  double **f=doubleAlign(max(A,B),L);
  Complex **g=ComplexAlign(max(A,B),L/2);

  for(size_t a=0; a < A; ++a) {
    double *fa=f[a];
    for(size_t j=0; j < L; ++j) {
      fa[j]=Output || testError ? (j % 2 ? (a+1)*j/2+3: j/2+a+1) : 0.0;
    }
    g[a]=(Complex *) f[a];
  }

  double *h=NULL;
  if(testError) {
    double **F=doubleAlign(max(A,B),L);
    for(size_t a=0; a < A; ++a) {
      double *Fa=F[a];
      double *fa=f[a];
      for(size_t j=0; j < L; ++j)
        Fa[j]=fa[j];
    }
    h=doubleAlign(L);
    directconv<double> C(L);
    C.convolve(h,F[0],F[1]);
  }

  if(!Output && !testError)
    Convolve.convolve(g);

  double sum=0.0;
  while(sum < s || T.size() < N) {
    double t;
    if(normalized || testError) {
      cpuTimer c;
      Convolve.convolve(g);
      t=c.nanoseconds();
    } else {
      cpuTimer c;
      Convolve.convolveRaw(g);
      t=c.nanoseconds();
    }
    T.push_back(t);
    sum += t;
  }

  cout << endl;
  timings("Hybrid",L,T.data(),T.size(),stats);
  cout << endl;

  if(Output) {
    if(testError)
      cout << "Hybrid:" << endl;
    for(size_t b=0; b < B; ++b)
      for(size_t j=0; j < L; ++j)
        cout << f[b][j] << endl;
  }

  if(testError) {
    if(Output) {
      cout << endl;
      cout << "Direct:" << endl;
      for(size_t j=0; j < L; ++j)
        cout << h[j] << endl;
      cout << endl;
    }
    double err=0.0;
    double norm=0.0;

    // Assumes B=1
    double* f0=f[0];
    for(size_t j=0; j < L; ++j) {
      double hj=h[j];
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
