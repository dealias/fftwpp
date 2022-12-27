#include <vector>

#include "convolve.h"
#include "timing.h"
#include "direct.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

size_t A=2; // number of inputs
size_t B=1; // number of outputs

int main(int argc, char *argv[])
{
  Lx=Ly=8;  // input data length
  Mx=My=16; // minimum padded length

  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv);

  cout << "Lx=" << Lx << endl;
  cout << "Ly=" << Ly << endl;
  cout << "Mx=" << Mx << endl;
  cout << "My=" << My << endl;

  if(Output || testError)
    K=0;
  if(K == 0) minCount=1;
  cout << "K=" << K << endl << endl;
  K *= 1.0e9;

  if(Sx == 0) Sx=Ly;

  vector<double> T;

  Application appx(A,B,multNone,fftw::maxthreads,0,mx,Dx,Ix);
  fftPad fftx(Lx,Mx,appx,Ly,Sx);
  Application appy(A,B,multbinary,appx,fftx.C,my,Dy,Iy);
  fftPad ffty(Ly,My,appy);
  Convolution2 Convolve2(&fftx,&ffty);

  Complex **f=ComplexAlign(max(A,B),fftx.inputSize());

  for(size_t a=0; a < A; ++a) {
    Complex *fa=f[a];
    for(size_t i=0; i < Lx; ++i) {
      for(size_t j=0; j < Ly; ++j) {
        fa[Sx*i+j]=Output || testError ? Complex((1.0+a)*i,j+a) : 0.0;
      }
    }
  }

  Complex *h=NULL;
  if(testError) {
    h=ComplexAlign(Lx*Ly);
    DirectConvolution2 C(Lx,Ly,Sx);
    C.convolve(h,f[0],f[1]);
  }

  double sum=0.0;
  while(sum <= K || T.size() < minCount) {
    double t;
    if(normalized || testError) {
      cpuTimer c;
      Convolve2.convolve(f);
      t=c.nanoseconds();
    } else {
      cpuTimer c;
      Convolve2.convolveRaw(f);
      t=c.nanoseconds();
    }
    T.push_back(t);
    sum += t;
  }

  cout << endl;
  timings("Hybrid",Lx*Ly,T.data(),T.size(),stats);
  cout << endl;

  if(Output) {
    if(testError)
      cout << "Hybrid:" << endl;
    for(size_t b=0; b < B; ++b) {
      for(size_t i=0; i < Lx; ++i) {
        for(size_t j=0; j < Ly; ++j) {
          cout << f[b][Sx*i+j] << " ";
        }
        cout << endl;
      }
    }
  }

  if(testError) {
    if(Output) {
      cout << endl;
      cout << "Direct:" << endl;
      for(size_t i=0; i < Lx; ++i) {
        for(size_t j=0; j < Ly; ++j) {
          cout << h[Ly*i+j] << " ";
        }
        cout << endl;
      }
      cout << endl;
    }
    double err=0.0;
    double norm=0.0;
    // Assumes B=1
    for(size_t i=0; i < Lx; ++i) {
      for(size_t j=0; j < Ly; ++j) {
        Complex hij=h[Ly*i+j];
        err += abs2(f[0][Sx*i+j]-hij);
        norm += abs2(hij);
      }
    }
    double relError=sqrt(err/norm);
    cout << "Error: "<< relError << endl;
    deleteAlign(h);
  }
  return 0;
}
