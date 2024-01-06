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
  Lx=Ly=4;  // input data length
  Mx=My=8; // minimum padded length

  fftw::maxthreads=parallel::get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv);

  cout << "Lx=" << Lx << endl;
  cout << "Ly=" << Ly << endl;
  cout << "Mx=" << Mx << endl;
  cout << "My=" << My << endl;

  if(Output || testError)
    s=0;
  if(s == 0) N=1;
  cout << "s=" << s << endl << endl;
  s *= 1.0e9;

  if(Sx == 0) Sx=Ly;

  vector<double> T;

  Application appx(A,B,multNone,fftw::maxthreads,true,true,mx,Dx,Ix);
  fftPadReal fftx(Lx,Mx,appx,Ly,Sx);
  Application appy(A,B,multBinary,appx,my,Dy,Iy);
  fftPad ffty(Ly,My,appy);
  Convolution2 Convolve(&fftx,&ffty);

  double **f=doubleAlign(max(A,B),Lx*Sx);

  for(size_t a=0; a < A; ++a) {
    double *fa=f[a];
    for(size_t i=0; i < Lx; ++i) {
      for(size_t j=0; j < Ly; ++j) {
        fa[Sx*i+j]=Output || testError ? i+a*j+1 : 0.0;
      }
    }
  }

  double *h=NULL;
  if(testError) {
    h=doubleAlign(Lx*Ly);
    directconv2<double> C(Lx,Ly,Sx);
    C.convolve(h,f[0],f[1]);
  }

  if(!Output && !testError)
    Convolve.convolve((Complex **) f);

  double sum=0.0;
  while(sum <= s || T.size() < N) {
    double t;
    if(normalized || testError) {
      cpuTimer c;
      Convolve.convolve((Complex **) f);
      t=c.nanoseconds();
    } else {
      cpuTimer c;
      Convolve.convolveRaw((Complex **) f);
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

  deleteAlign(f[0]); delete [] f;

  return 0;
}
