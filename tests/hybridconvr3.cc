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
  Lx=Ly=Lz=4;  // input data length
  Mx=My=Mz=8; // minimum padded length

  fftw::maxthreads=parallel::get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv);

  cout << "Lx=" << Lx << endl;
  cout << "Ly=" << Ly << endl;
  cout << "Lz=" << Lz << endl;
  cout << "Mx=" << Mx << endl;
  cout << "My=" << My << endl;
  cout << "Mz=" << Mz << endl;

  if(Output || testError)
    s=0;
  if(s == 0) N=1;
  cout << "s=" << s << endl << endl;
  s *= 1.0e9;

  vector<double> T;

  if(Sy == 0) Sy=Lz;
  if(Sx == 0) Sx=Ly*Sy;

  Application appx(A,B,multNone,fftw::maxthreads,true,true,mx,Dx,Ix);
  fftPadReal fftx(Lx,Mx,appx,Sy == Lz ? Ly*Lz : Lz,Sx);
  Application appy(A,B,multNone,appx,my,Dy,Iy);
  fftPad ffty(Ly,My,appy,Lz,Sy);
  Application appz(A,B,multBinary,appy,mz,Dz,Iz);
  fftPad fftz(Lz,Mz,appz);
  Convolution3 Convolve(&fftx,&ffty,&fftz);

  double **f=doubleAlign(max(A,B),Lx*Sx);

  for(size_t a=0; a < A; ++a) {
    double *fa=f[a];
    for(size_t i=0; i < Lx; ++i) {
      for(size_t j=0; j < Ly; ++j) {
        for(size_t k=0; k < Lz; ++k) {
          fa[Sx*i+Sy*j+k]=Output || testError ? i+(a+1)*j+a*k+1 : 0.0;
        }
      }
    }
  }

  double *h=NULL;
  if(testError) {
    h=doubleAlign(Lx*Ly*Lz);
    directconv3<double> C(Lx,Ly,Lz,Sx,Sy);
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
  timings("Hybrid",Lx*Ly*Lz,T.data(),T.size(),stats);
  cout << endl;

  if(Output) {
    if(testError)
      cout << "Hybrid:" << endl;
    for(size_t b=0; b < B; ++b) {
      double *fb=f[b];
      for(size_t i=0; i < Lx; ++i) {
        for(size_t j=0; j < Ly; ++j) {
          for(size_t k=0; k < Lz; ++k) {
            cout << fb[Sx*i+Sy*j+k] << " ";
          }
          cout << endl;
        }
        cout << endl;
      }
      cout << endl;
    }
  }
  if(testError) {
    if(Output) {
      cout << endl;
      cout << "Direct:" << endl;
      for(size_t i=0; i < Lx; ++i) {
        for(size_t j=0; j < Ly; ++j) {
          for(size_t k=0; k < Lz; ++k) {
            cout << h[Lz*(Ly*i+j)+k] << " ";
          }
          cout << endl;
        }
        cout << endl;
      }
      cout << endl;
    }
    double err=0.0;
    double norm=0.0;

    // Assumes B=1
    for(size_t i=0; i < Lx; ++i)
      for(size_t j=0; j < Ly; ++j)
        for(size_t k=0; k < Lz; ++k){
          double hijk=h[Lz*(Ly*i+j)+k];
          err += abs2(f[0][Sx*i+Sy*j+k]-hijk);
          norm += abs2(hijk);
        }
    double relError=sqrt(err/norm);
    cout << "Error: "<< relError << endl;
    deleteAlign(h);
  }

  deleteAlign(f[0]); delete [] f;

  return 0;
}
