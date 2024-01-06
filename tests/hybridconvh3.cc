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
  Lx=Ly=Lz=7;  // input data length
  Mx=My=Mz=10; // minimum padded length

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

  size_t Hx=ceilquotient(Lx,2);
  size_t Hy=ceilquotient(Ly,2);
  size_t Hz=ceilquotient(Lz,2);

  if(Sy == 0) Sy=Hz;
  if(Sx == 0) Sx=Ly*Sy;

  vector<double> T;

  Application appx(A,B,multNone,fftw::maxthreads,true,true,mx,Dx,Ix);
  fftPadCentered fftx(Lx,Mx,appx,Sy == Hz ? Ly*Hz : Hz,Sx);
  Application appy(A,B,multNone,appx,my,Dy,Iy);
  fftPadCentered ffty(Ly,My,appy,Hz,Sy);
  Application appz(A,B,realMultBinary,appy,mz,Dz,Iz);
  fftPadHermitian fftz(Lz,Mz,appz);
  Convolution3 Convolve(&fftx,&ffty,&fftz);

  Complex **f=ComplexAlign(max(A,B),Lx*Sx);

  for(size_t a=0; a < A; ++a) {
    Complex *fa=f[a];
    for(size_t i=0; i < Lx; ++i) {
      int I=Lx % 2 ? i : i-1;
      for(size_t j=0; j < Ly; ++j) {
        int J=Ly % 2 ? j : j-1;
        for(size_t k=0; k < Hz; ++k) {
          fa[Sx*i+Sy*j+k]=Output || testError ?
            Complex(I+a*k+1,(a+1)*J+3+k) : 0.0;
        }
      }
    }
    HermitianSymmetrizeXY(Hx,Hy,Hz,Lx/2,Ly/2,fa,Sx,Sy);
  }

  Complex *h=NULL;
  if(testError) {
    h=ComplexAlign(Lx*Ly*Lz);
    directconvh3 C(Hx,Hy,Hz,Lx%2,Ly%2,Sx,Sy);
    C.convolve(h,f[0],f[1],false);
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
  timings("Hybrid",Lx*Ly*Lz,T.data(),T.size(),stats);
  cout << endl;

  if(Output) {
    if(testError)
      cout << "Hybrid:" << endl;
    for(size_t b=0; b < B; ++b) {
      Complex *fb=f[b];
      for(size_t i=0; i < Lx; ++i) {
        for(size_t j=0; j < Ly; ++j) {
          for(size_t k=0; k < Hz; ++k)
            cout << fb[Sx*i+Sy*j+k] << " ";
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
          for(size_t k=0; k < Hz; ++k) {
            cout << h[Hz*(Ly*i+j)+k] << " ";
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
        for(size_t k=0; k < Hz; ++k){
          Complex hijk=h[Hz*(Ly*i+j)+k];
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
