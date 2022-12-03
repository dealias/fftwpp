#include <vector>

#include "convolve.h"
#include "timing.h"
#include "direct.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

unsigned int A=2; // number of inputs
unsigned int B=1; // number of outputs

int main(int argc, char *argv[])
{
  Lx=Ly=Lz=7;  // input data length
  Mx=My=Mz=10; // minimum padded length

  fftw::maxthreads=get_max_threads();

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
    K=0;
  if(K == 0) minCount=1;
  cout << "K=" << K << endl << endl;
  K *= 1.0e9;

  unsigned int Hx=ceilquotient(Lx,2);
  unsigned int Hy=ceilquotient(Ly,2);
  unsigned int Hz=ceilquotient(Lz,2);

  if(Sy == 0) Sy=Hz;
  if(Sx == 0) Sx=Ly*Sy;

  vector<double> T;
  unsigned int N=max(A,B);

  Application appx(A,B,multNone,fftw::maxthreads,0,mx,Dx,Ix);
  fftPadCentered fftx(Lx,Mx,appx,Sy == Hz? Ly*Hz : Hz,Sx);
  bool embed=fftx.embed();
  unsigned int size=embed ? fftx.outputSize() : fftx.inputSize();
  Complex **f=ComplexAlign(N,size);
  Application appy(A,B,multNone,appx.Threads(),fftx.l,my,Dy,Iy);
  fftPadCentered ffty(Ly,My,appy,Hz,Sy);
  Application appz(A,B,realmultbinary,appy.Threads(),ffty.l,mz,Dz,Iz);
  ConvolutionHermitian convolvez(Lz,Mz,appz);
  ConvolutionHermitian2 convolveyz(&ffty,&convolvez);
  ConvolutionHermitian3 Convolve3(&fftx,&convolveyz,embed ? f : NULL);

//  ConvolutionHermitian3 Convolve3(Lx,Mx,Ly,My,Lz,Mz,A,B);

  for(unsigned int a=0; a < A; ++a) {
    Complex *fa=f[a];
    for(unsigned int i=0; i < Lx; ++i) {
      for(unsigned int j=0; j < Ly; ++j) {
        for(unsigned int k=0; k < Hz; ++k) {
          int I=Lx % 2 ? i : -1+i;
          int J=Ly % 2 ? j : -1+j;
          fa[Sx*i+Sy*j+k]=Output || testError ?
            Complex((1.0+a)*I+k,J+a+k) : 0.0;
        }
      }
    }
    HermitianSymmetrizeXY(Hx,Hy,Hz,Lx/2,Ly/2,fa,Sx,Sy);
  }

  Complex *h=NULL;
  if(testError) {
    h=ComplexAlign(Lx*Ly*Lz);
    DirectHConvolution3 C(Hx,Hy,Hz,Lx%2,Ly%2,Sx,Sy);
    C.convolve(h,f[0],f[1],false);
  }

  double sum=0.0;
  while(sum <= K || T.size() < minCount) {
    double t;
    if(normalized || testError) {
      double t0=nanoseconds();
      Convolve3.convolve(f);
      t=nanoseconds()-t0;
    } else {
      double t0=nanoseconds();
      Convolve3.convolveRaw(f);
      t=nanoseconds()-t0;
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
    for(unsigned int b=0; b < B; ++b) {
      Complex *fb=f[b];
      for(unsigned int i=0; i < Lx; ++i) {
        for(unsigned int j=0; j < Ly; ++j) {
          for(unsigned int k=0; k < Hz; ++k)
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
      for(unsigned int i=0; i < Lx; ++i) {
        for(unsigned int j=0; j < Ly; ++j) {
          for(unsigned int k=0; k < Hz; ++k) {
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
    for(unsigned int i=0; i < Lx; ++i)
      for(unsigned int j=0; j < Ly; ++j)
        for(unsigned int k=0; k < Hz; ++k){
          Complex hijk=h[Hz*(Ly*i+j)+k];
          err += abs2(f[0][Sx*i+Sy*j+k]-hijk);
          norm += abs2(hijk);
        }
    double relError=sqrt(err/norm);
    cout << "Error: "<< relError << endl;
    deleteAlign(h);
  }
  return 0;
}
