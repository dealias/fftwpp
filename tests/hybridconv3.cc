#include "convolve.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

unsigned int A=2; // number of inputs
unsigned int B=1; // number of outputs
unsigned int C=1; // number of copies
unsigned int S=0; // stride between copies (0 means L)
unsigned int L=512; // input data length
unsigned int M=1024; // minimum padded length

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv);

  unsigned int Lx=L;
  unsigned int Ly=L;
  unsigned int Lz=L;
  unsigned int Mx=M;
  unsigned int My=M;
  unsigned int Mz=M;
  unsigned int Sx=S;
  unsigned int Sy=S;

  cout << "Lx=" << Lx << endl;
  cout << "Ly=" << Ly << endl;
  cout << "Lz=" << Lz << endl;
  cout << "Mx=" << Mx << endl;
  cout << "My=" << My << endl;
  cout << "Mz=" << Mz << endl;
  cout << endl;

  if(Sx == 0) Sx=Ly*Lz;
  if(Sy == 0) Sy=Lz;

  Complex **f=new Complex *[max(A,B)];
  for(unsigned int a=0; a < A; ++a)
    f[a]=ComplexAlign(Lx*Sx);

  ForwardBackward FBx(A,B);
  fftPad fftx(Lx,Mx,FBx,Ly*Lz,Sx);
  ForwardBackward FBy(A,B,FBx.Threads(),fftx.l);
  fftPad ffty(Ly,My,FBy,Lz,Sy);
  ForwardBackward FBz(A,B,FBy.Threads(),ffty.l);
  Convolution convolvez(Lz,Mz,FBz);
  Convolution2 convolveyz(&ffty,&convolvez);
  Convolution3 Convolve3(&fftx,&convolveyz);

//  Convolution3 Convolve3(Lx,Mx,Ly,My,Lz,Mz,A,B);

  double T=0;

  for(unsigned int c=0; c < C; ++c) {

    for(unsigned int a=0; a < A; ++a) {
      Complex *fa=f[a];
      for(unsigned int i=0; i < Lx; ++i) {
        for(unsigned int j=0; j < Ly; ++j) {
          for(unsigned int k=0; k < Lz; ++k) {
            fa[Sx*i+Sy*j+k]=Complex((1.0+a)*i+k,j+k+a);
          }
        }
      }
    }

    if(Lx*Ly*Lz < 200 && c == 0) {
      for(unsigned int a=0; a < A; ++a) {
        for(unsigned int i=0; i < Lx; ++i) {
          for(unsigned int j=0; j < Ly; ++j) {
            for(unsigned int k=0; k < Lz; ++k) {
              cout << f[a][Sx*i+Sy*j+k] << " ";
            }
            cout << endl;
          }
          cout << endl;
        }
        cout << endl;
      }
    }

    seconds();
    Convolve3.convolve(f,multbinary);
    T += seconds();
  }

  cout << "mean=" << T/C << endl;

  Complex sum=0.0;
  for(unsigned int b=0; b < B; ++b) {
    Complex *fb=f[b];
    for(unsigned int i=0; i < Lx; ++i) {
      for(unsigned int j=0; j < Ly; ++j) {
        for(unsigned int k=0; k < Lz; ++k) {
          sum += fb[Sx*i+Sy*j+k];
        }
      }
    }
  }

  cout << "sum=" << sum << endl;
  cout << endl;

  if(Lx*Ly < 200) {
    for(unsigned int b=0; b < B; ++b) {
      Complex *fb=f[b];
      for(unsigned int i=0; i < Lx; ++i) {
        for(unsigned int j=0; j < Ly; ++j) {
          for(unsigned int k=0; k < Lz; ++k) {
            cout << fb[Sx*i+Sy*j+k] << " ";
          }
          cout << endl;
        }
        cout << endl;
      }
      cout << endl;
    }
  }
  return 0;
}
