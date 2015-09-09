#ifndef __mpiconvolution_h__
#define __mpiconvolution_h__ 1
  
#include <mpi.h>
#include <fftw3-mpi.h>
#include "../convolution.h"
#include <vector>
#include "mpifftw++.h"
#include "mpi/mpitranspose.h"

namespace fftwpp {

// In-place implicitly dealiased 2D complex convolution.
class ImplicitConvolution2MPI : public ImplicitConvolution2 {
protected:
  split d;
  mpitranspose<Complex> *T;
  MPI_Comm global;
public:  
  
  void inittranspose() {
    T=new mpitranspose<Complex>(d.X,d.y,d.x,d.Y,1,u2,d.communicator,global);
  }

  // u1 is a temporary array of size my*A*threads.
  // u2 is a temporary array of size split(mx,my).n*A.
  // A is the number of inputs.
  // B is the number of outputs.
  // threads is the number of threads to use in the outer subconvolution loop.
  ImplicitConvolution2MPI(unsigned int mx, unsigned int my, const split& d,
                          Complex *u1, Complex *u2, 
                          unsigned int A=2, unsigned int B=1,
                          unsigned int threads=fftw::maxthreads,
                          MPI_Comm global=0) :
    ImplicitConvolution2(mx,my,u1,u2,A,B,threads,d.x,d.y,d.n), d(d),
    global(global ? global : d.communicator) {
    inittranspose();
  }
  
  ImplicitConvolution2MPI(unsigned int mx, unsigned int my, const split& d,
                          unsigned int A=2, unsigned int B=1,
                          unsigned int threads=fftw::maxthreads,
                          MPI_Comm global=0) :
    ImplicitConvolution2(mx,my,A,B,threads,d.x,d.y,d.n), d(d),
    global(global ? global : d.communicator) {
    inittranspose();
  }
  
  virtual ~ImplicitConvolution2MPI() {
    delete T;
  }
  
  // F is a pointer to A distinct data blocks each of size mx*d.y,
  // shifted by offset (contents not preserved).
  void convolve(Complex **F, multiplier *pmult, unsigned int offset=0);
  
  // Binary convolution:
  void convolve(Complex *f, Complex *g) {
    Complex *F[]={f,g};
    convolve(F,multbinary);
  }
};

// In-place implicitly dealiased 2D Hermitian convolution.
class ImplicitHConvolution2MPI : public ImplicitHConvolution2 {
protected:
  split d,du;
  mpitranspose<double> *T,*U;
  MPI_Comm global;
public:  
  
  void inittranspose(Complex *f) {
    T=new mpitranspose<double>(d.X,d.y,d.x,d.Y,2,(double *) f,NULL,
                               d.communicator,global);
    U=new mpitranspose<double>(du.X,du.y,du.x,du.Y,2,(double *)u2,NULL,
                               du.communicator,global);
  }    
  
  void transpose(mpitranspose<double> *T, unsigned int A, Complex **F,
                 bool inflag, bool outflag, unsigned int offset=0) {
    for(unsigned int a=0; a < A; ++a) {
      double *f=(double *) (F[a]+offset);
      T->transpose(f,inflag,outflag);
    }
  }
  
  // u1 is a temporary array of size (my/2+1)*A*threads.
  // u2 is a temporary array of size du.n*A.
  // A is the number of inputs.
  // B is the number of outputs.
  // threads is the number of threads to use in the outer subconvolution loop.
  // f is a temporary array of size d.n needed only during construction.
  ImplicitHConvolution2MPI(unsigned int mx, unsigned int my,
                           const split& d, const split& du,
                           Complex *f, Complex *u1, Complex *u2,
                           unsigned int A=2, unsigned int B=1,
                           bool compact=true,
                           unsigned int threads=fftw::maxthreads,
                          MPI_Comm global=0) :
    ImplicitHConvolution2(mx,my,u1,u2,A,B,compact,threads,d.x,d.y,du.n),
    d(d), du(du), global(global ? global : d.communicator) {
    inittranspose(f);
  }
  
  ImplicitHConvolution2MPI(unsigned int mx, unsigned int my,
                           const split& d, const split& du,
                           Complex *f,
                           unsigned int A=2, unsigned int B=1,
                           bool compact=true,
                           unsigned int threads=fftw::maxthreads,
                           MPI_Comm global=0) :
    ImplicitHConvolution2(mx,my,A,B,compact,threads,d.x,d.y,du.n), d(d),
    du(du), global(global ? global : d.communicator) {
    inittranspose(f);
  }
  
  virtual ~ImplicitHConvolution2MPI() {
    delete U;
    delete T;
  }

  // F is a pointer to A distinct data blocks each of size 
  // (2mx-compact)*d.y, shifted by offset (contents not preserved).
  void convolve(Complex **F, realmultiplier *pmult,
                bool symmetrize=true, unsigned int offset=0);

  // Binary convolution:
  void convolve(Complex *f, Complex *g, bool symmetrize=true) {
    Complex *F[]={f,g};
    convolve(F,multbinary,symmetrize);
  }
};

// In-place implicitly dealiased 3D complex convolution.
class ImplicitConvolution3MPI : public ImplicitConvolution3 {
protected:
  splityz d;
  fftw_plan intranspose,outtranspose;
  mpitranspose<Complex> *T;
public:  
  void inittranspose() {
    T=new mpitranspose<Complex>(d.X,d.y,d.x,d.Y,d.z,u3,d.xy.communicator,
                                d.communicator);
  }

  void initMPI() {
    if(d.z < d.Z) {
      yzconvolve=new ImplicitConvolution2*[threads];
      for(unsigned int t=0; t < threads; ++t)
        yzconvolve[t]=new ImplicitConvolution2MPI(my,mz,d.yz,
                                                  u1+t*mz*A*innerthreads,
                                                  u2+t*d.n2*A,A,B,
                                                  innerthreads,d.communicator);
      initpointers3(U3,u3,d.n);
    }
  }
  
  // u1 is a temporary array of size mz*A*threads.
  // u2 is a temporary array of size d.y*mz*A*(d.z < mz ? 1 : threads).
  // u3 is a temporary array of size d.n*A.
  // A is the number of inputs.
  // B is the number of outputs.
  // threads is the number of threads to use in the outer subconvolution loop.
  ImplicitConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                          const splityz& d,
                          Complex *u1, Complex *u2, Complex *u3, 
                          unsigned int A=2, unsigned int B=1,
                          unsigned int threads=fftw::maxthreads) :
    ImplicitConvolution3(mx,my,mz,u1,u2,u3,A,B,threads,d.y,d.z,d.n2,d.n), d(d) {
    inittranspose();
    initMPI();
  }

  ImplicitConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                          const splityz& d,
                          unsigned int A=2, unsigned int B=1,
                          unsigned int threads=fftw::maxthreads) :
    ImplicitConvolution3(mx,my,mz,A,B,threads,d.y,d.z,d.n2,d.n), d(d) {
    inittranspose();
    initMPI();
  }
  
  virtual ~ImplicitConvolution3MPI() {
    delete T;
  }
  
  // F is a pointer to A distinct data blocks each of size
  // 2mx*2d.y*d.z, shifted by offset (contents not preserved).
  void convolve(Complex **F, multiplier *pmult, unsigned int offset=0);
  
  // Binary convolution:
  void convolve(Complex *f, Complex *g) {
    Complex *F[]={f,g};
    convolve(F,multbinary);
  }
};

class range {
public:
  unsigned int n;
  int start;
};

void HermitianSymmetrizeXYMPI(unsigned int mx, unsigned int my,
			      splityz& d, bool compact, Complex *f,
                              unsigned int nu, Complex *u);
 
// In-place implicitly dealiased 3D complex convolution.
class ImplicitHConvolution3MPI : public ImplicitHConvolution3 {
protected:
  splityz d,du;
  mpitranspose<double> *T,*U;
  MPI_Comm global;
public:  
  void inittranspose(Complex *f) {
    if(d.y < d.Y) {
      T=new mpitranspose<double>(d.X,d.y,d.x,d.Y,2*d.z,(double *) f,NULL,
                                 mpioptions(1,1,1),d.xy.communicator,global);
      U=new mpitranspose<double>(du.X,du.y,du.x,du.Y,2*d.z,(double *)u2,NULL,
                                 mpioptions(1,1,1),du.xy.communicator,global);
    }
  }

  void initMPI(Complex *f) {
    if(d.z < d.Z) {
      yzconvolve=new ImplicitHConvolution2*[threads];
      for(unsigned int t=0; t < threads; ++t)
        yzconvolve[t]=
          new ImplicitHConvolution2MPI(my,mz,d.yz,du.yz,f,
                                       u1+t*(mz/2+1)*A*innerthreads,
                                       u2+t*du.n2*A,A,B,compact,innerthreads);
      initpointers3(U3,u3,du.n);
    }
  }
  
  // u1 is a temporary array of size (mz/2+1)*A*threads,
  // u2 is a temporary array of size du.n2*A*(d.z < mz ? 1 : threads).
  // u3 is a temporary array of size du.n2*A.
  // A is the number of inputs.
  // B is the number of outputs.
  // threads is the number of threads to use in the outer subconvolution loop.
  ImplicitHConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                           const splityz& d, const splityz& du,
                           Complex *f, Complex *u1, Complex *u2, Complex *u3,
                           unsigned int A=2, unsigned int B=1,
                           bool compact=true,
                           unsigned int threads=fftw::maxthreads,
                           MPI_Comm global=0) :
    ImplicitHConvolution3(mx,my,mz,u1,u2,u3,A,B,compact,threads,d.y,d.z,
                          du.n2,du.n), 
    d(d), du(du), global(global ? global : d.communicator) { 
    initMPI(f);
    inittranspose(f);
  }

  ImplicitHConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                           const splityz& d, const splityz& du,
                           Complex *f, unsigned int A=2, unsigned int B=1,
                           bool compact=true,
                           unsigned int threads=fftw::maxthreads,
                           MPI_Comm global=0) :
    ImplicitHConvolution3(mx,my,mz,A,B,compact,threads,d.y,d.z,du.n2,du.n),
    d(d), du(du), global(global ? global : d.communicator) {
    initMPI(f);
    inittranspose(f);
  }
  
  virtual ~ImplicitHConvolution3MPI() {
    if(d.y < d.Y) {
      delete U;
      delete T;
    }
  }
  
  void transpose(mpitranspose<double> *T, unsigned int A, Complex **F,
                 bool inflag, bool outflag, unsigned int offset=0) {
    for(unsigned int a=0; a < A; ++a) {
      double *f=(double *) (F[a]+offset);
      T->transpose(f,inflag,outflag);
    }
  }
  
  void HermitianSymmetrize(Complex *f, Complex *u) {
    HermitianSymmetrizeXYMPI(mx,my,d,compact,f,du.n,u);
  }
  
  // F is a pointer to A distinct data blocks each of size
  // (2mx-compact)*d.y*d.z, shifted by offset (contents not preserved).
  void convolve(Complex **F, realmultiplier *pmult, bool symmetrize=true,
                unsigned int offset=0);
  
  // Binary convolution:
  void convolve(Complex *f, Complex *g, bool symmetrize=true) {
    Complex *F[]={f,g};
    convolve(F,multbinary,symmetrize);
  }
};


} // namespace fftwpp

#endif
