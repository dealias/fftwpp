#ifndef __mpiconvolution_h__
#define __mpiconvolution_h__ 1
  
#include <mpi.h>
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
  
  void inittranspose(const mpiOptions& mpioptions) {
    T=new mpitranspose<Complex>(d.X,d.y,d.x,d.Y,1,u2,d.communicator,mpioptions,
                                global);
    d.Deactivate();
  }

  // u1 is a temporary array of size my*A*options.threads.
  // u2 is a temporary array of size split(mx,my).n*A.
  // A is the number of inputs.
  // B is the number of outputs.
  ImplicitConvolution2MPI(unsigned int mx, unsigned int my, const split& d,
                          Complex *u1, Complex *u2, 
                          unsigned int A=2, unsigned int B=1,
                          convolveOptions options=defaultconvolveOptions,
                          MPI_Comm global=0) :
    ImplicitConvolution2(mx,my,u1,u2,A,B,convolveOptions(options,d.x,d.y,
                                                         d.Activate())),
    d(d), global(global ? global : d.communicator) {
    inittranspose(options.mpi);
  }
  
  ImplicitConvolution2MPI(unsigned int mx, unsigned int my, const split& d,
                          unsigned int A=2, unsigned int B=1,
                          convolveOptions options=defaultconvolveOptions,
                          MPI_Comm global=0) :
    ImplicitConvolution2(mx,my,A,B,convolveOptions(options,d.x,d.y,
                                                   d.Activate())),
    d(d), global(global ? global : d.communicator) {
    inittranspose(options.mpi);
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
  
  void inittranspose(Complex *f, const mpiOptions& mpioptions) {
    T=new mpitranspose<double>(d.X,d.y,d.x,d.Y,2,(double *) f,
                               d.communicator,mpioptions,global);
    U=new mpitranspose<double>(du.X,du.y,du.x,du.Y,2,(double *) u2,
                               du.communicator,mpioptions,global);
    du.Deactivate();
  }    
  
  void transpose(mpitranspose<double> *T, unsigned int A, Complex **F,
                 bool inflag, bool outflag, unsigned int offset=0) {
    for(unsigned int a=0; a < A; ++a) {
      double *f=(double *) (F[a]+offset);
      T->transpose(f,inflag,outflag);
    }
  }
  
  // f is a temporary array of size d.n needed only during construction.
  // u1 is a temporary array of size (my/2+1)*A*options.threads.
  // u2 is a temporary array of size du.n*A.
  // A is the number of inputs.
  // B is the number of outputs.
  ImplicitHConvolution2MPI(unsigned int mx, unsigned int my,
                           const split& d, const split& du,
                           Complex *f, Complex *u1, Complex *u2,
                           unsigned int A=2, unsigned int B=1,
                           convolveOptions options=defaultconvolveOptions,
                           MPI_Comm global=0) :
    ImplicitHConvolution2(mx,my,u1,u2,A,B,
                          convolveOptions(options,d.x,d.y,du.Activate())),
    d(d), du(du), global(global ? global : d.communicator) {
    inittranspose(f,options.mpi);
  }
  
  ImplicitHConvolution2MPI(unsigned int mx, unsigned int my,
                           bool xcompact, bool ycompact,
                           const split& d, const split& du,
                           Complex *f, Complex *u1, Complex *u2,
                           unsigned int A=2, unsigned int B=1,
                           convolveOptions options=defaultconvolveOptions,
                           MPI_Comm global=0) :
    ImplicitHConvolution2(mx,my,xcompact,ycompact,u1,u2,A,B,
                          convolveOptions(options,d.x,d.y,du.Activate())),
    d(d), du(du), global(global ? global : d.communicator) {
    inittranspose(f,options.mpi);
  }
  
  ImplicitHConvolution2MPI(unsigned int mx, unsigned int my,
                           const split& d, const split& du,
                           Complex *f, unsigned int A=2, unsigned int B=1,
                           convolveOptions options=defaultconvolveOptions,
                           MPI_Comm global=0) :
    ImplicitHConvolution2(mx,my,A,B,convolveOptions(options,d.x,d.y,
                                          du.Activate())),
    d(d), du(du), global(global ? global : d.communicator) {
    inittranspose(f,options.mpi);
  }
  
  ImplicitHConvolution2MPI(unsigned int mx, unsigned int my,
                           bool xcompact, bool ycompact,
                           const split& d, const split& du,
                           Complex *f, unsigned int A=2, unsigned int B=1,
                           convolveOptions options=defaultconvolveOptions,
                           MPI_Comm global=0) :
    ImplicitHConvolution2(mx,my,xcompact,ycompact,A,B,
                          convolveOptions(options,d.x,d.y,du.Activate())),
    d(d), du(du), global(global ? global : d.communicator) {
    inittranspose(f,options.mpi);
  }
  
  virtual ~ImplicitHConvolution2MPI() {
    delete U;
    delete T;
  }

  // F is a pointer to A distinct data blocks each of size 
  // (2mx-xcompact)*d.y, shifted by offset (contents not preserved).
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
  split3 d;
  fftw_plan intranspose,outtranspose;
  mpitranspose<Complex> *T;
public:  
  void inittranspose(const mpiOptions& mpioptions) {
    T=d.xy.y < d.Y ? 
               new mpitranspose<Complex>(d.X,d.xy.y,d.x,d.Y,d.z,u3,
                                         d.xy.communicator,mpioptions,
                                         d.communicator) : NULL;
    d.Deactivate();
  }

  void initMPI(const convolveOptions& options) {
    if(d.z < d.Z) {
      yzconvolve=new ImplicitConvolution2*[threads];
      for(unsigned int t=0; t < threads; ++t)
        yzconvolve[t]=new ImplicitConvolution2MPI(my,mz,d.yz,
                                                  u1+t*mz*A*innerthreads,
                                                  u2+t*d.n2*A,A,B,
                                                  convolveOptions(options,
                                                                  innerthreads),
                                                  d.communicator);
      initpointers3(U3,u3,d.n);
    }
    inittranspose(options.mpi);
  }
  
  // u1 is a temporary array of size mz*A*options.threads.
  // u2 is a temporary array of size d.y*mz*A*(d.z < mz ? 1 : threads).
  // u3 is a temporary array of size d.n*A.
  // A is the number of inputs.
  // B is the number of outputs.
  ImplicitConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                          const split3& d,
                          Complex *u1, Complex *u2, Complex *u3, 
                          unsigned int A=2, unsigned int B=1,
                          convolveOptions options=defaultconvolveOptions) :
    ImplicitConvolution3(mx,my,mz,u1,u2,u3,A,B,
                         convolveOptions(options,d.xy.y,d.z,d.n2,d.Activate())),
    d(d) {
    initMPI(options);
  }

  ImplicitConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                          const split3& d,
                          unsigned int A=2, unsigned int B=1,
                          convolveOptions options=defaultconvolveOptions) :
    ImplicitConvolution3(mx,my,mz,A,B,
                         convolveOptions(options,d.xy.y,d.z,d.n2,d.Activate())),
    d(d) {
    initMPI(options);
  }
  
  virtual ~ImplicitConvolution3MPI() {
    if(T) delete T;
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
			      split3& d, bool xcompact, bool ycompact,
                              Complex *f, unsigned int nu=0, Complex *u=NULL);
 
// In-place implicitly dealiased 3D complex convolution.
class ImplicitHConvolution3MPI : public ImplicitHConvolution3 {
protected:
  split3 d,du;
  mpitranspose<double> *T,*U;
  MPI_Comm global;
public:  
  void inittranspose(Complex *f, const mpiOptions& mpioptions) {
    if(d.xy.y < d.Y) {
      T=new mpitranspose<double>(d.X,d.xy.y,d.x,d.Y,2*d.z,(double *) f,
                                 d.xy.communicator,mpioptions,global);
      U=new mpitranspose<double>(du.X,du.xy.y,du.x,du.Y,2*d.z,(double *) u3,
                                 du.xy.communicator,mpioptions,global);
    } else {T=U=NULL;}
    du.Deactivate();
  }

  void initMPI(Complex *f, const convolveOptions& options) {
    if(d.z < d.Z) {
      yzconvolve=new ImplicitHConvolution2*[threads];
      for(unsigned int t=0; t < threads; ++t)
        yzconvolve[t]=
          new ImplicitHConvolution2MPI(my,mz,d.yz,du.yz,f,
                                       u1+t*(mz/2+1)*A*innerthreads,
                                       u2+t*du.n2*A,A,B,
                                       convolveOptions(options,innerthreads));
      initpointers3(U3,u3,du.n);
    }
    inittranspose(f,options.mpi);
  }
  
  // f is a temporary array of size d.n needed only during construction.
  // u1 is a temporary array of size (mz/2+1)*A*options.threads,
  // u2 is a temporary array of size du.n2*A*(d.z < mz ? 1 : threads).
  // u3 is a temporary array of size du.n2*A.
  // A is the number of inputs.
  // B is the number of outputs.
  ImplicitHConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                           const split3& d, const split3& du,
                           Complex *f, Complex *u1, Complex *u2, Complex *u3,
                           unsigned int A=2, unsigned int B=1,
                           convolveOptions options=defaultconvolveOptions,
                           MPI_Comm global=0) :
    ImplicitHConvolution3(mx,my,mz,u1,u2,u3,A,B,
                          convolveOptions(options,d.xy.y,d.z,du.n2,
                                          du.Activate())), 
    d(d), du(du), global(global ? global : d.communicator) { 
    initMPI(f,options);
  }

  ImplicitHConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                           bool xcompact, bool ycompact, bool zcompact,
                           const split3& d, const split3& du,
                           Complex *f, Complex *u1, Complex *u2, Complex *u3,
                           unsigned int A=2, unsigned int B=1,
                           convolveOptions options=defaultconvolveOptions,
                           MPI_Comm global=0) :
    ImplicitHConvolution3(mx,my,mz,xcompact,ycompact,zcompact,u1,u2,u3,A,B,
                          convolveOptions(options,d.xy.y,d.z,du.n2,
                                          du.Activate())), 
    d(d), du(du), global(global ? global : d.communicator) { 
    initMPI(f,options);
  }

  ImplicitHConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                           const split3& d, const split3& du,
                           Complex *f, unsigned int A=2, unsigned int B=1,
                           convolveOptions options=defaultconvolveOptions,
                           MPI_Comm global=0) :
    ImplicitHConvolution3(mx,my,mz,A,B,
                          convolveOptions(options,d.xy.y,d.z,du.n2,
                                          du.Activate())),
    d(d), du(du), global(global ? global : d.communicator) {
    initMPI(f,options);
  }
  
  ImplicitHConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                           bool xcompact, bool ycompact, bool zcompact,
                           const split3& d, const split3& du,
                           Complex *f, unsigned int A=2, unsigned int B=1,
                           convolveOptions options=defaultconvolveOptions,
                           MPI_Comm global=0) :
    ImplicitHConvolution3(mx,my,mz,xcompact,ycompact,zcompact,A,B,
                          convolveOptions(options,d.xy.y,d.z,du.n2,
                                          du.Activate())),
    d(d), du(du), global(global ? global : d.communicator) {
    initMPI(f,options);
  }
  
  virtual ~ImplicitHConvolution3MPI() {
    if(T) {
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
    HermitianSymmetrizeXYMPI(mx,my,d,xcompact,ycompact,f,du.n,u);
  }
  
  // F is a pointer to A distinct data blocks each of size
  // (2mx-xcompact)*d.y*d.z, shifted by offset (contents not preserved).
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
