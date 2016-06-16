#ifndef __mpiconvolution_h__
#define __mpiconvolution_h__ 1
  
#include <mpi.h>
#include <vector>
#include "convolution.h"
#include "mpifftw++.h"
#include "mpitranspose.h"

namespace fftwpp {

// In-place implicitly dealiased 2D complex convolution.
class ImplicitConvolution2MPI : public ImplicitConvolution2 {
protected:
  utils::split d;
  utils::mpitranspose<Complex> *T,*U;
public:  
  
  void inittranspose(const utils::mpiOptions& mpioptions, Complex *work,
                     MPI_Comm& global) {
    global=global ? global : d.communicator;
    T=new utils::mpitranspose<Complex>(d.X,d.Y,d.x,d.y,1,u2,work,
                                       d.communicator,mpioptions,global);
    U=new utils::mpitranspose<Complex>(d.X,d.Y,d.x,d.y,1,u2,work,
                                       d.communicator,T->Options(),global);
    d.Deactivate();
  }

  void synchronizeWisdom(unsigned int threads) {
    for(unsigned int t=this->threads; t < threads; ++t)
      delete new ImplicitConvolution(my,u1,A,B,innerthreads);
  }
  
  // u1 is a temporary array of size my*A*options.threads.
  // u2 is a temporary array of size split(mx,my).n*A.
  // A is the number of inputs.
  // B is the number of outputs.
  ImplicitConvolution2MPI(unsigned int mx, unsigned int my,
                          const utils::split& d,
                          Complex *u1, Complex *u2, 
                          utils::mpiOptions mpi=utils::defaultmpiOptions,
                          unsigned int A=2, unsigned int B=1,
                          unsigned int threads=fftw::maxthreads,
                          Complex *work=NULL, MPI_Comm global=0,
                          bool toplevel=true) :
    ImplicitConvolution2(mx,my,u1,u2,A,B,threads,
                         convolveOptions(d.x,d.y,d.Activate(),mpi,toplevel)),
    d(d) {
    synchronizeWisdom(threads);
    inittranspose(mpi,work,global);
  }
  
  ImplicitConvolution2MPI(unsigned int mx, unsigned int my,
                          const utils::split& d,
                          utils::mpiOptions mpi=utils::defaultmpiOptions,
                          unsigned int A=2, unsigned int B=1,
                          unsigned int threads=fftw::maxthreads,
                          Complex *work=NULL, MPI_Comm global=0,
                          bool toplevel=true) :
    ImplicitConvolution2(mx,my,A,B,threads,
                         convolveOptions(d.x,d.y,d.Activate(),mpi,toplevel)),
    d(d) {
    synchronizeWisdom(threads);
    inittranspose(mpi,work,global);
  }
  
  virtual ~ImplicitConvolution2MPI() {
    delete T;
    delete U;
  }
  
  // F is a pointer to A distinct data blocks each of size mx*d.y,
  // shifted by offset (contents not preserved).
  void convolve(Complex **F, multiplier *pmult, unsigned int i=0,
                unsigned int offset=0);
  
  // Binary convolution:
  void convolve(Complex *f, Complex *g) {
    Complex *F[]={f,g};
    convolve(F,multbinary);
  }
};

// In-place implicitly dealiased 2D Hermitian convolution.
class ImplicitHConvolution2MPI : public ImplicitHConvolution2 {
protected:
  utils::split d,du;
  utils::mpitranspose<Complex> *T,*U;
public:  
  
  void inittranspose(Complex *f, const utils::mpiOptions& mpi, Complex *work,
                     MPI_Comm global) {
    global=global ? global : d.communicator;
    T=new utils::mpitranspose<Complex>(d.X,d.Y,d.x,d.y,1,f,work,
                                       d.communicator,mpi,global);
    U=new utils::mpitranspose<Complex>(du.X,du.Y,du.x,du.y,1,u2,work,
                                       du.communicator,mpi,global);
    du.Deactivate();
  }    
  
 void synchronizeWisdom(unsigned int threads) {
    for(unsigned int t=this->threads; t < threads; ++t)
      delete new ImplicitHConvolution(my,ycompact,u1,A,B,innerthreads);
  }
  
   // f is a temporary array of size d.n needed only during construction.
  // u1 is a temporary array of size (my/2+1)*A*options.threads.
  // u2 is a temporary array of size du.n*A.
  // A is the number of inputs.
  // B is the number of outputs.
  ImplicitHConvolution2MPI(unsigned int mx, unsigned int my,
                           const utils::split& d, const utils::split& du,
                           Complex *f, Complex *u1, Complex *u2,
                           utils::mpiOptions mpi=utils::defaultmpiOptions,
                           unsigned int A=2, unsigned int B=1,
                           unsigned int threads=fftw::maxthreads,
                           Complex *work=NULL, MPI_Comm global=0,
                           bool toplevel=true) :
    ImplicitHConvolution2(mx,my,u1,u2,A,B,threads,
                          convolveOptions(d.x,d.y,du.Activate(),mpi,toplevel)),
    d(d), du(du) {
    synchronizeWisdom(threads);
    inittranspose(f,mpi,work,global);
  }
  
  ImplicitHConvolution2MPI(unsigned int mx, unsigned int my,
                           bool xcompact, bool ycompact,
                           const utils::split& d, const utils::split& du,
                           Complex *f, Complex *u1, Complex *u2,
                           utils::mpiOptions mpi=utils::defaultmpiOptions,
                           unsigned int A=2, unsigned int B=1,
                           unsigned int threads=fftw::maxthreads,
                           Complex *work=NULL, MPI_Comm global=0,
                           bool toplevel=true) :
    ImplicitHConvolution2(mx,my,xcompact,ycompact,u1,u2,A,B,threads,
                          convolveOptions(d.x,d.y,du.Activate(),mpi,toplevel)),
    d(d), du(du) {
    synchronizeWisdom(threads);
    inittranspose(f,mpi,work,global);
  }
  
  ImplicitHConvolution2MPI(unsigned int mx, unsigned int my,
                           const utils::split& d, const utils::split& du,
                           Complex *f,
                           utils::mpiOptions mpi=utils::defaultmpiOptions,
                           unsigned int A=2, unsigned int B=1,
                           unsigned int threads=fftw::maxthreads,
                           Complex *work=NULL, MPI_Comm global=0,
                           bool toplevel=true) :
    ImplicitHConvolution2(mx,my,true,true,A,B,threads,
                          convolveOptions(d.x,d.y,du.Activate(),mpi,toplevel)),
    d(d), du(du) {
    synchronizeWisdom(threads);
    inittranspose(f,mpi,work,global);
  }
  
  ImplicitHConvolution2MPI(unsigned int mx, unsigned int my,
                           bool xcompact, bool ycompact,
                           const utils::split& d, const utils::split& du,
                           Complex *f,
                           utils::mpiOptions mpi=utils::defaultmpiOptions,
                           unsigned int A=2, unsigned int B=1,
                           unsigned int threads=fftw::maxthreads,
                           Complex *work=NULL, MPI_Comm global=0,
                           bool toplevel=true) :
    ImplicitHConvolution2(mx,my,xcompact,ycompact,A,B,threads,
                          convolveOptions(d.x,d.y,du.Activate(),mpi,toplevel)),
    d(d), du(du) {
    synchronizeWisdom(threads);
    inittranspose(f,mpi,work,global);
  }
  
  virtual ~ImplicitHConvolution2MPI() {
    delete U;
    delete T;
  }

  // F is a pointer to A distinct data blocks each of size 
  // (2mx-xcompact)*d.y, shifted by offset (contents not preserved).
  void convolve(Complex **F, realmultiplier *pmult, bool symmetrize=true,
                unsigned int i=0, unsigned int offset=0);

  // Binary convolution:
  void convolve(Complex *f, Complex *g, bool symmetrize=true) {
    Complex *F[]={f,g};
    convolve(F,multbinary,symmetrize);
  }
};

// In-place implicitly dealiased 3D complex convolution.
class ImplicitConvolution3MPI : public ImplicitConvolution3 {
protected:
  utils::split3 d;
  fftw_plan intranspose,outtranspose;
  utils::mpitranspose<Complex> *T,*U;
public:  
  void inittranspose(const utils::mpiOptions& mpi, Complex *work,
                     MPI_Comm global) {
    if(d.xy.y < d.Y) { 
      T=new utils::mpitranspose<Complex>(d.X,d.Y,d.x,d.xy.y,d.z,u3,work,
                                         d.xy.communicator,mpi,global);
      U=new utils::mpitranspose<Complex>(d.X,d.Y,d.x,d.xy.y,d.z,u3,work,
                                         d.xy.communicator,T->Options(),
                                         global);
    } else {
      T=U=NULL;
    }
    d.Deactivate();
  }

  void initMPI(const utils::mpiOptions& mpi, Complex *work, Complex *work2, 
               MPI_Comm global, unsigned int Threads) {
    global=global ? global : d.communicator;
    if(d.z < d.Z) {
      yzconvolve=new ImplicitConvolution2*[threads];
      for(unsigned int t=0; t < threads; ++t)
        yzconvolve[t]=new ImplicitConvolution2MPI(my,mz,d.yz,
                                                  u1+t*mz*A*innerthreads,
                                                  u2+t*d.n2*A,mpi,A,B,
                                                  innerthreads,work2,global,
                                                  false);
      initpointers3(U3,u3,d.n);
    }
    for(unsigned int t=d.z < d.Z ? threads : 0; t < Threads; ++t)
      delete new ImplicitConvolution2MPI(my,mz,d.yz,u1,u2,mpi,A,B,
                                         innerthreads,work2,global,
                                         false);
    inittranspose(mpi,work,global);
  }
  
  // u1 is a temporary array of size mz*A*options.threads.
  // u2 is a temporary array of size d.y*mz*A*(d.z < mz ? 1 : threads).
  // u3 is a temporary array of size d.n*A.
  // A is the number of inputs.
  // B is the number of outputs.
  ImplicitConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                          const utils::split3& d,
                          Complex *u1, Complex *u2, Complex *u3, 
                          utils::mpiOptions mpi=utils::defaultmpiOptions,
                          unsigned int A=2, unsigned int B=1,
                          unsigned int threads=fftw::maxthreads,
                          Complex *work=NULL, Complex *work2=NULL,
                          MPI_Comm global=0) :
    ImplicitConvolution3(mx,my,mz,u1,u2,u3,A,B,threads,
                         convolveOptions(d.xy.y,d.z,d.n2,d.Activate(),mpi)),
    d(d) {
    initMPI(mpi,work,work2,global,threads);
  }

  ImplicitConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                          const utils::split3& d,
                          utils::mpiOptions mpi=utils::defaultmpiOptions,
                          unsigned int A=2, unsigned int B=1,
                          unsigned int threads=fftw::maxthreads,
                          Complex *work=NULL, Complex *work2=NULL,
                          MPI_Comm global=0) :
    ImplicitConvolution3(mx,my,mz,A,B,threads,
                         convolveOptions(d.xy.y,d.z,d.n2,d.Activate(),mpi)),
    d(d) {
    initMPI(mpi,work,work2,global,threads);
  }
  
  virtual ~ImplicitConvolution3MPI() {
    if(T) {
      delete U;
      delete T;
    }
  }
  
  // F is a pointer to A distinct data blocks each of size
  // 2mx*2d.y*d.z, shifted by offset (contents not preserved).
  void convolve(Complex **F, multiplier *pmult, unsigned int i=0,
                unsigned int offset=0);
  
  // Binary convolution:
  void convolve(Complex *f, Complex *g) {
    Complex *F[]={f,g};
    convolve(F,multbinary);
  }
};

void HermitianSymmetrizeXYMPI(unsigned int mx, unsigned int my,
                              utils::split3& d, bool xcompact, bool ycompact,
                              Complex *f, unsigned int nu=0, Complex *u=NULL);
 
// In-place implicitly dealiased 3D complex convolution.
class ImplicitHConvolution3MPI : public ImplicitHConvolution3 {
protected:
  utils::split3 d,du;
  utils::mpitranspose<Complex> *T,*U;
public:  
  void inittranspose(Complex *f, const utils::mpiOptions& mpi, Complex *work,
                     MPI_Comm global) {
    if(d.xy.y < d.Y) {
      T=new utils::mpitranspose<Complex>(d.X,d.Y,d.x,d.xy.y,d.z,f,work,
                                         d.xy.communicator,mpi,global);
      U=new utils::mpitranspose<Complex>(du.X,du.Y,du.x,du.xy.y,du.z,u3,work,
                                         du.xy.communicator,mpi,global);
    } else {T=U=NULL;}
    du.Deactivate();
  }

  void initMPI(Complex *f, const utils::mpiOptions& mpi,
               Complex *work, Complex *work2, MPI_Comm global,
               unsigned int Threads) {
    global=global ? global : d.communicator;
    if(d.z < d.Z) {
      yzconvolve=new ImplicitHConvolution2*[threads];
      for(unsigned int t=0; t < threads; ++t)
        yzconvolve[t]=
          new ImplicitHConvolution2MPI(my,mz,ycompact,zcompact,
                                       d.yz,du.yz,f,
                                       u1+t*(mz/2+1)*A*innerthreads,
                                       u2+t*du.n2*A,mpi,A,B,innerthreads,
                                       work2,global,false);
      initpointers3(U3,u3,du.n);
    }
    for(unsigned int t=d.z < d.Z ? threads : 0; t < Threads; ++t)
      delete new ImplicitHConvolution2MPI(my,mz,ycompact,zcompact,
                                          d.yz,du.yz,f,u1,u2,mpi,A,B,
                                          innerthreads,work2,global,false);
    inittranspose(f,mpi,work,global);
  }
  
  // f is a temporary array of size d.n needed only during construction.
  // u1 is a temporary array of size (mz/2+1)*A*options.threads,
  // u2 is a temporary array of size du.n2*A*(d.z < mz ? 1 : threads).
  // u3 is a temporary array of size du.n2*A.
  // A is the number of inputs.
  // B is the number of outputs.
  ImplicitHConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                           const utils::split3& d, const utils::split3& du,
                           Complex *f, Complex *u1, Complex *u2, Complex *u3,
                           utils::mpiOptions mpi=utils::defaultmpiOptions,
                           unsigned int A=2, unsigned int B=1,
                           unsigned int threads=fftw::maxthreads,
                           Complex *work=NULL, Complex *work2=NULL,
                           MPI_Comm global=0) :
    ImplicitHConvolution3(mx,my,mz,u1,u2,u3,A,B,threads,
                          convolveOptions(d.xy.y,d.z,du.n2,du.Activate(),mpi)),
    d(d), du(du) {
    initMPI(f,mpi,work,work2,global,threads);
  }

  ImplicitHConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                           bool xcompact, bool ycompact, bool zcompact,
                           const utils::split3& d, const utils::split3& du,
                           Complex *f, Complex *u1, Complex *u2, Complex *u3,
                           utils::mpiOptions mpi=utils::defaultmpiOptions,
                           unsigned int A=2, unsigned int B=1,
                           unsigned int threads=fftw::maxthreads,
                           Complex *work=NULL, Complex *work2=NULL,
                           MPI_Comm global=0) :
    ImplicitHConvolution3(mx,my,mz,xcompact,ycompact,zcompact,u1,u2,u3,A,B,
                          threads,convolveOptions(d.xy.y,d.z,du.n2,
                                                  du.Activate(),mpi)),
    d(d), du(du) {
    initMPI(f,mpi,work,work2,global,threads);
  }

  ImplicitHConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                           const utils::split3& d, const utils::split3& du,
                           Complex *f,
                           utils::mpiOptions mpi=utils::defaultmpiOptions,
                           unsigned int A=2, unsigned int B=1,
                           unsigned int threads=fftw::maxthreads,
                           Complex *work=NULL, Complex *work2=NULL,
                           MPI_Comm global=0) :
    ImplicitHConvolution3(mx,my,mz,true,true,true,A,B,threads,
                          convolveOptions(d.xy.y,d.z,du.n2,
                                          du.Activate(),mpi)), d(d), du(du) {
    initMPI(f,mpi,work,work2,global,threads);
  }
  
  ImplicitHConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                           bool xcompact, bool ycompact, bool zcompact,
                           const utils::split3& d, const utils::split3& du,
                           Complex *f,
                           utils::mpiOptions mpi=utils::defaultmpiOptions,
                           unsigned int A=2, unsigned int B=1,
                           unsigned int threads=fftw::maxthreads,
                           Complex *work=NULL, Complex *work2=NULL,
                           MPI_Comm global=0) :
    ImplicitHConvolution3(mx,my,mz,xcompact,ycompact,zcompact,A,B,threads,
                          convolveOptions(d.xy.y,d.z,du.n2,
                                          du.Activate(),mpi)), d(d), du(du) {
    initMPI(f,mpi,work,work2,global,threads);
  }
  
  virtual ~ImplicitHConvolution3MPI() {
    if(T) {
      delete U;
      delete T;
    }
  }
  
  void HermitianSymmetrize(Complex *f, Complex *u) {
    HermitianSymmetrizeXYMPI(mx,my,d,xcompact,ycompact,f,du.n,u);
  }
  
  // F is a pointer to A distinct data blocks each of size
  // (2mx-xcompact)*d.y*d.z, shifted by offset (contents not preserved).
  void convolve(Complex **F, realmultiplier *pmult, bool symmetrize=true,
                unsigned int i=0, unsigned int offset=0);
  
  // Binary convolution:
  void convolve(Complex *f, Complex *g, bool symmetrize=true) {
    Complex *F[]={f,g};
    convolve(F,multbinary,symmetrize);
  }
};


} // namespace fftwpp

#endif
