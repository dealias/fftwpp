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
  utils::split d;
  utils::mpitranspose<Complex> *T;
public:  
  
  void inittranspose(const utils::mpiOptions& mpioptions,
                     const MPI_Comm& global) {
    T=new utils::mpitranspose<Complex>(d.X,d.y,d.x,d.Y,1,u2,d.communicator,
                                       mpioptions,
                                       global ? global : d.communicator);
    d.Deactivate();
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
                          MPI_Comm global=0) :
    ImplicitConvolution2(mx,my,u1,u2,A,B,threads,
                         convolveOptions(d.x,d.y,d.Activate(),mpi)), d(d) {
    inittranspose(mpi,global);
  }
  
  ImplicitConvolution2MPI(unsigned int mx, unsigned int my,
                          const utils::split& d,
                          utils::mpiOptions mpi=utils::defaultmpiOptions,
                          unsigned int A=2, unsigned int B=1,
                          unsigned int threads=fftw::maxthreads,
                          MPI_Comm global=0) :
    ImplicitConvolution2(mx,my,A,B,threads,
                         convolveOptions(d.x,d.y,d.Activate(),mpi)), d(d) {
    inittranspose(mpi,global);
  }
  
  virtual ~ImplicitConvolution2MPI() {
    delete T;
  }
  
  // F is a pointer to A distinct data blocks each of size mx*d.y,
  // shifted by offset (contents not preserved).
  void convolve(Complex **F, multiplier *pmult,
                std::vector<unsigned int>&index=index1, unsigned int offset=0);
  
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
  
  void inittranspose(Complex *f, const utils::mpiOptions& mpi,
                     MPI_Comm global) {
    global=global ? global : d.communicator;
    T=new utils::mpitranspose<Complex>(d.X,d.y,d.x,d.Y,1,f,
                               d.communicator,mpi,global);
    U=new utils::mpitranspose<Complex>(du.X,du.y,du.x,du.Y,1,u2,
                                du.communicator,mpi,global);
    du.Deactivate();
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
                           MPI_Comm global=0) :
    ImplicitHConvolution2(mx,my,u1,u2,A,B,threads,
                          convolveOptions(d.x,d.y,du.Activate(),mpi)),
    d(d), du(du) {
    inittranspose(f,mpi,global);
  }
  
  ImplicitHConvolution2MPI(unsigned int mx, unsigned int my,
                           bool xcompact, bool ycompact,
                           const utils::split& d, const utils::split& du,
                           Complex *f, Complex *u1, Complex *u2,
                           utils::mpiOptions mpi=utils::defaultmpiOptions,
                           unsigned int A=2, unsigned int B=1,
                           unsigned int threads=fftw::maxthreads,
                           MPI_Comm global=0) :
    ImplicitHConvolution2(mx,my,xcompact,ycompact,u1,u2,A,B,threads,
                          convolveOptions(d.x,d.y,du.Activate(),mpi)),
    d(d), du(du) {
    inittranspose(f,mpi,global);
  }
  
  ImplicitHConvolution2MPI(unsigned int mx, unsigned int my,
                           const utils::split& d, const utils::split& du,
                           Complex *f,
                           utils::mpiOptions mpi=utils::defaultmpiOptions,
                           unsigned int A=2, unsigned int B=1,
                           unsigned int threads=fftw::maxthreads,
                           MPI_Comm global=0) :
    ImplicitHConvolution2(mx,my,true,true,A,B,threads,
                          convolveOptions(d.x,d.y,du.Activate(),mpi)),
    d(d), du(du) {
    inittranspose(f,mpi,global);
  }
  
  ImplicitHConvolution2MPI(unsigned int mx, unsigned int my,
                           bool xcompact, bool ycompact,
                           const utils::split& d, const utils::split& du,
                           Complex *f,
                           utils::mpiOptions mpi=utils::defaultmpiOptions,
                           unsigned int A=2, unsigned int B=1,
                           unsigned int threads=fftw::maxthreads,
                           MPI_Comm global=0) :
    ImplicitHConvolution2(mx,my,xcompact,ycompact,A,B,threads,
                          convolveOptions(d.x,d.y,du.Activate(),mpi)),
    d(d), du(du) {
    inittranspose(f,mpi,global);
  }
  
  virtual ~ImplicitHConvolution2MPI() {
    delete U;
    delete T;
  }

  // F is a pointer to A distinct data blocks each of size 
  // (2mx-xcompact)*d.y, shifted by offset (contents not preserved).
  void convolve(Complex **F, realmultiplier *pmult,
                bool symmetrize=true,
                std::vector<unsigned int>&index=index1, unsigned int offset=0);

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
  utils::mpitranspose<Complex> *T;
public:  
  void inittranspose(const utils::mpiOptions& mpi, MPI_Comm global) {
    T=d.xy.y < d.Y ? 
               new utils::mpitranspose<Complex>(d.X,d.xy.y,d.x,d.Y,d.z,u3,
                                         d.xy.communicator,mpi,global) :
      NULL;
    d.Deactivate();
  }

  void initMPI(const utils::mpiOptions& mpi, MPI_Comm global) {
    global=global ? global : d.communicator;
    if(d.z < d.Z) {
      yzconvolve=new ImplicitConvolution2*[threads];
      for(unsigned int t=0; t < threads; ++t)
        yzconvolve[t]=new ImplicitConvolution2MPI(my,mz,d.yz,
                                                  u1+t*mz*A*innerthreads,
                                                  u2+t*d.n2*A,mpi,A,B,
                                                  innerthreads,global);
      initpointers3(U3,u3,d.n);
    }
    inittranspose(mpi,global);
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
                          MPI_Comm global=0) :
    ImplicitConvolution3(mx,my,mz,u1,u2,u3,A,B,threads,
                         convolveOptions(d.xy.y,d.z,d.n2,d.Activate(),mpi)),
    d(d) {
    initMPI(mpi,global);
  }

  ImplicitConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                          const utils::split3& d,
                          utils::mpiOptions mpi=utils::defaultmpiOptions,
                          unsigned int A=2, unsigned int B=1,
                          unsigned int threads=fftw::maxthreads,
                          MPI_Comm global=0) :
    ImplicitConvolution3(mx,my,mz,A,B,threads,
                         convolveOptions(d.xy.y,d.z,d.n2,d.Activate(),mpi)),
    d(d) {
    initMPI(mpi,global);
  }
  
  virtual ~ImplicitConvolution3MPI() {
    if(T) delete T;
  }
  
  // F is a pointer to A distinct data blocks each of size
  // 2mx*2d.y*d.z, shifted by offset (contents not preserved).
  void convolve(Complex **F, multiplier *pmult,
                std::vector<unsigned int>&index=index2, unsigned int offset=0);
  
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
			      utils::split3& d, bool xcompact, bool ycompact,
                              Complex *f, unsigned int nu=0, Complex *u=NULL);
 
// In-place implicitly dealiased 3D complex convolution.
class ImplicitHConvolution3MPI : public ImplicitHConvolution3 {
protected:
  utils::split3 d,du;
  utils::mpitranspose<Complex> *T,*U;
public:  
  void inittranspose(Complex *f, const utils::mpiOptions& mpi,
                     MPI_Comm global) {
    if(d.xy.y < d.Y) {
      T=new utils::mpitranspose<Complex>(d.X,d.xy.y,d.x,d.Y,d.z,f,
                                 d.xy.communicator,mpi,global);
      U=new utils::mpitranspose<Complex>(du.X,du.xy.y,du.x,du.Y,du.z,u3,
                                 du.xy.communicator,mpi,global);
    } else {T=U=NULL;}
    du.Deactivate();
  }

  void initMPI(Complex *f, const utils::mpiOptions& mpi, MPI_Comm global) {
    global=global ? global : d.communicator;
    if(d.z < d.Z) {
      yzconvolve=new ImplicitHConvolution2*[threads];
      for(unsigned int t=0; t < threads; ++t)
        yzconvolve[t]=
          new ImplicitHConvolution2MPI(my,mz,ycompact,zcompact,
                                       d.yz,du.yz,f,
                                       u1+t*(mz/2+1)*A*innerthreads,
                                       u2+t*du.n2*A,mpi,A,B,innerthreads,
                                       global);
      initpointers3(U3,u3,du.n);
    }
    inittranspose(f,mpi,global);
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
                           MPI_Comm global=0) :
    ImplicitHConvolution3(mx,my,mz,u1,u2,u3,A,B,threads,
                          convolveOptions(d.xy.y,d.z,du.n2,du.Activate(),mpi)),
    d(d), du(du) {
    initMPI(f,mpi,global);
  }

  ImplicitHConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                           bool xcompact, bool ycompact, bool zcompact,
                           const utils::split3& d, const utils::split3& du,
                           Complex *f, Complex *u1, Complex *u2, Complex *u3,
                           utils::mpiOptions mpi=utils::defaultmpiOptions,
                           unsigned int A=2, unsigned int B=1,
                           unsigned int threads=fftw::maxthreads,
                           MPI_Comm global=0) :
    ImplicitHConvolution3(mx,my,mz,xcompact,ycompact,zcompact,u1,u2,u3,A,B,
                          threads,convolveOptions(d.xy.y,d.z,du.n2,
                                                  du.Activate(),mpi)),
    d(d), du(du) {
    initMPI(f,mpi,global);
  }

  ImplicitHConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                           const utils::split3& d, const utils::split3& du,
                           Complex *f,
                           utils::mpiOptions mpi=utils::defaultmpiOptions,
                           unsigned int A=2, unsigned int B=1,
                           unsigned int threads=fftw::maxthreads,
                           MPI_Comm global=0) :
    ImplicitHConvolution3(mx,my,mz,true,true,true,A,B,threads,
                          convolveOptions(d.xy.y,d.z,du.n2,
                                          du.Activate(),mpi)), d(d), du(du) {
    initMPI(f,mpi,global);
  }
  
  ImplicitHConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                           bool xcompact, bool ycompact, bool zcompact,
                           const utils::split3& d, const utils::split3& du,
                           Complex *f,
                           utils::mpiOptions mpi=utils::defaultmpiOptions,
                           unsigned int A=2, unsigned int B=1,
                           unsigned int threads=fftw::maxthreads,
                           MPI_Comm global=0) :
    ImplicitHConvolution3(mx,my,mz,xcompact,ycompact,zcompact,A,B,threads,
                          convolveOptions(d.xy.y,d.z,du.n2,
                                          du.Activate(),mpi)), d(d), du(du) {
    initMPI(f,mpi,global);
  }
  
  virtual ~ImplicitHConvolution3MPI() {
    if(T) {
      delete U;
      delete T;
    }
  }
  
  void transpose(utils::mpitranspose<Complex> *T, unsigned int A, Complex **F,
                 bool inflag, bool outflag, unsigned int offset=0) {
    for(unsigned int a=0; a < A; ++a)
      T->transpose(F[a]+offset,inflag,outflag);
  }
  
  void HermitianSymmetrize(Complex *f, Complex *u) {
    HermitianSymmetrizeXYMPI(mx,my,d,xcompact,ycompact,f,du.n,u);
  }
  
  // F is a pointer to A distinct data blocks each of size
  // (2mx-xcompact)*d.y*d.z, shifted by offset (contents not preserved).
  void convolve(Complex **F, realmultiplier *pmult, bool symmetrize=true,
                std::vector<unsigned int>&index=index2, unsigned int offset=0);
  
  // Binary convolution:
  void convolve(Complex *f, Complex *g, bool symmetrize=true) {
    Complex *F[]={f,g};
    convolve(F,multbinary,symmetrize);
  }
};


} // namespace fftwpp

#endif
