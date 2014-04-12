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
  splity d;
  fftw_plan intranspose,outtranspose;
  bool alltoall; // Use experimental nonblocking transpose
  mpitranspose<Complex> *T;
public:  
  
  void inittranspose() {
    int size;
    MPI_Comm_size(d.communicator,&size);
    alltoall=mx % size == 0 && my % size == 0;
    alltoall=false;

    if(alltoall) {
      T=new mpitranspose<Complex>(mx,d.y,d.x,my,1,u2);
      int rank;
      MPI_Comm_rank(d.communicator,&rank);
      if(rank == 0) {
        std::cout << "Using fast alltoall block transpose";
#if MPI_VERSION >= 3
        std::cout << " (NONBLOCKING MPI 3.0 version)";
#endif
        std::cout << std::endl;
      }
    } else {
      intranspose=
        fftw_mpi_plan_many_transpose(my,mx,2,d.block,0,(double*) u2,
                                     (double*) u2,d.communicator,
                                     FFTW_MPI_TRANSPOSED_IN);
      if(!intranspose) transposeError("in2");

      outtranspose=
        fftw_mpi_plan_many_transpose(mx,my,2,0,d.block,(double*) u2,
                                     (double*) u2,d.communicator,
                                     FFTW_MPI_TRANSPOSED_OUT);
      if(!outtranspose) transposeError("out2");
      MPISaveWisdom(d.communicator);
    }
  }

  // u1 is a temporary array of size my*A*threads.
  // u2 is a temporary array of size splity(mx,my).n*A.
  // A is the number of inputs.
  // B is the number of outputs.
  // threads is the number of threads to use in the outer subconvolution loop.
  ImplicitConvolution2MPI(unsigned int mx, unsigned int my, const splity& d,
                          Complex *u1, Complex *u2, 
                          unsigned int A=2, unsigned int B=1,
                          unsigned int threads=fftw::maxthreads) :
    ImplicitConvolution2(mx,my,u1,u2,A,B,threads,d.x,d.y,d.n), d(d) {
    inittranspose();
  }
  
  ImplicitConvolution2MPI(unsigned int mx, unsigned int my, const splity& d,
                          unsigned int A=2, unsigned int B=1,
                          unsigned int threads=fftw::maxthreads) :
    ImplicitConvolution2(mx,my,A,B,threads,d.x,d.y,d.n), d(d) {
    inittranspose();
  }
  
  virtual ~ImplicitConvolution2MPI() {
    if(!alltoall) {
      fftw_destroy_plan(outtranspose);
      fftw_destroy_plan(intranspose);
    }
  }
  
  void transpose(fftw_plan plan, unsigned int A, Complex **F,
                 unsigned int offset=0) {
    for(unsigned int a=0; a < A; ++a) {
      double *f=(double *) (F[a]+offset);
      fftw_mpi_execute_r2r(plan,f,f);
    }
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
  splity d,du;
  fftw_plan intranspose,outtranspose;
  fftw_plan uintranspose,uouttranspose;
public:  
  
  void inittranspose(Complex *f) {
    unsigned int nx=2*mx-compact;
    unsigned int mx1=mx+compact;
    unsigned int ny=my+!compact;
    intranspose=
      fftw_mpi_plan_many_transpose(ny,nx,2,d.block,0,(double*) f,(double*) f,
                                   d.communicator,FFTW_MPI_TRANSPOSED_IN);
    if(!intranspose) transposeError("inH2");
    uintranspose=
      fftw_mpi_plan_many_transpose(ny,mx1,2,du.block,0,(double*) u2,
                                   (double*) u2,du.communicator,
                                   FFTW_MPI_TRANSPOSED_IN);
    if(!uintranspose) transposeError("uinH2");
    outtranspose=
      fftw_mpi_plan_many_transpose(nx,ny,2,0,d.block,(double*) f,(double*) f,
                                   d.communicator,FFTW_MPI_TRANSPOSED_OUT);
    if(!outtranspose) transposeError("outH2");
    uouttranspose=
      fftw_mpi_plan_many_transpose(mx1,ny,2,0,du.block,(double*) u2,
                                   (double*) u2,du.communicator,
                                   FFTW_MPI_TRANSPOSED_OUT);
    if(!uouttranspose) transposeError("uoutH2");
    MPISaveWisdom(d.communicator);
  }
  
  // u1 is a temporary array of size (my/2+1)*A*threads.
  // u2 is a temporary array of size du.n*A.
  // A is the number of inputs.
  // B is the number of outputs.
  // threads is the number of threads to use in the outer subconvolution loop.
  // f is a temporary array of size d.n needed only during construction.
  ImplicitHConvolution2MPI(unsigned int mx, unsigned int my,
                           const splity& d, const splity& du,
                           Complex *f, Complex *u1, Complex *u2,
                           unsigned int A=2, unsigned int B=1,
                           bool compact=true,
                           unsigned int threads=fftw::maxthreads) :
    ImplicitHConvolution2(mx,my,u1,u2,A,B,compact,threads,d.x,d.y,du.n),
    d(d), du(du) {
    inittranspose(f);
  }
  
  ImplicitHConvolution2MPI(unsigned int mx, unsigned int my,
                           const splity& d, const splity& du,
                           Complex *f,
                           unsigned int A=2, unsigned int B=1,
                           bool compact=true,
                           unsigned int threads=fftw::maxthreads) :
    ImplicitHConvolution2(mx,my,A,B,compact,threads,d.x,d.y,du.n), d(d),
    du(du) {
    inittranspose(f);
  }
  
  virtual ~ImplicitHConvolution2MPI() {
    fftw_destroy_plan(uouttranspose);
    fftw_destroy_plan(outtranspose);
    fftw_destroy_plan(uintranspose);
    fftw_destroy_plan(intranspose);
  }

  void transpose(fftw_plan plan, unsigned int A, Complex **F,
                 unsigned int offset=0) {
    for(unsigned int a=0; a < A; ++a) {
      double *f=(double *) (F[a]+offset);
      fftw_mpi_execute_r2r(plan,f,f);
    }
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
  bool alltoall; // Use experimental nonblocking transpose
  mpitranspose<Complex> *T;
public:  
  void inittranspose() {
    int size;
    MPI_Comm_size(d.communicator,&size);
    alltoall=mx % size == 0 && my % size == 0;
//    alltoall=false;

    if(alltoall) {
      T=new mpitranspose<Complex>(mx,d.y,d.x,my,d.z,u3);
      int rank;
      MPI_Comm_rank(d.communicator,&rank);
      if(rank == 0) {
        std::cout << "Using fast alltoall block transpose";
#if MPI_VERSION >= 3
        std::cout << " (NONBLOCKING MPI 3.0 version)";
#endif
        std::cout << std::endl;        
      }
    } else {
      intranspose=
        fftw_mpi_plan_many_transpose(my,mx,2*d.z,d.yblock,0,
                                     (double*) u3,(double*) u3,
                                     d.xy.communicator,FFTW_MPI_TRANSPOSED_IN);
      if(!intranspose) transposeError("in3");
      outtranspose=
        fftw_mpi_plan_many_transpose(mx,my,2*d.z,0,d.yblock,
                                     (double*) u3,(double*) u3,
                                     d.xy.communicator,FFTW_MPI_TRANSPOSED_OUT);
      if(!outtranspose) transposeError("out3");
      MPISaveWisdom(d.xy.communicator);
    }
  }

  void initMPI() {
    if(d.z < d.nz) {
      yzconvolve=new ImplicitConvolution2*[threads];
      for(unsigned int t=0; t < threads; ++t)
        yzconvolve[t]=new ImplicitConvolution2MPI(my,mz,d.yz,
                                                  u1+t*mz*A*innerthreads,
                                                  u2+t*d.n2*A,A,B,
                                                  innerthreads);
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
    initMPI();
    inittranspose();
  }

  ImplicitConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                          const splityz& d,
                          unsigned int A=2, unsigned int B=1,
                          unsigned int threads=fftw::maxthreads) :
    ImplicitConvolution3(mx,my,mz,A,B,threads,d.y,d.z,d.n2,d.n), d(d) {
    initMPI();
    inittranspose();
  }
  
  virtual ~ImplicitConvolution3MPI() {
    if(!alltoall) {
      fftw_destroy_plan(intranspose);
      fftw_destroy_plan(outtranspose);
    }
  }
  
  void transpose(fftw_plan plan, unsigned int A, Complex **F,
                 unsigned int offset=0) {
    for(unsigned int a=0; a < A; ++a) {
      double *f=(double *) (F[a]+offset);
      fftw_mpi_execute_r2r(plan,f,f);
    }
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
  fftw_plan intranspose,outtranspose;
  fftw_plan uintranspose,uouttranspose;
public:  
  void inittranspose(Complex *f) {
    if(d.y < d.ny) {
      unsigned int mx1=mx+compact;
      intranspose=
        fftw_mpi_plan_many_transpose(d.ny,d.nx,2*d.z,d.yblock,0,
                                     (double*) f,(double*) f,
                                     d.xy.communicator,FFTW_MPI_TRANSPOSED_IN);
      if(!intranspose) transposeError("inH3");
      uintranspose=
        fftw_mpi_plan_many_transpose(d.ny,mx1,2*du.z,du.yblock,0,
                                     (double*) u3,(double*) u3,
                                     du.xy.communicator,FFTW_MPI_TRANSPOSED_IN);
      if(!uintranspose) transposeError("uinH3");
      outtranspose=
        fftw_mpi_plan_many_transpose(d.nx,d.ny,2*d.z,0,d.yblock,
                                     (double*) f,(double*) f,
                                     d.xy.communicator,FFTW_MPI_TRANSPOSED_OUT);
      if(!outtranspose) transposeError("outH3");
      uouttranspose=
        fftw_mpi_plan_many_transpose(mx1,d.ny,2*du.z,0,du.yblock,
                                     (double*) u3,(double*) u3,
                                     du.xy.communicator,
                                     FFTW_MPI_TRANSPOSED_OUT);
      if(!uouttranspose) transposeError("uoutH3");
      MPISaveWisdom(d.xy.communicator);
    }
  }

  void initMPI(Complex *f) {
    if(d.z < d.nz) {
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
                           unsigned int threads=fftw::maxthreads) :
    ImplicitHConvolution3(mx,my,mz,u1,u2,u3,A,B,compact,threads,d.y,d.z,
                          du.n2,du.n),
    d(d), du(du) { 
    initMPI(f);
    inittranspose(f);
  }

  ImplicitHConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                           const splityz& d, const splityz& du,
                           Complex *f, unsigned int A=2, unsigned int B=1,
                           bool compact=true,
                           unsigned int threads=fftw::maxthreads) :
    ImplicitHConvolution3(mx,my,mz,A,B,compact,threads,d.y,d.z,du.n2,du.n),
    d(d), du(du) { 
    initMPI(f);
    inittranspose(f);
  }
  
  virtual ~ImplicitHConvolution3MPI() {
    if(d.y < d.ny) {
      fftw_destroy_plan(uouttranspose);
      fftw_destroy_plan(outtranspose);
      fftw_destroy_plan(uintranspose);
      fftw_destroy_plan(intranspose);
    }
  }
  
  void transpose(fftw_plan plan, unsigned int A, Complex **F,
                 unsigned int offset=0) {
    for(unsigned int a=0; a < A; ++a) {
      double *f=(double *) (F[a]+offset);
      fftw_mpi_execute_r2r(plan,f,f);
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
