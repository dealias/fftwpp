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
  dimensions d;
  fftw_plan intranspose,outtranspose;
  bool alltoall; // Use experimental nonblocking transpose
  mpitranspose *T;
public:  
  
  void inittranspose() {
    int size;
    MPI_Comm_size(d.communicator,&size);
    alltoall=A == 2 && B == 1 && mx % size == 0 && my % size == 0;
    alltoall=false;

    if(alltoall) {
      T=new mpitranspose(mx,d.y,d.x,my,1,u2);
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
      SaveWisdom(d.communicator);
    }
  }

  // u1 is a temporary array of size my*A*threads.
  // u2 is a temporary array of size dimensions(mx,my).n*A.
  // A is the number of inputs.
  // B is the number of outputs.
  // threads is the number of threads to use in the outer subconvolution loop.
  ImplicitConvolution2MPI(unsigned int mx, unsigned int my, const dimensions& d,
                          Complex *u1, Complex *u2, 
                          unsigned int A=2, unsigned int B=1,
                          unsigned int threads=fftw::maxthreads) :
    ImplicitConvolution2(mx,my,u1,u2,A,B,threads,d.x,d.y,d.n), d(d) {
    inittranspose();
  }
  
  ImplicitConvolution2MPI(unsigned int mx, unsigned int my, const dimensions& d,
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
  dimensions d,du;
  fftw_plan intranspose,outtranspose;
  fftw_plan uintranspose,uouttranspose;
public:  
  
  void inittranspose(Complex *f) {
    unsigned int nx=2*mx-1;
    unsigned int mx1=mx+1;
    intranspose=
      fftw_mpi_plan_many_transpose(my,nx,2,d.block,0,(double*) f,(double*) f,
                                   d.communicator,FFTW_MPI_TRANSPOSED_IN);
    if(!intranspose) transposeError("inH2");
    uintranspose=
      fftw_mpi_plan_many_transpose(my,mx1,2,du.block,0,(double*) u2,
                                   (double*) u2,du.communicator,
                                   FFTW_MPI_TRANSPOSED_IN);
    if(!uintranspose) transposeError("uinH2");
    outtranspose=
      fftw_mpi_plan_many_transpose(nx,my,2,0,d.block,(double*) f,(double*) f,
                                   d.communicator,FFTW_MPI_TRANSPOSED_OUT);
    if(!outtranspose) transposeError("outH2");
    uouttranspose=
      fftw_mpi_plan_many_transpose(mx1,my,2,0,du.block,(double*) u2,
                                   (double*) u2,du.communicator,
                                   FFTW_MPI_TRANSPOSED_OUT);
    if(!uouttranspose) transposeError("uoutH2");
    SaveWisdom(d.communicator);
  }
  
  // u1 and v1 are temporary arrays of size (my/2+1)*M*threads.
  // w1 is a temporary array of size 3*M*threads.
  // u2 and v2 are temporary arrays of size du.n*M.
  // M is the number of data blocks (each corresponding to a dot product term).
  // threads is the number of threads to use in the outer subconvolution loop.
  // f is a temporary array of size d.n needed only during construction.
  ImplicitHConvolution2MPI(unsigned int mx, unsigned int my,
                           const dimensions& d, const dimensions& du,
                           Complex *f, Complex *u1, Complex *v1, Complex *w1,
                           Complex *u2, Complex *v2, unsigned int M=1,
                           unsigned int threads=fftw::maxthreads) :
    ImplicitHConvolution2(mx,my,u1,v1,w1,u2,v2,M,threads,d.x,d.y,10*du.n),
    d(d), du(du) {
    inittranspose(f);
  }
  
  ImplicitHConvolution2MPI(unsigned int mx, unsigned int my,
                           const dimensions& d, const dimensions& du,
                           Complex *f, unsigned int M=1,
                           unsigned int threads=fftw::maxthreads) :
    ImplicitHConvolution2(mx,my,M,threads,d.x,d.y,du.n), d(d), du(du) {
    inittranspose(f);
  }
  
  virtual ~ImplicitHConvolution2MPI() {
    fftw_destroy_plan(intranspose);
    fftw_destroy_plan(outtranspose);
    fftw_destroy_plan(uintranspose);
    fftw_destroy_plan(uouttranspose);
  }

  void pretranspose(Complex **F, unsigned int offset=0) {
    for(unsigned int s=0; s < M; ++s) {
      double *f=(double *) (F[s]+offset);
      fftw_mpi_execute_r2r(intranspose,f,f);
    }
  }
  
  void pretranspose(Complex *u2) {
    unsigned int stride=du.n;
    for(unsigned int s=0; s < M; ++s) {
      double *u=(double *) (u2+s*stride);
      fftw_mpi_execute_r2r(uintranspose,u,u);
    }
  }
  
  void posttranspose(const fftw_plan& plan, Complex *f) {
    fftw_mpi_execute_r2r(plan,(double *) f,(double *) f);
  }
  
  // F and G are distinct pointers to M distinct data blocks each of size 
  // (2mx-1)*my, shifted by offset (contents not preserved).
  // The output is returned in F[0].
  void convolve(Complex **F, Complex **G, Complex ***U, Complex **v,
                Complex **w, Complex **U2, Complex **V2, bool symmetrize=true,
                unsigned int offset=0);

  
  void convolve(Complex **F, Complex **G, bool symmetrize=true,
                unsigned int offset=0) {
    convolve(F,G,U,v,w,U2,V2,symmetrize,offset);
  }
  
  // Constructor for special case M=1:
  void convolve(Complex *f, Complex *g, bool symmetrize=true) {
    convolve(&f,&g,symmetrize);
  }
};

// In-place implicitly dealiased 3D complex convolution.
class ImplicitConvolution3MPI : public ImplicitConvolution3 {
protected:
  dimensions3 d;
  unsigned int innermostthreads;
  fftw_plan intranspose,outtranspose;
  bool alltoall; // Use experimental nonblocking transpose
  mpitranspose *T;
public:  
  void inittranspose() {
    int size;
    MPI_Comm_size(d.communicator,&size);
    alltoall=mx % size == 0 && my % size == 0;
    alltoall=false;

    if(alltoall) {
      T=new mpitranspose(mx,d.y,d.x,my,d.z,u3);
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
      SaveWisdom(d.xy.communicator);
    }
  }

  void initMPI() {
    if(d.z < mz) {
      for(unsigned int t=0; t < threads; ++t)
        yzconvolve[t]=new ImplicitConvolution2MPI(my,mz,d.yz,
                                                  u1+t*mz*A*innermostthreads,
                                                  u2+t*d.n2*A,A,B,innermostthreads);
      initpointers3(U3,u3,d.n);
    }
  }
  
  // u1 is a temporary array of size mz*A*threads.
  // u2 is a temporary array of size d.y*mz*A*(d.z < mz ? 1 : threads).
  // u3 is a temporary array of size d.n*M.
  // A is the number of inputs.
  // B is the number of outputs.
  // threads is the number of threads to use within the outer subconvolution loop.
  ImplicitConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                          const dimensions3& d,
                          Complex *u1, Complex *u2, Complex *u3, 
                          unsigned int A=2, unsigned int B=1,
                          unsigned int threads=fftw::maxthreads) :
    ImplicitConvolution3(mx,my,mz,u1,u2,u3,A,B,
                         d.z < mz ? 1 : threads,
                         d.y,d.z,d.n2,d.n), d(d), innermostthreads(threads) {
    initMPI();
    inittranspose();
  }

  ImplicitConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                          const dimensions3& d,
                          unsigned int A=2, unsigned int B=1,
                          unsigned int threads=fftw::maxthreads) :
    ImplicitConvolution3(mx,my,mz,A,B,
                         d.z < mz ? 1 : threads,
                         d.z < mz ? threads : 1,
                         d.y,d.z,d.n2,d.n), d(d), innermostthreads(threads) {
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
			      dimensions3& d, Complex *f, Complex *u);
 
// In-place implicitly dealiased 3D complex convolution.
class ImplicitHConvolution3MPI : public ImplicitHConvolution3 {
protected:
  dimensions3 d,du;
  unsigned int innermostthreads;
  fftw_plan intranspose,outtranspose;
  fftw_plan uintranspose,uouttranspose;
public:  
  void inittranspose(Complex *f) {
    if(d.y < d.ny) {
      unsigned int mx1=mx+1;
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
      SaveWisdom(d.xy.communicator);
    }
  }

  void initMPI(Complex *f) {
    if(d.z < mz) {
      yzconvolve=new ImplicitHConvolution2MPI(my,mz,d.yz,du.yz,f,u1,v1,w1,u2,v2,
                                              M,innermostthreads);
      yzconvolve->initpointers(U,v,w,innermostthreads);
      initpointers2(U2,V2,1,du.n2);
      initpointers3(U3,V3,u3,v3,du.n);
    }
  }
  
  // u1 and v1 are temporary arrays of size (mz/2+1)*M*threads,
  // w1 is a temporary array of size 3*M*threads.
  // u2 and v2 are temporary arrays of size du.n2*M*(d.z < mz ? 1 : threads).
  // u3 and v3 are temporary arrays of size du.n2*M.
  // M is the number of data blocks (each corresponding to a dot product term).
  // threads is the number of threads to use within the outer subconvolution loop.
  // f is a temporary array of size d.n2 needed only during construction.
  ImplicitHConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                           const dimensions3& d, const dimensions3& du,
                           Complex *f, Complex *u1, Complex *v1, Complex *w1,
                           Complex *u2, Complex *v2,
                           Complex *u3, Complex *v3, unsigned int M=1,
                           unsigned int threads=fftw::maxthreads) :
    ImplicitHConvolution3(mx,my,mz,u1,v1,w1,u2,v2,u3,v3,M,
                          d.z < mz ? 1 : threads,
                          d.y,d.z,du.n2,du.n),
    d(d), du(du), innermostthreads(threads) { 
    initMPI(f);
    inittranspose(f);
  }

  ImplicitHConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                           const dimensions3& d, const dimensions3& du,
                           Complex *f, unsigned int M=1,
                           unsigned int threads=fftw::maxthreads) :
    ImplicitHConvolution3(mx,my,mz,M,
                          d.z < mz ? 1 : threads,
                          d.z < mz ? threads : 1,
                          d.y,d.z,du.n2,du.n),
    d(d), du(du), innermostthreads(threads) { 
    initMPI(f);
    inittranspose(f);
  }
  
  virtual ~ImplicitHConvolution3MPI() {
    if(d.y < d.ny) {
      fftw_destroy_plan(intranspose);
      fftw_destroy_plan(outtranspose);
      fftw_destroy_plan(uintranspose);
      fftw_destroy_plan(uouttranspose);
    }
  }
  
  void pretranspose(Complex **F, unsigned int offset=0) {
    for(unsigned int s=0; s < M; ++s) {
      double *f=(double *) (F[s]+offset);
      fftw_mpi_execute_r2r(intranspose,f,f);
    }
  }
  
  void pretranspose(Complex *u3) {
    unsigned int stride=du.n;
    for(unsigned int s=0; s < M; ++s) {
      double *u=(double *) (u3+s*stride);
      fftw_mpi_execute_r2r(uintranspose,u,u);
    }
  }
  
  void posttranspose(const fftw_plan& plan, Complex *f) {
    fftw_mpi_execute_r2r(plan,(double *) f,(double *) f);
  }
  
  void HermitianSymmetrize(Complex *f, Complex *u) {
    HermitianSymmetrizeXYMPI(mx,my,d,f,u);
  }
  
  void convolve(Complex **F, Complex **G, Complex ***U, Complex **v,
                Complex **w, Complex ***U2, Complex ***V2, Complex **U3,
                Complex **V3, bool symmetrize=true, unsigned int offset=0);
  
  void convolve(Complex **F, Complex **G, bool symmetrize=true,
                unsigned int offset=0) {
    convolve(F,G,U,v,w,U2,V2,U3,V3,symmetrize,offset);
  }
  
  // Constructor for special case M=1:
  void convolve(Complex *f, Complex *g, bool symmetrize=true) {
    convolve(&f,&g,symmetrize);
  }
};


} // namespace fftwpp

#endif
