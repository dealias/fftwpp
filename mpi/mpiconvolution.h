#define NEW 0

#ifndef __mpiconvolution_h__
#define __mpiconvolution_h__ 1
  
#include <mpi.h>
#include <fftw3-mpi.h>
#include "convolution.h"
#include <vector>
#include "mpi/mpitranspose.h"

namespace fftwpp {

inline unsigned int ceilquotient(unsigned int a, unsigned int b)
{
  return (a+b-1)/b;
}

extern MPI_Comm *active;

class MPIgroup {
public:  
  unsigned int my;
  int rank,size;
  unsigned int yblock,zblock;
  MPI_Comm active;                     // active communicator 
  MPI_Comm communicator,communicator2; // 3D transpose communicators
  
  void init() {
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size); 
  }
  
  void activate() {
    MPI_Comm_split(MPI_COMM_WORLD,rank < size,0,&active);
    fftwpp::active=&active;
  }
  
  void matrix() {
    int major=ceilquotient(size,my);
    int p=rank % major;
    int q=rank / major;
  
    /* Split nodes into row and columns */ 
    MPI_Comm_split(active,p,q,&communicator); 
    MPI_Comm_split(active,q,p,&communicator2);
  }
  
  MPIgroup(unsigned int my) : my(my) {
    init();
    yblock=ceilquotient(my,size);
    size=ceilquotient(my,yblock);
    activate();
  }
    
  MPIgroup(unsigned int my, unsigned int mz, bool allowPencil=true) : my(my) {
    init();
    yblock=ceilquotient(my,size);
    zblock=allowPencil ? ceilquotient(mz,size*yblock/my) : mz;
    size=ceilquotient(my,yblock)*ceilquotient(mz,zblock);
    activate();
    if(rank < size)
      matrix();
  }
};

void show(Complex *f, unsigned int nx, unsigned int ny, const MPIgroup& group);
void show(Complex *f, unsigned int nx, unsigned int ny, unsigned int nz,
          const MPIgroup& group);
int hash(Complex *f, unsigned int nx, unsigned int ny, const MPIgroup& group);
int hash(Complex *f, unsigned int nx, unsigned int ny, unsigned int nz,
         const MPIgroup& group);

// Class to compute the local array dimensions and storage requirements for
// distributing the y index among multiple MPI processes.
class dimensions {
public:
  unsigned int nx,ny; // problem size
  unsigned int x;     // local transposed matrix is x X ny
  unsigned int y;     // local matrix is nx X y
  unsigned int x0;    // local starting x value
  unsigned int y0;    // local starting y value
  unsigned int n;     // total required storage (Complex words)
  MPI_Comm communicator;
  unsigned int block; // requested block size
  unsigned int M;     // number of Complex words per matrix element 
  dimensions() {}
  dimensions(unsigned int nx, unsigned int ny, MPI_Comm communicator,
             unsigned int Block=0, unsigned int M=1) 
    : nx(nx), ny(ny), communicator(communicator), block(Block), M(M) {
    if(block == 0) {
      int size;
      MPI_Comm_size(communicator,&size);
      block=ceilquotient(ny,size);
    }
    
    ptrdiff_t N[2]={ny,nx};
    ptrdiff_t local0,local1;
    ptrdiff_t start0,start1;
    n=fftw_mpi_local_size_many_transposed(2,N,2*M,block,0,
                                          communicator,&local0,
                                          &start0,&local1,&start1)*
      sizeof(double)/sizeof(Complex);
    x=local1;
    y=local0;
    x0=start1;
    y0=start0;
  }
};

//         local matrix is nx X   y  X z
// xy transposed matrix is  x X  ny  X z  allocated n  Complex words
// yz transposed matrix is  x X yz.x X nz allocated n2 Complex words
class dimensions3 {
public:
  unsigned int n,n2;
  unsigned int nx,ny,nz;
  unsigned int x;
  unsigned int y;
  unsigned int z;
  unsigned int x0;
  unsigned int y0;
  unsigned int z0;
  dimensions xy;
  dimensions yz;
  MPI_Comm communicator;
  unsigned int yblock,zblock; // requested block size
  MPI_Comm *XYplane;          // Used by HermitianSymmetrizeXYMPI
  int *reflect;               // Used by HermitianSymmetrizeXYMPI
  dimensions3() {}
  dimensions3(unsigned int nx, unsigned int ny,
              unsigned int Ny, unsigned int nz,
              const MPIgroup& group) : nx(nx), ny(ny), nz(nz),
                                       communicator(group.active),
                                       yblock(group.yblock),
                                       zblock(group.zblock), XYplane(NULL) {
    xy=dimensions(nx,ny,group.communicator,yblock,zblock);
    yz=dimensions(Ny,nz,group.communicator2,zblock);
    n=xy.n=std::max(xy.n,(xy.x+1)*yz.n);
    n2=yz.n;
    x=xy.x;
    y=xy.y;
    z=yz.y;
    x0=0;
    y0=xy.y0;
    z0=yz.y0;
  }
};
  
void LoadWisdom(const MPI_Comm& active);
void SaveWisdom(const MPI_Comm& active);
  
inline void transposeError(const char *text) {
  std::cout << "Cannot construct " << text << " transpose plan." << std::endl;
  exit(-1);
}

// In-place implicitly dealiased 2D complex convolution.
class ImplicitConvolution2MPI : public ImplicitConvolution2 {
protected:
  dimensions d;
  fftw_plan intranspose,outtranspose;
  bool alltoall; // Use experimental nonblocking transpose
  transpose *T;
public:  
  
  void inittranspose() {
    int size;
    MPI_Comm_size(d.communicator,&size);
    alltoall=mx % size == 0 && my % size == 0;
    alltoall=false;
#if NEW    
    if(!alltoall) T=new transpose(mx,d.y,d.x,my);
#endif    

    if(alltoall) {
      T=new transpose(mx,d.y,d.x,my);
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

  // u1 and v1 are temporary arrays of size my*M*threads.
  // u2 and v2 are temporary arrays of size dimensions(mx,my).n*M.
  // M is the number of data blocks (each corresponding to a dot product term).
  // threads is the number of threads to use in the outer subconvolution loop.
  ImplicitConvolution2MPI(unsigned int mx, unsigned int my, const dimensions& d,
                          Complex *u1, Complex *v1, Complex *u2, Complex *v2,
                          unsigned int M=1,
                          unsigned int threads=fftw::maxthreads) :
    ImplicitConvolution2(mx,my,u1,v1,u2,v2,M,threads,d.x,d.y,d.n), d(d) {
    inittranspose();
  }
  
  ImplicitConvolution2MPI(unsigned int mx, unsigned int my, const dimensions& d,
                          unsigned int M=1,
                          unsigned int threads=fftw::maxthreads) :
    ImplicitConvolution2(mx,my,M,threads,d.x,d.y,d.n), d(d) {
    inittranspose();
  }
  
  virtual ~ImplicitConvolution2MPI() {
    if(!alltoall) {
      fftw_destroy_plan(outtranspose);
      fftw_destroy_plan(intranspose);
    }
  }
  
  void pretranspose(Complex **F, unsigned int offset=0) {
    for(unsigned int s=0; s < M; ++s) {
#if NEW      
      T->InTransposed(F[s]+offset);
#else      
      double *f=(double *) (F[s]+offset);
      fftw_mpi_execute_r2r(intranspose,f,f);
#endif      
    }
  }
  
  void pretranspose(Complex *u2) {
    unsigned int stride=d.n;
    for(unsigned int s=0; s < M; ++s) {
#if NEW      
      T->InTransposed(u2+s*stride);
#else      
      double *u=(double *) (u2+s*stride);
      fftw_mpi_execute_r2r(intranspose,u,u);
#endif      
    }
  }
  
  void posttranspose(Complex *f) {
//    T->OutTransposed(f);
    fftw_mpi_execute_r2r(outtranspose,(double *) f,(double *) f);
  }
  
  void convolve(Complex **F, Complex **G, Complex **u, Complex ***V,
                Complex **U2, Complex **V2, unsigned int offset=0);
  
  // F and G are distinct pointers to M distinct data blocks each of size
  // mx*d.y, shifted by offset (contents not preserved).
  // The output is returned in F[0].
  void convolve(Complex **F, Complex **G, unsigned int offset=0) {
    convolve(F,G,u,V,U2,V2,offset);
  }

  // Convolution for special case M=1:
  void convolve(Complex *f, Complex *g) {
    convolve(&f,&g);
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
  unsigned int innerthreads;
  fftw_plan intranspose,outtranspose;
  bool alltoall; // Use experimental nonblocking transpose
  transpose *T;
public:  
  void inittranspose() {
    int size;
    MPI_Comm_size(d.communicator,&size);
    alltoall=mx % size == 0 && my % size == 0;
    alltoall=false;
#if NEW    
    if(!alltoall) T=new transpose(mx,d.y,d.x,my,d.z);
#endif    

    if(alltoall) {
      T=new transpose(mx,d.y,d.x,my,d.z);
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
      yzconvolve=new ImplicitConvolution2MPI(my,mz,d.yz,u1,v1,u2,v2,M,
                                             innerthreads);
      yzconvolve->initpointers(u,V,innerthreads);
      initpointers2(U2,V2,1,d.n2);
      initpointers3(U3,V3,u3,v3,d.n);
    }
  }
  
  // u1 and v1 are temporary arrays of size mz*M*threads.
  // u2 and v2 are temporary arrays of size d.y*mz*M*(d.z < mz ? 1 : threads).
  // u3 and v3 are temporary arrays of size d.n*M.
  // M is the number of data blocks (each corresponding to a dot product term).
  // threads is the number of threads to use within the innermost MPI node.
  ImplicitConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                          const dimensions3& d,
                          Complex *u1, Complex *v1,
                          Complex *u2, Complex *v2,
                          Complex *u3, Complex *v3, unsigned int M=1,
                          unsigned int threads=fftw::maxthreads) :
    ImplicitConvolution3(mx,my,mz,u1,v1,u2,v2,u3,v3,M,
                         d.z < mz ? 1 : threads,
                         d.y,d.z,d.n2,d.n), d(d), innerthreads(threads) {
    initMPI();
    inittranspose();
  }

  ImplicitConvolution3MPI(unsigned int mx, unsigned int my, unsigned int mz,
                          const dimensions3& d, unsigned int M=1,
                          unsigned int threads=fftw::maxthreads) :
    ImplicitConvolution3(mx,my,mz,M,
                         d.z < mz ? 1 : threads,
                         d.z < mz ? threads : 1,
                         d.y,d.z,d.n2,d.n), d(d), innerthreads(threads) {
    initMPI();
    inittranspose();
  }
  
  virtual ~ImplicitConvolution3MPI() {
    if(!alltoall) {
      fftw_destroy_plan(intranspose);
      fftw_destroy_plan(outtranspose);
    }
  }
  
  void pretranspose(Complex **F, unsigned int offset=0) {
    for(unsigned int s=0; s < M; ++s) {
#if NEW    
      T->InTransposed(F[s]+offset);
#else      
      double *f=(double *) (F[s]+offset);
      fftw_mpi_execute_r2r(intranspose,f,f);
#endif      
    }
  }
  
  void pretranspose(Complex *u3) {
    unsigned int stride=d.n;
    for(unsigned int s=0; s < M; ++s) {
#if NEW    
      T->InTransposed(u2+s*stride);
#else      
      double *u=(double *) (u3+s*stride);
      fftw_mpi_execute_r2r(intranspose,u,u);
#endif      
    }
  }
  
  void posttranspose(Complex *f) {
#if NEW    
    T->OutTransposed(f);
#else
    fftw_mpi_execute_r2r(outtranspose,(double *) f,(double *) f);
#endif    
  }
  
  void convolve(Complex **F, Complex **G, Complex **u, Complex ***V,
                Complex ***U2, Complex ***V2, Complex **U3, Complex **V3,
                unsigned int offset=0);
  
  void convolve(Complex **F, Complex **G, unsigned int offset=0) {
    convolve(F,G,u,V,U2,V2,U3,V3,offset);
  }
  
  // Constructor for special case M=1:
  void convolve(Complex *f, Complex *g) {
    convolve(&f,&g);
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
  unsigned int innerthreads;
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
                                              M,innerthreads);
      yzconvolve->initpointers(U,v,w,innerthreads);
      initpointers2(U2,V2,1,du.n2);
      initpointers3(U3,V3,u3,v3,du.n);
    }
  }
  
  // u1 and v1 are temporary arrays of size (mz/2+1)*M*threads,
  // w1 is a temporary array of size 3*M*threads.
  // u2 and v2 are temporary arrays of size du.n2*M*(d.z < mz ? 1 : threads).
  // u3 and v3 are temporary arrays of size du.n2*M.
  // M is the number of data blocks (each corresponding to a dot product term).
  // threads is the number of threads to use within the innermost MPI node.
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
    d(d), du(du), innerthreads(threads) { 
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
    d(d), du(du), innerthreads(threads) { 
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
