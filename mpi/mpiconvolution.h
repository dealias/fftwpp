#ifndef __mpiconvolution_h__
#define __mpiconvolution_h__ 1
  
#include <mpi.h>
#include <fftw3-mpi.h>
#include "convolution.h"
#include <vector>

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
  MPI_Comm active;
  unsigned int yblock,zblock; // requested block size
  MPI_Comm *XYplane;          // Used by HermitianSymmetrizeXYMPI
  int *reflect;               // Used by HermitianSymmetrizeXYMPI
  dimensions3() {}
  dimensions3(unsigned int nx, unsigned int ny,
              unsigned int Ny, unsigned int nz,
              const MPIgroup& group) : nx(nx), ny(ny), nz(nz),
                                       active(group.active),
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
public:  
  
  void inittranspose() {
    intranspose=
      fftw_mpi_plan_many_transpose(my,mx,2,d.block,0,(double*) u2,(double*) u2,
                                   d.communicator,FFTW_MPI_TRANSPOSED_IN);
    if(!intranspose) transposeError("in2");

    outtranspose=
      fftw_mpi_plan_many_transpose(mx,my,2,0,d.block,(double*) u2,(double*) u2,
                                   d.communicator,FFTW_MPI_TRANSPOSED_OUT);
    if(!outtranspose) transposeError("out2");
    SaveWisdom(d.communicator);
  }

  // u1 and v1 are temporary arrays of size my*M*threads.
  // u2 and v2 are temporary arrays of size dimensions(mx,my).n*M.
  // M is the number of data blocks (each corresponding to a dot product term).
  // threads is the number of threads to use in the outer subconvolution loop.
  ImplicitConvolution2MPI(unsigned int mx, unsigned int my, const dimensions& d,
                          Complex *u1, Complex *v1, Complex *u2, Complex *v2,
                          unsigned int M=1,
                          unsigned int threads=fftw::maxthreads) :
    ImplicitConvolution2(mx,my,u1,v1,u2,v2,M,threads,d.y,d.n), d(d) {
    inittranspose();
  }
  
  ImplicitConvolution2MPI(unsigned int mx, unsigned int my, const dimensions& d,
                          unsigned int M=1,
                          unsigned int threads=fftw::maxthreads) :
    ImplicitConvolution2(mx,my,M,threads,d.y,d.n), d(d) {
    inittranspose();
  }
  
  virtual ~ImplicitConvolution2MPI() {
    fftw_destroy_plan(intranspose);
    fftw_destroy_plan(outtranspose);
  }
  
  void pretranspose(Complex **F, unsigned int offset=0) {
    for(unsigned int s=0; s < M; ++s) {
      double *f=(double *) (F[s]+offset);
      fftw_mpi_execute_r2r(intranspose,f,f);
    }
  }
  
  void pretranspose(Complex *u2) {
    unsigned int stride=d.n;
    for(unsigned int s=0; s < M; ++s) {
      double *u=(double *) u2+s*stride;
      fftw_mpi_execute_r2r(intranspose,u,u);
    }
  }
  
  void posttranspose(Complex *f) {
    fftw_mpi_execute_r2r(outtranspose,(double *) f,(double *) f);
  }
  
  void convolve(Complex **F, Complex **G, Complex **u, Complex ***V,
                Complex **U2, Complex **V2, unsigned int offset=0) {
    Complex *u2=U2[0];
    Complex *v2=V2[0];
    
    backwards(F,u2,d.n,offset);
    backwards(G,v2,d.n,offset);
    
    pretranspose(F,offset);
    pretranspose(u2);
    pretranspose(G,offset);
    pretranspose(v2);
    unsigned int size=d.x*my;
    
    subconvolution(F,G,u,V,offset,size+offset);
    subconvolution(U2,V2,u,V,0,size);
    
    Complex *f=F[0]+offset;
    posttranspose(f);
    posttranspose(u2);
    
    forwards(f,u2);
  }
  
  // F and G are distinct pointers to M distinct data blocks each of size mx*d.y,
  // shifted by offset (contents not preserved).
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
      fftw_mpi_plan_many_transpose(my,mx1,2,du.block,0,
				   (double*) u2,(double*) u2,
                                   du.communicator,FFTW_MPI_TRANSPOSED_IN);
    if(!uintranspose) transposeError("uinH2");
    outtranspose=
      fftw_mpi_plan_many_transpose(nx,my,2,0,d.block,(double*) f,(double*) f,
                                   d.communicator,FFTW_MPI_TRANSPOSED_OUT);
    if(!outtranspose) transposeError("outH2");
    uouttranspose=
      fftw_mpi_plan_many_transpose(mx1,my,2,0,du.block,
				   (double*) u2,(double*) u2,
                                   du.communicator,FFTW_MPI_TRANSPOSED_OUT);
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
    ImplicitHConvolution2(mx,my,u1,v1,w1,u2,v2,M,threads,d.y,10*du.n),
    d(d), du(du) {
    inittranspose(f);
  }
  
  ImplicitHConvolution2MPI(unsigned int mx, unsigned int my,
                           const dimensions& d, const dimensions& du,
                           Complex *f, unsigned int M=1,
                           unsigned int threads=fftw::maxthreads) :
    ImplicitHConvolution2(mx,my,M,threads,d.y,du.n), d(d), du(du) {
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
      double *u=(double *) u2+s*stride;
      fftw_mpi_execute_r2r(uintranspose,u,u);
    }
  }
  
  void posttranspose(const fftw_plan& plan, Complex *f) {
    fftw_mpi_execute_r2r(plan,(double *) f,(double *) f);
  }
  
  // F and G are distinct pointers to M distinct data blocks each of size 
  // (2mx-1)*my, shifted by offset (contents not preserved).
  // The output is returned in F[0].
  void convolve(Complex **F, Complex **G, Complex ***U, 
		Complex **v, Complex **w,
                Complex **U2, Complex **V2, bool symmetrize=true,
                unsigned int offset=0) {
    Complex *u2=U2[0];
    Complex *v2=V2[0];
    
    if(d.y0 > 0) symmetrize=false;
    
    backwards(F,u2,d.y,du.n,symmetrize,offset);
    backwards(G,v2,d.y,du.n,symmetrize,offset);
    
    pretranspose(F,offset);
    pretranspose(u2);
    pretranspose(G,offset);
    pretranspose(v2);
    
    subconvolution(F,G,U,v,w,offset,d.x*my+offset);
    subconvolution(U2,V2,U,v,w,0,du.x*my);
    
    Complex *f=F[0]+offset;
    posttranspose(outtranspose,f);
    posttranspose(uouttranspose,u2);
    
    forwards(f,u2);
  }
  
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
public:  
  void inittranspose() {
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
    fftw_destroy_plan(intranspose);
    fftw_destroy_plan(outtranspose);
  }
  
  void pretranspose(Complex **F, unsigned int offset=0) {
    for(unsigned int s=0; s < M; ++s) {
      double *f=(double *) (F[s]+offset);
      fftw_mpi_execute_r2r(intranspose,f,f);
    }
  }
  
  void pretranspose(Complex *u3) {
    unsigned int stride=d.n;
    for(unsigned int s=0; s < M; ++s) {
      double *u=(double *) u3+s*stride;
      fftw_mpi_execute_r2r(intranspose,u,u);
    }
  }
  
  void posttranspose(Complex *f) {
    fftw_mpi_execute_r2r(outtranspose,(double *) f,(double *) f);
  }
  
  void convolve(Complex **F, Complex **G, Complex **u, Complex ***V,
                Complex ***U2, Complex ***V2, Complex **U3, Complex **V3,
                unsigned int offset=0) {
    Complex *u3=U3[0];
    Complex *v3=V3[0];
    
    backwards(F,u3,d.n,offset);
    backwards(G,v3,d.n,offset);
    
    pretranspose(F,offset);
    pretranspose(u3);
    pretranspose(G,offset);
    pretranspose(v3);
    
    unsigned int stride=my*d.z;
    unsigned int size=d.x*stride;
    
    subconvolution(F,G,u,V,U2,V2,offset,size+offset,stride);
    subconvolution(U3,V3,u,V,U2,V2,0,size,stride);
   
    Complex *f=F[0]+offset;
    posttranspose(f);
    posttranspose(u3);
    
    forwards(f,u3);
  }
  
  void convolve(Complex **F, Complex **G, unsigned int offset=0) {
    convolve(F,G,u,V,U2,V2,U3,V3,offset);
  }
  
  // Constructor for special case M=1:
  void convolve(Complex *f, Complex *g) {
    convolve(&f,&g);
  }
};

// Enforce 3D Hermiticity using specified (x,y > 0,z=0) and (x >= 0,y=0,z=0)
// data.
// u is a work array of size d.nx.
inline void HermitianSymmetrizeXYMPI(unsigned int mx, unsigned int my,
                                     dimensions3& d, Complex *f, Complex *u)
{
  if(d.y == d.ny && d.z == d.nz) {
    HermitianSymmetrizeXY(mx,my,d.nz,f);
    return;
  }

  MPI_Status stat;
  int rank,size;
  unsigned int yorigin=my-1;
  if(d.XYplane == NULL) {
    d.XYplane=new MPI_Comm;
    MPI_Comm_split(d.active,d.z0 == 0,0,d.XYplane);
    if(d.z0 != 0) return;
    d.reflect=new int[d.y];
    MPI_Comm_rank(*d.XYplane,&rank);
    MPI_Comm_size(*d.XYplane,&size);
    if(rank == 0) {
      int *process=new int[d.ny];
      unsigned int *y0=new unsigned int[size];
      unsigned int *y=new unsigned int[size];
      y0[0]=d.y0;
      y[0]=d.y;
      unsigned int stop=d.y0+d.y;
      for(unsigned int j=d.y0; j < stop; ++j)
        process[j]=0;
      for(int p=1; p < size; ++p) {
        unsigned int indices[2];
        MPI_Recv(indices,2,MPI_UNSIGNED,p,0,*d.XYplane,&stat);
        y0[p]=indices[0];
        y[p]=indices[1];
        unsigned int stop=y0[p]+y[p];
        for(unsigned int j=y0[p]; j < stop; ++j)
          process[j]=p;
      }
      for(unsigned int j=0; j < y[0]; ++j)
        d.reflect[j]=process[2*yorigin-y0[0]-j];
      for(int p=1; p < size; ++p) {
        for(unsigned int j=0; j < y[p]; ++j)
          MPI_Send(process+2*yorigin-y0[p]-j,1,MPI_INT,p,j,*d.XYplane);
      }
      delete [] y;
      delete [] y0;
      delete [] process;
    } else {
      unsigned int indices[2]={d.y0,d.y};
      MPI_Send(indices,2,MPI_UNSIGNED,0,0,*d.XYplane);
        
      for(unsigned int j=0; j < d.y; ++j)
        MPI_Recv(d.reflect+j,1,MPI_INT,0,j,*d.XYplane,&stat);
    }
  }
  if(d.z0 != 0) return;
  
  MPI_Comm_rank(*d.XYplane,&rank);
  MPI_Comm_size(*d.XYplane,&size);
  
  unsigned int stride=d.y*d.z;
  unsigned int offset=2*yorigin-d.y0;
  unsigned int start=(yorigin > d.y0) ? yorigin-d.y0 : 0;
  for(unsigned int j=start; j < d.y; ++j) {
    for(unsigned int i=0; i < d.nx; ++i)
      u[i]=conj(f[stride*(d.nx-1-i)+d.z*j]);
    int J=d.reflect[j];
    if(J != rank)
      MPI_Send(u,2*d.nx,MPI_DOUBLE,J,0,*d.XYplane);
    else {
      if(d.y0+j != yorigin) {
        for(unsigned int i=0; i < d.nx; ++i)
          f[stride*i+d.z*(offset-j)]=u[i];
      } else {
        unsigned int origin=stride*(mx-1)+d.z*j;
        f[origin].im=0.0;
        unsigned int mxstride=mx*stride;
        for(unsigned int i=stride; i < mxstride; i += stride)
          f[origin-i]=conj(f[origin+i]);
      }
    }
  }
  for(int j=std::min((int)d.y,(int)(yorigin-d.y0))-1; j >= 0; --j) {
    int J=d.reflect[j];
    if(J != rank) {
      MPI_Recv(u,2*d.nx,MPI_DOUBLE,J,0,*d.XYplane,&stat);
      for(unsigned int i=0; i < d.nx; ++i)
        f[stride*i+d.z*j]=u[i];
    }
  }
}

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
      double *u=(double *) u3+s*stride;
      fftw_mpi_execute_r2r(uintranspose,u,u);
    }
  }
  
  void posttranspose(const fftw_plan& plan, Complex *f) {
    fftw_mpi_execute_r2r(plan,(double *) f,(double *) f);
  }
  
  void HermitianSymmetrize(Complex *f, Complex *u) {
      HermitianSymmetrizeXYMPI(mx,my,d,f,u);
  }
  
  void convolve(Complex **F, Complex **G, Complex ***U, Complex **v, Complex **w,
                Complex ***U2, Complex ***V2, Complex **U3, Complex **V3,
                bool symmetrize=true, unsigned int offset=0) {
    Complex *u3=U3[0];
    Complex *v3=V3[0];
    
    backwards(F,u3,du.n,symmetrize,offset);
    backwards(G,v3,du.n,symmetrize,offset);
    
    if(d.y < d.ny) {
      pretranspose(F,offset);
      pretranspose(u3);
      pretranspose(G,offset);
      pretranspose(v3);
    }
    
    unsigned int stride=d.ny*d.z;
    subconvolution(F,G,U,v,w,U2,V2,offset,d.x*stride+offset,stride);
    subconvolution(U3,V3,U,v,w,U2,V2,0,du.x*stride,stride);
    
    Complex *f=F[0]+offset;
    if(d.y < d.ny) {
      posttranspose(outtranspose,f);
      posttranspose(uouttranspose,u3);
    }
    
    forwards(f,u3);
  }
  
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
