#ifndef __mpifftwpp_h__
#define __mpifftwpp_h__ 1

#include "mpi/mpitranspose.h"
#include <fftw3-mpi.h>
#include "fftw++.h"

namespace fftwpp {

// defined in fftw++.cc
extern bool mpi;

void MPILoadWisdom(const MPI_Comm& active);
void MPISaveWisdom(const MPI_Comm& active);

inline unsigned int ceilquotient(unsigned int a, unsigned int b)
{
  return (a+b-1)/b;
}

// Distribute first over x, then (if allowpencil=true), over y.
class MPIgroup {
public:  
  int rank,size;
  unsigned int block,block2;
  MPI_Comm active;                     // active communicator 
  MPI_Comm communicator,communicator2; // 3D transpose communicators
  
  void init(const MPI_Comm& comm) {
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);
  }
  
  void activate(const MPI_Comm& comm) {
    MPI_Comm_split(comm,rank < size,0,&active);
    fftw::mpi=true;
  }
  
  MPIgroup(const MPI_Comm& comm, unsigned int x) {
    init(comm);
    block=ceilquotient(x,size);
    size=ceilquotient(x,block);
    activate(comm);
  }
  
  MPIgroup(const MPI_Comm& comm, unsigned int x, unsigned int y, 
	   bool allowPencil=true) {
    init(comm);
    block=ceilquotient(x,size);
    block2=allowPencil ? ceilquotient(y,size*block/x) : y;
    size=ceilquotient(x,block)*ceilquotient(y,block2);
    activate(comm);
    if(rank < size) {
      int major=ceilquotient(size,x);
      int p=rank % major;
      int q=rank / major;
  
      /* Split nodes into row and columns */ 
      MPI_Comm_split(active,p,q,&communicator);
      MPI_Comm_split(active,q,p,&communicator2);
    }
  }
};

void show(Complex *f, unsigned int nx, unsigned int ny, const MPIgroup& group);
void show(Complex *f, unsigned int nx, unsigned int ny, unsigned int nz,
          const MPIgroup& group);
int hash(Complex *f, unsigned int nx, unsigned int ny, const MPIgroup& group);
int hash(Complex *f, unsigned int nx, unsigned int ny, unsigned int nz,
         const MPIgroup& group);

// Class to compute the local array splity and storage requirements for
// distributing the y index among multiple MPI processes and transposing.
//            local matrix is nx X   y
// local transposed matrix is  x X  ny
class splity {
public:
  unsigned int nx,ny;    // matrix splity
  unsigned int x,y;      // local splity
  unsigned int x0,y0;    // local starting values
  unsigned int n;     // total required storage (Complex words)
  MPI_Comm communicator;
  unsigned int block; // requested block size
  unsigned int M;     // number of Complex words per matrix element 
  splity() {}
  splity(unsigned int nx, unsigned int ny, MPI_Comm communicator,
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

  void show() {
    std::cout << "nx=" << nx << "\tny="<<ny << std::endl;
    std::cout << "x=" << x << "\ty="<<y << std::endl;
    std::cout << "x0=" << x0 << "\ty0="<< y0 << std::endl;
    std::cout << "n=" << n << std::endl;
    std::cout << "block=" << block << std::endl;
  }
};

// Class to compute the local array splity and storage requirements for
// distributing the x index among multiple MPI processes and transposing.
//            local matrix is  x X ny
// local transposed matrix is nx X  y
class splitx {
public:
  unsigned int nx,ny;    // matrix splity
  unsigned int x,y;      // local splity
  unsigned int x0,y0;    // local starting values
  unsigned int n;     // total required storage (Complex words)
  MPI_Comm communicator;
  unsigned int block; // requested block size
  unsigned int M;     // number of Complex words per matrix element
  splitx() {}
  splitx(unsigned int nx, unsigned int ny, MPI_Comm communicator,
         unsigned int Block=0, unsigned int M=1) 
    : nx(nx), ny(ny), communicator(communicator), block(Block), M(M) {
    if(block == 0) {
      int size;
      MPI_Comm_size(communicator,&size);
      block=ceilquotient(nx,size);
    }
    
    ptrdiff_t N[2]={nx,ny};
    ptrdiff_t local0,local1;
    ptrdiff_t start0,start1;
    n=fftw_mpi_local_size_many_transposed(2,N,2*M,block,0,
                                          communicator,&local0,
                                          &start0,&local1,&start1)*
      sizeof(double)/sizeof(Complex);
    x=local0;
    y=local1;
    x0=start0;
    y0=start1;
  }

  void show() {
    std::cout << "nx=" << nx << "\tny="<<ny << std::endl;
    std::cout << "x=" << x << "\ty="<<y << std::endl;
    std::cout << "x0=" << x0 << "\ty0="<< y0 << std::endl;
    std::cout << "n=" << n << std::endl;
    std::cout << "block=" << block << std::endl;
  }
};

// Distribute first over y, then over z.
//         local matrix is nx X   y  X z
// xy transposed matrix is  x X  ny  X z  allocated n  Complex words
// yz transposed matrix is  x X yz.x X nz allocated n2 Complex words [omit for slab]
class splityz {
public:
  unsigned int n,n2;
  unsigned int nx,ny,nz;
  unsigned int x,y,z;
  unsigned int x0,y0,z0;
  splity xy,yz;
  MPI_Comm communicator;
  unsigned int yblock,zblock; // requested block size
  MPI_Comm *XYplane;          // Used by HermitianSymmetrizeXYMPI
  int *reflect;               // Used by HermitianSymmetrizeXYMPI
  splityz() {}
  splityz(unsigned int nx, unsigned int ny, unsigned int nz,
              const MPIgroup& group, unsigned int Ny=0) : nx(nx), ny(ny), nz(nz),
                                       communicator(group.active),
                                       yblock(group.block),
                                       zblock(group.block2), XYplane(NULL) {
    if(Ny == 0) Ny=ny;
    xy=splity(nx,ny,group.communicator,yblock,zblock);
    yz=splity(Ny,nz,group.communicator2,zblock);
    x=xy.x;
    y=xy.y;
    z=yz.y;
    x0=xy.x0;
    y0=xy.y0;
    z0=yz.y0;
    n=xy.n=std::max(xy.n,(ceilquotient(xy.n,y)+1)*(yz.n+1));
    n2=yz.n;
  }
  
  void show() {
    std::cout << "nx=" << nx << "\tny="<<ny << "\tnz="<<nz << std::endl;
    std::cout << "x=" << x << "\ty="<<y << "\tz="<<z << std::endl;
    std::cout << "x0=" << x0 << "\ty0="<< y0 << "\tz0="<<z0 << std::endl;
    std::cout << "yz.x=" << yz.x << std::endl;
    std::cout << "n=" << n << std::endl;
    std::cout << "yblock=" << yblock << "\tzblock=" << zblock << std::endl;
  }
};
  
// Distribute first over x, then over y.
//         local matrix is  x X  y X nz
// yz transposed matrix is  x X ny X z allocated n2 Complex words [omit for slab]
// xy transposed matrix is nx X xy.y X z allocated n Complex words
class splitxy {
public:
  unsigned int n,n2;
  unsigned int nx,ny,nz;
  unsigned int x,y,z;
  unsigned int x0,y0,z0;
  splitx yz,xy;
  MPI_Comm communicator;
  unsigned int xblock,yblock; // requested block size
  splitxy() {}
  splitxy(unsigned int nx, unsigned int ny, unsigned int nz,
          const MPIgroup& group) : nx(nx), ny(ny), nz(nz),
                                       communicator(group.active),
                                       xblock(group.block),
                                       yblock(group.block2) {
    xy=splitx(nx,ny,group.communicator,xblock,yblock);
    yz=splitx(ny,nz,group.communicator2,yblock);
    x=xy.x;
    y=yz.x;
    z=yz.y;
    x0=xy.x0;
    y0=yz.x0;
    z0=yz.y0;
    n=xy.n=std::max(xy.n,xy.x*yz.n);
    n2=yz.n;
  }
  
  void show() {
    std::cout << "nx=" << nx << "\tny="<<ny << "\tnz="<<nz << std::endl;
    std::cout << "x=" << x << "\ty="<<y << "\tz="<<z << std::endl;
    std::cout << "x0=" << x0 << "\ty0="<< y0 << "\tz0="<<z0 << std::endl;
    std::cout << "xy.y=" << xy.y << std::endl;
    std::cout << "n=" << n << std::endl;
    std::cout << "xblock=" << yblock << "\tyblock=" << yblock << std::endl;
  }
};
  
// In-place OpenMP/MPI 2D complex FFT.
// Fourier transform an mx x my array, distributed first over x.
// The array must be allocated as splitx::n Complex words.
//
// Example:
// MPIgroup group(MPI_COMM_WORLD,mx);
// splitx d(mx,my,group.active,group.block);
// Complex *f=ComplexAlign(d.n);
// fft2dMPI fft(d,f);
// fft.Forwards(f);
// fft.Backwards(f);
// fft.Normalize(f);
// deleteAlign(f);

class fft2dMPI {
 private:
  splitx d;
  mfft1d *xForwards,*xBackwards;
  mfft1d *yForwards,*yBackwards;
  Complex *f;
  fftw_plan intranspose,outtranspose;
  bool tranfftwpp;
  mpitranspose<Complex> *T;
 public:
  void inittranspose(Complex* f) {
    int size;
    MPI_Comm_size(d.communicator,&size);
    tranfftwpp=T->divisible(size,d.nx,d.ny);

    if(tranfftwpp) {
      T=new mpitranspose<Complex>(d.nx,d.y,d.x,d.ny,1,f,d.communicator);
    } else {
      MPILoadWisdom(d.communicator);
      intranspose=
	fftw_mpi_plan_many_transpose(d.ny,d.nx,2,
				     0,d.block,
				     (double*) f,(double*) f,
				     d.communicator,
				     FFTW_MPI_TRANSPOSED_IN);
      if(!intranspose) transposeError("in");

      outtranspose=
	fftw_mpi_plan_many_transpose(d.ny,d.nx,2,
				     0,d.block,
				     (double*) f,(double*) f,
				     d.communicator,
				     FFTW_MPI_TRANSPOSED_OUT);
      if(!outtranspose) transposeError("out");

    }
  }
  
 fft2dMPI(const splitx& d, Complex *f) : d(d) {
    inittranspose(f);

    xForwards=new mfft1d(d.nx,-1,d.y,d.y,1,f,f); 
    xBackwards=new mfft1d(d.nx,1,d.y,d.y,1,f,f);
 
    yForwards=new mfft1d(d.ny,-1,d.x,1,d.ny,f,f);
    yBackwards=new mfft1d(d.ny,1,d.x,1,d.ny,f,f);
  }
  
  virtual ~fft2dMPI() {
    if(!tranfftwpp) {
      fftw_destroy_plan(intranspose);
      fftw_destroy_plan(outtranspose);
    }
  }

  void Forwards(Complex *f);
  void Backwards(Complex *f);
  void Normalize(Complex *f);
  void BackwardsNormalized(Complex *f);
};

// In-place OpenMP/MPI 3D complex FFT.
// Fourier transform an mx x my x mz array, distributed first over x and
// then over y. The array must be allocated as splitxy::n Complex words.
//
// Example:
// MPIgroup group(MPI_COMM_WORLD,mx,my);
// splitxy d(mx,my,mz,group);
// Complex *f=ComplexAlign(d.n);
// fft3dMPI fft(d,f);
// fft.Forwards(f);
// fft.Backwards(f);
// fft.Normalize(f);
// deleteAlign(f);
//
class fft3dMPI {
 private:
  splitxy d;
  mfft1d *xForwards,*xBackwards;
  mfft1d *yForwards,*yBackwards;
  mfft1d *zForwards,*zBackwards;
  fft2d *yzForwards,*yzBackwards;
  Complex *f;
  mpitranspose<Complex> *Txy,*Tyz;
 public:
  
 fft3dMPI(const splitxy& d, Complex *f) : d(d) {
    Txy=new mpitranspose<Complex>(d.nx,d.xy.y,d.x,d.ny,d.z,f,d.xy.communicator);
    
    xForwards=new mfft1d(d.nx,-1,d.xy.y*d.z,d.xy.y*d.z,1);
    xBackwards=new mfft1d(d.nx,1,d.xy.y*d.z,d.xy.y*d.z,1);

    if(d.y < d.ny) {
      Tyz=new mpitranspose<Complex>(d.ny,d.z,d.y,d.nz,1,f,d.yz.communicator);
      
      yForwards=new mfft1d(d.ny,-1,d.z,d.z,1);
      yBackwards=new mfft1d(d.ny,1,d.z,d.z,1);

      zForwards=new mfft1d(d.nz,-1,d.x*d.y,1,d.nz);
      zBackwards=new mfft1d(d.nz,1,d.x*d.y,1,d.nz);
    } else {
      yzForwards=new fft2d(d.ny,d.nz,-1,f);
      yzBackwards=new fft2d(d.ny,d.nz,1,f);
    }
  }
  
  virtual ~fft3dMPI() {}

  void Forwards(Complex *f);
  void Backwards(Complex *f);
  void Normalize(Complex *f);
};

#if 0
// rcfft2dMPI:
// Real-to-complex and complex-to-real in-place and out-of-place
// distributed FFTs.
//
// Basic interface:
// Forwards(double *f, Complex * g);
// Backwards(Complex *g, double *f);
// 
// Shift Fourier origin from (0,0) to (nx/2,0):
// Forwards0(double *f, Complex * g);
// Backwards0(Complex *g, double *f);
//
// Normalize:
// BackwardsNormalized(Complex *g, double *f);
// Backwards0Normalized(Complex *g, double *f);
class rcfft2dMPI {
 private:
  unsigned int mx, my;
  splity dr,dc;
  mfft1d *xForwards;
  mfft1d *xBackwards;
  mrcfft1d *yForwards;
  mcrfft1d *yBackwards;
  Complex *f;
  bool inplace;
  unsigned int rdist;
  bool tranfftwpp;
  fftw_plan intranspose,outtranspose;
 protected:
  mpitranspose<Complex> *T;
 public:
  void inittranspose(Complex* out) {
    int size;
    MPI_Comm_size(dc.communicator,&size);

    tranfftwpp=T->divisible(size,dc.nx,dc.ny);

    if(tranfftwpp) {
      T=new mpitranspose<Complex>(dc.nx,dc.y,dc.x,dc.ny,1,out,dc.communicator);
    } else {
      fftw_mpi_init();
      MPILoadWisdom(dc.communicator);
      intranspose=
      	fftw_mpi_plan_many_transpose(dc.nx,dc.ny,2,
      				     0,dc.block,
      				     (double*) out,(double*) out,
      				     dc.communicator,
      				     FFTW_MPI_TRANSPOSED_OUT);
      if(!intranspose) transposeError("in");

      outtranspose=
      	fftw_mpi_plan_many_transpose(dc.ny,dc.nx,2,
      				     dc.block,0,
      				     (double*) out,(double*) out,
      				     dc.communicator,
      				     FFTW_MPI_TRANSPOSED_IN);
      if(!outtranspose) transposeError("out");
    }
  }
  
 rcfft2dMPI(const splity& dr, const splity& dc,
	   double *f, Complex *g) : dr(dr), dc(dc), inplace((double*) g == f){
    mx=dr.nx;
    my=dc.ny;
    inittranspose(g);
    
    rdist=inplace ? dr.nx+2 : dr.nx;

    xForwards=new mfft1d(dc.nx,-1,dc.y,dc.y,1);
    xBackwards=new mfft1d(dc.nx,1,dc.y,dc.y,1);
    yForwards=new mrcfft1d(dr.ny,dr.x,1,rdist,f,g);
    yBackwards=new mcrfft1d(dr.ny,dc.x,1,dc.ny,g,f);
  }
  
  virtual ~rcfft2dMPI() {
    if(!tranfftwpp) {
      fftw_destroy_plan(intranspose);
      fftw_destroy_plan(outtranspose);
    }
  }

  void Forwards(double *f, Complex * g);
  void Forwards0(double *f, Complex * g);
  void Backwards(Complex *g, double *f);
  void Backwards0(Complex *g, double *f);
  void BackwardsNormalized(Complex *g, double *f);
  void Backwards0Normalized(Complex *g, double *f);
  void Shift(double *f);
  void Normalize(double *f);
  //  void BackwardsNormalized(Complex *f);
};
#endif

  
} // end namespace fftwpp

#endif
