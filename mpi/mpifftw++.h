#ifndef __mpifftwpp_h__
#define __mpifftwpp_h__ 1

#include "mpitranspose.h"
#include <fftw3-mpi.h>
#include "fftw++.h"

namespace fftwpp {

void MPILoadWisdom(const MPI_Comm& active);
void MPISaveWisdom(const MPI_Comm& active);

// Distribute first over x, then (if allowpencil=true), over y.
class MPIgroup {
public:  
  int rank,size;
  unsigned int block,block2; // TODO: Remove? ***********
  MPI_Comm active;                     // active communicator 
  MPI_Comm communicator,communicator2; // 3D transpose communicators
  
  void init(const MPI_Comm& comm) {
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);
  }
  
  void activate(const MPI_Comm& comm) {
    MPI_Comm_split(comm,rank < size,0,&active);
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
class split {
public:
  unsigned int nx,ny;   // matrix dimensions
  unsigned int x,y;     // local dimensions
  unsigned int x0,y0;   // local starting values
  unsigned int n;       // total required storage (Complex words)
  MPI_Comm communicator;
  unsigned int M;       // number of Complex words per matrix element 
  split() {}
  split(unsigned int nx, unsigned int ny, MPI_Comm communicator,
        unsigned int M=1) 
    : nx(nx), ny(ny), communicator(communicator), M(M) {
    int size;
    int rank;
      
    MPI_Comm_rank(communicator,&rank);
    MPI_Comm_size(communicator,&size);
    
    x=localdimension(nx,rank,size);
    y=localdimension(ny,rank,size);
  
    x0=localstart(nx,rank,size);
    y0=localstart(ny,rank,size);
    n=std::max(nx*y,x*ny);
  }

  void show() {
    std::cout << "nx=" << nx << "\tny="<<ny << std::endl;
    std::cout << "x=" << x << "\ty="<<y << std::endl;
    std::cout << "x0=" << x0 << "\ty0="<< y0 << std::endl;
    std::cout << "n=" << n << std::endl;
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
  split xy,yz;
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
    xy=split(nx,ny,group.communicator,zblock);
    yz=split(Ny,nz,group.communicator2);
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
  split yz,xy;
  MPI_Comm communicator;
  unsigned int xblock,yblock; // requested block size
  splitxy() {}
  splitxy(unsigned int nx, unsigned int ny, unsigned int nz,
          const MPIgroup& group) : nx(nx), ny(ny), nz(nz),
                                       communicator(group.active),
                                       xblock(group.block),
                                       yblock(group.block2) {
    xy=split(nx,ny,group.communicator,yblock);
    yz=split(ny,nz,group.communicator2);
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
// split d(mx,my,group.active);
// Complex *f=ComplexAlign(d.n);
// fft2dMPI fft(d,f);
// fft.Forwards(f);
// fft.Backwards(f);
// fft.Normalize(f);
// deleteAlign(f);

class fft2dMPI {
 private:
  split d;
  mfft1d *xForwards,*xBackwards;
  mfft1d *yForwards,*yBackwards;
  Complex *f;
  bool tranfftwpp;
  mpitranspose<Complex> *T;
 public:
  void inittranspose(Complex* f) {
    int size;
    MPI_Comm_size(d.communicator,&size);
    T=new mpitranspose<Complex>(d.nx,d.y,d.x,d.ny,1,f,d.communicator);
  }
  
 fft2dMPI(const split& d, Complex *f) : d(d) {
    inittranspose(f);

    xForwards=new mfft1d(d.nx,-1,d.y,d.y,1,f,f); 
    xBackwards=new mfft1d(d.nx,1,d.y,d.y,1,f,f);
 
    yForwards=new mfft1d(d.ny,-1,d.x,1,d.ny,f,f);
    yBackwards=new mfft1d(d.ny,1,d.x,1,d.ny,f,f);
  }
  
  virtual ~fft2dMPI() {
    delete yBackwards;
    delete yForwards;
    delete xBackwards;
    delete xForwards;
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
  
  virtual ~fft3dMPI() {
    if(d.y < d.ny) {
      delete zBackwards;
      delete zForwards;
      delete yBackwards;
      delete yForwards;
      delete Tyz;
    } else {
      delete yzBackwards;
      delete yzForwards;
    }
    
    delete xBackwards;
    delete xForwards;
    delete Txy;
  }

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
 protected:
  mpitranspose<Complex> *T;
 public:
  void inittranspose(Complex* out) {
    int size;
    MPI_Comm_size(dc.communicator,&size);

    T=new mpitranspose<Complex>(dc.nx,dc.y,dc.x,dc.ny,1,out,dc.communicator);
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
  
  virtual ~rcfft2dMPI() {}

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
