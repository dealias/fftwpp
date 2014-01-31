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
    fftw::mpi=true;
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

  void show() {
    std::cout << "nx=" << nx << "\tny="<<ny << std::endl;
    std::cout << "x=" << x << "\ty="<<y << std::endl;
    std::cout << "x0=" << x0 << "\ty0="<< y0 << std::endl;
    std::cout << "n=" << n << std::endl;
    std::cout << "block=" << block << std::endl;
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

  void show() {
    std::cout << "nx=" << nx << "\tny="<<ny << "\tnz="<<nz << std::endl;
    std::cout << "x=" << x << "\ty="<<y << "\tz="<<z << std::endl;
    std::cout << "x0=" << x0 << "\ty0="<< y0 << "\tz0="<<z0 << std::endl;
    std::cout << "yz.x=" << yz.x << std::endl;
    std::cout << "n=" << n << std::endl;
    std::cout << "yblock=" << yblock << "\tzblock=" << zblock << std::endl;
  }
};
  
// In-place OpenMP/MPI 2D complex FFT.
// The input array of size mx,my,mz is Fourier-transformed.
// The output may be left transposed or returned to the original
// orientation.
// Uses dimensions class for array (and transposed
// array sizes).  Data must be allocated to size dimenions3::n.
//
// Example:
// dimensions d(mx,my,group.active,group.yblock);
// Complex *f=ComplexAlign(d.n);
// cfft2dMPI fft(d,f);
// fft.Forwards(f,true);
// fft.Backwards(f,true);
// fft.Normalize(f);
// 
// If a transposed output is desired, call with fft.Forwards(f,false).
class cfft2dMPI {
 private:
  unsigned int mx, my;
  dimensions d;
  mfft1d *xForwards;
  mfft1d *xBackwards;
  mfft1d *yForwards;
  mfft1d *yBackwards;
  Complex *f;
  fftw_plan intranspose,outtranspose;
  bool tranfftwpp;
 protected:
  mpitranspose<Complex> *T;
 public:
  void inittranspose(Complex* f) {
    int size;
    MPI_Comm_size(d.communicator,&size);
    tranfftwpp=T->divisible(size,d.nx,d.ny);

    if(tranfftwpp) {
      T=new mpitranspose<Complex>(d.nx,d.y,d.x,d.ny,1,f,d.communicator);
    } else {
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
  
 cfft2dMPI(const dimensions& d, Complex *f) : d(d) {
    mx=d.nx;
    my=d.ny;

    inittranspose(f);

    xForwards=new mfft1d(d.nx,-1,d.y,d.y,1,f,f); 
    xBackwards=new mfft1d(d.nx,1,d.y,d.y,1,f,f);
 
    yForwards=new mfft1d(d.ny,-1,d.x,1,d.ny,f,f);
    yBackwards=new mfft1d(d.ny,1,d.x,1,d.ny,f,f);
  }
  
  virtual ~cfft2dMPI() {
    if(!tranfftwpp) {
      fftw_destroy_plan(intranspose);
      fftw_destroy_plan(outtranspose);
    }
  }

  void Forwards(Complex *f, bool finaltranspose=true);
  void Backwards(Complex *f, bool finaltranspose=true);
  void Normalize(Complex *f);
  void BackwardsNormalized(Complex *f, bool finaltranspose=true);
};

// In-place OpenMP/MPI 3D complex FFT.
// The input array of size mx,my,mz is Fourier-transformed with 
// the output tranposed.  Uses dimensions3 class for array (and transposed
// array sizes).  Data must be allocated to size dimenions3::n.
//
// Example:
// dimensions3 d(mx,my,my,mz,group);
// Complex *f=ComplexAlign(d.n);
// cfft3dMPI fft(d,f);
// fft.Forwards(f,dofinaltranspose);
// fft.Backwards(f,dofinaltranspose);
// fft.Normalize(f);
//
// TODO: allow for non-transposed output.
class cfft3dMPI {
 private:
  unsigned int mx, my, mz;
  dimensions3 d;
  mfft1d *xForwards;
  mfft1d *xBackwards;
  mfft1d *yForwards;
  mfft1d *yBackwards;
  mfft1d *zForwards;
  mfft1d *zBackwards;
  Complex *f;
 protected:
  mpitranspose<Complex> *Txy, *Tyz;
 public:
  void inittranspose(Complex* f) {
    int size;
    MPI_Comm_size(d.communicator,&size);
    Txy=new mpitranspose<Complex>(d.nx,d.y,d.x,d.ny,d.z,f,d.xy.communicator);
    Tyz=new mpitranspose<Complex>(d.ny,d.z,d.yz.x,d.nz,d.x,f,d.yz.communicator);
    // FIXME: xz tranpose?
    MPISaveWisdom(d.communicator);
  }
  
 cfft3dMPI(const dimensions3& d, Complex *f) : d(d) {
    mx=d.nx;
    my=d.ny;
    mz=d.nz;
    inittranspose(f);
 
    xForwards=new mfft1d(d.nx,-1,
			 d.y*d.z, // M=howmany
			 d.y*d.z, // stride
			 1); // dist
    xBackwards=new mfft1d(d.nx,1,d.y*d.z,d.y*d.z,1);
    yForwards=new mfft1d(d.ny,-1,
			 d.z, // M=howmany
			 d.z, // stride
			 1); // dist
    yBackwards=new mfft1d(d.ny,1,d.z,d.z,1);

    zForwards=new mfft1d(d.nz,-1,
			 d.x*d.yz.x, // M=howmany
			 1, // stride
			 d.nz); // dist
    zBackwards=new mfft1d(d.nz,1,d.x*d.yz.x,1,d.nz);
  }
  
  virtual ~cfft3dMPI() {}

  void Forwards(Complex *f, bool finaltranspose=true);
  void Backwards(Complex *f, bool finaltranspose=true);
  void Normalize(Complex *f);
};

// rcfft2dMPI:
// Real-to-complex and complex-to-real in-place and out-of-place
// distributed FFTs.
//
// Basic interface:
// Forwards(double *f, Complex * g, bool finaltranspose=true);
// Backwards(Complex *g, double *f, bool finaltranspose=true);
// 
// Shift Fourier origin from (0,0) to (nx/2,0):
// Forwards0(double *f, Complex * g, bool finaltranspose=true);
// Backwards0(Complex *g, double *f, bool finaltranspose=true);
//
// Normalize:
// BackwardsNormalized(Complex *g, double *f, bool finaltranspose=true);
// Backwards0Normalized(Complex *g, double *f, bool finaltranspose=true);
class rcfft2dMPI {
 private:
  unsigned int mx, my;
  dimensions dr,dc;
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
    std::cout <<  "tranfftwpp: " << tranfftwpp << std::endl;
    if(tranfftwpp) {
      T=new mpitranspose<Complex>(dc.nx,dc.y,dc.x,dc.ny,1,out,dc.communicator);
    } else {
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

    MPISaveWisdom(dc.communicator);
  }
  
 rcfft2dMPI(const dimensions& dr, const dimensions& dc,
	   double *f, Complex *g) : dr(dr), dc(dc), inplace((double*) g == f){
    mx=dr.nx;
    my=dc.ny;
    inittranspose(g);
    
    rdist=inplace ? dr.nx+2 : dr.nx;

    xForwards=new mfft1d(dc.nx,-1,dc.y,dc.y,1);
    xBackwards=new mfft1d(dc.nx,1,dc.y,dc.y,1);
    yForwards=new mrcfft1d(dr.ny, // length
			   dr.x, // howmany
			   1, // stride
			   rdist, // dist between the first elements of inputs
			   f, // input
			   g); // output
    yBackwards=new mcrfft1d(dr.ny, // length or real output
			   dc.x, // howmany
			   1, // stride
			   dc.ny,// dist between the first elements of inputs
			   g, // input
			   f); // output
  }
  
  virtual ~rcfft2dMPI() {
    if(!tranfftwpp) {
      fftw_destroy_plan(intranspose);
      fftw_destroy_plan(outtranspose);
    }
  }

  void Forwards(double *f, Complex * g, bool finaltranspose=true);
  void Forwards0(double *f, Complex * g, bool finaltranspose=true);
  void Backwards(Complex *g, double *f, bool finaltranspose=true);
  void Backwards0(Complex *g, double *f, bool finaltranspose=true);
  void BackwardsNormalized(Complex *g, double *f, bool finaltranspose=true);
  void Backwards0Normalized(Complex *g, double *f, bool finaltranspose=true);
  void Shift(double *f);
  void Normalize(double *f);
  //  void BackwardsNormalized(Complex *f, bool finaltranspose=true);
};


  
} // end namespace fftwpp

#endif
