#ifndef __mpifftwpp_h__
#define __mpifftwpp_h__ 1

#include "mpi/mpitranspose.h"
#include <fftw3-mpi.h>
#include <fftw++.h>

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
  

// In-place MPI/OpenMPI  2D complex FFT.
// FIXME: complete documentation.
class cfft2MPI : public ThreadBase {
 private:
  unsigned int mx, my;
  dimensions d;
  mfft1d *xForwards;
  mfft1d *xBackwards;
  mfft1d *yForwards;
  mfft1d *yBackwards;
  Complex *f;
 protected:
  fftw_plan intranspose,outtranspose;
 public:
  void inittranspose(Complex* f) {
    int size;
    MPI_Comm_size(d.communicator,&size);
    intranspose=
      fftw_mpi_plan_many_transpose(my,mx,2,
				   d.block,0,
				   (double*) f,(double*) f,
				   d.communicator,
				   FFTW_MPI_TRANSPOSED_OUT);
    if(!intranspose) transposeError("in");

    outtranspose=
      fftw_mpi_plan_many_transpose(mx,my,2,
				   0,d.block,
				   (double*) f,(double*) f,
				   d.communicator,
				   FFTW_MPI_TRANSPOSED_IN);
    if(!outtranspose) transposeError("out");
    SaveWisdom(d.communicator);

  }
  
  // threads is the number of threads to use in the outer subconvolution loop.
  // FIXME: is threads set correctly?
 cfft2MPI(const dimensions& d, Complex *f,
	  unsigned int threads=fftw::maxthreads) : d(d) {
    mx=d.nx;
    my=d.ny;
    inittranspose(f);

    xForwards=new mfft1d(d.nx,-1,d.y,d.y,1);
    xBackwards=new mfft1d(d.nx,1,d.y,d.y,1);
 
    yForwards=new mfft1d(d.ny,-1,d.x,1,d.ny);
    yBackwards=new mfft1d(d.ny,1,d.x,1,d.ny);
  }
  
  virtual ~cfft2MPI() {
    fftw_destroy_plan(intranspose);
    fftw_destroy_plan(outtranspose);
  }

  void Forwards(Complex *f, bool finaltranspose=true);
  void Backwards(Complex *f, bool finaltranspose=true);
  void Normalize(Complex *f);
  void BackwardsNormalized(Complex *f, bool finaltranspose=true);
};

// In-place MPI/OpenMPI 3D complex FFT.
// FIXME: complete documentation.
class cfft3MPI : public ThreadBase {
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
  fftw_plan xyintranspose, xyouttranspose;
  fftw_plan yzintranspose, yzouttranspose;
 public:
  void inittranspose(Complex* f) {
    int size;
    MPI_Comm_size(d.communicator,&size);
    xyintranspose=
      fftw_mpi_plan_many_transpose(d.nx,d.ny,2*d.z,d.yblock,0,
				   (double*) f,(double*) f,
				   d.xy.communicator,
				   FFTW_MPI_TRANSPOSED_OUT);
    if(!xyintranspose) transposeError("xyin");

    xyouttranspose=
      fftw_mpi_plan_many_transpose(d.ny,d.nx,2*d.z,d.yblock,0,
				   (double*) f,(double*) f,
				   d.xy.communicator,
				   FFTW_MPI_TRANSPOSED_IN);
    if(!xyouttranspose) transposeError("xyout");
    
    yzintranspose=
      fftw_mpi_plan_many_transpose(d.ny,d.nz,2*d.x,d.zblock,0,
				   (double*) f,(double*) f,
				   d.yz.communicator,
				   FFTW_MPI_TRANSPOSED_OUT);
    if(!yzintranspose) transposeError("yzin");

    yzouttranspose=
      fftw_mpi_plan_many_transpose(d.nz,d.ny,2*d.x,d.zblock,0,
				   (double*) f,(double*) f,
				   d.yz.communicator,
				   FFTW_MPI_TRANSPOSED_IN);
    if(!yzouttranspose) transposeError("yzout");

    // FIXME: xz tranpose?  Do I have to do both xy and yz?

    SaveWisdom(d.communicator);
  }
  
  // threads is the number of threads to use in the outer subconvolution loop.
  // FIXME: is threads set correctly?
 cfft3MPI(const dimensions3& d, Complex *f,
	  unsigned int threads=fftw::maxthreads) : d(d) {
    mx=d.nx;
    my=d.ny;
    mz=d.nz;
    inittranspose(f);
 
    xForwards=new mfft1d(d.nx,-1,d.y*d.z,d.y*d.z,1);
    xBackwards=new mfft1d(d.nx,1,d.y*d.z,d.y*d.z,1);

    yForwards=new mfft1d(d.ny,-1,d.x*d.z,d.x*d.z,1);
    yBackwards=new mfft1d(d.ny,1,d.x*d.z,d.x*d.z,1);

    zForwards=new mfft1d(d.nz,-1,d.x*d.yz.x,d.x*d.yz.x,1);
    zBackwards=new mfft1d(d.nz,1,d.x*d.yz.x,d.x*d.yz.x,1);
  }
  
  virtual ~cfft3MPI() {
    // FIXME
  }

  void Forwards(Complex *f, bool finaltranspose=true);
  void Backwards(Complex *f, bool finaltranspose=true);
  void Normalize(Complex *f);
};

  
} // end namespace fftwpp

#endif
