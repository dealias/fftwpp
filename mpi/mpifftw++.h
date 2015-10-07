#ifndef __mpifftwpp_h__
#define __mpifftwpp_h__ 1

#include "mpitranspose.h"
#include "fftw++.h"

namespace fftwpp {

fftw_plan MPIplanner(fftw *F, Complex *in, Complex *out);

extern MPI_Comm Active;

// Distribute first Y, then (if allowpencil=true) Z.
class MPIgroup {
public:  
  int rank,size;
  MPI_Comm active;                     // active communicator 
  MPI_Comm communicator,communicator2; // 3D transpose communicators
  
  void init(const MPI_Comm& comm) {
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);
  }
  
  void activate(const MPI_Comm& comm) {
    MPI_Comm_split(comm,rank < size,0,&active);
  }
  
  MPIgroup(const MPI_Comm& comm, unsigned int Y) {
    init(comm);
    unsigned int yblock=ceilquotient(Y,size);
    size=ceilquotient(Y,yblock);
    activate(comm);
  }
  
  MPIgroup(const MPI_Comm& comm, unsigned int X, unsigned int Y,
           unsigned int Z, bool allowPencil=true) {
    init(comm);
    unsigned int x=ceilquotient(X,size);
    unsigned int y=ceilquotient(Y,size);
    unsigned int z=allowPencil && X*y == x*Y ? ceilquotient(Z,size*y/Y) : Z;
    size=ceilquotient(Y,y)*ceilquotient(Z,z);
    
    activate(comm);
    if(rank < size) {
      int major=ceilquotient(size,Y);
      int p=rank % major;
      int q=rank / major;
  
      /* Split nodes into row and columns */ 
      MPI_Comm_split(active,p,q,&communicator);
      MPI_Comm_split(active,q,p,&communicator2);
    }
  }

};

// Class to compute the local array dimensions and storage requirements for
// distributing the Y dimension among multiple MPI processes and transposing.
// Big letters denote global dimensions; small letters denote local dimensions.
//            local matrix is X * y
// local transposed matrix is x * Y
class split {
public:
  unsigned int X,Y;     // global matrix dimensions
  unsigned int x,y;     // local matrix dimensions
  unsigned int x0,y0;   // local starting values
  unsigned int n;       // total required storage (words)
  MPI_Comm communicator;
  split() {}
  split(unsigned int X, unsigned int Y, MPI_Comm communicator)
    : X(X), Y(Y), communicator(communicator) {
    int size;
    int rank;
      
    MPI_Comm_rank(communicator,&rank);
    MPI_Comm_size(communicator,&size);
    
    x=localdimension(X,rank,size);
    y=localdimension(Y,rank,size);
    
    x0=localstart(X,rank,size);
    y0=localstart(Y,rank,size);
    n=std::max(X*y,x*Y);
  }

  int Activate() const {
    Active=communicator;
    fftw::planner=MPIplanner;
    return n;
  }

  void Deactivate() const {
    Active=MPI_COMM_NULL;
  }
  
  void show() {
    std::cout << "X=" << X << "\tY=" <<Y << std::endl;
    std::cout << "x=" << x << "\ty=" <<y << std::endl;
    std::cout << "x0=" << x0 << "\ty0=" << y0 << std::endl;
    std::cout << "n=" << n << std::endl;
  }
};

// Class to compute the local array dimensions and storage requirements for
// distributing X and Y among multiple MPI processes and transposing.
//         local matrix is x * y * Z
// yz transposed matrix is x * Y * z allocated n2 words [omit for slab]
// xy transposed matrix is X * xy.y * z allocated n words
//
// If spectral=true, for convenience rename xy.y to y and xy.y0 to y0.
class split3 {
public:
  unsigned int n;             // Total storage (words) for xy transpose
  unsigned int n2;            // Total storage (words) for yz transpose
  unsigned int X,Y,Z;         // Global dimensions
  unsigned int x,y,z;         // Local dimensions
  unsigned int x0,y0,z0;      // Local offsets
  split yz,xy;
  MPI_Comm communicator;
  MPI_Comm *XYplane;          // Used by HermitianSymmetrizeXYMPI
  int *reflect;               // Used by HermitianSymmetrizeXYMPI
  split3() {}
  split3(unsigned int X, unsigned int Y, unsigned int Z,
         const MPIgroup& group, bool spectral=false) : 
    X(X), Y(Y), Z(Z), communicator(group.active), XYplane(NULL) {
    xy=split(X,Y,group.communicator);
    yz=split(Y,Z,group.communicator2);
    x=xy.x;
    x0=xy.x0;
    if(spectral) {
      y=xy.y;
      y0=xy.y0;
    } else {
      y=yz.x;
      y0=yz.x0;
    }
    z=yz.y;
    z0=yz.y0;
    n2=yz.n;
    n=std::max(xy.n*z,x*n2);
  }
  
  int Activate() const {
    xy.Activate();
    return n;
  }

  void Deactivate() const {
    xy.Deactivate();
  }

  void show() {
    std::cout << "X=" << X << "\tY=" << Y << "\tZ=" << Z << std::endl;
    std::cout << "x=" << x << "\ty=" << y << "\tz=" << z << std::endl;
    std::cout << "x0=" << x0 << "\ty0=" << y0 << "\tz0=" << z0 << std::endl;
    std::cout << "xy.y=" << xy.y << "\txy.y0=" << xy.y0 << std::endl;
    std::cout << "yz.x=" << yz.x << "\tyz.x0=" << yz.x0 << std::endl;
    std::cout << "n=" << n << std::endl;
  }
};
  
// In-place OpenMP/MPI 2D complex FFT.
// Fourier transform an mx x my array, distributed first over x.
// The array must be allocated as split::n Complex words.
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
  unsigned int threads;
  mfft1d *xForwards,*xBackwards;
  mfft1d *yForwards,*yBackwards;
  bool tranfftwpp;
  mpitranspose<Complex> *T;
public:
  fft2dMPI(const split& d, Complex *f, const mpiOptions& options) : d(d) {
    d.Activate();
    threads=options.threads;

    T=new mpitranspose<Complex>(d.X,d.y,d.x,d.Y,1,f,d.communicator,options);
    
    unsigned int n=d.X;
    unsigned int M=d.y;
    unsigned int stride=d.y;
    unsigned int dist=1;
    xForwards=new mfft1d(n,-1,M,stride,dist,f,f,threads);
    xBackwards=new mfft1d(n,1,M,stride,dist,f,f,threads);

    n=d.Y;
    M=d.x;
    stride=1;
    dist=d.Y;
    yForwards=new mfft1d(n,-1,M,stride,dist,f,f,threads);
    yBackwards=new mfft1d(n,1,M,stride,dist,f,f,threads);
    d.Deactivate();
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
// then over y. The array must be allocated as split3::n Complex words.
//
// Example:
// MPIgroup group(MPI_COMM_WORLD,mx,my);
// split3 d(mx,my,mz,group);
// Complex *f=ComplexAlign(d.n);
// fft3dMPI fft(d,f);
// fft.Forwards(f);
// fft.Backwards(f);
// fft.Normalize(f);
// deleteAlign(f);
//
class fft3dMPI {
private:
  split3 d;
  unsigned int xythreads,yzthreads;
  mfft1d *xForwards,*xBackwards;
  mfft1d *yForwards,*yBackwards;
  mfft1d *zForwards,*zBackwards;
  fft2d *yzForwards,*yzBackwards;
  mpitranspose<Complex> *Txy,*Tyz;
public:
  
  void init(Complex *f, const mpiOptions& xy, const mpiOptions &yz) {
    d.Activate();
    xythreads=xy.threads;
    yzthreads=yz.threads; 
    if(d.z > 0)
      Txy=new mpitranspose<Complex>(d.X,d.xy.y,d.x,d.Y,d.z,f,d.xy.communicator,
                                    xy);
    unsigned int M=d.xy.y*d.z;

    xForwards=new mfft1d(d.X,-1,M,M,1,f,f,xythreads);
    xBackwards=new mfft1d(d.X,1,M,M,1,f,f,xythreads);
    
    if(d.y < d.Y) {
      Tyz=new mpitranspose<Complex>(d.Y,d.z,d.y,d.Z,1,f,d.yz.communicator,yz);
      
      yForwards=new mfft1d(d.Y,-1,d.z,d.z,1,f,f,yzthreads);
      yBackwards=new mfft1d(d.Y,1,d.z,d.z,1,f,f,yzthreads);

      unsigned int M=d.x*d.y;
      zForwards=new mfft1d(d.Z,-1,M,1,d.Z,f,f,yzthreads);
      zBackwards=new mfft1d(d.Z,1,M,1,d.Z,f,f,yzthreads);
    } else {
      yzForwards=new fft2d(d.Y,d.Z,-1,f,f,yzthreads);
      yzBackwards=new fft2d(d.Y,d.Z,1,f,f,yzthreads);
    }
    d.Deactivate();
  }
  
  fft3dMPI(const split3& d, Complex *f, const mpiOptions& xy,
           const mpiOptions& yz) : d(d) {
    init(f,xy,yz);
  }
  
  fft3dMPI(const split3& d, Complex *f, const mpiOptions& xy) : d(d) {
    init(f,xy,xy);
  }
    
  virtual ~fft3dMPI() {
    if(d.y < d.Y) {
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

    if(d.z > 0)
      delete Txy;
  }

  void Forwards(Complex *f);
  void Backwards(Complex *f);
  void Normalize(Complex *f);
  void BackwardsNormalized(Complex *f);
};

// rcfft2dMPI:
// Real-to-complex and complex-to-real in-place and out-of-place
// distributed FFTs.
//
// The input has size mx x my, distributed in the x-direction.
// The output has size mx x (my / 2 + 1), distributed in the y-direction. 
// Basic interface:
// Forwards(double *f, Complex * g);
// Backwards(Complex *g, double *f);
//
// TODO:
// Shift Fourier origin from (0,0) to (X/2,0):
// Forwards0(double *f, Complex * g);
// Backwards0(Complex *g, double *f);
//
// Normalize:
// BackwardsNormalized(Complex *g, double *f);
// Backwards0Normalized(Complex *g, double *f);
class rcfft2dMPI {
private:
  split dr,dc; // real and complex MPI dimensions.
  unsigned int threads;
  mfft1d *xForwards,*xBackwards;
  mrcfft1d *yForwards;
  mcrfft1d *yBackwards;
  bool inplace;
  //unsigned int rdist;
protected:
  mpitranspose<Complex> *T;
public:
  rcfft2dMPI(const split& dr, const split& dc,
             double *f, Complex *g, const mpiOptions& options) : dr(dr), dc(dc)
  {
    threads=options.threads;
    dc.Activate();
    
    T=new mpitranspose<Complex>(dc.X,dc.y,dc.x,dc.Y,1,g,dc.communicator,
                                options);

    bool inplace=f == (double*) g;
    if(inplace) {
      std::cerr << "In-place transform not yet implemented: TODO!" << std::endl;
      exit(1);
    }

    // Set up y-direction transforms
    {
      unsigned int n=dr.Y;
      unsigned int M=dr.x;
      ptrdiff_t rstride=1;
      ptrdiff_t cstride=1;
      ptrdiff_t rdist=inplace ? dr.Y+2 : dr.Y;
      ptrdiff_t cdist=dr.Y/2+1;
      yForwards=new mrcfft1d(n,M,rstride,cstride,rdist,cdist,f,g);
      yBackwards=new mcrfft1d(n,M,rstride,cstride,rdist,cdist,g,f);
    }

    // Set up x-direction transforms
    {    
      unsigned int n=dc.X;
      unsigned int M=dc.y;
      unsigned int stride=dc.y;
      unsigned int dist=1;
      xForwards=new mfft1d(n,-1,M,stride,dist,g,g,threads);
      xBackwards=new mfft1d(n,1,M,stride,dist,g,g,threads);
    }
    dc.Deactivate();
  }
   
  virtual ~rcfft2dMPI() {
    // FIXME
  }

  void Forwards(double *f, Complex * g);
  void Backwards(Complex *g, double *f);
  void Normalize(double *f);
  void BackwardsNormalized(Complex *g, double *f);

  // FIXME: implement shift!
  //void Shift(double *f);
};


  
} // end namespace fftwpp

#endif
