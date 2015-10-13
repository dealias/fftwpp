#ifndef __mpifftwpp_h__
#define __mpifftwpp_h__ 1

#include "mpitranspose.h"
#include "fftw++.h"

namespace fftwpp {

fftw_plan MPIplanner(fftw *F, Complex *in, Complex *out);

extern MPI_Comm Active;

// Distribute first X, then (if allowpencil=true) Y.
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
  
  MPIgroup(const MPI_Comm& comm, unsigned int X) {
    init(comm);
    unsigned int xblock=ceilquotient(X,size);
    size=ceilquotient(X,xblock);
    activate(comm);
  }
  
  MPIgroup(const MPI_Comm& comm, unsigned int X, unsigned int Y,
           unsigned int Z, bool allowPencil=true) {
    init(comm);
    unsigned int x=ceilquotient(X,size);
    unsigned int z=ceilquotient(Z,size);
    unsigned int y=allowPencil && x*Z == X*z ? ceilquotient(Y,size*x/X) : Y;
    size=ceilquotient(X,x)*ceilquotient(Y,y);
    
    activate(comm);
    if(rank < size) {
      int major=ceilquotient(size,X);
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
  unsigned int X,Y,Y2,Z;      // Global dimensions
  unsigned int x,y,z;         // Local dimensions
  unsigned int x0,y0,z0;      // Local offsets
  split yz,xy;
  MPI_Comm communicator;
  MPI_Comm *XYplane;          // Used by HermitianSymmetrizeXYMPI
  int *reflect;               // Used by HermitianSymmetrizeXYMPI
  split3() {}
  void init(const MPIgroup& group, bool spectral) {
    xy=split(X,Y,group.communicator);
    yz=split(Y2,Z,group.communicator2);
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
    n=std::max(xy.n*z,group.size*n2);
    show();
  }
  
  split3(unsigned int X, unsigned int Y, unsigned int Z,
         const MPIgroup& group, bool spectral=false) :
    X(X), Y(Y), Y2(Y), Z(Z), communicator(group.active), XYplane(NULL) {
    init(group,spectral);
  }
    
  split3(unsigned int X, unsigned int Y, unsigned int Y2, unsigned int Z,
         const MPIgroup& group, bool spectral=false) : 
    X(X), Y(Y), Y2(Y2), Z(Z), communicator(group.active), XYplane(NULL) {
    init(group,spectral);
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
    std::cout << "n2=" << n << std::endl;
  }
};
  
// 2D OpenMP/MPI complex in-place and out-of-place 
// xY -> Xy distributed FFT.
// Fourier transform an nx x ny array, distributed first over x.
// The array must be allocated as split::n Complex words.
//
// Example:
// MPIgroup group(MPI_COMM_WORLD,nx);
// split d(nx,ny,group.active);
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

// 3D OpenMP/MPI complex in-place and out-of-place 
// xyZ -> Xyz distributed FFT.
// Fourier transform an nx x ny x nz array, distributed first over x and
// then over y. The array must be allocated as split3::n Complex words.
//
// Example:
// MPIgroup group(MPI_COMM_WORLD,nx,ny,nz);
// split3 d(nx,ny,nz,group);
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
    Txy=d.z > 0 ? 
      new mpitranspose<Complex>(d.X,d.xy.y,d.x,d.Y,d.z,f,d.xy.communicator,xy,
                                d.communicator) :
      NULL;

    // Set up x-direction transforms
    {
      unsigned int n=d.X;
      unsigned int M=d.xy.y*d.z;
      size_t stride=M;
      size_t dist=1;
      xForwards=new mfft1d(n,-1,M,stride,dist,f,f,xythreads);
      xBackwards=new mfft1d(n,1,M,stride,dist,f,f,xythreads);
    }
    
    if(d.yz.x < d.Y) {
      Tyz=new mpitranspose<Complex>(d.Y,d.z,d.yz.x,d.Z,1,f,d.yz.communicator,yz,
        d.communicator);

      // Set up y-direction transforms
      {
        unsigned int n=d.Y;
        unsigned int M=d.z;
        size_t stride=d.z;
        size_t dist=1;
        yForwards=new mfft1d(n,-1,M,stride,dist,f,f,yzthreads);
        yBackwards=new mfft1d(n,1,M,stride,dist,f,f,yzthreads);
      }

      // Set up z-direction transforms
      {
        unsigned int n=d.Z;
        unsigned int M=d.x*d.yz.x;
        size_t stride=1;
        size_t dist=d.Z;
        zForwards=new mfft1d(n,-1,M,stride,dist,f,f,yzthreads);
        zBackwards=new mfft1d(n,1,M,stride,dist,f,f,yzthreads);
      }
      
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
    if(d.yz.x < d.Y) {
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

    if(Txy)
      delete Txy;
  }

  void Forwards(Complex *f);
  void Backwards(Complex *f);
  void Normalize(Complex *f);
  void BackwardsNormalized(Complex *f);
};

// 2D OpenMP/MPI real-to-complex and complex-to-real in-place and out-of-place
// xY->Xy distributed FFT.
//
// The input has size nx x ny, distributed in the x direction.
// The output has size nx x (ny/2+1), distributed in the y direction. 
// The arrays must be allocated as split3::n Complex words.
//
// Basic interface:
// Forwards(double *f, Complex * g);
// Backwards(Complex *g, double *f);
//
// Forwards0(double *f, Complex * g); // Fourier origin at (nx/2,0)
// Backwards0(Complex *g, double *f); // Fourier origin at (nx/2,0)
//
// Normalize:
// BackwardsNormalized(Complex *g, double *f);
// Backwards0Normalized(Complex *g, double *f); // Fourier origin at (nx/2,0)
class rcfft2dMPI {
private:
  split dr,dc; // real and complex MPI dimensions.
  unsigned int threads;
  mfft1d *xForwards,*xBackwards;
  mrcfft1d *yForwards;
  mcrfft1d *yBackwards;
  bool inplace;
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
      size_t rstride=1;
      size_t cstride=1;
      size_t rdist=inplace ? dr.Y+2 : dr.Y;
      size_t cdist=dr.Y/2+1;
      yForwards=new mrcfft1d(n,M,rstride,cstride,rdist,cdist,f,g,threads);
      yBackwards=new mcrfft1d(n,M,cstride,rstride,cdist,rdist,g,f,threads);
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
    delete xBackwards;
    delete xForwards;
    delete yBackwards;
    delete yForwards;
    delete T;
  }

  // Remove the Nyquist mode for even transforms.
  void deNyquist(Complex *f) {
    if(dr.X % 2 == 0)
      for(unsigned int j=0; j < dc.y; ++j)
        f[j]=0.0;
    
    if(dr.Y % 2 == 0 && dc.y0+dc.y == dc.Y) // Last process
      for(unsigned int i=0; i < dc.X; ++i)
        f[(i+1)*dc.y-1]=0.0;
  }
  
  void Normalize(double *f);
  void Shift(double *f);
  void Forwards(double *f, Complex * g);
  void Backwards(Complex *g, double *f);
  void BackwardsNormalized(Complex *g, double *f);
  void Forwards0(double *f, Complex * g);
  void Backwards0(Complex *g, double *f);
  void Backwards0Normalized(Complex *g, double *f);
};
  
// 3D real-to-complex and complex-to-real in-place and out-of-place
// xyZ -> Xyz distributed FFT.
//
// The input has size nx x ny x nz, distributed in the x and y directions.
// The output has size nx x ny x (nz/2+1), distributed in the y and z directions.
// The arrays must be allocated as split3::n Complex words.
//
// Example:
// MPIgroup group(MPI_COMM_WORLD,nx,ny);
// split3 d(nx,ny,nz,group);
// Complex *f=ComplexAlign(d.n);
// fft3dMPI fft(d,f);
// fft.Forwards(f);
// fft.Backwards(f);
// fft.Normalize(f);
// deleteAlign(f);
//
class rcfft3dMPI {
private:
  split3 dr; // real-space dimensions
  split3 dc; // complex-space dimensions
  unsigned int xythreads,yzthreads;
  mfft1d *xForwards,*xBackwards;
  mfft1d *yForwards,*yBackwards;
  mrcfft1d *zForwards;
  mcrfft1d *zBackwards;
  mpitranspose<Complex> *Txy,*Tyz;
  bool inplace;
  
public:
  void init(double *f, Complex *g,
            const mpiOptions& xy, const mpiOptions &yz) {
    inplace=f == (double *)g;

    dc.Activate();
    xythreads=xy.threads;
    yzthreads=yz.threads; 

    Txy=dc.z > 0 ? new mpitranspose<Complex>(dc.X,dc.xy.y,dc.x,dc.Y,dc.z,
                                             g,dc.xy.communicator,xy,
      dc.communicator) : NULL;

    Tyz=dc.yz.x < dc.Y ? new mpitranspose<Complex>(dc.Y,dc.z,dc.yz.x,dc.Z,1,
                                                   g,dc.yz.communicator,yz,
      dc.communicator) : NULL;
    
    // Set up z-direction transforms
    {
      unsigned int n=dr.Z;
      unsigned int M=dr.x*dr.yz.x;
      size_t rstride=1;
      size_t cstride=1;
      size_t rdist=inplace ? dr.Z+2 : dr.Z;
      size_t cdist=dr.Z/2+1;
      zForwards=new mrcfft1d(n,M,rstride,cstride,rdist,cdist,f,g,xythreads);
      zBackwards=new mcrfft1d(n,M,cstride,rstride,cdist,rdist,g,f,xythreads);
    }

    // Set up y-direction transforms
    {
      unsigned int n=dc.Y;
      unsigned int M=dc.z;
      size_t stride=dc.z;
      size_t dist=1;
      yForwards=new mfft1d(n,-1,M,stride,dist,g,g,yzthreads);
      yBackwards=new mfft1d(n,1,M,stride,dist,g,g,yzthreads);
    }

    // Set up x-direction transforms
    {
      unsigned int n=dc.X;
      unsigned int M=dc.xy.y*dc.z;
      size_t stride=M;
      size_t dist=1;
      xForwards=new mfft1d(n,-1,M,stride,dist,g,g,xythreads);
      xBackwards=new mfft1d(n,1,M,stride,dist,g,g,xythreads);
    }
        
    dc.Deactivate();
  }
  
  rcfft3dMPI(const split3& dr, const split3& dc,
             double *f, Complex *g, const mpiOptions& xy,
             const mpiOptions& yz) : dr(dr) , dc(dc) {
    init(f,g,xy,yz);
  }
  
  rcfft3dMPI(const split3& dr, const split3& dc,
             double *f, Complex *g, const mpiOptions& xy) :
    dr(dr), dc(dc) {
    init(f,g,xy,xy);
  }
    
  virtual ~rcfft3dMPI() {
    delete xBackwards;
    delete xForwards;
    delete yBackwards;
    delete yForwards;
    delete zBackwards;
    delete zForwards;
    
    if(Tyz) delete Tyz;
    if(Txy) delete Txy;
  }

  // Remove the Nyquist mode for even transforms.
  void deNyquist(Complex *f) {
    if(dr.X % 2 == 0) {
      unsigned int stop=dc.xy.y*dc.z;
      for(unsigned int k=0; k < stop; ++k)
        f[k]=0.0;
    }
    unsigned int yz=dc.xy.y*dc.z;
    
    if(dr.Y % 2 == 0 && dc.xy.y0 == 0) {
      for(unsigned int i=0; i < dc.X; ++i) {
        unsigned int iyz=i*yz;
        for(unsigned int k=0; k < dc.z; ++k)
          f[iyz+k]=0.0;
      }
    }
        
    if(dr.Z % 2 == 0 && dc.z0+dc.z == dc.Z) // Last process
      for(unsigned int i=0; i < dc.X; ++i)
        for(unsigned int j=0; j < dc.xy.y; ++j)
          f[i*yz+(j+1)*dc.z-1]=0.0;
  }
  
  void Forwards(double *f, Complex *g);
  void Backwards(Complex *g, double* f);
  void Normalize(double *f);
  void BackwardsNormalized(Complex *g, double* f);
  void Shift(double *f);
  void Forwards0(double *f, Complex *g);
  void Backwards0(Complex *g, double* f);
  void Backwards0Normalized(Complex *g, double* f);

};

} // end namespace fftwpp

#endif
