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
    if(rank == 0 && y < Y) std::cout << "Using pencil mode." << std::endl;
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
    n=std::max(xy.n*z,ceilquotient(Y,y)*n2);
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
    std::cout << "n2=" << n << " Y2=" << Y2 << std::endl;
  }
};
  
// 2D OpenMP/MPI complex in-place and out-of-place 
// xY -> Xy distributed FFT.
// Fourier transform an nx*ny array, distributed first over x.
// The array must be allocated as split::n Complex words.
//
// Example:
// MPIgroup group(MPI_COMM_WORLD,nx);
// split d(nx,ny,group.active);
// Complex *f=ComplexAlign(d.n);
// fft2dMPI fft(d,f);
// fft.Forward(f);
// fft.Backward(f);
// fft.Normalize(f);
// deleteAlign(f);

class fft2dMPI : public fftw {
  split d;
  mfft1d *xForward,*xBackward;
  mfft1d *yForward,*yBackward;
  mpitranspose<Complex> *T;
public:
  void init(Complex *in, Complex *out, const mpiOptions& options) {
    d.Activate();
    out=CheckAlign(in,out);
    inplace=(in == out);

    yForward=new mfft1d(d.Y,-1,d.x,1,d.Y,in,out,threads);
    yBackward=new mfft1d(d.Y,1,d.x,1,d.Y,out,out,threads);

    T=new mpitranspose<Complex>(d.X,d.y,d.x,d.Y,1,out,d.communicator,options);
    
    xForward=new mfft1d(d.X,-1,d.y,d.y,1,out,out,threads);
    xBackward=new mfft1d(d.X,1,d.y,d.y,1,in,out,threads);
    
    d.Deactivate();
  }
  
  fft2dMPI(const split& d, Complex *in,
           const mpiOptions& options=defaultmpiOptions) : 
    fftw(2*d.x*d.Y,0,options.threads,d.X*d.Y), d(d) {
    init(in,in,options);
  }
    
  fft2dMPI(const split& d, Complex *in, Complex *out,
           const mpiOptions& options=defaultmpiOptions) : 
    fftw(2*d.x*d.Y,0,options.threads,d.X*d.Y), d(d) {
    init(in,out,options);
  }
  
  virtual ~fft2dMPI() {
    delete yBackward;
    delete yForward;
    delete T;
    delete xBackward;
    delete xForward;
  }

  void Forward(Complex *in, Complex *out=NULL);
  void Backward(Complex *in, Complex *out=NULL);
};

// 3D OpenMP/MPI complex in-place and out-of-place 
// xyZ -> Xyz distributed FFT.
// Fourier transform an nx*ny*nz array, distributed first over x and
// then over y. The array must be allocated as split3::n Complex words.
//
// Example:
// MPIgroup group(MPI_COMM_WORLD,nx,ny,nz);
// split3 d(nx,ny,nz,group);
// Complex *f=ComplexAlign(d.n);
// fft3dMPI fft(d,f);
// fft.Forward(f);
// fft.Backward(f);
// fft.Normalize(f);
// deleteAlign(f);
//
class fft3dMPI : public fftw {
  split3 d;
  mfft1d *xForward,*xBackward;
  mfft1d *yForward,*yBackward;
  mfft1d *zForward,*zBackward;
  fft2d *yzForward,*yzBackward;
  mpitranspose<Complex> *Txy,*Tyz;
public:
  void init(Complex *in, Complex *out, const mpiOptions& xy,
            const mpiOptions &yz) {
    d.Activate();
    multithread(d.x);
    out=CheckAlign(in,out);
    inplace=(in == out);
    
    if(d.yz.x < d.Y) {
      unsigned int M=d.x*d.yz.x;
      zForward=new mfft1d(d.Z,-1,M,1,d.Z,in,out,threads);
      zBackward=new mfft1d(d.Z,1,M,1,d.Z,out,out,threads);
      Tyz=new mpitranspose<Complex>(d.Y,d.z,d.yz.x,d.Z,1,out,d.yz.communicator,
                                    yz,d.communicator);
      yForward=new mfft1d(d.Y,-1,d.z,d.z,1,in,out,innerthreads);
      yBackward=new mfft1d(d.Y,1,d.z,d.z,1,out,out,innerthreads);
    } else {
      yzForward=new fft2d(d.Y,d.Z,-1,in,out,innerthreads);
      yzBackward=new fft2d(d.Y,d.Z,1,in,out,innerthreads);
      Tyz=NULL;
    }
    
    Txy=d.z > 0 ?
      new mpitranspose<Complex>(d.X,d.xy.y,d.x,d.Y,d.z,out,d.xy.communicator,xy,
                                d.communicator) : NULL;
    unsigned int M=d.xy.y*d.z;
    xForward=new mfft1d(d.X,-1,M,M,1,out,out,threads);
    xBackward=new mfft1d(d.X,1,M,M,1,in,out,threads);
    
    d.Deactivate();
  }
  
  fft3dMPI(const split3& d, Complex *in, Complex *out, const mpiOptions& xy,
           const mpiOptions& yz) : fftw(2*d.x*d.y*d.Z,0,xy.threads,d.X*d.Y*d.Z),
                                   d(d) {init(in,out,xy,yz);}
  
  fft3dMPI(const split3& d, Complex *in, Complex *out,
           const mpiOptions& xy=defaultmpiOptions) :
    fftw(2*d.x*d.y*d.Z,0,xy.threads,d.X*d.Y*d.Z), d(d) {init(in,out,xy,xy);}
    
  fft3dMPI(const split3& d, Complex *in, const mpiOptions& xy,
           const mpiOptions& yz) : fftw(2*d.x*d.y*d.Z,0,xy.threads,d.X*d.Y*d.Z),
                                   d(d) {init(in,in,xy,yz);}
  
  fft3dMPI(const split3& d, Complex *in,
           const mpiOptions& xy=defaultmpiOptions) :
    fftw(2*d.x*d.y*d.Z,0,xy.threads,d.X*d.Y*d.Z), d(d) {init(in,in,xy,xy);}
    
  virtual ~fft3dMPI() {
    if(Tyz) {
      delete zBackward;
      delete zForward;
      delete yBackward;
      delete yForward;
      delete Tyz;
    } else {
      delete yzBackward;
      delete yzForward;
    }
    
    delete xBackward;
    delete xForward;

    if(Txy)
      delete Txy;
  }

  void Forward(Complex *in, Complex *out=NULL);
  void Backward(Complex *in, Complex *out=NULL);
};

// 2D OpenMP/MPI real-to-complex and complex-to-real in-place and out-of-place
// xY->Xy distributed FFT.
//
// The array in has size nx*ny, distributed in the x direction.
// The array out has size nx*(ny/2+1), distributed in the y direction. 
// The arrays must be allocated as split3::n words.
// The arrays in and out may coincide, dimensioned according to out.
//
// Basic interface:
// Forward(double *in, Complex *out=NULL);   // Fourier origin at (0,0)
// Forward0(double *in, Complex *out=NULL);  // Fourier origin at (nx/2,0);
//                                              input destroyed.
//
// Backward(Complex *in, double *out=NULL);  // Fourier origin at (0,0);
//                                              input destroyed.
// Backward0(Complex *in, double *out=NULL); // Fourier origin at (nx/2,0);
//                                              input destroyed.
//
// Normalize(Complex *out);
class rcfft2dMPI : public fftw {
  split dr,dc; // real and complex MPI dimensions.
  mfft1d *xForward,*xBackward;
  mrcfft1d *yForward;
  mcrfft1d *yBackward;
  mpitranspose<Complex> *T;
  unsigned int rdist;
public:
  
  void init(double *in, Complex *out, const mpiOptions& options) {
    dc.Activate();
    out=CheckAlign((Complex *) in,out);
    inplace=((Complex *) in == out);
    
    T=new mpitranspose<Complex>(dc.X,dc.y,dc.x,dc.Y,1,out,dc.communicator,
                                options);
    size_t cdist=dr.Y/2+1;
    yForward=new mrcfft1d(dr.Y,dr.x,1,1,rdist,cdist,in,out,threads);
    yBackward=new mcrfft1d(dr.Y,dr.x,1,1,cdist,rdist,out,in,threads);

    xForward=new mfft1d(dc.X,-1,dc.y,dc.y,1,out,out,threads);
    xBackward=new mfft1d(dc.X,1,dc.y,dc.y,1,out,out,threads);
    dc.Deactivate();
  }
   
  rcfft2dMPI(const split& dr, const split& dc, double *in, Complex *out,
             const mpiOptions& options=defaultmpiOptions) :
    fftw(dr.x*realsize(dr.Y,in,out),0,options.threads,dr.X*dr.Y), dr(dr),
    dc(dc), rdist(realsize(dr.Y,in,out)) {init(in,out,options);}
  
  rcfft2dMPI(const split& dr, const split& dc, Complex *out,
             const mpiOptions& options=defaultmpiOptions) :
    fftw(dr.x*2*(dr.Y/2+1),0,options.threads,dr.X*dr.Y), dr(dr), dc(dc),
    rdist(2*(dr.Y/2+1)) {init((double *) out,out,options);}
  
  virtual ~rcfft2dMPI() {
    delete xBackward;
    delete xForward;
    delete yBackward;
    delete yForward;
    delete T;
  }

  // Set Nyquist modes of even shifted transforms to zero.
  void deNyquist(Complex *f) {
    if(dr.X % 2 == 0)
      for(unsigned int j=0; j < dc.y; ++j)
        f[j]=0.0;
    
    if(dr.Y % 2 == 0 && dc.y0+dc.y == dc.Y) // Last process
      for(unsigned int i=0; i < dc.X; ++i)
        f[(i+1)*dc.y-1]=0.0;
  }
  
  void Shift(double *out);
  void Forward(double *in, Complex *out);
  void Forward0(double *in, Complex *out);
  void Backward(Complex *in, double *out=NULL);
  void Backward0(Complex *in, double *out=NULL);
  
  void Forward(Complex *out) {Forward((double *) out,out);}
};
  
// 3D real-to-complex and complex-to-real in-place and out-of-place
// xyZ -> Xyz distributed FFT.
//
// The input has size nx*ny*nz, distributed in the x and y directions.
// The output has size nx*ny*(nz/2+1), distributed in the y and z directions.
// The arrays must be allocated as split3::n Complex words.
//
// Example:
// MPIgroup group(MPI_COMM_WORLD,nx,ny);
// split3 d(nx,ny,nz,group);
// Complex *f=ComplexAlign(d.n);
// fft3dMPI fft(d,f);
// fft.Forward(f);
// fft.Backward(f);
// fft.Normalize(f);
// deleteAlign(f);
//
class rcfft3dMPI : public fftw {
  split3 dr,dc; // real and complex MPI dimensions
  mfft1d *xForward,*xBackward;
  mfft1d *yForward,*yBackward;
  mrcfft1d *zForward;
  mcrfft1d *zBackward;
  mpitranspose<Complex> *Txy,*Tyz;
  unsigned int rdist;
public:
  void init(double *in, Complex *out, const mpiOptions& xy,
            const mpiOptions &yz) {
    dc.Activate();
    multithread(dc.x);
    out=CheckAlign((Complex *) in,out);
    inplace=((Complex *) in == out);

    Txy=dc.z > 0 ? new mpitranspose<Complex>(dc.X,dc.xy.y,dc.x,dc.Y,dc.z,
                                             out,dc.xy.communicator,xy,
      dc.communicator) : NULL;

    Tyz=dc.yz.x < dc.Y ? new mpitranspose<Complex>(dc.Y,dc.z,dc.yz.x,dc.Z,1,
                                                   out,dc.yz.communicator,yz,
      dc.communicator) : NULL;
    
    unsigned int M=dr.x*dr.yz.x;
    size_t cdist=dr.Z/2+1;
    zForward=new mrcfft1d(dr.Z,M,1,1,rdist,cdist,in,out,threads);
    zBackward=new mcrfft1d(dr.Z,M,1,1,cdist,rdist,out,in,threads);

    yForward=new mfft1d(dc.Y,-1,dc.z,dc.z,1,out,out,innerthreads);
    yBackward=new mfft1d(dc.Y,1,dc.z,dc.z,1,out,out,innerthreads);

    M=dc.xy.y*dc.z;
    xForward=new mfft1d(dc.X,-1,M,M,1,out,out,threads);
    xBackward=new mfft1d(dc.X,1,M,M,1,out,out,threads);
        
    dc.Deactivate();
  }
  
  rcfft3dMPI(const split3& dr, const split3& dc, double *in, Complex *out,
             const mpiOptions& xy, const mpiOptions& yz) : 
    fftw(dr.x*dr.yz.x*realsize(dr.Z,in,out),0,xy.threads,dr.X*dr.Y*dr.Z),
    dr(dr), dc(dc), rdist(realsize(dr.Z,in,out)) {
    init(in,out,xy,yz);
  }
  
  rcfft3dMPI(const split3& dr, const split3& dc, double *in, Complex *out,
             const mpiOptions& xy=defaultmpiOptions) : 
    fftw(dr.x*dr.yz.x*realsize(dr.Z,in,out),0,xy.threads,dr.X*dr.Y*dr.Z),
    dr(dr), dc(dc), rdist(realsize(dr.Z,in,out)) {
    init(in,out,xy,xy);
  }
  
  rcfft3dMPI(const split3& dr, const split3& dc, Complex *out,
             const mpiOptions& xy, const mpiOptions& yz) : 
    fftw(dr.x*dr.yz.x*2*(dr.Z/2+1),0,xy.threads,dr.X*dr.Y*dr.Z),
    dr(dr), dc(dc), rdist(2*(dr.Z/2+1)) {
    init((double *) out,out,xy,yz);
  }
  
  rcfft3dMPI(const split3& dr, const split3& dc, Complex *out,
             const mpiOptions& xy=defaultmpiOptions) : 
    fftw(dr.x*dr.yz.x*2*(dr.Z/2+1),0,xy.threads,dr.X*dr.Y*dr.Z),
    dr(dr), dc(dc), rdist(2*(dr.Z/2+1)) {
    init((double *) out,out,xy,xy);
  }
  
  virtual ~rcfft3dMPI() {
    delete xBackward;
    delete xForward;
    delete yBackward;
    delete yForward;
    delete zBackward;
    delete zForward;
    
    if(Tyz) delete Tyz;
    if(Txy) delete Txy;
  }

  // Set Nyquist modes of even shifted transforms to zero.
  void deNyquist(Complex *f) {
    unsigned int yz=dc.xy.y*dc.z;
    if(dr.X % 2 == 0) {
      for(unsigned int k=0; k < yz; ++k)
        f[k]=0.0;
    }
    
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
  
  void Shift(double *out);
  void Forward(double *in, Complex *out);
  void Forward0(double *in, Complex *out);
  void Backward(Complex *in, double *out=NULL);
  void Backward0(Complex *in, double *out=NULL);
  
  void Forward(Complex *out) {Forward((double *) out,out);}
};

} // end namespace fftwpp

#endif
