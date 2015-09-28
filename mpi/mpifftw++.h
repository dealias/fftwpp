#ifndef __mpifftwpp_h__
#define __mpifftwpp_h__ 1

#include "mpitranspose.h"
#include "fftw++.h"

namespace fftwpp {

void MPIInitWisdom(const MPI_Comm& active);

// Distribute first y, then (if allowpencil=true) z.
class MPIgroup {
public:  
  int rank,size;
  unsigned int z;
  MPI_Comm active;                     // active communicator 
  MPI_Comm communicator,communicator2; // 3D transpose communicators
  
  void init(const MPI_Comm& comm) {
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);
  }
  
  void activate(const MPI_Comm& comm) {
    MPI_Comm_split(comm,rank < size,0,&active);
    MPIInitWisdom(active);
  }
  
  MPIgroup(const MPI_Comm& comm, unsigned int Y) {
    init(comm);
    unsigned int yblock=ceilquotient(Y,size);
    size=ceilquotient(Y,yblock);
    activate(comm);
  }
  
  MPIgroup(const MPI_Comm& comm, unsigned int X, unsigned int Y, unsigned int Z,
           bool allowPencil=true) {
    init(comm);
    unsigned int x=ceilquotient(X,size);
    unsigned int y=ceilquotient(Y,size);
    z=allowPencil && X*y == x*Y ? ceilquotient(Z,size*y/Y) : Z;
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
// distributing the y index among multiple MPI processes and transposing.
//            local matrix is X * y
// local transposed matrix is x * Y
class split {
public:
  unsigned int X,Y;     // global matrix dimensions
  unsigned int x,y;     // local matrix dimensions
  unsigned int x0,y0;   // local starting values
  unsigned int n;       // total required storage (Complex words)
  MPI_Comm communicator;
  unsigned int Z;       // number of Complex words per matrix element 
  split() {}
  split(unsigned int X, unsigned int Y, MPI_Comm communicator,
        unsigned int Z=1) 
    : X(X), Y(Y), communicator(communicator), Z(Z) {
    int size;
    int rank;
      
    MPI_Comm_rank(communicator,&rank);
    MPI_Comm_size(communicator,&size);
    
    x=localdimension(X,rank,size);
    y=localdimension(Y,rank,size);
    
    x0=localstart(X,rank,size);
    y0=localstart(Y,rank,size);
    n=std::max(X*y,x*Y)*Z;
  }

  void show() {
    std::cout << "X=" << X << "\tY="<<Y << std::endl;
    std::cout << "x=" << x << "\ty="<<y << std::endl;
    std::cout << "x0=" << x0 << "\ty0="<< y0 << std::endl;
    std::cout << "n=" << n << std::endl;
  }
};

// Distribute first over y, then over z.
//         local matrix is X * y * z
// xy transposed matrix is x * Y * z  allocated n  Complex words
// yz transposed matrix is x * yz.x * Z allocated n2 Complex words [omit for slab]
class splityz {
public:
  unsigned int n,n2;
  unsigned int X,Y,Z;
  unsigned int x,y,z;
  unsigned int x0,y0,z0;
  split xy,yz;
  MPI_Comm communicator;
  MPI_Comm *XYplane;          // Used by HermitianSymmetrizeXYMPI
  int *reflect;               // Used by HermitianSymmetrizeXYMPI
  splityz() {}
  splityz(unsigned int X, unsigned int Y, unsigned int Z,
          const MPIgroup& group, unsigned int Y2=0) : 
    X(X), Y(Y), Z(Z), communicator(group.active),
    XYplane(NULL) {
    if(Y2 == 0) Y2=Y;
    xy=split(X,Y,group.communicator,group.z);
    yz=split(Y2,Z,group.communicator2);
    x=xy.x;
    y=xy.y;
    z=yz.y;
    x0=xy.x0;
    y0=xy.y0;
    z0=yz.y0;
    n2=yz.n;
    n=std::max(xy.n,x*n2);
  }
  
  void show() {
    std::cout << "X=" << X << "\tY="<<Y << "\tZ="<<Z << std::endl;
    std::cout << "x=" << x << "\ty="<<y << "\tz="<<z << std::endl;
    std::cout << "x0=" << x0 << "\ty0="<< y0 << "\tz0="<<z0 << std::endl;
    std::cout << "yz.x=" << yz.x << std::endl;
    std::cout << "n=" << n << "\tn2=" << n2 << std::endl;
  }
};
  
// Distribute first over x, then over y.
//         local matrix is x * y * Z
// yz transposed matrix is x * Y * z allocated n2 Complex words [omit for slab]
// xy transposed matrix is X * xy.y * z allocated n Complex words
class splitxy {
public:
  unsigned int n,n2;
  unsigned int X,Y,Z;
  unsigned int x,y,z;
  unsigned int x0,y0,z0;
  split yz,xy;
  MPI_Comm communicator;
  splitxy() {}
  splitxy(unsigned int X, unsigned int Y, unsigned int Z,
          const MPIgroup& group) : X(X), Y(Y), Z(Z),
				   communicator(group.active) {
    xy=split(X,Y,group.communicator,group.z);
    yz=split(Y,Z,group.communicator2);
    x=xy.x;
    y=yz.x;
    z=yz.y;
    x0=xy.x0;
    y0=yz.x0;
    z0=yz.y0;
    n2=yz.n;
    n=std::max(xy.n,x*n2);
  }
  
  void show() {
    std::cout << "X=" << X << "\tY="<<Y << "\tZ="<<Z << std::endl;
    std::cout << "x=" << x << "\ty="<<y << "\tz="<<z << std::endl;
    std::cout << "x0=" << x0 << "\ty0="<< y0 << "\tz0="<<z0 << std::endl;
    std::cout << "xy.y=" << xy.y << std::endl;
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
  mfft1d *xForwards,*xBackwards;
  mfft1d *yForwards,*yBackwards;
  Complex *f;
  bool tranfftwpp;
  mpitranspose<Complex> *T;
public:
  void inittranspose(Complex* f) {
    int size;
    MPI_Comm_size(d.communicator,&size);
    T=new mpitranspose<Complex>(d.X,d.y,d.x,d.Y,1,f,d.communicator);
  }
  
  fft2dMPI(const split& d, Complex *f) : d(d) {
    inittranspose(f);

    unsigned int n=d.X;
    unsigned int M=d.y;
    unsigned int stride=d.y;
    unsigned int dist=1;
    xForwards=new mfft1d(n,-1,M,stride,dist,f,f);
    xBackwards=new mfft1d(n,1,M,stride,dist,f,f);

    n=d.Y;
    M=d.x;
    stride=1;
    dist=d.Y;
    yForwards=new mfft1d(n,-1,M,stride,dist,f,f);
    yBackwards=new mfft1d(n,1,M,stride,dist,f,f);
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
    Txy=new mpitranspose<Complex>(d.X,d.xy.y,d.x,d.Y,d.z,f,d.xy.communicator);
    
    xForwards=new mfft1d(d.X,-1,d.xy.y*d.z,d.xy.y*d.z,1);
    xBackwards=new mfft1d(d.X,1,d.xy.y*d.z,d.xy.y*d.z,1);

    if(d.y < d.Y) {
      Tyz=new mpitranspose<Complex>(d.Y,d.z,d.y,d.Z,1,f,d.yz.communicator);
      
      yForwards=new mfft1d(d.Y,-1,d.z,d.z,1);
      yBackwards=new mfft1d(d.Y,1,d.z,d.z,1);

      zForwards=new mfft1d(d.Z,-1,d.x*d.y,1,d.Z);
      zBackwards=new mfft1d(d.Z,1,d.x*d.y,1,d.Z);
    } else {
      yzForwards=new fft2d(d.Y,d.Z,-1,f);
      yzBackwards=new fft2d(d.Y,d.Z,1,f);
    }
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
    delete Txy;
  }

  void Forwards(Complex *f);
  void Backwards(Complex *f);
  void Normalize(Complex *f);
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
  unsigned int mx, my;
  split dr,dc;
  mfft1d *xForwards;
  mfft1d *xBackwards;
  mrcfft1d *yForwards;
  mcrfft1d *yBackwards;
  Complex *f;
  //bool inplace; // we stick with out-of-place transforms for now
  //unsigned int rdist;
protected:
  mpitranspose<Complex> *T;
public:
   rcfft2dMPI(const split& dr, const split& dc,
	      double *f, Complex *g) : dr(dr), dc(dc) {
    mx=dr.X;
    my=dc.Y;
    inittranspose(g);
    
    //rdist=inplace ? dr.X+2 : dr.X;

    xForwards=new mfft1d(dc.X,-1,dc.y,dc.y,1);
    xBackwards=new mfft1d(dc.X,1,dc.y,dc.y,1);
    
    yForwards=new mrcfft1d(dr.Y, // n
			   dr.x, // M
			   1,    // stride
			   dr.Y,// dist
			   f,g);
    yBackwards=new mcrfft1d(dr.Y,dc.x,1,dc.Y,g,f);
  }

 void inittranspose(Complex* out) {
    int size;
    MPI_Comm_size(dc.communicator,&size);

    T=new mpitranspose<Complex>(dc.X,dc.y,dc.x,dc.Y,1,out,dc.communicator);
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

  
} // end namespace fftwpp

#endif
