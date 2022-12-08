#ifndef __explicit_h__
#define __explicit_h__ 1

#include "timing.h"

namespace fftwpp {

// Specialized mult routines for testing explicit convolutions:
typedef void Multiplier(Complex **, size_t m,
                        size_t threads);
typedef void Realmultiplier(double **, size_t m,
                            size_t threads);

Multiplier multbinary,multbinaryUnNormalized;
Realmultiplier multbinary,multbinaryUnNormalized;

class ExplicitPad : public ThreadBase {
protected:
  size_t n; // Number of modes including padding
  size_t m; // Number of dealiased modes
public:
  ExplicitPad(size_t n, size_t m) : n(n), m(m) {}

  void pad(Complex *f) {
    PARALLELIF(
      n-m > threshold,
      for(size_t i=m; i < n; ++i) f[i]=0.0;
      );
  }
};

class ExplicitPad2 : public ThreadBase {
protected:
  size_t nx,ny; // Number of modes including padding in each direction
  size_t mx,my; // Number of dealiased modes in each direction
public:
  ExplicitPad2(size_t nx, size_t ny,
               size_t mx, size_t my) :
    nx(nx), ny(ny), mx(mx), my(my) {}

  void pad(Complex *f) {
    // zero pad upper block
    PARALLELIF(
      mx*ny > threshold,
      for(size_t i=0; i < mx; ++i) {
        size_t nyi=ny*i;
        size_t stop=nyi+ny;
        for(size_t j=nyi+my; j < stop; ++j)
          f[j]=0.0;
      });

    // zero pad right-hand block
    const size_t start=mx*ny;
    const size_t stop=nx*ny;
    PARALLELIF(
      stop-start > threshold,
      for(size_t i=start; i < stop; ++i)
        f[i]=0.0;
      );
  }
};

class ExplicitHPad2 : public ThreadBase {
protected:
  size_t nx,ny; // Number of modes including padding in each direction
  size_t mx,my; // Number of dealiased modes in each direction
public:
  ExplicitHPad2(size_t nx, size_t ny,
                size_t mx, size_t my) :
    nx(nx), ny(ny), mx(mx), my(my) {}

  void pad(Complex *f) {
    size_t nyp=ny/2+1;
    size_t nx2=nx/2;

    // zero pad left block
    size_t stop=(nx2-mx+1)*nyp;
    PARALLELIF(
      stop > threshold,
      for(size_t i=0; i < stop; ++i)
        f[i]=0.0;
      );

    // zero pad top-middle block
    size_t stop2=stop+2*mx*nyp;
    size_t diff=nyp-my;
    PARALLELIF(
      2*mx*diff > threshold,
      for(size_t i=stop+nyp; i < stop2; i += nyp) {
        for(size_t j=i-diff; j < i; ++j)
          f[j]=0.0;
      }
      );

    // zero pad right block
    stop=nx*nyp;
    PARALLELIF(
      stop-(nx2+mx)*nyp > threshold,
      for(size_t i=(nx2+mx)*nyp; i < stop; ++i)
        f[i]=0.0;
      );
  }
};

class ExplicitHTPad2 : public ThreadBase {
protected:
  size_t nx,ny; // Number of modes including padding in each direction
  size_t mx,my; // Number of dealiased modes in each direction
public:
  ExplicitHTPad2(size_t nx, size_t ny,
                 size_t mx, size_t my) :
    nx(nx), ny(ny), mx(mx), my(my) {}

  void pad(Complex *f) {
    size_t nyp=ny/2+1;
    size_t nx2=nx/2;
    size_t end=nx2-mx;
    PARALLEL(
      for(size_t i=0; i <= end; ++i) {
        size_t nypi=nyp*i;
        size_t stop=nypi+nyp;
        for(size_t j=nypi; j < stop; ++j)
          f[j]=0.0;
      }
      );

    PARALLEL(
      for(size_t i=nx2+mx; i < nx; ++i) {
        size_t nypi=nyp*i;
        size_t stop=nypi+nyp;
        for(size_t j=nypi; j < stop; ++j)
          f[j]=0.0;
      }
      );

    PARALLEL(
      for(size_t i=0; i < nx; ++i) {
        size_t nypi=nyp*i;
        size_t stop=nypi+nyp;
        for(size_t j=nypi+my; j < stop; ++j)
          f[j]=0.0;
      }
      );
  }
};

class ExplicitPad3 : public ThreadBase {
protected:
  size_t nx,ny,nz; // Number of modes including padding in each direction
  size_t mx,my,mz; // Number of dealiased modes in each direction
public:
  ExplicitPad3(size_t nx, size_t ny, size_t nz,
               size_t mx, size_t my, size_t mz) :
    nx(nx), ny(ny), nz(nz), mx(mx), my(my), mz(mz) {}

  void pad(Complex *f) {
    PARALLEL(
      for(size_t i=0; i < mx; ++i) {
        size_t nyi=ny*i;
        for(size_t j=0; j < my; ++j) {
          size_t nyzij=nz*(nyi+j);
          size_t stop=nyzij+nz;
          for(size_t k=nyzij+mz; k < stop; ++k)
            f[k]=0.0;
        }
      }
      );

    size_t nyz=ny*nz;
    PARALLEL(
      for(size_t i=mx; i < nx; ++i) {
        size_t nyzi=nyz*i;
        for(size_t j=0; j < ny; ++j) {
          size_t nyzij=nyzi+nz*j;
          size_t stop=nyzij+nz;
          for(size_t k=nyzij; k < stop; ++k)
            f[k]=0.0;
        }
      }
      );

    PARALLEL(
      for(size_t i=0; i < nx; ++i) {
        size_t nyzi=nyz*i;
        for(size_t j=my; j < ny; ++j) {
          size_t nyzij=nyzi+nz*j;
          size_t stop=nyzij+nz;
          for(size_t k=nyzij; k < stop; ++k)
            f[k]=0.0;
        }
      }
      );
  }
};

class ExplicitHPad3 : public ThreadBase {
protected:
  size_t nx,ny,nz; // Number of modes including padding in each direction
  size_t mx,my,mz; // Number of dealiased modes in each direction
public:
  ExplicitHPad3(size_t nx, size_t ny, size_t nz,
               size_t mx, size_t my, size_t mz) :
    nx(nx), ny(ny), nz(nz), mx(mx), my(my), mz(mz) {}

  void pad(Complex *f) {
    // Not yet implemented
  }
};

// In-place explicitly dealiased 1D complex convolution.
class ExplicitConvolution : public ExplicitPad {
protected:
  fft1d *Backwards,*Forwards;
public:

  // u is a temporary array of size n.
  ExplicitConvolution(size_t n, size_t m, Complex *u, Complex *v=NULL) :
    ExplicitPad(n,m) {
    if(v == NULL) v=u;
    Backwards=new fft1d(n,1,u,v);
    Forwards=new fft1d(n,-1,v,u);

    threads=Forwards->Threads();
  }

  ~ExplicitConvolution() {
    delete Forwards;
    delete Backwards;
  }

  void backwards(Complex *f, Complex *F);
  void forwards(Complex *F, Complex *f);

  // F is an array of pointers to distinct data blocks each of size n.
  void convolve(Complex **F, Multiplier *mult, Complex **G=NULL);

  // Compute f (*) g. The distinct input arrays f and g are each of size n
  // (contents not preserved). The output is returned in f.
  void convolve(Complex *f, Complex *g);
};

// In-place explicitly dealiased 1D Hermitian convolution.
class ExplicitHConvolution : public ExplicitPad {
protected:
  size_t n,m;
  rcfft1d *rc;
  crfft1d *cr;
  size_t threads;
public:
  // u is a temporary array of size n.
  ExplicitHConvolution(size_t n, size_t m, Complex *u, double *v) :
    ExplicitPad(n/2+1,m), n(n), m(m) {
    if(v == NULL) v=(double *) u;
    rc=new rcfft1d(n,v,u);
    cr=new crfft1d(n,u,v);

    threads=cr->Threads();
  }

  ~ExplicitHConvolution() {
    delete cr;
    delete rc;
  }

  void backwards(Complex *F, double *G);
  void forwards(double *G, Complex *F);

  // F is an array of pointers to distinct data blocks each of size n.
  void convolve(Complex **F, Realmultiplier *mult, double **G=NULL);

// Compute f (*) g, where f and g contain the m non-negative Fourier
// components of real functions. Dealiasing is internally implemented via
// explicit zero-padding to size n >= 3*m.
//
// The (distinct) input arrays f and g must each be allocated to size n/2+1
// (contents not preserved). The output is returned in the first m elements
// of f.
//  void convolve(Complex *f, Complex *g);
};

// In-place explicitly dealiased 2D complex convolution.
class ExplicitConvolution2 : public ExplicitPad2 {
protected:
  bool prune; // Skip Fourier transforming rows containing all zeroes?
  mfft1d *xBackwards, *xForwards;
  mfft1d *yBackwards, *yForwards;
  fft2d *Backwards, *Forwards;
public:
  ExplicitConvolution2(size_t nx, size_t ny,
                       size_t mx, size_t my,
                       Complex *f, bool prune=false) :
    ExplicitPad2(nx,ny,mx,my), prune(prune) {
    if(prune) {
      xBackwards=new mfft1d(nx,1,my,ny,1,f,f);
      yBackwards=new mfft1d(ny,1,nx,1,ny,f,f);
      yForwards=new mfft1d(ny,-1,nx,1,ny,f,f);
      xForwards=new mfft1d(nx,-1,my,ny,1,f,f);
      threads=xForwards->Threads();
    } else {
      Backwards=new fft2d(nx,ny,1,f);
      Forwards=new fft2d(nx,ny,-1,f);
      threads=Forwards->Threads();
    }
  }

  ~ExplicitConvolution2() {
    if(prune) {
      delete xForwards;
      delete yForwards;
      delete yBackwards;
      delete xBackwards;
    } else {
      delete Forwards;
      delete Backwards;
    }
  }

  void backwards(Complex *f);
  void forwards(Complex *f);

  // F is an array of pointers to distinct data blocks each of size n.
//  void convolve(Complex **F, Multiplier *mult);

  void convolve(Complex *f, Complex *g);
};

// In-place explicitly dealiased 2D Hermitian convolution.
class ExplicitHConvolution2 : public ExplicitHPad2 {
protected:
  size_t M;
  bool prune; // Skip Fourier transforming rows containing all zeroes?
  mfft1d *xBackwards;
  mfft1d *xForwards;
  mcrfft1d *yBackwards;
  mrcfft1d *yForwards;
  crfft2d *Backwards;
  rcfft2d *Forwards;

  size_t s;
  Complex *ZetaH,*ZetaL;
public:
  ExplicitHConvolution2(size_t nx, size_t ny,
                        size_t mx, size_t my,
                        Complex *f, size_t M=1, bool pruned=false) :
    ExplicitHPad2(nx,ny,mx,my), M(M), prune(pruned) {
    threads=fftw::maxthreads;
    size_t nyp=ny/2+1;
    // Odd nx requires interleaving of shift with x and y transforms.
    size_t My=my;
    if(nx % 2) {
      if(!prune) My=nyp;
      prune=true;
      s=BuildZeta(2*nx,nx,ZetaH,ZetaL);
    }

    if(prune) {
      xBackwards=new mfft1d(nx,1,My,nyp,1,f);
      xForwards=new mfft1d(nx,-1,My,nyp,1,f);

      {
        ptrdiff_t cdist=nx/2+1;
        ptrdiff_t rdist=2*cdist; // in-place transform
        yForwards=new mrcfft1d(ny,nx,1,1,rdist,cdist,(double*) f);
        yBackwards=new mcrfft1d(ny,nx,1,1,cdist,rdist,f);
      }

    } else {
      Backwards=new crfft2d(nx,ny,f);
      Forwards=new rcfft2d(nx,ny,f);
    }
  }

  ~ExplicitHConvolution2() {
    if(prune) {
      delete xForwards;
      delete yForwards;
      delete yBackwards;
      delete xBackwards;
    } else {
      delete Forwards;
      delete Backwards;
    }
  }

  void backwards(Complex *f, bool shift=true);
  void forwards(Complex *f);
  void convolve(Complex **F, Complex **G, bool symmetrize=true);

  // Constructor for special case M=1:
  void convolve(Complex *f, Complex *g, bool symmetrize=true) {
    convolve(&f,&g,symmetrize);
  }
};

// In-place explicitly dealiased 3D complex convolution.
class ExplicitConvolution3 : public ExplicitPad3 {
protected:
  bool prune; // Skip Fourier transforming rows containing all zeroes?
  mfft1d *xBackwards, *xForwards;
  mfft1d *yBackwards, *yForwards;
  mfft1d *zBackwards, *zForwards;
  fft3d *Backwards, *Forwards;
  size_t threads;
public:
  ExplicitConvolution3(size_t nx, size_t ny, size_t nz,
                       size_t mx, size_t my, size_t mz,
                       Complex *f, bool prune=false) :
    ExplicitPad3(nx,ny,nz,mx,my,mz), prune(prune) {
    threads=fftw::maxthreads;
    size_t nxy=nx*ny;
    size_t nyz=ny*nz;
    if(prune) {
      xBackwards=new mfft1d(nx,1,mz,nyz,1,f);
      yBackwards=new mfft1d(ny,1,mz,nz,1,f);
      zBackwards=new mfft1d(nz,1,nxy,1,nz,f);
      zForwards=new mfft1d(nz,-1,nxy,1,nz,f);
      yForwards=new mfft1d(ny,-1,mz,nz,1,f);
      xForwards=new mfft1d(nx,-1,mz,nyz,1,f);
    } else {
      Backwards=new fft3d(nx,ny,nz,1,f);
      Forwards=new fft3d(nx,ny,nz,-1,f);
    }
  }

  ~ExplicitConvolution3() {
    if(prune) {
      delete xForwards;
      delete yForwards;
      delete zForwards;
      delete zBackwards;
      delete yBackwards;
      delete xBackwards;
    } else {
      delete Forwards;
      delete Backwards;
    }
  }

  void backwards(Complex *f);
  void forwards(Complex *f);
  void convolve(Complex *f, Complex *g);
};

// In-place explicitly dealiased 3D Hermitian convolution.
class ExplicitHConvolution3 : public ExplicitHPad3 {
protected:
  size_t M;
  crfft3d *Backwards;
  rcfft3d *Forwards;

  size_t s;
  Complex *ZetaH,*ZetaL;
public:
  ExplicitHConvolution3(size_t nx, size_t ny, size_t nz,
                        size_t mx, size_t my, size_t mz,
                        Complex *f, size_t M=1) :
    ExplicitHPad3(nx,ny,nz,mx,my,mz), M(M) {
    threads=fftw::maxthreads;
    if(nx % 2 || ny % 2)
      std::cout << "nx and ny must be even" << std::endl;

    Backwards=new crfft3d(nx,ny,nz,f);
    Forwards=new rcfft3d(nx,ny,nz,f);
  }

  ~ExplicitHConvolution3() {
    delete Forwards;
    delete Backwards;
  }

  void backwards(Complex *f);
  void forwards(Complex *f);
  void convolve(Complex **F, Complex **G, bool symmetrize=true);

  // Constructor for special case M=1:
  void convolve(Complex *f, Complex *g, bool symmetrize=true) {
    convolve(&f,&g,symmetrize);
  }
};

// In-place explicitly dealiased Hermitian ternary convolution.
class ExplicitHTConvolution : public ExplicitPad  {
protected:
  size_t n,m;
  rcfft1d *rc;
  crfft1d *cr;
  size_t threads;
public:
  // u is a temporary array of size n.
  ExplicitHTConvolution(size_t n, size_t m, Complex *u) :
    ExplicitPad(n/2+1,m), n(n), m(m) {
    rc=new rcfft1d(n,u);
    cr=new crfft1d(n,u);

    threads=cr->Threads();
  }

  ~ExplicitHTConvolution() {
    delete cr;
    delete rc;
  }

  void backwards(Complex *f);
  void forwards(Complex *f);

// Compute the ternary convolution of f, and g, and h, where f, and g, and h
// contain the m non-negative Fourier components of real
// functions. Dealiasing is internally implemented via explicit
// zero-padding to size n >= 3*m. The (distinct) input arrays f, g, and h
// must each be allocated to size n/2+1 (contents not preserved).
// The output is returned in the first m elements of f.
  void convolve(Complex *f, Complex *g, Complex *h);
};

// In-place explicitly dealiased 2D Hermitian ternary convolution.
class ExplicitHTConvolution2 : public ExplicitHTPad2 {
protected:
  bool prune; // Skip Fourier transforming rows containing all zeroes?
  mfft1d *xBackwards;
  mfft1d *xForwards;
  mcrfft1d *yBackwards;
  mrcfft1d *yForwards;
  crfft2d *Backwards;
  rcfft2d *Forwards;
  size_t s;
  Complex *ZetaH,*ZetaL;
  size_t threads;

public:
  ExplicitHTConvolution2(size_t nx, size_t ny,
                         size_t mx, size_t my, Complex *f,
                         bool pruned=false) :
    ExplicitHTPad2(nx,ny,mx,my), prune(pruned) {
    size_t nyp=ny/2+1;
    // Odd nx requires interleaving of shift with x and y transforms.
    size_t My=my;
    if(nx % 2) {
      if(!prune) My=nyp;
      prune=true;
      s=BuildZeta(2*nx,nx,ZetaH,ZetaL);
    }

    if(prune) {
      xBackwards=new mfft1d(nx,1,My,nyp,1,f);
      xForwards=new mfft1d(nx,-1,My,nyp,1,f);
      {
        ptrdiff_t cdist=nx/2+1;
        ptrdiff_t rdist=2*cdist; // in-place transform
        yForwards=new mrcfft1d(ny,nx,1,1,rdist,cdist,(double*) f);
        yBackwards=new mcrfft1d(ny,nx,1,1,cdist,rdist,f);
      }

      threads=xForwards->Threads();
    } else {
      Backwards=new crfft2d(nx,ny,f);
      Forwards=new rcfft2d(nx,ny,f);
      threads=Forwards->Threads();
    }
  }

  ~ExplicitHTConvolution2() {
    if(prune) {
      delete xForwards;
      delete yForwards;
      delete yBackwards;
      delete xBackwards;
    } else {
      delete Forwards;
      delete Backwards;
    }
  }

  void backwards(Complex *f, bool shift=true);
  void forwards(Complex *f, bool shift=true);
  void convolve(Complex *f, Complex *g, Complex *h, bool symmetrize=true);
};

}

#endif
