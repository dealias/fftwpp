#ifndef __explicit_h__
#define __explicit_h__ 1

namespace fftwpp {

// In-place explicitly dealiased 1D complex convolution.
class ExplicitConvolution : public ThreadBase {
protected:
  unsigned int n,m;
  fft1d *Backwards,*Forwards;
public:  
  
  // u is a temporary array of size n.
  ExplicitConvolution(unsigned int n, unsigned int m, Complex *u) :
    n(n), m(m) {
    Backwards=new fft1d(n,1,u);
    Forwards=new fft1d(n,-1,u);

    threads=Forwards->Threads();
  }
  
  ~ExplicitConvolution() {
    delete Forwards;
    delete Backwards;
  }    
  
  void pad(Complex *f);
  void backwards(Complex *f);
  void forwards(Complex *f);
  
  // Compute f (*) g. The distinct input arrays f and g are each of size n 
  // (contents not preserved). The output is returned in f.
  void convolve(Complex *f, Complex *g);
};

// In-place explicitly dealiased 1D Hermitian convolution.
class ExplicitHConvolution : public ThreadBase {
protected:
  unsigned int n,m;
  rcfft1d *rc;
  crfft1d *cr;
  unsigned int threads;
public:
  // u is a temporary array of size n.
  ExplicitHConvolution(unsigned int n, unsigned int m, Complex *u) :
    n(n), m(m) {
    rc=new rcfft1d(n,u);
    cr=new crfft1d(n,u);

    threads=cr->Threads();
  }
  
  ~ExplicitHConvolution() {
    delete cr;
    delete rc;
  }
    
  void pad(Complex *f);
  void backwards(Complex *f);
  void forwards(Complex *f);
  
// Compute f (*) g, where f and g contain the m non-negative Fourier
// components of real functions. Dealiasing is internally implemented via
// explicit zero-padding to size n >= 3*m.
//
// The (distinct) input arrays f and g must each be allocated to size n/2+1
// (contents not preserved). The output is returned in the first m elements
// of f.
  void convolve(Complex *f, Complex *g);
};

// In-place explicitly dealiased 2D complex convolution.
class ExplicitConvolution2 : public ThreadBase {
protected:
  unsigned int nx,ny;
  unsigned int mx,my;
  bool prune; // Skip Fourier transforming rows containing all zeroes?
  mfft1d *xBackwards, *xForwards;
  mfft1d *yBackwards, *yForwards;
  fft2d *Backwards, *Forwards;
public:
  ExplicitConvolution2(unsigned int nx, unsigned int ny,
                       unsigned int mx, unsigned int my,
                       Complex *f, bool prune=false) :
    nx(nx), ny(ny), mx(mx), my(my), prune(prune) {
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
  
  void pad(Complex *f);
  void backwards(Complex *f);
  void forwards(Complex *f);
  void convolve(Complex *f, Complex *g);
};

// In-place explicitly dealiased 2D Hermitian convolution.
class ExplicitHConvolution2 : public ThreadBase {
protected:
  unsigned int nx,ny;
  unsigned int mx,my;
  unsigned int M;
  bool prune; // Skip Fourier transforming rows containing all zeroes?
  mfft1d *xBackwards;
  mfft1d *xForwards;
  mcrfft1d *yBackwards;
  mrcfft1d *yForwards;
  crfft2d *Backwards;
  rcfft2d *Forwards;
  
  unsigned int s;
  Complex *ZetaH,*ZetaL;
public:  
  ExplicitHConvolution2(unsigned int nx, unsigned int ny, 
                        unsigned int mx, unsigned int my,
                        Complex *f, unsigned int M=1,
                        bool pruned=false) :
    nx(nx), ny(ny), mx(mx), my(my), M(M), prune(pruned) {
    threads=fftw::maxthreads;
    unsigned int nyp=ny/2+1;
    // Odd nx requires interleaving of shift with x and y transforms.
    unsigned int My=my;
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
  
  void pad(Complex *f);
  void backwards(Complex *f, bool shift=true);
  void forwards(Complex *f);
  void convolve(Complex **F, Complex **G, bool symmetrize=true);
  
  // Constructor for special case M=1:
  void convolve(Complex *f, Complex *g, bool symmetrize=true) {
    convolve(&f,&g,symmetrize);
  }
};

// In-place explicitly dealiased 3D complex convolution.
class ExplicitConvolution3 : public ThreadBase {
protected:
  unsigned int nx,ny,nz;
  unsigned int mx,my,mz;
  bool prune; // Skip Fourier transforming rows containing all zeroes?
  mfft1d *xBackwards, *xForwards;
  mfft1d *yBackwards, *yForwards;
  mfft1d *zBackwards, *zForwards;
  fft3d *Backwards, *Forwards;
  unsigned int threads;
public:
  ExplicitConvolution3(unsigned int nx, unsigned int ny, unsigned int nz,
                       unsigned int mx, unsigned int my, unsigned int mz,
                       Complex *f, bool prune=false) :
    nx(nx), ny(ny), nz(nz), mx(mx), my(my), mz(mz), prune(prune) {
    threads=fftw::maxthreads;
    unsigned int nxy=nx*ny;
    unsigned int nyz=ny*nz;
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
  
  void pad(Complex *f);
  void backwards(Complex *f);
  void forwards(Complex *f);
  void convolve(Complex *f, Complex *g);
};

// In-place explicitly dealiased Hermitian ternary convolution.
class ExplicitHTConvolution : public ThreadBase  {
protected:
  unsigned int n;
  unsigned int m;
  rcfft1d *rc;
  crfft1d *cr;
  unsigned int threads;
public:
  // u is a temporary array of size n.
  ExplicitHTConvolution(unsigned int n, unsigned int m, Complex *u) :
    n(n), m(m) {
    rc=new rcfft1d(n,u);
    cr=new crfft1d(n,u);

    threads=cr->Threads();
  }
  
  ~ExplicitHTConvolution() {
    delete cr;
    delete rc;
  }
    
  void pad(Complex *f);
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
class ExplicitHTConvolution2 : public ThreadBase {
protected:
  unsigned int nx,ny;
  unsigned int mx,my;
  bool prune; // Skip Fourier transforming rows containing all zeroes?
  mfft1d *xBackwards;
  mfft1d *xForwards;
  mcrfft1d *yBackwards;
  mrcfft1d *yForwards;
  crfft2d *Backwards;
  rcfft2d *Forwards;
  unsigned int s;
  Complex *ZetaH,*ZetaL;
  unsigned int threads;

public:  
  ExplicitHTConvolution2(unsigned int nx, unsigned int ny, 
                         unsigned int mx, unsigned int my, Complex *f,
                         bool pruned=false) :
    nx(nx), ny(ny), mx(mx), my(my), prune(pruned) {
    unsigned int nyp=ny/2+1;
    // Odd nx requires interleaving of shift with x and y transforms.
    unsigned int My=my;
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
  
  void pad(Complex *f);
  void backwards(Complex *f, bool shift=true);
  void forwards(Complex *f, bool shift=true);
  void convolve(Complex *f, Complex *g, Complex *h, bool symmetrize=true);
};

}

#endif

