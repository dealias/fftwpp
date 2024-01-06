#pragma once

#include <mpi.h>

#include "convolve.h"
#include "mpifftw++.h"
#include "mpitranspose.h"

namespace fftwpp {

class param {
public:
  size_t m;
  size_t D;
  size_t I;
  size_t l;

  param() : m(0), D(0), I(0), l(0) {}

  void init(fftBase *fft) {
    m=fft->m;
    D=fft->D;
    I=fft->inplace;
    l=fft->l;
  }
};

class params {
public:
  param x;
  param y;
  param z;
};

// Return the output buffer decomposition
inline utils::split outputSplit(fftBase *fftx, fftBase *ffty, const MPI_Comm &communicator) {
  return utils::split(fftx->l*fftx->D,ffty->inputLength(),communicator);
}

inline utils::split3 outputSplit(fftBase *fftx, fftBase *ffty, fftBase *fftz, const utils::MPIgroup &group, bool spectral=false) {
  return utils::split3(fftx->l*fftx->D,ffty->inputLength(),fftz->inputLength(),
                       group,spectral);
}

// Return the minimum buffer size for the transpose
inline size_t bufferSize(fftBase *fftx, fftBase *ffty,
                         const MPI_Comm &communicator) {
  return outputSplit(fftx,ffty,communicator).n;
}

// Return the minimum buffer size for the transpose
inline size_t bufferSize(fftBase *fftx, fftBase *ffty, fftBase *fftz,
                         const utils::MPIgroup &group) {
  return outputSplit(fftx,ffty,fftz,group).n;
}

// Allocate the outputBuffer;
inline Complex **outputBuffer(fftBase *fftx, fftBase *ffty,
                              const MPI_Comm &communicator) {
  return utils::ComplexAlign(std::max(fftx->app.A,fftx->app.B),
                             bufferSize(fftx,ffty,communicator));
}

// Allocate the outputBuffer;
inline Complex **outputBuffer(fftBase *fftx, fftBase *ffty, fftBase *fftz,
                              const utils::MPIgroup &group) {
  return utils::ComplexAlign(std::max(fftx->app.A,fftx->app.B),
                             bufferSize(fftx,ffty,fftz,group));
}

// In-place implicitly dealiased 2D complex convolution.
class Convolution2MPI : public Convolution2 {
protected:
  utils::split d;
  utils::mpitranspose<Complex> **T;
  size_t overwrite;
  bool overlap;
public:

  void inittranspose(const utils::mpiOptions& mpi) {
    overwrite=fftx->overwrite ? fftx->n-1 : 0;
    overlap=overwrite ? false : fftx->loop2();

    if(d.y < d.Y) {
      size_t C=std::max(A,B);
      T=new utils::mpitranspose<Complex> *[C];
      for(size_t a=0; a < C; ++a)
        T[a]=new utils::mpitranspose<Complex>(d.X,d.Y,d.x,d.y,1,
                                              F[a],NULL,
                                              d.communicator,mpi,d.communicator);
    } else {
      overlap=false;
      T=NULL;
    }
  }

  Convolution2MPI(fftBase *fftx, fftBase *ffty,
                  const MPI_Comm& communicator,
                  utils::mpiOptions mpi=utils::defaultmpiOptions,
                  Complex **F=NULL, Complex *W=NULL, Complex *V=NULL) :
    Convolution2(fftx,ffty,F ? F : AllocateF(fftx,ffty,communicator),W,V),
    d(outputSplit(fftx,ffty,communicator)) {
    inittranspose(mpi);
  }

  virtual ~Convolution2MPI() {
    if(T) {
      size_t C=std::max(A,B);
      for(size_t a=0; a < C; ++a)
        delete T[a];
      delete [] T;
    }
  }

  Complex **AllocateF(fftBase *fftx, fftBase *ffty,
                      const MPI_Comm &communicator) {
    allocateF=true;
    return outputBuffer(fftx,ffty,communicator);
  }

  void forward(Complex **f, Complex **F, size_t rx,
               size_t start, size_t stop,
               size_t offset=0) {
    size_t incr=Sx*lx;
    size_t cutoff=overlap && rx != 0 ? ((B-C)/C+1)*C : 0;
    for(size_t a=start; a < stop; ++a) {
      (fftx->*Forward)(f[a]+offset,F[a],rx,W);
      if(T) {
        if(a < cutoff)
          T[a]->ilocalize1(F[a]);
        else
          T[a]->localize1(F[a]);
        for(size_t r=0; r < overwrite; ++r)
          T[a]->localize1(f[a]+offset+incr*r);
      }
    }
    size_t Stop=std::min(stop,cutoff);
    for(size_t a=start; a < Stop; ++a)
      T[a]->wait();
  }

  virtual size_t stridex() {
    return d.Y;
  }

  virtual size_t blocksizex(size_t rx) {
    return d.x;
  }

  void backward(Complex **F, Complex **f, size_t rx,
                size_t start, size_t stop,
                size_t offset=0, Complex *W0=NULL) {
    size_t incr=Sx*lx;
    if(T)
      for(size_t b=start; b < stop; ++b) {
        if(overlap && rx == 0)
          T[b]->ilocalize0(F[b]);
        else
          T[b]->localize0(F[b]);
        for(size_t r=0; r < overwrite; ++r)
          T[b]->localize0(f[b]+offset+incr*r);
      }
    for(size_t b=start; b < stop; ++b) {
      if(overlap && rx == 0)
        T[b]->wait();
      (fftx->*Backward)(F[b],f[b]+offset,rx,W0);
    }
    if(W && W == W0) (fftx->*Pad)(W0);
  }

  size_t inputLengthy() {
    return d.y;
  }

};

// In-place implicitly dealiased 2D complex convolution.
class Convolution3MPI : public Convolution3 {
protected:
  utils::split3 d;
  utils::mpitranspose<Complex> **T;
  size_t overwrite;
  bool overlap;
public:

  void inittranspose(const utils::mpiOptions& mpi) {
    if(d.xy.y < d.Y) {
      overwrite=fftx->overwrite ? fftx->n-1 : 0;
      overlap=overwrite ? false : fftx->loop2();

      size_t C=std::max(A,B);
      T=new utils::mpitranspose<Complex> *[C];
      for(size_t a=0; a < C; ++a)
        T[a]=new utils::mpitranspose<Complex>(d.X,d.Y,d.x,d.xy.y,d.z,
                                              F[a],NULL,
                                              d.xy.communicator,
                                              mpi,d.communicator);
    } else {
      overlap=false;
      T=NULL;
    }
  }

  void initMPI(const utils::mpiOptions& mpi, unsigned int Threads) {
    for(unsigned int t=0; t < threads; ++t) {
      convolveyz[t]=d.z < d.Z ?
        new Convolution2MPI(ffty,fftz,d.yz.communicator,mpi) :
        new Convolution2(ffty,fftz);
    }
    scale=1.0/normalization();
    inittranspose(mpi);
  }

  Convolution3MPI(fftBase *fftx, fftBase *ffty, fftBase *fftz,
                  const utils::MPIgroup& group,
                  utils::mpiOptions mpi=utils::defaultmpiOptions,
                  Complex **F=NULL, Complex *W=NULL, Complex *V=NULL) :
    Convolution3(fftx,ffty,fftz,F ? F : AllocateF(fftx,ffty,fftz,group),
                 W,V,true),
    d(outputSplit(fftx,ffty,fftz,group)) {
    initMPI(mpi,threads);
  }

  virtual ~Convolution3MPI() {
    if(T) {
      size_t C=std::max(A,B);
      for(size_t a=0; a < C; ++a)
        delete T[a];
      delete [] T;
    }
  }

  Complex **AllocateF(fftBase *fftx, fftBase *ffty, fftBase *fftz,
                      const utils::MPIgroup &group) {
    allocateF=true;
    return outputBuffer(fftx,ffty,fftz,group);
  }

  void forward(Complex **f, Complex **F, size_t rx,
               size_t start, size_t stop,
               size_t offset=0) {
    size_t incr=Sx*lx;
    size_t cutoff=overlap && rx != 0 ? ((B-C)/C+1)*C : 0;
    for(size_t a=start; a < stop; ++a) {
      (fftx->*Forward)(f[a]+offset,F[a],rx,W);
      if(T) {
        if(a < cutoff)
          T[a]->ilocalize1(F[a]);
        else
          T[a]->localize1(F[a]);
        for(size_t r=0; r < overwrite; ++r)
          T[a]->localize1(f[a]+offset+incr*r);
      }
    }
    size_t Stop=std::min(stop,cutoff);
    for(size_t a=start; a < Stop; ++a)
      T[a]->wait();
  }

  virtual size_t stridex() {
    return d.Y*d.z;
  }

  virtual size_t blocksizex(size_t rx) {
    return d.x;
  }

  void backward(Complex **F, Complex **f, size_t rx,
                size_t start, size_t stop,
                size_t offset=0, Complex *W0=NULL) {
    size_t incr=Sx*lx;
    if(T)
      for(size_t b=start; b < stop; ++b) {
        if(overlap && rx == 0)
          T[b]->ilocalize0(F[b]);
        else
          T[b]->localize0(F[b]);
        for(size_t r=0; r < overwrite; ++r)
          T[b]->localize0(f[b]+offset+incr*r);
      }
    for(size_t b=start; b < stop; ++b) {
      if(overlap && rx == 0)
        T[b]->wait();
      (fftx->*Backward)(F[b],f[b]+offset,rx,W0);
    }
    if(W && W == W0) (fftx->*Pad)(W0);
  }

  size_t inputLengthy() {
    return d.xy.y;
  }

  size_t inputLengthz() {
    return d.z;
  }

};

// Enforce 2D Hermiticity using specified (x >= 0,y=0) data.
inline void HermitianSymmetrizeX(const utils::split& d, Complex *f,
                                 size_t threads=fftw::maxthreads)
{
  size_t Hx=utils::ceilquotient(d.X,2);
  if(d.y0 == 0)
    HermitianSymmetrizeX(Hx,d.y,d.X/2,f);
  else
    // Zero out Nyquist modes
    if(d.X/2 == Hx) {
      PARALLELIF(
        d.y > threshold,
        for(size_t j=0; j < d.y; ++j)
          f[j]=0.0;
        );
    }

}

// Enforce 3D Hermiticity using specified (x >= 0,y=0,z=0) and (x,y > 0,z=0).
// data.
void HermitianSymmetrizeXY(utils::split3& d, Complex *f, Complex *u=NULL,
                           size_t threads=fftw::maxthreads);

} // namespace fftwpp
