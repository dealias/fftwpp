#pragma once

#include <mpi.h>
#include <vector>

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

  void init(fftBase &fft) {
    m=fft.m;
    D=fft.D;
    I=fft.inplace;
    l=fft.l;
  }
};

class params {
public:
  param x;
  param y;
  param z;
};

// Return the output buffer decomposition
utils::split outputSplit(fftBase *fftx, fftBase *ffty, const MPI_Comm &communicator) {
  return utils::split(fftx->l*fftx->D,ffty->L,communicator);
}

utils::split3 outputSplit(fftBase *fftx, fftBase *ffty, fftBase *fftz, const utils::MPIgroup &group, bool spectral=false) {
  return utils::split3(fftx->l*fftx->D,ffty->L,fftz->L,group,spectral);
}

// Return the minimum buffer size for the transpose
size_t bufferSize(fftBase *fftx, fftBase *ffty,
                  const MPI_Comm &communicator) {
  return outputSplit(fftx,ffty,communicator).n;
}

// Return the minimum buffer size for the transpose
size_t bufferSize(fftBase *fftx, fftBase *ffty, fftBase *fftz,
                  const utils::MPIgroup &group) {
  return outputSplit(fftx,ffty,fftz,group).n;
}

// Allocate the outputBuffer;
Complex **outputBuffer(fftBase *fftx, fftBase *ffty,
                       const MPI_Comm &communicator) {
  return utils::ComplexAlign(std::max(fftx->app.A,fftx->app.B),
                             bufferSize(fftx,ffty,communicator));
}

// Allocate the outputBuffer;
Complex **outputBuffer(fftBase *fftx, fftBase *ffty, fftBase *fftz,
                       const utils::MPIgroup &group) {
  return utils::ComplexAlign(std::max(fftx->app.A,fftx->app.B),
                             bufferSize(fftx,ffty,fftz,group));
}

// In-place implicitly dealiased 2D complex convolution.
class Convolution2MPI : public Convolution2 {
protected:
  utils::split d;
  utils::mpitranspose<Complex> **T;
public:

  void inittranspose(const utils::mpiOptions& mpi) {
    size_t C=std::max(A,B);
    T=new utils::mpitranspose<Complex> *[C];
    for(size_t a=0; a < C; ++a)
      T[a]=new utils::mpitranspose<Complex>(d.X,d.Y,d.x,d.y,1,
                                            F[a],NULL,
                                            d.communicator,mpi,d.communicator);
    d.Deactivate();
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
    size_t C=std::max(A,B);
    for(size_t a=0; a < C; ++a)
      delete T[a];
  }

  Complex **AllocateF(fftBase *fftx, fftBase *ffty,
                      const MPI_Comm &communicator) {
    allocateF=true;
    return outputBuffer(fftx,ffty,communicator);
  }

  void forward(Complex **f, Complex **F, size_t rx,
               size_t start, size_t stop,
               size_t offset=0) {
    for(size_t a=start; a < stop; ++a) {
      (fftx->*Forward)(f[a]+offset,F[a],rx,W);
      T[a]->ilocalize1(F[a]);
    }
    for(size_t a=start; a < stop; ++a)
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
    for(size_t b=start; b < stop; ++b)
      T[b]->ilocalize0(F[b]);
    for(size_t b=start; b < stop; ++b) {
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
public:

  void inittranspose(const utils::mpiOptions& mpi) {
    if(d.xy.y < d.Y) {
      size_t C=std::max(A,B);
      T=new utils::mpitranspose<Complex> *[C];
      for(size_t a=0; a < C; ++a)
        T[a]=new utils::mpitranspose<Complex>(d.X,d.Y,d.x,d.xy.y,d.z,
                                              F[a],NULL,
                                              d.xy.communicator,
                                              mpi,d.communicator);
    } else
      T=NULL;
    d.Deactivate();
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
    for(size_t a=start; a < stop; ++a) {
      (fftx->*Forward)(f[a]+offset,F[a],rx,W);
      if(T)
        T[a]->ilocalize1(F[a]);
    }
    if(T)
      for(size_t a=start; a < stop; ++a)
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
    if(T)
      for(size_t b=start; b < stop; ++b)
        T[b]->ilocalize0(F[b]);
    for(size_t b=start; b < stop; ++b) {
      if(T)
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

} // namespace fftwpp
