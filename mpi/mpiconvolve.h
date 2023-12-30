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

  void init(size_t m, size_t D, size_t I, size_t l) {
    this->m=m;
    this->D=D;
    this->I=I;
    this->l=l;
  }
};

class params {
public:
  param x;
  param y;
};

// Return the output buffer decomposition
utils::split outputSplit(fftBase *fftx, fftBase *ffty, const MPI_Comm &communicator) {
  return utils::split(fftx->l*fftx->D,ffty->L,communicator);
}

// Return the minimum buffer size for the transpose
size_t bufferSize(fftBase *fftx, fftBase *ffty, const MPI_Comm &communicator) {
  return outputSplit(fftx,ffty,communicator).n;
}

// Allocate the outputBuffer;
Complex **outputBuffer(fftBase *fftx, fftBase *ffty, const MPI_Comm &communicator) {
  return utils::ComplexAlign(std::max(fftx->app.A,fftx->app.B),
                             bufferSize(fftx,ffty,communicator));
}

// In-place implicitly dealiased 2D complex convolution.
class Convolution2MPI : public Convolution2 {
protected:
  utils::split d;
  utils::mpitranspose<Complex> **T;
public:

  void inittranspose(const utils::mpiOptions& mpioptions, Complex *work,
                     MPI_Comm& global) {
    global=global ? global : d.communicator;
    size_t C=std::max(A,B);
    T=new utils::mpitranspose<Complex> *[C];
    for(size_t a=0; a < C; ++a)
      T[a]=new utils::mpitranspose<Complex>(d.X,d.Y,d.x,d.y,1,
                                            F[a],work,
                                            d.communicator,mpioptions,global);
    d.Deactivate();
  }

  Convolution2MPI(fftBase *fftx, fftBase *ffty,
                  const MPI_Comm& communicator,
                  utils::mpiOptions mpi=utils::defaultmpiOptions,
                  Complex **F=NULL, Complex *W=NULL, Complex *V=NULL,
                  MPI_Comm global=0) :
    Convolution2(fftx,ffty,F ? F : AllocateF(fftx,ffty,communicator),W,V),
    d(outputSplit(fftx,ffty,communicator)) {
    inittranspose(mpi,NULL,global);
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

} // namespace fftwpp
