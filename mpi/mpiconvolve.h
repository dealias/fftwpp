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

// Return the minimum buffer size for the transpose
size_t bufferSize(fftBase *fftx, fftBase *ffty, const utils::split &d) {
  return utils::split(fftx->l*fftx->D,ffty->l*ffty->D,d.communicator).n;
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

  Convolution2MPI(fftBase *fftx, fftBase *ffty, const utils::split& d,
                  utils::mpiOptions mpi=utils::defaultmpiOptions,
                  Complex *W=NULL, Complex *V=NULL,
                  MPI_Comm global=0, bool toplevel=true) :
    // TODO: handle mpi options.
    Convolution2(fftx,ffty,
                 utils::ComplexAlign(fftx->app.A,bufferSize(fftx,ffty,d)),W,V),
    d(d) {
    inittranspose(mpi,NULL,global);
    allocateF=true;
  }

  Convolution2MPI(fftBase *fftx, fftBase *ffty, const utils::split& d,
                  Complex **F, utils::mpiOptions mpi=utils::defaultmpiOptions,
                  Complex *W=NULL, Complex *V=NULL,
                  MPI_Comm global=0, bool toplevel=true) :
    Convolution2(fftx,ffty,F,W,V), d(d) {
    inittranspose(mpi,NULL,global);
  }

  virtual ~Convolution2MPI() {
    size_t C=std::max(A,B);
    for(size_t a=0; a < C; ++a)
      delete T[a];
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
