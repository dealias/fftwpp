/* Implicitly dealiased convolution routines.
   Copyright (C) 2010-2015 John C. Bowman and Malcolm Roberts, Univ. of Alberta

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA. */

#include <iostream>

#include "Complex.h"
#include "fftw++.h"
#include "cmult-sse2.h"
#include "transposeoptions.h"

namespace fftwpp {

#ifndef __convolution_h__
#define __convolution_h__ 1

extern const double sqrt3;
extern const double hsqrt3;

extern const Complex hSqrt3;
extern const Complex mhsqrt3;
extern const Complex mhalf;
extern const Complex zeta3;
extern const double twopi;

inline size_t min(size_t a, size_t b)
{
  return (a < b) ? a : b;
}

inline size_t max(size_t a, size_t b)
{
  return (a > b) ? a : b;
}

// Build the factored zeta tables.
size_t BuildZeta(double arg, size_t m,
                       Complex *&ZetaH, Complex *&ZetaL,
                       size_t threads=1, size_t s=0);

size_t BuildZeta(size_t n, size_t m,
                       Complex *&ZetaH, Complex *&ZetaL,
                       size_t threads=1, size_t s=0);

struct convolveOptions {
  size_t nx,ny,nz;           // |
  size_t stride2,stride3;    // | Used internally by the MPI interface.
  utils::mpiOptions mpi;           // |
  bool toplevel;

  convolveOptions(size_t nx, size_t ny, size_t nz,
                  size_t stride2, size_t stride3) :
    nx(nx), ny(ny), nz(nz), stride2(stride2), stride3(stride3),
    toplevel(true) {}

  convolveOptions(size_t nx, size_t ny, size_t stride2,
                  utils::mpiOptions mpi, bool toplevel=true) :
    nx(nx), ny(ny), stride2(stride2), mpi(mpi), toplevel(toplevel) {}

  convolveOptions(size_t ny, size_t nz,
                  size_t stride2, size_t stride3,
                  utils::mpiOptions mpi, bool toplevel=true) :
    ny(ny), nz(nz), stride2(stride2), stride3(stride3), mpi(mpi),
    toplevel(toplevel) {}

  convolveOptions(bool toplevel=true) : nx(0), ny(0), nz(0),
                                        toplevel(toplevel) {}
};

static const convolveOptions defaultconvolveOptions;

typedef void multiplier(Complex **, size_t m,
                        const size_t indexsize,
                        const size_t *index,
                        size_t r, size_t threads);
typedef void realmultiplier(double **, size_t m,
                            const size_t indexsize,
                            const size_t *index,
                            size_t r, size_t threads);

// Multipliers for binary convolutions.

multiplier multautoconvolution;
multiplier multautocorrelation;
multiplier multbinary;
multiplier multcorrelation;
multiplier multbinary2;
multiplier multbinary3;
multiplier multbinary4;
multiplier multbinary8;

realmultiplier multbinary;
realmultiplier multbinary2;
realmultiplier multadvection2;

struct general {};
struct pretransform1 {};
struct pretransform2 {};
struct pretransform3 {};
struct pretransform4 {};

// In-place implicitly dealiased 1D complex convolution using
// function pointers for multiplication
class ImplicitConvolution : public ThreadBase {
private:
  size_t m;
  Complex **U;
  size_t A;
  size_t B;
  Complex *u;
  size_t s;
  Complex *ZetaH, *ZetaL;
  fft1d *BackwardsO,*ForwardsO;
  fft1d *Backwards,*Forwards;
  bool pointers;
  bool allocated;
  size_t indexsize;
public:
  size_t *index;

  void initpointers(Complex **&U, Complex *u) {
    size_t C=max(A,B);
    U=new Complex *[C];
    for(size_t a=0; a < C; ++a)
      U[a]=u+a*m;
    pointers=true;
  }

  void deletepointers(Complex **&U) {
    delete [] U;
  }

  void allocateindex(size_t n, size_t *i) {
    indexsize=n;
    index=i;
  }

  void init() {
    indexsize=0;

    Complex* U0=U[0];
    Complex* U1=A == 1 ? utils::ComplexAlign(m) : U[1];

    BackwardsO=new fft1d(m,1,U0,U1);
    ForwardsO=new fft1d(m,-1,U0,U1);
    threads=std::min(threads,max(BackwardsO->Threads(),ForwardsO->Threads()));

    if(A == B) {
      Backwards=new fft1d(m,1,U0);
      threads=std::min(threads,Backwards->Threads());
    }
    if(A <= B) {
      Forwards=new fft1d(m,-1,U0);
      threads=std::min(threads,Forwards->Threads());
    }

    if(A == 1) utils::deleteAlign(U1);

    s=BuildZeta(2*m,m,ZetaH,ZetaL,threads);
  }

  // m is the number of Complex data values.
  // U is an array of C distinct work arrays each of size m, where C=max(A,B)
  // A is the number of inputs.
  // B is the number of outputs.
  ImplicitConvolution(size_t m, Complex **U, size_t A=2,
                      size_t B=1,
                      size_t threads=fftw::maxthreads)
    : ThreadBase(threads), m(m), U(U), A(A), B(B), pointers(false),
      allocated(false) {
    init();
  }

  // m is the number of Complex data values.
  // u is a work array of C*m Complex values.
  // A is the number of inputs.
  // B is the number of outputs.
  ImplicitConvolution(size_t m, Complex *u,
                      size_t A=2, size_t B=1,
                      size_t threads=fftw::maxthreads)
    : ThreadBase(threads), m(m), A(A), B(B), u(u), allocated(false) {
    initpointers(U,u);
    init();
  }

  // m is the number of Complex data values.
  // A is the number of inputs.
  // B is the number of outputs.
  ImplicitConvolution(size_t m,
                      size_t A=2, size_t B=1,
                      size_t threads=fftw::maxthreads)
    : ThreadBase(threads), m(m), A(A), B(B), allocated(true) {
    u=utils::ComplexAlign(max(A,B)*m);
    initpointers(U,u);
    init();
  }

  ~ImplicitConvolution() {
    utils::deleteAlign(ZetaH);
    utils::deleteAlign(ZetaL);

    if(pointers) deletepointers(U);
    if(allocated) utils::deleteAlign(u);

    if(A == B)
      delete Backwards;
    if(A <= B)
      delete Forwards;

    delete ForwardsO;
    delete BackwardsO;
  }

  // F is an array of C pointers to distinct data blocks each of
  // size m, shifted by offset (contents not preserved).
  void convolve(Complex **F, multiplier *pmult, size_t i=0,
                size_t offset=0);

  void autoconvolve(Complex *f) {
    Complex *F[]={f};
    convolve(F,multautoconvolution);
  }

  void autocorrelate(Complex *f) {
    Complex *F[]={f};
    convolve(F,multautocorrelation);
  }

  // Binary convolution:
  void convolve(Complex *f, Complex *g) {
    Complex *F[]={f,g};
    convolve(F,multbinary);
  }

  // Binary correlation:
  void correlate(Complex *f, Complex *g) {
    Complex *F[]={f,g};
    convolve(F,multcorrelation);
  }

  template<class T>
  inline void pretransform(Complex **F, size_t k, Vec& Zetak);

  template<class T>
  void pretransform(Complex **F);

  void posttransform(Complex *f, Complex *u);
};

// In-place implicitly dealiased 1D Hermitian convolution.
class ImplicitHConvolution : public ThreadBase {
protected:
  size_t m;
  size_t c;
  bool compact;
  Complex **U;
  size_t A;
  size_t B;
  Complex *u;
  size_t s;
  Complex *ZetaH,*ZetaL;
  rcfft1d *rc,*rco,*rcO;
  crfft1d *cr,*cro,*crO;
  Complex *w; // Work array of size max(A,B) to hold f[c] in even case.
  bool pointers;
  bool allocated;
  bool even;
  size_t indexsize;
public:
  size_t *index;

  void initpointers(Complex **&U, Complex *u) {
    size_t C=max(A,B);
    U=new Complex *[C];
    size_t stride=c+1;
    for(size_t a=0; a < C; ++a)
      U[a]=u+a*stride;
    pointers=true;
  }

  void deletepointers(Complex **&U) {
    delete [] U;
  }

  void allocateindex(size_t n, size_t *i) {
    indexsize=n;
    index=i;
  }

  void init() {
    even=m == 2*c;
    indexsize=0;
    Complex* U0=U[0];

    rc=new rcfft1d(m,U0);
    cr=new crfft1d(m,U0);

    Complex* U1=A == 1 ? utils::ComplexAlign(m) : U[1];
    rco=new rcfft1d(m,(double *) U0,U1);
    cro=new crfft1d(m,U1,(double *) U0);
    if(A == 1) utils::deleteAlign(U1);

    if(A != B) {
      rcO=rco;
      crO=cro;
    } else {
      rcO=rc;
      crO=cr;
    }

    threads=std::min(threads,std::max(rco->Threads(),cro->Threads()));
    s=BuildZeta(3*m,c+2,ZetaH,ZetaL,threads);
    w=even ? utils::ComplexAlign(max(A,B)) : u;
  }

  // m is the number of independent data values
  // U is an array of max(A,B) distinct work arrays of size c+1, where c=m/2
  // A is the number of inputs.
  // B is the number of outputs.
  ImplicitHConvolution(size_t m, Complex **U, size_t A=2,
                       size_t B=1, size_t threads=fftw::maxthreads)
    : ThreadBase(threads), m(m), c(m/2), compact(true), U(U), A(A), B(B),
      pointers(false), allocated(false) {
    init();
  }

  ImplicitHConvolution(size_t m, bool compact, Complex **U,
                       size_t A=2, size_t B=1,
                       size_t threads=fftw::maxthreads)
    : ThreadBase(threads), m(m), c(m/2), compact(compact), U(U), A(A), B(B),
      pointers(false), allocated(false) {
    init();
  }

  // m is the number of independent data values
  // u is a work array of max(A,B)*(c+1) Complex values, where c=m/2
  // A is the number of inputs.
  // B is the number of outputs.
  ImplicitHConvolution(size_t m, Complex *u,
                       size_t A=2, size_t B=1,
                       size_t threads=fftw::maxthreads)
    : ThreadBase(threads), m(m), c(m/2), compact(true), A(A), B(B), u(u),
      allocated(false) {
    initpointers(U,u);
    init();
  }

  ImplicitHConvolution(size_t m, bool compact, Complex *u,
                       size_t A=2, size_t B=1,
                       size_t threads=fftw::maxthreads)
    : ThreadBase(threads), m(m), c(m/2), compact(compact), A(A), B(B), u(u),
      allocated(false) {
    initpointers(U,u);
    init();
  }

  // m is the number of independent data values
  // u is a work array of max(A,B)*(c+1) Complex values, where c=m/2
  // A is the number of inputs.
  // B is the number of outputs.
  ImplicitHConvolution(size_t m, bool compact=true, size_t A=2,
                       size_t B=1, size_t threads=fftw::maxthreads)
    : ThreadBase(threads), m(m), c(m/2), compact(compact), A(A), B(B),
      u(utils::ComplexAlign(max(A,B)*(c+1))), allocated(true) {
    initpointers(U,u);
    init();
  }

  virtual ~ImplicitHConvolution() {
    if(even) utils::deleteAlign(w);
    utils::deleteAlign(ZetaH);
    utils::deleteAlign(ZetaL);

    if(pointers) deletepointers(U);
    if(allocated) utils::deleteAlign(u);

    if(A != B) {
      delete cro;
      delete rco;
    }

    delete cr;
    delete rc;
  }

  // F is an array of A pointers to distinct data blocks each of size m,
  // shifted by offset (contents not preserved).
  void convolve(Complex **F, realmultiplier *pmult, size_t i=0,
                size_t offset=0);

  void pretransform(Complex *F, Complex *f1c, Complex *U);
  void posttransform(Complex *F, const Complex& f1c, Complex *U);

  // Binary convolution:
  void convolve(Complex *f, Complex *g) {
    Complex *F[]={f,g};
    convolve(F,multbinary);
  }
};


// Compute the scrambled implicitly m-padded complex Fourier transform of M
// complex vectors, each of length m.
// The arrays in and out (which may coincide), along with the array u, must
// be allocated as Complex[M*m].
//
//   fftpad fft(m,M,stride);
//   fft.backwards(in,u);
//   fft.forwards(in,u);
//
// Notes:
//   stride is the spacing between the elements of each Complex vector.
//
class fftpad {
  size_t m;
  size_t M;
  size_t stride;
  size_t s;
  Complex *ZetaH, *ZetaL;
  size_t threads;
public:
  mfft1d *Backwards;
  mfft1d *Forwards;

  fftpad(size_t m, size_t M,
         size_t stride, Complex *u=NULL,
         size_t Threads=fftw::maxthreads)
    : m(m), M(M), stride(stride), threads(Threads) {
    Backwards=new mfft1d(m,1,M,stride,1,u,NULL,threads);
    Forwards=new mfft1d(m,-1,M,stride,1,u,NULL,threads);

    threads=std::max(Backwards->Threads(),Forwards->Threads());

    s=BuildZeta(2*m,m,ZetaH,ZetaL,threads);
  }

  ~fftpad() {
    utils::deleteAlign(ZetaL);
    utils::deleteAlign(ZetaH);
    delete Forwards;
    delete Backwards;
  }

  void expand(Complex *f, Complex *u);
  void reduce(Complex *f, Complex *u);

  void backwards(Complex *f, Complex *u);
  void forwards(Complex *f, Complex *u);
};

// Compute the scrambled implicitly m-padded complex Fourier transform of M
// complex vectors, each of length 2m-1 with the origin at index m-1,
// containing physical data for wavenumbers -m+1 to m-1.
// The arrays in and out (which may coincide) must be allocated as
// Complex[M*(2m-1)]. The array u must be allocated as Complex[M*(m+1)].
//
//   fft0pad fft(m,M,stride,u);
//   fft.backwards(in,u);
//   fft.forwards(in,u);
//
// Notes:
//   stride is the spacing between the elements of each Complex vector.
//
class fft0pad {
protected:
  size_t m;
  size_t M;
  size_t s;
  size_t stride;
  Complex *ZetaH, *ZetaL;
  size_t threads;
public:
  mfft1d *Forwards;
  mfft1d *Backwards;

  fft0pad(size_t m, size_t M, size_t stride, Complex *u=NULL,
          size_t Threads=fftw::maxthreads)
    : m(m), M(M), stride(stride), threads(Threads) {
    Backwards=new mfft1d(m,1,M,stride,1,u,NULL,threads);
    Forwards=new mfft1d(m,-1,M,stride,1,u,NULL,threads);

    s=BuildZeta(3*m,m,ZetaH,ZetaL);
  }

  virtual ~fft0pad() {
    utils::deleteAlign(ZetaL);
    utils::deleteAlign(ZetaH);
    delete Forwards;
    delete Backwards;
  }

  // Unscramble indices, returning spatial index stored at position i
  inline static size_t findex(size_t i, size_t m) {
    return i < m-1 ? 3*i : 3*i+4-3*m; // for i >= m-1: j=3*(i-(m-1))+1
  }

  inline static size_t uindex(size_t i, size_t m) {
    return i > 0 ? (i < m ? 3*i-1 : 3*m-3) : 3*m-1;
  }

  virtual void expand(Complex *f, Complex *u);
  virtual void reduce(Complex *f, Complex *u);

  void backwards(Complex *f, Complex *u);
  virtual void forwards(Complex *f, Complex *u);

  virtual void Backwards1(Complex *f, Complex *u);
  virtual void Forwards0(Complex *f);
  virtual void Forwards1(Complex *f, Complex *u);
};

// Compute the scrambled implicitly m-padded complex Fourier transform of M
// complex vectors, each of length 2m with the origin at index m,
// corresponding to wavenumbers -m to m-1.
// The arrays in and out (which may coincide) must be allocated as
// Complex[M*2m]. The array u must be allocated as Complex[M*m].
//
//   fft1pad fft(m,M,stride,u);
//   fft.backwards(in,u);
//   fft.forwards(in,u);
//
// Notes:
//   stride is the spacing between the elements of each Complex vector.
//
class fft1pad : public fft0pad {
public:
  fft1pad(size_t m, size_t M, size_t stride,
          Complex *u=NULL, size_t threads=fftw::maxthreads) :
    fft0pad(m,M,stride,u,threads) {}

  // Unscramble indices, returning spatial index stored at position i
  inline static size_t findex(size_t i, size_t m) {
    return i < m ? 3*i : 3*(i-m)+1;
  }

  inline static size_t uindex(size_t i, size_t m) {
    return i > 0 ? 3*i-1 : 3*m-1;
  }

  void expand(Complex *f, Complex *u);
  void reduce(Complex *f, Complex *u);

  void forwards(Complex *f, Complex *u);

  void Backwards1(Complex *f, Complex *u);
  void Forwards0(Complex *f);
  void Forwards1(Complex *f, Complex *u);
};

// In-place implicitly dealiased 2D complex convolution.
class ImplicitConvolution2 : public ThreadBase {
protected:
  size_t mx,my;
  Complex *u1;
  Complex *u2;
  size_t A,B;
  fftpad *xfftpad;
  ImplicitConvolution **yconvolve;
  Complex **U2;
  bool allocated;
  size_t indexsize;
  bool toplevel;
public:
  size_t *index;

  void initpointers2(Complex **&U2, Complex *u2, size_t stride) {
    U2=new Complex *[A];
    for(size_t a=0; a < A; ++a)
      U2[a]=u2+a*stride;

    if(toplevel) allocateindex(1,new size_t[1]);
  }

  void deletepointers2(Complex **&U2) {
    if(toplevel) {
      delete [] index;

      for(size_t t=1; t < threads; ++t)
        delete [] yconvolve[t]->index;
    }

    delete [] U2;
  }

  void allocateindex(size_t n, size_t *i) {
    indexsize=n;
    index=i;
    yconvolve[0]->allocateindex(n,i);
    for(size_t t=1; t < threads; ++t)
      yconvolve[t]->allocateindex(n,new size_t[n]);
  }

  void init(const convolveOptions& options) {
    toplevel=options.toplevel;
    xfftpad=new fftpad(mx,options.ny,options.ny,u2,threads);
    size_t C=max(A,B);
    yconvolve=new ImplicitConvolution*[threads];
    for(size_t t=0; t < threads; ++t)
      yconvolve[t]=new ImplicitConvolution(my,u1+t*my*C,A,B,innerthreads);
    initpointers2(U2,u2,options.stride2);
  }

  void set(convolveOptions& options) {
    if(options.nx == 0) options.nx=mx;
    if(options.ny == 0) {
      options.ny=my;
      options.stride2=mx*my;
    }
  }

  // u1 is a temporary array of size my*C*threads.
  // u2 is a temporary array of size mx*my*C.
  // A is the number of inputs.
  // B is the number of outputs.
  // Here C=max(A,B).
  ImplicitConvolution2(size_t mx, size_t my,
                       Complex *u1, Complex *u2,
                       size_t A=2, size_t B=1,
                       size_t threads=fftw::maxthreads,
                       convolveOptions options=defaultconvolveOptions) :
    ThreadBase(threads), mx(mx), my(my), u1(u1), u2(u2), A(A), B(B),
    allocated(false) {
    set(options);
    multithread(options.nx);
    init(options);
  }

  ImplicitConvolution2(size_t mx, size_t my,
                       size_t A=2, size_t B=1,
                       size_t threads=fftw::maxthreads,
                       convolveOptions options=defaultconvolveOptions) :
    ThreadBase(threads), mx(mx), my(my), A(A), B(B), allocated(true) {
    set(options);
    multithread(options.nx);
    size_t C=max(A,B);
    u1=utils::ComplexAlign(my*C*threads);
    u2=utils::ComplexAlign(options.stride2*C);
    init(options);
  }

  virtual ~ImplicitConvolution2() {
    deletepointers2(U2);

    for(size_t t=0; t < threads; ++t)
      delete yconvolve[t];
    delete [] yconvolve;

    delete xfftpad;

    if(allocated) {
      utils::deleteAlign(u2);
      utils::deleteAlign(u1);
    }
  }

  void backwards(Complex **F, Complex **U2, size_t offset) {
    for(size_t a=0; a < A; ++a)
      xfftpad->backwards(F[a]+offset,U2[a]);
  }

  void subconvolution(Complex **F, multiplier *pmult,
                      size_t r, size_t M, size_t stride,
                      size_t offset=0) {
    if(threads > 1) {
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif
      for(size_t i=0; i < M; ++i)
        yconvolve[get_thread_num()]->convolve(F,pmult,2*i+r,offset+i*stride);
    } else {
      ImplicitConvolution *yconvolve0=yconvolve[0];
      for(size_t i=0; i < M; ++i)
        yconvolve0->convolve(F,pmult,2*i+r,offset+i*stride);
    }
  }

  void forwards(Complex **F, Complex **U2, size_t offset) {
    for(size_t b=0; b < B; ++b)
      xfftpad->forwards(F[b]+offset,U2[b]);
  }

  // F is a pointer to A distinct data blocks each of size mx*my,
  // shifted by offset (contents not preserved).
  virtual void convolve(Complex **F, multiplier *pmult, size_t i=0,
                        size_t offset=0) {
    if(!toplevel) {
      index[indexsize-2]=i;
      if(threads > 1) {
        for(size_t t=1; t < threads; ++t) {
          size_t *Index=yconvolve[t]->index;
          for(size_t i=0; i < indexsize; ++i)
            Index[i]=index[i];
        }
      }
    }
    backwards(F,U2,offset);
    subconvolution(F,pmult,0,mx,my,offset);
    subconvolution(U2,pmult,1,mx,my);
    forwards(F,U2,offset);
  }

  // Binary convolution:
  void convolve(Complex *f, Complex *g) {
    Complex *F[]={f,g};
    convolve(F,multbinary);
  }

  // Binary correlation:
  void correlate(Complex *f, Complex *g) {
    Complex *F[]={f, g};
    convolve(F,multcorrelation);
  }

  void autoconvolve(Complex *f) {
    Complex *F[]={f};
    convolve(F,multautoconvolution);
  }

  void autocorrelate(Complex *f) {
    Complex *F[]={f};
    convolve(F,multautocorrelation);
  }
};

inline void HermitianSymmetrizeX(size_t mx, size_t stride,
                                 size_t xorigin, Complex *f,
                                 int threads=fftw::maxthreads)
{
  Complex *F=f+xorigin*stride;
  F[0].im=0.0;
  PARALLELIF(
    mx > threshold,
  for(size_t i=1; i < mx; ++i) {
    size_t istride=i*stride;
    *(F-istride)=conj(F[istride]);
  });
}

// Enforce 3D Hermiticity using specified (x >= 0,y=0,z=0) and (x,y > 0,z=0).
// data.
inline void HermitianSymmetrizeXY(size_t Hx, size_t Hy,
                                  size_t Hz,
                                  size_t x0, size_t y0,
                                  Complex *f,
                                  size_t Sx, size_t Sy,
                                  size_t threads=fftw::maxthreads)
{
  size_t origin=x0*Sx+y0*Sy;
  Complex *F=f+origin;
  size_t stop=Hx*Sx;
  for(size_t i=Sx; i < stop; i += Sx)
    *(F-i)=conj(F[i]);

  F[0].im=0.0;

  PARALLELIF(
    2*Hx*Hy > threshold,
    for(int i=(-Hx+1)*Sx; i < (int) stop; i += Sx) {
      size_t m=origin-i;
      size_t p=origin+i;
      size_t Stop=Sy*Hy;
      for(size_t j=Sy; j < Stop; j += Sy) {
        f[m-j]=conj(f[p+j]);
      }
    });

  // Zero out Nyquist modes
  if(x0 == Hx) {
    size_t Ly=y0+Hy;
    PARALLELIF(
      Ly*Hz > threshold,
      for(size_t j=0; j < Ly; ++j) {
        for(size_t k=0; k < Hz; ++k) {
          f[Sy*j+k]=0.0;
        }
      });
  }

  if(y0 == Hy) {
    size_t Lx=x0+Hx;
    PARALLELIF(
      Lx*Hz > threshold,
      for(size_t i=0; i < Lx; ++i) {
        for(size_t k=0; k < Hz; ++k) {
          f[Sx*i+k]=0.0;
        }
      });
  }
}

inline void HermitianSymmetrizeXY(size_t Hx, size_t Hy,
                                  size_t Hz,
                                  size_t x0, size_t y0,
                                  Complex *f,
                                  size_t threads=fftw::maxthreads)
{
  size_t Ly=y0+Hy;
  HermitianSymmetrizeXY(Hx,Hy,Hz,x0,y0,f,Ly*Hz,Hz,threads);
}

typedef size_t IndexFunction(size_t, size_t m);

class ImplicitHConvolution2 : public ThreadBase {
protected:
  size_t mx,my;
  bool xcompact,ycompact;
  Complex *u1;
  Complex *u2;
  size_t A,B;
  fft0pad *xfftpad;
  ImplicitHConvolution **yconvolve;
  Complex **U2;
  bool allocated;
  size_t indexsize;
  bool toplevel;
public:
  size_t *index;

  void initpointers2(Complex **&U2, Complex *u2, size_t stride)
  {
    size_t C=max(A,B);
    U2=new Complex *[C];
    for(size_t a=0; a < C; ++a)
      U2[a]=u2+a*stride;

    if(toplevel) allocateindex(1,new size_t[1]);
  }

  void deletepointers2(Complex **&U2) {
    if(toplevel) {
      delete [] index;

      for(size_t t=1; t < threads; ++t)
        delete [] yconvolve[t]->index;
    }

    delete [] U2;
  }

  void allocateindex(size_t n, size_t *i) {
    indexsize=n;
    index=i;
    yconvolve[0]->allocateindex(n,i);
    for(size_t t=1; t < threads; ++t)
      yconvolve[t]->allocateindex(n,new size_t[n]);
  }

  void init(const convolveOptions& options) {
    size_t C=max(A,B);
    toplevel=options.toplevel;
    xfftpad=xcompact ? new fft0pad(mx,options.ny,options.ny,u2) :
      new fft1pad(mx,options.ny,options.ny,u2);

    yconvolve=new ImplicitHConvolution*[threads];
    for(size_t t=0; t < threads; ++t)
      yconvolve[t]=new ImplicitHConvolution(my,ycompact,u1+t*(my/2+1)*C,A,B,
                                            innerthreads);
    initpointers2(U2,u2,options.stride2);
  }

  void set(convolveOptions& options) {
    if(options.nx == 0) options.nx=mx;
    if(options.ny == 0) {
      options.ny=my+!ycompact;
      options.stride2=(mx+xcompact)*options.ny;
    }
  }

  // u1 is a temporary array of size (my/2+1)*C*threads.
  // u2 is a temporary array of size (mx+xcompact)*(my+!ycompact)*C;
  // A is the number of inputs.
  // B is the number of outputs.
  // Here C=max(A,B).
  ImplicitHConvolution2(size_t mx, size_t my,
                        Complex *u1, Complex *u2,
                        size_t A=2, size_t B=1,
                        size_t threads=fftw::maxthreads,
                        convolveOptions options=defaultconvolveOptions) :
    ThreadBase(threads), mx(mx), my(my), xcompact(true), ycompact(true),
    u1(u1), u2(u2), A(A), B(B), allocated(false) {
    set(options);
    multithread(options.nx);
    init(options);
  }

  ImplicitHConvolution2(size_t mx, size_t my,
                        bool xcompact, bool ycompact,
                        Complex *u1, Complex *u2,
                        size_t A=2, size_t B=1,
                        size_t threads=fftw::maxthreads,
                        convolveOptions options=defaultconvolveOptions) :
    ThreadBase(threads), mx(mx), my(my),
    xcompact(xcompact), ycompact(ycompact), u1(u1), u2(u2), A(A), B(B),
    allocated(false) {
    set(options);
    multithread(options.nx);
    init(options);
  }

  ImplicitHConvolution2(size_t mx, size_t my,
                        bool xcompact=true, bool ycompact=true,
                        size_t A=2, size_t B=1,
                        size_t threads=fftw::maxthreads,
                        convolveOptions options=defaultconvolveOptions) :
    ThreadBase(threads), mx(mx), my(my),
    xcompact(xcompact), ycompact(ycompact), A(A), B(B), allocated(true) {
    set(options);
    multithread(options.nx);
    size_t C=max(A,B);
    u1=utils::ComplexAlign((my/2+1)*C*threads);
    u2=utils::ComplexAlign(options.stride2*C);
    init(options);
  }

  virtual ~ImplicitHConvolution2() {
    deletepointers2(U2);

    for(size_t t=0; t < threads; ++t)
      delete yconvolve[t];
    delete [] yconvolve;

    delete xfftpad;

    if(allocated) {
      utils::deleteAlign(u2);
      utils::deleteAlign(u1);
    }
  }

  void backwards(Complex **F, Complex **U2, size_t ny,
                 bool symmetrize, size_t offset) {
    for(size_t a=0; a < A; ++a) {
      Complex *f=F[a]+offset;
      if(symmetrize)
        HermitianSymmetrizeX(mx,ny,mx-xcompact,f,threads);
      xfftpad->backwards(f,U2[a]);
    }
  }

  void subconvolution(Complex **F, realmultiplier *pmult,
                      IndexFunction indexfunction,
                      size_t M, size_t stride,
                      size_t offset=0) {
    if(threads > 1) {
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif
      for(size_t i=0; i < M; ++i)
        yconvolve[get_thread_num()]->convolve(F,pmult,indexfunction(i,mx),
                                              offset+i*stride);
    } else {
      ImplicitHConvolution *yconvolve0=yconvolve[0];
      for(size_t i=0; i < M; ++i)
        yconvolve0->convolve(F,pmult,indexfunction(i,mx),offset+i*stride);
    }
  }

  void forwards(Complex **F, Complex **U2, size_t offset) {
    for(size_t b=0; b < B; ++b)
      xfftpad->forwards(F[b]+offset,U2[b]);
  }

  // F is a pointer to A distinct data blocks each of size
  // (2mx-xcompact)*(my+!ycompact), shifted by offset (contents not preserved).
  virtual void convolve(Complex **F, realmultiplier *pmult,
                        bool symmetrize=true, size_t i=0,
                        size_t offset=0) {
    if(!toplevel) {
      index[indexsize-2]=i;
      if(threads > 1) {
        for(size_t t=1; t < threads; ++t) {
          size_t *Index=yconvolve[t]->index;
          for(size_t i=0; i < indexsize; ++i)
            Index[i]=index[i];
        }
      }
    }
    size_t stride=my+!ycompact;
    backwards(F,U2,stride,symmetrize,offset);
    subconvolution(F,pmult,xfftpad->findex,2*mx-xcompact,stride,offset);
    subconvolution(U2,pmult,xfftpad->uindex,mx+xcompact,stride);
    forwards(F,U2,offset);
  }

  // Binary convolution:
  void convolve(Complex *f, Complex *g, bool symmetrize=true) {
    Complex *F[]={f,g};
    convolve(F,multbinary,symmetrize);
  }
};

// In-place implicitly dealiased 3D complex convolution.
class ImplicitConvolution3 : public ThreadBase {
protected:
  size_t mx,my,mz;
  Complex *u1;
  Complex *u2;
  Complex *u3;
  size_t A,B;
  fftpad *xfftpad;
  ImplicitConvolution2 **yzconvolve;
  Complex **U3;
  bool allocated;
  size_t indexsize;
  bool toplevel;
public:
  size_t *index;

  void initpointers3(Complex **&U3, Complex *u3, size_t stride) {
    size_t C=max(A,B);
    U3=new Complex *[C];
    for(size_t a=0; a < C; ++a)
      U3[a]=u3+a*stride;

    if(toplevel) allocateindex(2,new size_t[2]);
  }

  void deletepointers3(Complex **&U3) {
    if(toplevel) {
      delete [] index;

      for(size_t t=1; t < threads; ++t)
        delete [] yzconvolve[t]->index;
    }

    delete [] U3;
  }

  void allocateindex(size_t n, size_t *i) {
    indexsize=n;
    index=i;
    yzconvolve[0]->allocateindex(n,i);
    for(size_t t=1; t < threads; ++t)
      yzconvolve[t]->allocateindex(n,new size_t[n]);
  }

  void init(const convolveOptions& options) {
    toplevel=options.toplevel;
    size_t nyz=options.ny*options.nz;
    xfftpad=new fftpad(mx,nyz,nyz,u3,threads);

    if(options.nz == mz) {
      size_t C=max(A,B);
      yzconvolve=new ImplicitConvolution2*[threads];
      for(size_t t=0; t < threads; ++t)
        yzconvolve[t]=new ImplicitConvolution2(my,mz,u1+t*mz*C*innerthreads,
                                               u2+t*options.stride2*C,A,B,
                                               innerthreads,false);
      initpointers3(U3,u3,options.stride3);
    } else yzconvolve=NULL;
  }

  void set(convolveOptions &options)
  {
    if(options.ny == 0) {
      options.ny=my;
      options.nz=mz;
      options.stride2=my*mz;
      options.stride3=mx*my*mz;
    }
  }

  // u1 is a temporary array of size mz*C*threads.
  // u2 is a temporary array of size my*mz*C*threads.
  // u3 is a temporary array of size mx*my*mz*C.
  // A is the number of inputs.
  // B is the number of outputs.
  // Here C=max(A,B).
  ImplicitConvolution3(size_t mx, size_t my, size_t mz,
                       Complex *u1, Complex *u2, Complex *u3,
                       size_t A=2, size_t B=1,
                       size_t threads=fftw::maxthreads,
                       convolveOptions options=defaultconvolveOptions) :
    ThreadBase(threads), mx(mx), my(my), mz(mz),
    u1(u1), u2(u2), u3(u3), A(A), B(B), allocated(false) {
    set(options);
    multithread(mx);
    init(options);
  }

  ImplicitConvolution3(size_t mx, size_t my, size_t mz,
                       size_t A=2, size_t B=1,
                       size_t threads=fftw::maxthreads,
                       convolveOptions options=defaultconvolveOptions) :
    ThreadBase(threads), mx(mx), my(my), mz(mz), A(A), B(B),
    allocated(true) {
    set(options);
    multithread(mx);
    size_t C=max(A,B);
    u1=utils::ComplexAlign(mz*C*threads*innerthreads);
    u2=utils::ComplexAlign(options.stride2*C*threads);
    u3=utils::ComplexAlign(options.stride3*C);
    init(options);
  }

  virtual ~ImplicitConvolution3() {
    if(yzconvolve) {
      deletepointers3(U3);

      for(size_t t=0; t < threads; ++t)
        delete yzconvolve[t];
      delete [] yzconvolve;
    }

    delete xfftpad;

    if(allocated) {
      utils::deleteAlign(u3);
      utils::deleteAlign(u2);
      utils::deleteAlign(u1);
    }
  }

  void backwards(Complex **F, Complex **U3, size_t offset) {
    for(size_t a=0; a < A; ++a)
      xfftpad->backwards(F[a]+offset,U3[a]);
  }

  void subconvolution(Complex **F, multiplier *pmult,
                      size_t r, size_t M, size_t stride,
                      size_t offset=0) {
    if(threads > 1) {
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif
      for(size_t i=0; i < M; ++i)
        yzconvolve[get_thread_num()]->convolve(F,pmult,2*i+r,offset+i*stride);
    } else {
      ImplicitConvolution2 *yzconvolve0=yzconvolve[0];
      for(size_t i=0; i < M; ++i) {
        yzconvolve0->convolve(F,pmult,2*i+r,offset+i*stride);
      }
    }
  }

  void forwards(Complex **F, Complex **U3, size_t offset=0) {
    for(size_t b=0; b < B; ++b)
      xfftpad->forwards(F[b]+offset,U3[b]);
  }

  // F is a pointer to A distinct data blocks each of size mx*my*mz,
  // shifted by offset
  virtual void convolve(Complex **F, multiplier *pmult, size_t i=0,
                        size_t offset=0)
  {
    if(!toplevel) {
      index[indexsize-3]=i;
      if(threads > 1) {
        for(size_t t=1; t < threads; ++t) {
          size_t *Index=yzconvolve[t]->index;
          for(size_t i=0; i < indexsize; ++i)
            Index[i]=index[i];
        }
      }
    }
    size_t stride=my*mz;
    backwards(F,U3,offset);
    subconvolution(F,pmult,0,mx,stride,offset);
    subconvolution(U3,pmult,1,mx,stride);
    forwards(F,U3,offset);
  }

  // Binary convolution:
  void convolve(Complex *f, Complex *g) {
    Complex *F[]={f,g};
    convolve(F,multbinary);
  }

  // Binary correlation:
  void correlate(Complex *f, Complex *g) {
    Complex *F[]={f, g};
    convolve(F,multcorrelation);
  }

  void autoconvolve(Complex *f) {
    Complex *F[]={f};
    convolve(F,multautoconvolution);
  }

  void autocorrelate(Complex *f) {
    Complex *F[]={f};
    convolve(F,multautocorrelation);
  }
};

// In-place implicitly dealiased 3D Hermitian convolution.
class ImplicitHConvolution3 : public ThreadBase {
protected:
  size_t mx,my,mz;
  bool xcompact,ycompact,zcompact;
  Complex *u1;
  Complex *u2;
  Complex *u3;
  size_t A,B;
  fft0pad *xfftpad;
  ImplicitHConvolution2 **yzconvolve;
  Complex **U3;
  bool allocated;
  size_t indexsize;
  bool toplevel;
public:
  size_t *index;

  void initpointers3(Complex **&U3, Complex *u3, size_t stride) {
    size_t C=max(A,B);
    U3=new Complex *[C];
    for(size_t a=0; a < C; ++a)
      U3[a]=u3+a*stride;

    if(toplevel) allocateindex(2,new size_t[2]);
  }

  void deletepointers3(Complex **&U3) {
    if(toplevel) {
      delete [] index;

      for(size_t t=1; t < threads; ++t)
        delete [] yzconvolve[t]->index;
    }

    delete [] U3;
  }

  void allocateindex(size_t n, size_t *i) {
    indexsize=n;
    index=i;
    yzconvolve[0]->allocateindex(n,i);
    for(size_t t=1; t < threads; ++t)
      yzconvolve[t]->allocateindex(n,new size_t[n]);
  }

  void init(const convolveOptions& options) {
    toplevel=options.toplevel;
    size_t nyz=options.ny*options.nz;
    xfftpad=xcompact ? new fft0pad(mx,nyz,nyz,u3) :
      new fft1pad(mx,nyz,nyz,u3);

    if(options.nz == mz+!zcompact) {
      size_t C=max(A,B);
      yzconvolve=new ImplicitHConvolution2*[threads];
      for(size_t t=0; t < threads; ++t)
        yzconvolve[t]=new ImplicitHConvolution2(my,mz,
                                                ycompact,zcompact,
                                                u1+t*(mz/2+1)*C*innerthreads,
                                                u2+t*options.stride2*C,
                                                A,B,innerthreads,false);
      initpointers3(U3,u3,options.stride3);
    } else yzconvolve=NULL;
  }

  void set(convolveOptions& options) {
    if(options.ny == 0) {
      options.ny=2*my-ycompact;
      options.nz=mz+!zcompact;
      options.stride2=(my+ycompact)*options.nz;
      options.stride3=(mx+xcompact)*options.ny*options.nz;
    }
  }

  // u1 is a temporary array of size (mz/2+1)*C*threads.
  // u2 is a temporary array of size (my+ycompact)*(mz+!zcompact)*C*threads.
  // u3 is a temporary array of size
  //                             (mx+xcompact)*(2my-ycompact)*(mz+!zcompact)*C.
  // A is the number of inputs.
  // B is the number of outputs.
  // Here C=max(A,B).
  ImplicitHConvolution3(size_t mx, size_t my, size_t mz,
                        Complex *u1, Complex *u2, Complex *u3,
                        size_t A=2, size_t B=1,
                        size_t threads=fftw::maxthreads,
                        convolveOptions options=defaultconvolveOptions) :
    ThreadBase(threads), mx(mx), my(my), mz(mz),
    xcompact(true), ycompact(true), zcompact(true), u1(u1), u2(u2), u3(u3),
    A(A), B(B),
    allocated(false) {
    set(options);
    multithread(mx);
    init(options);
  }

  ImplicitHConvolution3(size_t mx, size_t my, size_t mz,
                        bool xcompact, bool ycompact, bool zcompact,
                        Complex *u1, Complex *u2, Complex *u3,
                        size_t A=2, size_t B=1,
                        size_t threads=fftw::maxthreads,
                        convolveOptions options=defaultconvolveOptions) :
    ThreadBase(threads), mx(mx), my(my), mz(mz),
    xcompact(xcompact), ycompact(ycompact), zcompact(zcompact),
    u1(u1), u2(u2), u3(u3), A(A), B(B), allocated(false) {
    set(options);
    multithread(mx);
    init(options);
  }

  ImplicitHConvolution3(size_t mx, size_t my, size_t mz,
                        bool xcompact=true, bool ycompact=true,
                        bool zcompact=true,
                        size_t A=2, size_t B=1,
                        size_t threads=fftw::maxthreads,
                        convolveOptions options=defaultconvolveOptions) :
    ThreadBase(threads), mx(mx), my(my), mz(mz),
    xcompact(xcompact), ycompact(ycompact), zcompact(zcompact), A(A), B(B),
    allocated(true) {
    set(options);
    multithread(mx);
    size_t C=max(A,B);
    u1=utils::ComplexAlign((mz/2+1)*C*threads*innerthreads);
    u2=utils::ComplexAlign(options.stride2*C*threads);
    u3=utils::ComplexAlign(options.stride3*C);
    init(options);
  }

  virtual ~ImplicitHConvolution3() {
    if(yzconvolve) {
      deletepointers3(U3);

      for(size_t t=0; t < threads; ++t)
        delete yzconvolve[t];
      delete [] yzconvolve;
    }

    delete xfftpad;

    if(allocated) {
      utils::deleteAlign(u3);
      utils::deleteAlign(u2);
      utils::deleteAlign(u1);
    }
  }

  virtual void HermitianSymmetrize(Complex *f, Complex *u)
  {
    HermitianSymmetrizeXY(mx,my,mz+!zcompact,mx-xcompact,my-ycompact,f,
                          threads);
  }

  void backwards(Complex **F, Complex **U3, bool symmetrize,
                 size_t offset) {
    for(size_t a=0; a < A; ++a) {
      Complex *f=F[a]+offset;
      Complex *u=U3[a];
      if(symmetrize)
        HermitianSymmetrize(f,u);
      xfftpad->backwards(f,u);
    }
  }

  void subconvolution(Complex **F, realmultiplier *pmult,
                      IndexFunction indexfunction,
                      size_t M, size_t stride,
                      size_t offset=0) {
    if(threads > 1) {
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif
      for(size_t i=0; i < M; ++i)
        yzconvolve[get_thread_num()]->convolve(F,pmult,false,
                                               indexfunction(i,mx),
                                               offset+i*stride);
    } else {
      ImplicitHConvolution2 *yzconvolve0=yzconvolve[0];
      for(size_t i=0; i < M; ++i)
        yzconvolve0->convolve(F,pmult,false,indexfunction(i,mx),
                              offset+i*stride);
    }
  }

  void forwards(Complex **F, Complex **U3, size_t offset=0) {
    for(size_t b=0; b < B; ++b)
      xfftpad->forwards(F[b]+offset,U3[b]);
  }

  // F is a pointer to A distinct data blocks each of size
  // (2mx-compact)*(2my-ycompact)*(mz+!zcompact), shifted by offset
  // (contents not preserved).
  virtual void convolve(Complex **F, realmultiplier *pmult,
                        bool symmetrize=true, size_t i=0,
                        size_t offset=0) {
    if(!toplevel) {
      index[indexsize-3]=i;
      if(threads > 1) {
        for(size_t t=1; t < threads; ++t) {
          size_t *Index=yzconvolve[t]->index;
          for(size_t i=0; i < indexsize; ++i)
            Index[i]=index[i];
        }
      }
    }
    size_t stride=(2*my-ycompact)*(mz+!zcompact);
    backwards(F,U3,symmetrize,offset);
    subconvolution(F,pmult,xfftpad->findex,2*mx-xcompact,stride,offset);
    subconvolution(U3,pmult,xfftpad->uindex,mx+xcompact,stride);
    forwards(F,U3,offset);
  }

  // Binary convolution:
  void convolve(Complex *f, Complex *g, bool symmetrize=true) {
    Complex *F[]={f,g};
    convolve(F,multbinary,symmetrize);
  }
};

// In-place implicitly dealiased Hermitian ternary convolution.
class ImplicitHTConvolution : public ThreadBase {
protected:
  size_t m;
  Complex *u,*v,*w;
  size_t M;
  size_t s;
  rcfft1d *rc, *rco;
  crfft1d *cr, *cro;
  Complex *ZetaH, *ZetaL;
  Complex **W;
  bool allocated;
  size_t twom;
  size_t stride;
public:
  void initpointers(Complex **&W, Complex *w) {
    W=new Complex *[M];
    size_t m1=m+1;
    for(size_t s=0; s < M; ++s)
      W[s]=w+s*m1;
  }

  void deletepointers(Complex **&W) {
    delete [] W;
  }

  void init() {
    twom=2*m;
    stride=twom+2;

    rc=new rcfft1d(twom,u);
    cr=new crfft1d(twom,u);

    rco=new rcfft1d(twom,(double *) u,v);
    cro=new crfft1d(twom,v,(double *) u);

    threads=std::min(threads,std::max(rco->Threads(),cro->Threads()));

    s=BuildZeta(4*m,m,ZetaH,ZetaL,threads);

    initpointers(W,w);
  }

  // u, v, and w are distinct temporary arrays each of size (m+1)*M.
  ImplicitHTConvolution(size_t m, Complex *u, Complex *v,
                        Complex *w, size_t M=1) :
    m(m), u(u), v(v), w(w), M(M), allocated(false) {
    init();
  }

  ImplicitHTConvolution(size_t m, size_t M=1) :
    m(m), u(utils::ComplexAlign(m*M+M)), v(utils::ComplexAlign(m*M+M)),
    w(utils::ComplexAlign(m*M+M)), M(M), allocated(true) {
    init();
  }

  ~ImplicitHTConvolution() {
    deletepointers(W);

    if(allocated) {
      utils::deleteAlign(w);
      utils::deleteAlign(v);
      utils::deleteAlign(u);
    }
    utils::deleteAlign(ZetaL);
    utils::deleteAlign(ZetaH);
    delete cro;
    delete rco;
    delete cr;
    delete rc;
  }

  void mult(double *a, double *b, double **C, size_t offset=0);

  void convolve(Complex **F, Complex **G, Complex **H,
                Complex *u, Complex *v, Complex **W,
                size_t offset=0);

  // F, G, and H are distinct pointers to M distinct data blocks each of size
  // m+1, shifted by offset (contents not preserved).
  // The output is returned in F[0].
  void convolve(Complex **F, Complex **G, Complex **H, size_t offset=0) {
    convolve(F,G,H,u,v,W,offset);
  }

  // Constructor for special case M=1:
  void convolve(Complex *f, Complex *g, Complex *h) {
    convolve(&f,&g,&h);
  }
};

// In-place implicitly dealiased Hermitian ternary convolution.
// Special case G=H, M=1.
class ImplicitHFGGConvolution : public ThreadBase {
protected:
  size_t m;
  Complex *u,*v;
  size_t s;
  rcfft1d *rc, *rco;
  crfft1d *cr, *cro;
  Complex *ZetaH, *ZetaL;
  bool allocated;
  size_t twom;
  size_t stride;
public:
  void init() {
    twom=2*m;
    stride=twom+2;

    rc=new rcfft1d(twom,u);
    cr=new crfft1d(twom,u);

    rco=new rcfft1d(twom,(double *) u,v);
    cro=new crfft1d(twom,v,(double *) u);

    threads=std::min(threads,std::max(rco->Threads(),cro->Threads()));

    s=BuildZeta(4*m,m,ZetaH,ZetaL,threads);
  }

  // u and v are distinct temporary arrays each of size m+1.
  ImplicitHFGGConvolution(size_t m, Complex *u, Complex *v) :
    m(m), u(u), v(v), allocated(false) {
    init();
  }

  ImplicitHFGGConvolution(size_t m) :
    m(m), u(utils::ComplexAlign(m+1)), v(utils::ComplexAlign(m+1)),
    allocated(true) {
    init();
  }

  ~ImplicitHFGGConvolution() {
    if(allocated) {
      utils::deleteAlign(v);
      utils::deleteAlign(u);
    }
    utils::deleteAlign(ZetaL);
    utils::deleteAlign(ZetaH);
    delete cro;
    delete rco;
    delete cr;
    delete rc;
  }

  void mult(double *a, double *b);

  void convolve(Complex *f, Complex *g, Complex *u, Complex *v);

  // f and g are distinct pointers to data of size m+1 (contents not
  // preserved). The output is returned in f.
  void convolve(Complex *f, Complex *g) {
    convolve(f,g,u,v);
  }
};

// In-place implicitly dealiased Hermitian ternary convolution.
// Special case F=G=H, M=1.
class ImplicitHFFFConvolution : public ThreadBase {
protected:
  size_t m;
  Complex *u;
  size_t s;
  rcfft1d *rc;
  crfft1d *cr;
  Complex *ZetaH, *ZetaL;
  bool allocated;
  size_t twom;
  size_t stride;
public:
  void mult(double *a);

  void init() {
    twom=2*m;
    stride=twom+2;

    rc=new rcfft1d(twom,u);
    cr=new crfft1d(twom,u);

    threads=std::min(threads,std::max(rc->Threads(),cr->Threads()));

    s=BuildZeta(4*m,m,ZetaH,ZetaL,threads);
  }

  // u is a distinct temporary array of size m+1.
  ImplicitHFFFConvolution(size_t m, Complex *u) :
    m(m), u(u), allocated(false) {
    init();
  }

  ImplicitHFFFConvolution(size_t m) :
    m(m), u(utils::ComplexAlign(m+1)), allocated(true) {
    init();
  }

  ~ImplicitHFFFConvolution() {
    if(allocated)
      utils::deleteAlign(u);

    utils::deleteAlign(ZetaL);
    utils::deleteAlign(ZetaH);
    delete cr;
    delete rc;
  }

  void convolve(Complex *f, Complex *u);
  // f is a pointer to data of size m+1 (contents not preserved).
  // The output is returned in f.
  void convolve(Complex *f) {
    convolve(f,u);
  }
};

// Compute the scrambled implicitly 2m-padded complex Fourier transform of M
// complex vectors, each of length 2m with the Fourier origin at index m.
// The arrays in and out (which may coincide), along
// with the array u, must be allocated as Complex[M*2m].
//
//   fft0bipad fft(m,M,stride);
//   fft.backwards(in,u);
//   fft.forwards(in,u);
//
// Notes:
//   stride is the spacing between the elements of each Complex vector.
//
class fft0bipad {
  size_t m;
  size_t M;
  size_t stride;
  size_t s;
  mfft1d *Backwards;
  mfft1d *Forwards;
  Complex *ZetaH, *ZetaL;
  size_t threads;
public:
  fft0bipad(size_t m, size_t M, size_t stride,
            Complex *f, size_t Threads=fftw::maxthreads) :
    m(m), M(M), stride(stride), threads(Threads) {
    size_t twom=2*m;
    Backwards=new mfft1d(twom,1,M,stride,1,f,NULL,threads);
    Forwards=new mfft1d(twom,-1,M,stride,1,f,NULL,threads);

    threads=std::min(threads,
                     std::max(Backwards->Threads(),Forwards->Threads()));

    s=BuildZeta(4*m,twom,ZetaH,ZetaL,threads);
  }

  ~fft0bipad() {
    utils::deleteAlign(ZetaL);
    utils::deleteAlign(ZetaH);
    delete Forwards;
    delete Backwards;
  }

  void backwards(Complex *f, Complex *u);
  void forwards(Complex *f, Complex *u);
};

// In-place implicitly dealiased 2D Hermitian ternary convolution.
class ImplicitHTConvolution2 : public ThreadBase {
protected:
  size_t mx,my;
  Complex *u1,*v1,*w1;
  Complex *u2,*v2,*w2;
  size_t M;
  fft0bipad *xfftpad;
  ImplicitHTConvolution *yconvolve;
  Complex **U2,**V2,**W2;
  bool allocated;
  Complex **u,**v;
  Complex ***W;
public:
  void initpointers(Complex **&u, Complex **&v, Complex ***&W,
                    size_t threads) {
    u=new Complex *[threads];
    v=new Complex *[threads];
    W=new Complex **[threads];
    size_t my1M=(my+1)*M;
    for(size_t i=0; i < threads; ++i) {
      size_t imy1M=i*my1M;
      u[i]=u1+imy1M;
      v[i]=v1+imy1M;
      Complex *wi=w1+imy1M;
      yconvolve->initpointers(W[i],wi);
    }
  }

  void deletepointers(Complex **&u, Complex **&v, Complex ***&W,
                      size_t threads) {
    for(size_t i=0; i < threads; ++i)
      yconvolve->deletepointers(W[i]);
    delete [] W;
    delete [] v;
    delete [] u;
  }

  void initpointers(Complex **&U2, Complex **&V2, Complex **&W2,
                    Complex *u2, Complex *v2, Complex *w2) {
    U2=new Complex *[M];
    V2=new Complex *[M];
    W2=new Complex *[M];
    size_t mu=2*mx*(my+1);
    for(size_t s=0; s < M; ++s) {
      size_t smu=s*mu;
      U2[s]=u2+smu;
      V2[s]=v2+smu;
      W2[s]=w2+smu;
    }
  }

  void deletepointers(Complex **&U2, Complex **&V2, Complex **&W2) {
    delete [] W2;
    delete [] V2;
    delete [] U2;
  }

  void init() {
    xfftpad=new fft0bipad(mx,my,my+1,u2,threads);

    yconvolve=new ImplicitHTConvolution(my,u1,v1,w1,M);
    yconvolve->Threads(1);

    initpointers(u,v,W,threads);
    initpointers(U2,V2,W2,u2,v2,w2);
  }

  // u1, v1, and w1 are temporary arrays of size (my+1)*M*threads;
  // u2, v2, and w2 are temporary arrays of size 2mx*(my+1)*M.
  // M is the number of data blocks (each corresponding to a dot product term).
  // threads is the number of threads to use in the outer subconvolution loop.
  ImplicitHTConvolution2(size_t mx, size_t my,
                         Complex *u1, Complex *v1, Complex *w1,
                         Complex *u2, Complex *v2, Complex *w2,
                         size_t M=1,
                         size_t threads=fftw::maxthreads) :
    ThreadBase(threads), mx(mx), my(my), u1(u1), v1(v1), w1(w1),
    u2(u2), v2(v2), w2(w2), M(M), allocated(false) {
    init();
  }

  ImplicitHTConvolution2(size_t mx, size_t my,
                         size_t M=1,
                         size_t threads=fftw::maxthreads) :
    ThreadBase(threads), mx(mx), my(my),
    u1(utils::ComplexAlign((my+1)*M*threads)),
    v1(utils::ComplexAlign((my+1)*M*threads)),
    w1(utils::ComplexAlign((my+1)*M*threads)),
    u2(utils::ComplexAlign(2*mx*(my+1)*M)),
    v2(utils::ComplexAlign(2*mx*(my+1)*M)),
    w2(utils::ComplexAlign(2*mx*(my+1)*M)),
    M(M), allocated(true) {
    init();
  }

  ~ImplicitHTConvolution2() {
    deletepointers(U2,V2,W2);
    deletepointers(u,v,W,threads);

    delete yconvolve;
    delete xfftpad;

    if(allocated) {
      utils::deleteAlign(w2);
      utils::deleteAlign(v2);
      utils::deleteAlign(u2);
      utils::deleteAlign(w1);
      utils::deleteAlign(v1);
      utils::deleteAlign(u1);
    }
  }

  void convolve(Complex **F, Complex **G, Complex **H,
                Complex **u, Complex **v, Complex ***W,
                Complex **U2, Complex **V2, Complex **W2,
                bool symmetrize=true, size_t offset=0) {
    Complex *u2=U2[0];
    Complex *v2=V2[0];
    Complex *w2=W2[0];

    size_t my1=my+1;
    size_t mu=2*mx*my1;

    for(size_t s=0; s < M; ++s) {
      Complex *f=F[s]+offset;
      if(symmetrize)
        HermitianSymmetrizeX(mx,my1,mx,f,threads);
      xfftpad->backwards(f,u2+s*mu);
    }

    for(size_t s=0; s < M; ++s) {
      Complex *g=G[s]+offset;
      if(symmetrize)
        HermitianSymmetrizeX(mx,my1,mx,g,threads);
      xfftpad->backwards(g,v2+s*mu);
    }

    for(size_t s=0; s < M; ++s) {
      Complex *h=H[s]+offset;
      if(symmetrize)
        HermitianSymmetrizeX(mx,my1,mx,h,threads);
      xfftpad->backwards(h,w2+s*mu);
    }

#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif
    for(size_t i=0; i < mu; i += my1) {
      size_t thread=get_thread_num();
      yconvolve->convolve(F,G,H,u[thread],v[thread],W[thread],i+offset);
    }

#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif
    for(size_t i=0; i < mu; i += my1) {
      size_t thread=get_thread_num();
      yconvolve->convolve(U2,V2,W2,u[thread],v[thread],W[thread],i+offset);
    }

    xfftpad->forwards(F[0]+offset,u2);
  }

  // F, G, and H are distinct pointers to M distinct data blocks each of size
  // 2mx*(my+1), shifted by offset (contents not preserved).
  // The output is returned in F[0].
  void convolve(Complex **F, Complex **G, Complex **H, bool symmetrize=true,
                size_t offset=0) {
    convolve(F,G,H,u,v,W,U2,V2,W2,symmetrize,offset);
  }

  // Constructor for special case M=1:
  void convolve(Complex *f, Complex *g, Complex *h, bool symmetrize=true) {
    convolve(&f,&g,&h,symmetrize);
  }
};

// In-place implicitly dealiased 2D Hermitian ternary convolution.
// Special case G=H, M=1.
class ImplicitHFGGConvolution2 : public ThreadBase {
protected:
  size_t mx,my;
  Complex *u1,*v1;
  Complex *u2,*v2;
  fft0bipad *xfftpad;
  ImplicitHFGGConvolution *yconvolve;
  bool allocated;
  Complex **u,**v;
public:
  void initpointers(Complex **&u, Complex **&v, size_t threads) {
    u=new Complex *[threads];
    v=new Complex *[threads];
    size_t my1=my+1;
    for(size_t i=0; i < threads; ++i) {
      size_t imy1=i*my1;
      u[i]=u1+imy1;
      v[i]=v1+imy1;
    }
  }

  void deletepointers(Complex **&u, Complex **&v) {
    delete [] v;
    delete [] u;
  }

  void init() {
    xfftpad=new fft0bipad(mx,my,my+1,u2,threads);

    yconvolve=new ImplicitHFGGConvolution(my,u1,v1);
    yconvolve->Threads(1);

    initpointers(u,v,threads);
  }

  // u1 and v1 are temporary arrays of size (my+1)*threads.
  // u2 and v2 are temporary arrays of size 2mx*(my+1).
  // threads is the number of threads to use in the outer subconvolution loop.
  ImplicitHFGGConvolution2(size_t mx, size_t my,
                           Complex *u1, Complex *v1,
                           Complex *u2, Complex *v2,
                           size_t threads=fftw::maxthreads) :
    ThreadBase(threads), mx(mx), my(my), u1(u1), v1(v1), u2(u2), v2(v2),
    allocated(false) {
    init();
  }

  ImplicitHFGGConvolution2(size_t mx, size_t my,
                           size_t threads=fftw::maxthreads) :
    ThreadBase(threads), mx(mx), my(my),
    u1(utils::ComplexAlign((my+1)*threads)),
    v1(utils::ComplexAlign((my+1)*threads)),
    u2(utils::ComplexAlign(2*mx*(my+1))),
    v2(utils::ComplexAlign(2*mx*(my+1))),
    allocated(true) {
    init();
  }

  ~ImplicitHFGGConvolution2() {
    deletepointers(u,v);

    delete yconvolve;
    delete xfftpad;

    if(allocated) {
      utils::deleteAlign(v2);
      utils::deleteAlign(u2);
      utils::deleteAlign(v1);
      utils::deleteAlign(u1);
    }
  }

  void convolve(Complex *f, Complex *g,
                Complex **u, Complex **v,
                Complex *u2, Complex *v2, bool symmetrize=true) {
    size_t my1=my+1;
    size_t mu=2*mx*my1;

    if(symmetrize)
      HermitianSymmetrizeX(mx,my1,mx,f,threads);
    xfftpad->backwards(f,u2);

    if(symmetrize)
      HermitianSymmetrizeX(mx,my1,mx,g,threads);
    xfftpad->backwards(g,v2);

#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif
    for(size_t i=0; i < mu; i += my1) {
      size_t thread=get_thread_num();
      yconvolve->convolve(f+i,g+i,u[thread],v[thread]);
    }

#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif
    for(size_t i=0; i < mu; i += my1) {
      size_t thread=get_thread_num();
      yconvolve->convolve(u2+i,v2+i,u[thread],v[thread]);
    }

    xfftpad->forwards(f,u2);
  }

  void convolve(Complex *f, Complex *g, bool symmetrize=true) {
    convolve(f,g,u,v,u2,v2,symmetrize);
  }
};

// In-place implicitly dealiased 2D Hermitian ternary convolution.
// Special case F=G=H, M=1.
class ImplicitHFFFConvolution2 : public ThreadBase {
protected:
  size_t mx,my;
  Complex *u1;
  Complex *u2;
  fft0bipad *xfftpad;
  ImplicitHFFFConvolution *yconvolve;
  bool allocated;
  Complex **u;
public:
  void initpointers(Complex **&u, size_t threads) {
    u=new Complex *[threads];
    size_t my1=my+1;
    for(size_t i=0; i < threads; ++i)
      u[i]=u1+i*my1;
  }

  void deletepointers(Complex **&u) {
    delete [] u;
  }

  void init() {
    xfftpad=new fft0bipad(mx,my,my+1,u2,threads);

    yconvolve=new ImplicitHFFFConvolution(my,u1);
    yconvolve->Threads(1);
    initpointers(u,threads);
  }

  // u1 is a temporary array of size (my+1)*threads.
  // u2 is a temporary array of size 2mx*(my+1).
  // threads is the number of threads to use in the outer subconvolution loop.
  ImplicitHFFFConvolution2(size_t mx, size_t my,
                           Complex *u1, Complex *u2,
                           size_t threads=fftw::maxthreads) :
    ThreadBase(threads), mx(mx), my(my),
    u1(u1), u2(u2), allocated(false) {
    init();
  }

  ImplicitHFFFConvolution2(size_t mx, size_t my,
                           size_t threads=fftw::maxthreads) :
    ThreadBase(threads), mx(mx), my(my),
    u1(utils::ComplexAlign((my+1)*threads)),
    u2(utils::ComplexAlign(2*mx*(my+1))),
    allocated(true) {
    init();
  }

  ~ImplicitHFFFConvolution2() {
    deletepointers(u);
    delete yconvolve;
    delete xfftpad;

    if(allocated) {
      utils::deleteAlign(u2);
      utils::deleteAlign(u1);
    }
  }

  void convolve(Complex *f, Complex **u, Complex *u2, bool symmetrize=true) {
    size_t my1=my+1;
    size_t mu=2*mx*my1;

    if(symmetrize)
      HermitianSymmetrizeX(mx,my1,mx,f,threads);
    xfftpad->backwards(f,u2);

#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif
    for(size_t i=0; i < mu; i += my1)
      yconvolve->convolve(f+i,u[get_thread_num()]);

#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif
    for(size_t i=0; i < mu; i += my1)
      yconvolve->convolve(u2+i,u[get_thread_num()]);

    xfftpad->forwards(f,u2);
  }

  void convolve(Complex *f, bool symmetrize=true) {
    convolve(f,u,u2,symmetrize);
  }
};

} //end namespace fftwpp

#endif
