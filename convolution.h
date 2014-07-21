/* Implicitly dealiased convolution routines.
   Copyright (C) 2010-2012 John C. Bowman and Malcolm Roberts, Univ. of Alberta

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

#include "Complex.h"
#include "fftw++.h"
#include "cmult-sse2.h"

#ifndef FFTWPP_SINGLE_THREAD
#define PARALLEL(code)                                  \
  if(threads > 1) {                                     \
    _Pragma("omp parallel for num_threads(threads)")    \
      code                                              \
      } else {                                          \
    code                                                \
      }
#else
#define PARALLEL(code)                          \
  {                                             \
    code                                        \
      }
#endif

namespace fftwpp {

#ifndef __convolution_h__
#define __convolution_h__ 1

extern const double sqrt3;
extern const double hsqrt3;

extern const Complex hSqrt3;
extern const Complex mhsqrt3;
extern const Complex mhalf;
extern const Complex zeta3;

inline unsigned int min(unsigned int a, unsigned int b)
{
  return (a < b) ? a : b;
}

inline unsigned int max(unsigned int a, unsigned int b)
{
  return (a > b) ? a : b;
}

// Build the factored zeta tables.
unsigned int BuildZeta(unsigned int n, unsigned int m,
                       Complex *&ZetaH, Complex *&ZetaL,
                       unsigned int threads=1);

class ThreadBase
{
protected:
  unsigned int threads;
  unsigned int innerthreads;
public:  
  ThreadBase() {threads=fftw::maxthreads;}
  ThreadBase(unsigned int threads) : threads(threads) {}
  void Threads(unsigned int nthreads) {threads=nthreads;}
  unsigned int Threads() {return threads;}
  
  void multithread(unsigned int nx) {
    if(nx >= threads) {
      innerthreads=1;
    } else {
      innerthreads=threads;
      threads=1;
    }
  }
  
};

#ifndef __SSE2__
#define Vec Complex
#endif

typedef void multiplier(Complex **, unsigned int m, unsigned int threads); 
typedef void realmultiplier(double **, unsigned int m, unsigned int threads); 
  
// Sample multiplier for binary convolutions for use with
// function-pointer convolutions.
multiplier mult_autoconvolution;
multiplier mult_autocorrelation;
multiplier multbinary;
multiplier multbinary2;
multiplier multbinary3;
multiplier multbinary4;
multiplier multbinary8;

realmultiplier multbinary;
realmultiplier multbinary2;

struct general {};
struct premult1 {};
struct premult2 {};
struct premult3 {};
struct premult4 {};

// In-place implicitly dealiased 1D complex convolution using
// function pointers for multiplication
class ImplicitConvolution : public ThreadBase {
private:
  unsigned int m;
  Complex **U;
  unsigned int A;
  unsigned int B;
  Complex *u;
  unsigned int s;
  Complex *ZetaH, *ZetaL;
  fft1d *BackwardsO,*ForwardsO;
  fft1d *Backwards,*Forwards;
  bool pointers;
  bool allocated;
  bool out_of_place;
public:
  void initpointers(Complex **&U, Complex *u) {
    unsigned int C=max(A,B);
    U=new Complex *[C];
    for(unsigned int a=0; a < C; ++a) 
      U[a]=u+a*m;
    pointers=true;
  }
  
  void deletepointers(Complex **&U) {
    delete [] U;
  }

  void init() {
    out_of_place = A < B;

    Complex* U0=U[0];
    Complex* U1=A == 1 ? ComplexAlign(m) : U[1];
    
    BackwardsO=new fft1d(m,1,U0,U1);
    Backwards=new fft1d(m,1,U0);
    Forwards=new fft1d(m,-1,U0);
    if(out_of_place) {
      threads=std::min(threads,
                       std::max(Backwards->Threads(),Forwards->Threads()));
    } else {
      ForwardsO=new fft1d(m,-1,U0,U1);
      threads=std::min(threads,
                       std::max(BackwardsO->Threads(),ForwardsO->Threads()));
    }
    
    if(A == 1) deleteAlign(U1);

    s=BuildZeta(2*m,m,ZetaH,ZetaL,threads);
  }
  
  // m is the number of Complex data values.
  // U is an array of C distinct work arrays each of size m, where C=max(A,B)
  // A is the number of inputs.
  // B is the number of outputs.
  ImplicitConvolution(unsigned int m, Complex **U, unsigned int A=2,
                      unsigned int B=1,
                      unsigned int threads=fftw::maxthreads)
    : ThreadBase(threads), m(m), U(U), A(A), B(B), pointers(false),
      allocated(false) {
    init();
  }
  
  // m is the number of Complex data values.
  // u is a work array of C*m Complex values.
  // A is the number of inputs.
  // B is the number of outputs.
  ImplicitConvolution(unsigned int m, Complex *u,
                      unsigned int A=2, unsigned int B=1,
                      unsigned int threads=fftw::maxthreads)
    : ThreadBase(threads), m(m), A(A), B(B), u(u), allocated(false) {
    initpointers(U,u);
    init();
  }
  
  // m is the number of Complex data values.
  // A is the number of inputs.
  // B is the number of outputs.
  ImplicitConvolution(unsigned int m,
		      unsigned int A=2, unsigned int B=1,
		      unsigned int threads=fftw::maxthreads)
    : ThreadBase(threads), m(m), A(A), B(B), allocated(true) {
    u=ComplexAlign(max(A,B)*m);
    initpointers(U,u);
    init();
  }
 
  ~ImplicitConvolution() {
    deleteAlign(ZetaH);
    deleteAlign(ZetaL);
    
    if(pointers) deletepointers(U);
    if(allocated) deleteAlign(u);
    
    delete ForwardsO;
    delete BackwardsO;    
    if(out_of_place) {
      delete Forwards;
      delete Backwards;
    }
    
  }

  // F is an array of A pointers to distinct data blocks each of size m,
  // shifted by offset (contents not preserved).
  void convolve(Complex **F, multiplier *pmult, unsigned int offset=0);
  
  void autoconvolve(Complex *f) {
    Complex *F[]={f};
    convolve(F,mult_autoconvolution);
  }

  void autocorrelate(Complex *f) {
    Complex *F[]={f};
    convolve(F,mult_autocorrelation);
  }

  // Binary convolution:
  void convolve(Complex *f, Complex *g) {
    Complex *F[]={f,g};
    convolve(F,multbinary);
  }
    
  template<class T>
  inline void premult(Complex **F, unsigned int k, Vec& Zetak);

  void premult_nonSSE2(Complex **F);
  
  template<class T>
  void premult(Complex **F);
  
  void postmultadd(Complex *f, Complex *u);
};

// In-place implicitly dealiased 1D Hermitian convolution.
class ImplicitHConvolution : public ThreadBase {
protected:
  unsigned int m;
  unsigned int c;
  Complex **U;
  unsigned int A;
  unsigned int B;
  Complex *u;
  unsigned int s;
  Complex *ZetaH,*ZetaL;
  rcfft1d *rc,*rco;
  crfft1d *cr,*cro;
  bool pointers;
  bool allocated;
public:

  void initpointers(Complex **&U, Complex *u) {
    unsigned int C=max(A,B);
    U=new Complex *[C];
    unsigned stride=c+1;
    for(unsigned int a=0; a < C; ++a) 
      U[a]=u+a*stride;
    pointers=true;
  }
  
  void deletepointers(Complex **&U) {
    delete [] U;
  }
  
  void init() {
    Complex* U0=U[0];
    Complex* U1=A == 1 ? ComplexAlign(m) : U[1];
    
    rc=new rcfft1d(m,U0);
    cr=new crfft1d(m,U0);

    rco=new rcfft1d(m,(double *) U0,U1);
    cro=new crfft1d(m,U1,(double *) U0);
    
    if(A == 1) deleteAlign(U1);
    
    threads=std::min(threads,std::max(rco->Threads(),cro->Threads()));
    s=BuildZeta(3*m,c+2,ZetaH,ZetaL,threads);
  }
  
  // m is the number of independent data values
  // U is an array of max(A,B) distinct work arrays of size c+1, where c=m/2
  // A is the number of inputs.
  // B is the number of outputs.
  ImplicitHConvolution(unsigned int m, Complex **U, unsigned int A=2,
                       unsigned int B=1, unsigned int threads=fftw::maxthreads)
    : ThreadBase(threads), m(m), c(m/2), U(U), A(A), B(B), pointers(false),
      allocated(false) {
    init();
  }

  // m is the number of independent data values
  // u is a work array of max(A,B)*(c+1) Complex values, where c=m/2
  // A is the number of inputs.
  // B is the number of outputs.
  ImplicitHConvolution(unsigned int m, Complex *u, unsigned int A=2,
                       unsigned int B=1, unsigned int threads=fftw::maxthreads)
    : ThreadBase(threads), m(m), c(m/2), A(A), B(B), u(u), allocated(false) {
    initpointers(U,u);
    init();
  }

  // m is the number of independent data values
  // u is a work array of max(A,B)*(c+1) Complex values, where c=m/2
  // A is the number of inputs.
  // B is the number of outputs.
  ImplicitHConvolution(unsigned int m, unsigned int A=2,
                       unsigned int B=1, unsigned int threads=fftw::maxthreads)
    : ThreadBase(threads), m(m), c(m/2), A(A), B(B),
      u(ComplexAlign(max(A,B)*(c+1))), allocated(true) {
    initpointers(U,u);
    init();
  }

  virtual ~ImplicitHConvolution() {
    deleteAlign(ZetaH);
    deleteAlign(ZetaL);
    
    if(pointers) deletepointers(U);
    if(allocated) deleteAlign(u);

    delete cro;
    delete rco;
    
    delete cr;
    delete rc;
  }
  
  // F is an array of A pointers to distinct data blocks each of size m,
  // shifted by offset (contents not preserved).
  void convolve(Complex **F, realmultiplier *pmult, unsigned int offset=0);

  void premult(Complex **F, 
	       //Complex **crm, Complex **cr0, Complex **crp,
	       unsigned int offset,
	       Complex* f1c);
  
  void postmultadd(Complex **cr2, Complex **cr0, Complex **Q);
  void postmultadd0(Complex **crm, Complex **cr0, Complex **crp, Complex *f1c);

  // Binary convolution:
  void convolve(Complex *f, Complex *g) {
    Complex *F[]={f,g};
    convolve(F,multbinary);
  }
  
  template<class T>
  inline void premult(Complex **F, unsigned int k, Vec& Zetak);

  void premult_nonSSE2(Complex **F);
  
  template<class T>
  void premult(Complex **F);
  
  void postmultadd(Complex *f, Complex *u);
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
  unsigned int m;
  unsigned int M;
  unsigned int stride;
  unsigned int dist;
  unsigned int s;
  Complex *ZetaH, *ZetaL;
  unsigned int threads;
public:  
  mfft1d *Backwards; 
  mfft1d *Forwards;
  
  fftpad(unsigned int m, unsigned int M,
         unsigned int stride, Complex *u=NULL,
         unsigned int Threads=fftw::maxthreads)
    : m(m), M(M), stride(stride), threads(std::min(m,Threads)) {
    Backwards=new mfft1d(m,1,M,stride,1,u,NULL,threads);
    Forwards=new mfft1d(m,-1,M,stride,1,u,NULL,threads);
    
    threads=std::max(Backwards->Threads(),Forwards->Threads());
    
    s=BuildZeta(2*m,m,ZetaH,ZetaL,threads);
  }
  
  ~fftpad() {
    deleteAlign(ZetaL);
    deleteAlign(ZetaH);
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
  unsigned int m;
  unsigned int M;
  unsigned int s;
  unsigned int stride;
  mfft1d *Forwards;
  mfft1d *Backwards;
  Complex *ZetaH, *ZetaL;
  unsigned int threads;
public:  
  fft0pad(unsigned int m, unsigned int M, unsigned int stride, Complex *u=NULL,
          unsigned int Threads=fftw::maxthreads)
    : m(m), M(M), stride(stride), threads(std::min(m,Threads)) {
    Backwards=new mfft1d(m,1,M,stride,1,u,NULL,threads);
    Forwards=new mfft1d(m,-1,M,stride,1,u,NULL,threads);
    
    s=BuildZeta(3*m,m,ZetaH,ZetaL);
  }
  
  virtual ~fft0pad() {
    deleteAlign(ZetaL);
    deleteAlign(ZetaH);
    delete Forwards;
    delete Backwards;
  }
  
  // TODO: Update also for noncompact data format
  // Unscramble indices.
  inline unsigned findex(unsigned i) {
    return i < m-1 ? 3*i : 3*i+4-3*m;
  }

  inline unsigned uindex(unsigned i) {
    return i > 0 ? (i < m ? 3*i-1 : 3*m-3) : 3*m-1;
  }

  virtual void backwards(Complex *f, Complex *u);
  virtual void forwards(Complex *f, Complex *u);
};
  
// Compute the scrambled implicitly m-padded complex Fourier transform of M
// complex vectors, each of length 2m with the origin at index m.
// The first component of each vector is ignored; subsequent components
// store the physical data for wavenumbers -m+1 to m-1).
// The arrays in and out (which may coincide) must be allocated as
// Complex[M*2m]. The array u must be allocated as Complex[M*m].
//
//   fft0pad fft(m,M,stride,u);
//   fft.backwards(in,u);
//   fft.forwards(in,u);
//
// Notes:
//   stride is the spacing between the elements of each Complex vector.
//
class fft0padwide : public fft0pad {
public:  
  fft0padwide(unsigned int m, unsigned int M, unsigned int stride,
              Complex *u=NULL, unsigned int threads=fftw::maxthreads) :
    fft0pad(m,M,stride,u,threads) {}

  void backwards(Complex *f, Complex *u);
  void forwards(Complex *f, Complex *u);
};
  
// In-place implicitly dealiased 2D complex convolution.
class ImplicitConvolution2 : public ThreadBase {
protected:
  unsigned int mx,my;
  Complex *u1;
  Complex *u2;
  unsigned int A,B;
  fftpad *xfftpad;
  ImplicitConvolution **yconvolve;
  Complex **U2;
  bool allocated;
public:  
  unsigned int getmx() {return mx;}
  unsigned int getmy() {return my;}
  unsigned int getA() {return A;}
  unsigned int getB() {return B;}

  void initpointers2(Complex **&U2, Complex *u2, unsigned int stride) {
    U2=new Complex *[A];
    for(unsigned int a=0; a < A; ++a)
      U2[a]=u2+a*stride;
  }
  
  void deletepointers2(Complex **&U2) {
    delete [] U2;
  }
  
  void init(unsigned int nx, unsigned int ny, unsigned int stride) {
    xfftpad=new fftpad(mx,ny,ny,u2);
    yconvolve=new ImplicitConvolution*[threads];
    for(unsigned int t=0; t < threads; ++t)
      yconvolve[t]=new ImplicitConvolution(my,u1+t*my*A,A,B,innerthreads);
    initpointers2(U2,u2,stride);
  }
  
  void set(unsigned int& nx, unsigned int& ny, unsigned int& stride) {
    if(nx == 0) nx=mx;
    if(ny == 0) {
      ny=my;
      stride=mx*my;
    }
  }
  
  // u1 is a temporary array of size my*A*threads.
  // u2 is a temporary array of size mx*my*A.
  // A is the number of inputs.
  // B is the number of outputs.
  // threads is the number of threads to use in the outer subconvolution loop.
  // nx, ny, and stride are for the MPI interface; should not be set directly.
  ImplicitConvolution2(unsigned int mx, unsigned int my,
                       Complex *u1, Complex *u2,
                       unsigned int A=2, unsigned int B=1,
                       unsigned int threads=fftw::maxthreads,
                       unsigned int nx=0, unsigned int ny=0,
                       unsigned int stride=0) :
    ThreadBase(threads), mx(mx), my(my), u1(u1), u2(u2), A(A), B(B),
    allocated(false) {
    set(nx,ny,stride);
    multithread(nx);
    init(nx,ny,stride);
  }
  
  ImplicitConvolution2(unsigned int mx, unsigned int my,
                       unsigned int A=2, unsigned int B=1,
                       unsigned int threads=fftw::maxthreads,
                       unsigned int nx=0, unsigned int ny=0,
                       unsigned int stride=0) : 
    ThreadBase(threads), mx(mx), my(my), A(A), B(B) {
    set(nx,ny,stride);
    multithread(nx);
    u1=ComplexAlign(my*A*threads);
    u2=ComplexAlign(stride*A);
    allocated=true;
    init(nx,ny,stride);
  }
  
  virtual ~ImplicitConvolution2() {
    deletepointers2(U2);
    
    for(unsigned int t=0; t < threads; ++t)
      delete yconvolve[t];
    
    delete [] yconvolve;
    delete xfftpad;
    
    if(allocated) {
      deleteAlign(u2);
      deleteAlign(u1);
    }
  }
  
  void backwards(Complex **F, Complex **U2, unsigned int offset) {
    for(unsigned int a=0; a < A; ++a)
      xfftpad->backwards(F[a]+offset,U2[a]);
  }

  void subconvolution(Complex **F, multiplier *pmult, 
                      unsigned int start, unsigned int stop) {
    if(threads > 1) {
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif    
      for(unsigned int i=start; i < stop; i += my)
        yconvolve[get_thread_num()]->convolve(F,pmult,i);
    } else {
      ImplicitConvolution *yconvolve0=yconvolve[0];
      for(unsigned int i=start; i < stop; i += my)
        yconvolve0->convolve(F,pmult,i);
    }
  }
  
  void forwards(Complex **F, Complex **U2, unsigned int offset) {
    for(unsigned int b=0; b < B; ++b)
      xfftpad->forwards(F[b]+offset,U2[b]);
  }
  
  // F is a pointer to A distinct data blocks each of size mx*my,
  // shifted by offset (contents not preserved).
  virtual void convolve(Complex **F, multiplier *pmult,
                        unsigned int offset=0) {
    unsigned int size=mx*my;
    backwards(F,U2,offset);
    subconvolution(F,pmult,offset,size+offset);
    subconvolution(U2,pmult,0,size);
    forwards(F,U2,offset);
  }
  
  // Binary convolution:
  void convolve(Complex *f, Complex *g) {
    Complex *F[]={f,g};
    convolve(F,multbinary);
  }
};

inline void HermitianSymmetrizeX(unsigned int mx, unsigned int my,
                                 unsigned int xorigin, Complex *f)
{
  unsigned int offset=xorigin*my;
  unsigned int stop=mx*my;
  f[offset].im=0.0;
  for(unsigned int i=my; i < stop; i += my)
    f[offset-i]=conj(f[offset+i]);
}

// Enforce 3D Hermiticity using specified (x,y > 0,z=0) and (x >= 0,y=0,z=0)
// data.
inline void HermitianSymmetrizeXY(unsigned int mx, unsigned int my,
                                  unsigned int mz, unsigned int xorigin,
                                  unsigned int yorigin, Complex *f)
{
  int stride=(yorigin+my)*mz;
  int mxstride=mx*stride;
  unsigned int myz=my*mz;
  unsigned int origin=xorigin*stride+yorigin*mz;
  
  f[origin].im=0.0;

  for(int i=stride; i < mxstride; i += stride)
    f[origin-i]=conj(f[origin+i]);
  
  for(int i=stride-mxstride; i < mxstride; i += stride) {
    int stop=i+myz;
    for(int j=i+mz; j < stop; j += mz) {
      f[origin-j]=conj(f[origin+j]);
    }
  }
}

class ImplicitHConvolution2Base : public ThreadBase {
protected:
  unsigned int mx,my;
  Complex *u1;
  Complex *u2;
  unsigned int A,B;
  fft0pad *xfftpad;
  Complex **U2,**V2;
  bool compact;
  bool allocated;
public:

  void initpointers2(Complex **&U2, Complex *u2, unsigned int stride) 
  {
    U2=new Complex *[A];
    for(unsigned int a=0; a < A; ++a)
      U2[a]=u2+a*stride;
  }
  
  void deletepointers2(Complex **&U2) {
    delete [] U2;
  }
  
  void init(unsigned int nx, unsigned int ny, unsigned int stride) {
    xfftpad=compact ? new fft0pad(mx,ny,ny,u2,threads) :
      new fft0padwide(mx,ny,ny,u2,threads);
    initpointers2(U2,u2,stride);
  }

  void set(unsigned int &nx, unsigned int& ny, unsigned int& stride) {
    if(nx == 0) nx=mx;
    if(ny == 0) {
      ny=my+!compact;
      stride=(mx+compact)*ny;
    }
  }
  
  ImplicitHConvolution2Base(unsigned int mx, unsigned int my,
                            Complex *u1, Complex *u2,
                            unsigned int A=2, unsigned int B=1,
                            bool compact=true,
                            unsigned int threads=fftw::maxthreads,
                            unsigned int nx=0, unsigned int ny=0,
                            unsigned int stride=0) :
    ThreadBase(threads), mx(mx), my(my), u1(u1), u2(u2), A(A), B(B),
    compact(compact), allocated(false) {
    set(nx,ny,stride);
    multithread(nx);
    init(nx,ny,stride);
  }
  
  ImplicitHConvolution2Base(unsigned int mx, unsigned int my,
                            unsigned int A=2, unsigned int B=1,
                            bool compact=true,
                            unsigned int threads=fftw::maxthreads,
                            unsigned int nx=0, unsigned int ny=0,
                            unsigned int stride=0) :
    ThreadBase(threads), mx(mx), my(my), A(A), B(B), compact(compact) {
    set(nx,ny,stride);
    multithread(nx);
    u1=ComplexAlign((my/2+1)*A*threads);
    u2=ComplexAlign(stride*A);
    allocated=true;
    init(nx,ny,stride);
  }
  
  ~ImplicitHConvolution2Base() {
    deletepointers2(U2);
    
    if(allocated) {
      deleteAlign(u2);
      deleteAlign(u1);
    }
    delete xfftpad;
  }
};
  
// In-place implicitly dealiased 2D Hermitian convolution.
class ImplicitHConvolution2 : public ImplicitHConvolution2Base {
protected:
  ImplicitHConvolution **yconvolve;
  Complex ***U;
public:
  void initconvolve() {
    yconvolve=new ImplicitHConvolution*[threads];
    for(unsigned int t=0; t < threads; ++t)
      yconvolve[t]=new ImplicitHConvolution(my,u1+t*(my/2+1)*A,A,B,
                                            innerthreads);
  }
    
  // u1 is a temporary array of size (my/2+1)*A*threads.
  // u2 is a temporary array of size (mx+compact)*(my+!compact)*A;
  // A is the number of inputs.
  // B is the number of outputs.
  // threads is the number of threads to use in the outer subconvolution loop.
  // nx, ny, and stride are for the MPI interface; should not be set directly.
  ImplicitHConvolution2(unsigned int mx, unsigned int my,
                        Complex *u1, Complex *u2,
                        unsigned int A=2, unsigned int B=1,
                        bool compact=true,
                        unsigned int threads=fftw::maxthreads,
                        unsigned int nx=0, unsigned int ny=0,
                        unsigned int stride=0) :
    ImplicitHConvolution2Base(mx,my,u1,u2,A,B,compact,threads,nx,ny,stride) {
    initconvolve();
  }
  
  ImplicitHConvolution2(unsigned int mx, unsigned int my,
                        unsigned int A=2, unsigned int B=1,
                        bool compact=true,
                        unsigned int threads=fftw::maxthreads,
                        unsigned int nx=0, unsigned int ny=0,
                        unsigned int stride=0) :
    ImplicitHConvolution2Base(mx,my,A,B,compact,threads,nx,ny,stride) {
    initconvolve();
  }
  
  virtual ~ImplicitHConvolution2() {
    for(unsigned int t=0; t < threads; ++t)
      delete yconvolve[t];
    delete [] yconvolve;
  }
  
  void backwards(Complex **F, Complex **U2, unsigned int ny,
                 bool symmetrize, unsigned int offset) {
    for(unsigned int a=0; a < A; ++a) {
      Complex *f=F[a]+offset;
      if(symmetrize)
        HermitianSymmetrizeX(mx,ny,mx-compact,f);
      xfftpad->backwards(f,U2[a]);
    }
  }

  void subconvolution(Complex **F, realmultiplier *pmult,
                      unsigned int start, unsigned int stop, unsigned int stride) {
    if(threads > 1) {
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif    
      for(unsigned int i=start; i < stop; i += stride)
        yconvolve[get_thread_num()]->convolve(F,pmult,i);
    } else {
      ImplicitHConvolution *yconvolve0=yconvolve[0];
      for(unsigned int i=start; i < stop; i += stride)
        yconvolve0->convolve(F,pmult,i);
    }
  }  
  
  void forwards(Complex **F, Complex **U2, unsigned int offset) {
    for(unsigned int b=0; b < B; ++b)
      xfftpad->forwards(F[b]+offset,U2[b]);
  }
  
  // F is a pointer to A distinct data blocks each of size 
  // (2mx-compact)*(my+!compact), shifted by offset (contents not preserved).
  virtual void convolve(Complex **F, realmultiplier *pmult,
                        bool symmetrize=true, unsigned int offset=0) {
    unsigned ny=my+!compact;
    backwards(F,U2,ny,symmetrize,offset);
    subconvolution(F,pmult,offset,(2*mx-compact)*ny+offset,ny);
    subconvolution(U2,pmult,0,(mx+compact)*ny,ny);
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
  unsigned int mx,my,mz;
  Complex *u1;
  Complex *u2;
  Complex *u3;
  unsigned int A,B;
  fftpad *xfftpad;
  ImplicitConvolution2 **yzconvolve;
  Complex **U3;
  bool allocated;
public:  
  unsigned int getmx() {return mx;}
  unsigned int getmy() {return my;}
  unsigned int getmz() {return mz;}
  unsigned int getA() {return A;}
  unsigned int getB() {return B;}

  void initpointers3(Complex **&U3, Complex *u3, unsigned int stride) {
    U3=new Complex *[A];
    for(unsigned int a=0; a < A; ++a)
      U3[a]=u3+a*stride;
  }
  
  void deletepointers3(Complex **&U3) {
    delete [] U3;
  }
  
  void init(unsigned int ny, unsigned int nz, unsigned int stride2,
            unsigned int stride3) {
    unsigned int nyz=ny*nz;
    xfftpad=new fftpad(mx,nyz,nyz,u3);
    
    if(nz == mz) {
      yzconvolve=new ImplicitConvolution2*[threads];
      for(unsigned int t=0; t < threads; ++t)
        yzconvolve[t]=new ImplicitConvolution2(my,mz,u1+t*mz*A*innerthreads,
                                               u2+t*stride2*A,A,B,innerthreads);
      initpointers3(U3,u3,stride3);
    } else yzconvolve=NULL;
  }
  
  void set(unsigned int& ny,unsigned int& nz,
           unsigned int& stride2, unsigned int& stride3) {
    if(ny == 0) {
      ny=my;
      nz=mz;
      stride2=my*mz;
      stride3=mx*my*mz;
    }
  }
  
  // u1 is a temporary array of size mz*A*threads.
  // u2 is a temporary array of size my*mz*A*threads.
  // u3 is a temporary array of size mx*my*mz*A.
  // A is the number of inputs.
  // B is the number of outputs.
  // threads is the number of threads to use in the outer subconvolution loop.
  // ny, nz, stride2, and stride3 are used by the MPI interface
  // and should not be set directly.
  ImplicitConvolution3(unsigned int mx, unsigned int my, unsigned int mz,
                       Complex *u1, Complex *u2, Complex *u3,
                       unsigned int A=2, unsigned int B=1, 
                       unsigned int threads=fftw::maxthreads,
                       unsigned int ny=0, unsigned int nz=0,
                       unsigned int stride2=0, unsigned int stride3=0) :
    ThreadBase(threads), mx(mx), my(my), mz(mz), u1(u1), u2(u2), u3(u3),
    A(A), B(B), allocated(false) {
    set(ny,nz,stride2,stride3);
    multithread(mx);
    init(ny,nz,stride2,stride3);
  }

  // ny, nz, stride2, and stride3 are used by the MPI
  // interface and should not be set directly.
  ImplicitConvolution3(unsigned int mx, unsigned int my, unsigned int mz,
                       unsigned int A=2, unsigned int B=1, 
                       unsigned int threads=fftw::maxthreads,
                       unsigned int ny=0, unsigned int nz=0,
                       unsigned int stride2=0, unsigned int stride3=0) :
    ThreadBase(threads), mx(mx), my(my), mz(mz), A(A), B(B), allocated(true) {
    multithread(mx);
    u1=ComplexAlign(mz*A*threads*innerthreads);
    set(ny,nz,stride2,stride3);
    u2=ComplexAlign(stride2*A*threads);
    u3=ComplexAlign(stride3*A);
    init(ny,nz,stride2,stride3);
  }
  
  virtual ~ImplicitConvolution3() {
    if(yzconvolve) {
      deletepointers3(U3);

      for(unsigned int t=0; t < threads; ++t)
        delete yzconvolve[t];
      delete [] yzconvolve;
    }
    delete xfftpad;
    
    if(allocated) {
      deleteAlign(u3);
      deleteAlign(u2);
      deleteAlign(u1);
    }
  }
  
  void backwards(Complex **F, Complex **U3, unsigned int offset) {
    for(unsigned int a=0; a < A; ++a)
      xfftpad->backwards(F[a]+offset,U3[a]);
  }

  void subconvolution(Complex **F, multiplier *pmult, 
                      unsigned int start, unsigned int stop,
                      unsigned int stride) {
    if(threads > 1) {
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif    
      for(unsigned int i=start; i < stop; i += stride)
        yzconvolve[get_thread_num()]->convolve(F,pmult,i);
    } else {
      ImplicitConvolution2 *yzconvolve0=yzconvolve[0];
      for(unsigned int i=start; i < stop; i += stride)
        yzconvolve0->convolve(F,pmult,i);
    }
  }
  
  void forwards(Complex **F, Complex **U3, unsigned int offset=0) {
    for(unsigned int b=0; b < B; ++b)
      xfftpad->forwards(F[b]+offset,U3[b]);
  }
  
  // F is a pointer to A distinct data blocks each of size mx*my*mz,
  // shifted by offset
  virtual void convolve(Complex **F, multiplier *pmult, unsigned int offset=0) {
    unsigned int myz=my*mz;
    unsigned int size=mx*myz;
    
    backwards(F,U3,offset);
    subconvolution(F,pmult,offset,size+offset,myz);
    subconvolution(U3,pmult,0,size,myz);
    forwards(F,U3,offset);
  }
  
  // Binary convolution:
  void convolve(Complex *f, Complex *g) {
    Complex *F[]={f,g};
    convolve(F,multbinary);
  }
};

// In-place implicitly dealiased 3D Hermitian convolution.
class ImplicitHConvolution3 : public ThreadBase {
protected:
  unsigned int mx,my,mz;
  Complex *u1;
  Complex *u2;
  Complex *u3;
  unsigned int A,B;
  fft0pad *xfftpad;
  ImplicitHConvolution2 **yzconvolve;
  Complex **U3;
  bool compact;
  bool allocated;
public:     
  unsigned int getmx() {return mx;}
  unsigned int getmy() {return my;}
  unsigned int getmz() {return mz;}
  unsigned int getA() {return A;}
  unsigned int getB() {return B;}
  
  void initpointers3(Complex **&U3, Complex *u3, unsigned int stride) {
    U3=new Complex *[A];
    for(unsigned int a=0; a < A; ++a)
      U3[a]=u3+a*stride;
  }
  
  void deletepointers3(Complex **&U3) {
    delete [] U3;
  }
    
  void init(unsigned int ny, unsigned int nz, unsigned int stride2,
            unsigned int stride3) {
    unsigned int nyz=ny*nz;
    xfftpad=compact ? new fft0pad(mx,nyz,nyz,u3,threads) :
      new fft0padwide(mx,nyz,nyz,u3,threads);

    if(nz == mz+!compact) {
      yzconvolve=new ImplicitHConvolution2*[threads];
      for(unsigned int t=0; t < threads; ++t)
        yzconvolve[t]=new ImplicitHConvolution2(my,mz,
                                                u1+t*(mz/2+1)*A*innerthreads,
                                                u2+t*stride2*A,A,B,compact,
                                                innerthreads);
      initpointers3(U3,u3,stride3);
    } else yzconvolve=NULL;
  }
  
  void set(unsigned int& ny,unsigned int& nz,
           unsigned int& stride2, unsigned int& stride3) {
    if(ny == 0) {
      ny=2*my-compact;
      nz=mz+!compact;
      stride2=(my+compact)*nz;
      stride3=(mx+compact)*ny*nz;
    }
  }
  
  // u1 is a temporary array of size (mz/2+1)*A*threads.
  // u2 is a temporary array of size (my+compact)*(mz+!compact)*A*threads.
  // u3 is a temporary array of size (mx+compact)*(2my-compact)*(mz+!compact)*A.
  // A is the number of inputs.
  // B is the number of outputs.
  // threads is the number of threads to use in the outer subconvolution loop.
  // ny, nz, stride2, and stride3 are used by the MPI
  // interface and should not be set directly.
  ImplicitHConvolution3(unsigned int mx, unsigned int my, unsigned int mz,
                        Complex *u1, Complex *u2, Complex *u3,
                        unsigned int A=2, unsigned int B=1, bool compact=true,
                        unsigned int threads=fftw::maxthreads,
                        unsigned int ny=0, unsigned int nz=0,
                        unsigned int stride2=0, unsigned int stride3=0) :
    ThreadBase(threads), mx(mx), my(my), mz(mz), u1(u1), u2(u2), u3(u3),
    A(A), B(B), compact(compact), allocated(false) {
    set(ny,nz,stride2,stride3);
    multithread(mx);
    init(ny,nz,stride2,stride3);
  }
  
  // ny, nz, stride2, and stride3 are used by the MPI
  // interface and should not be set directly.
  ImplicitHConvolution3(unsigned int mx, unsigned int my, unsigned int mz,
                        unsigned int A=2, unsigned int B=1, bool compact=true,
                        unsigned int threads=fftw::maxthreads,
                        unsigned int ny=0, unsigned int nz=0,
                        unsigned int stride2=0, unsigned int stride3=0) :
    ThreadBase(threads), mx(mx), my(my), mz(mz), A(A), B(B), compact(compact),
    allocated(true) {
    multithread(mx);
    u1=ComplexAlign((mz/2+1)*A*threads*innerthreads);
    set(ny,nz,stride2,stride3);
    u2=ComplexAlign(stride2*A*threads);
    u3=ComplexAlign(stride3*A);
    init(ny,nz,stride2,stride3);
  }
  
  virtual ~ImplicitHConvolution3() {
    if(yzconvolve) {
      deletepointers3(U3);
      
      for(unsigned int t=0; t < threads; ++t)
        delete yzconvolve[t];
      delete [] yzconvolve;
    }

    delete xfftpad;
    
    if(allocated) {
      deleteAlign(u3);
      deleteAlign(u2);
      deleteAlign(u1);
    }
  }
  
  virtual void HermitianSymmetrize(Complex *f, Complex *u) {
    HermitianSymmetrizeXY(mx,my,mz+!compact,mx-compact,my-compact,f);
  }
  
  void backwards(Complex **F, Complex **U3, bool symmetrize,
                 unsigned int offset) {
    for(unsigned int a=0; a < A; ++a) {
      Complex *f=F[a]+offset;
      Complex *u=U3[a];
      if(symmetrize)
        HermitianSymmetrize(f,u);
      xfftpad->backwards(f,u);
    }
  }

  void subconvolution(Complex **F, realmultiplier *pmult,
                      unsigned int start, unsigned int stop,
                      unsigned int stride) {
    if(threads > 1) {
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif    
      for(unsigned int i=start; i < stop; i += stride)
        yzconvolve[get_thread_num()]->convolve(F,pmult,false,i);
    } else {
      ImplicitHConvolution2 *yzconvolve0=yzconvolve[0];
      for(unsigned int i=start; i < stop; i += stride)
        yzconvolve0->convolve(F,pmult,false,i);
    }
  }

  void forwards(Complex **F, Complex **U3, unsigned int offset=0) {
    for(unsigned int b=0; b < B; ++b)
      xfftpad->forwards(F[b]+offset,U3[b]);
  }
  
  // F is a pointer to A distinct data blocks each of size
  // (2mx-compact)*(2my-compact)*(mz+!compact), shifted by offset 
  // (contents not preserved).
  virtual void convolve(Complex **F, realmultiplier *pmult,
                        bool symmetrize=true, unsigned int offset=0) {
    unsigned int nynz=(2*my-compact)*(mz+!compact);
    backwards(F,U3,symmetrize,offset);
    subconvolution(F,pmult,offset,(2*mx-compact)*nynz+offset,nynz);
    subconvolution(U3,pmult,0,(mx+compact)*nynz,nynz);
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
  unsigned int m;
  Complex *u,*v,*w;
  unsigned int M;
  unsigned int s;
  rcfft1d *rc, *rco;
  crfft1d *cr, *cro;
  Complex *ZetaH, *ZetaL;
  Complex **W;
  bool allocated;
  unsigned int twom;
  unsigned int stride;
public:  
  void initpointers(Complex **&W, Complex *w) {
    W=new Complex *[M];
    unsigned int m1=m+1;
    for(unsigned int s=0; s < M; ++s) 
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
  ImplicitHTConvolution(unsigned int m, Complex *u, Complex *v,
                        Complex *w, unsigned int M=1) :
    m(m), u(u), v(v), w(w), M(M), allocated(false) {
    init();
  }
  
  ImplicitHTConvolution(unsigned int m, unsigned int M=1) : 
    m(m), u(ComplexAlign(m*M+M)), v(ComplexAlign(m*M+M)),
    w(ComplexAlign(m*M+M)), M(M), allocated(true) {
    init();
  }
  
  ~ImplicitHTConvolution() {
    deletepointers(W);
    
    if(allocated) {
      deleteAlign(w);
      deleteAlign(v);
      deleteAlign(u);
    }
    deleteAlign(ZetaL);
    deleteAlign(ZetaH);
    delete cro;
    delete rco;
    delete cr;
    delete rc;
  }
  
  void mult(double *a, double *b, double **C, unsigned int offset=0);
  
  void convolve(Complex **F, Complex **G, Complex **H, 
                Complex *u, Complex *v, Complex **W,
                unsigned int offset=0);
  
  // F, G, and H are distinct pointers to M distinct data blocks each of size
  // m+1, shifted by offset (contents not preserved).
  // The output is returned in F[0].
  void convolve(Complex **F, Complex **G, Complex **H, unsigned int offset=0) {
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
  unsigned int m;
  Complex *u,*v;
  unsigned int s;
  rcfft1d *rc, *rco;
  crfft1d *cr, *cro;
  Complex *ZetaH, *ZetaL;
  bool allocated;
  unsigned int twom;
  unsigned int stride;
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
  ImplicitHFGGConvolution(unsigned int m, Complex *u, Complex *v) :
    m(m), u(u), v(v), allocated(false) {
    init();
  }
  
  ImplicitHFGGConvolution(unsigned int m) : 
    m(m), u(ComplexAlign(m+1)), v(ComplexAlign(m+1)), allocated(true) {
    init();
  }
  
  ~ImplicitHFGGConvolution() {
    if(allocated) {
      deleteAlign(v);
      deleteAlign(u);
    }
    deleteAlign(ZetaL);
    deleteAlign(ZetaH);
    delete cro;
    delete rco;
    delete cr;
    delete rc;
  }
  
  void mult(double *a, double *b);
  
  void convolve(Complex *f, Complex *g, Complex *u, Complex *v);
  
  // f and g are distinct pointers to data of size m+1 (contents not preserved).
  // The output is returned in f.
  void convolve(Complex *f, Complex *g) {
    convolve(f,g,u,v);
  }
};

// In-place implicitly dealiased Hermitian ternary convolution.
// Special case F=G=H, M=1.
class ImplicitHFFFConvolution : public ThreadBase {
protected:
  unsigned int m;
  Complex *u;
  unsigned int s;
  rcfft1d *rc;
  crfft1d *cr;
  Complex *ZetaH, *ZetaL;
  bool allocated;
  unsigned int twom;
  unsigned int stride;
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
  ImplicitHFFFConvolution(unsigned int m, Complex *u) :
    m(m), u(u), allocated(false) {
    init();
  }
  
  ImplicitHFFFConvolution(unsigned int m) :
    m(m), u(ComplexAlign(m+1)), allocated(true) {
    init();
  }
  
  ~ImplicitHFFFConvolution() {
    if(allocated)
      deleteAlign(u);
    
    deleteAlign(ZetaL);
    deleteAlign(ZetaH);
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
  unsigned int m;
  unsigned int M;
  unsigned int stride;
  unsigned int s;
  mfft1d *Backwards;
  mfft1d *Forwards;
  Complex *ZetaH, *ZetaL;
  unsigned int threads;
public:  
  fft0bipad(unsigned int m, unsigned int M, unsigned int stride,
            Complex *f) : m(m), M(M), stride(stride) {
    unsigned int twom=2*m;
    Backwards=new mfft1d(twom,1,M,stride,1,f);
    Forwards=new mfft1d(twom,-1,M,stride,1,f);
    
    threads=std::min(threads,
                     std::max(Backwards->Threads(),Forwards->Threads()));
    
    s=BuildZeta(4*m,twom,ZetaH,ZetaL,threads);
  }
  
  ~fft0bipad() {
    deleteAlign(ZetaL);
    deleteAlign(ZetaH);
    delete Forwards;
    delete Backwards;
  }
  
  void backwards(Complex *f, Complex *u);
  void forwards(Complex *f, Complex *u);
};

// In-place implicitly dealiased 2D Hermitian ternary convolution.
class ImplicitHTConvolution2 : public ThreadBase {
protected:
  unsigned int mx,my;
  Complex *u1,*v1,*w1;
  Complex *u2,*v2,*w2;
  unsigned int M;
  fft0bipad *xfftpad;
  ImplicitHTConvolution *yconvolve;
  Complex **U2,**V2,**W2;
  bool allocated;
  Complex **u,**v;
  Complex ***W;
public:  
  void initpointers(Complex **&u, Complex **&v, Complex ***&W,
                    unsigned int threads) {
    u=new Complex *[threads];
    v=new Complex *[threads];
    W=new Complex **[threads];
    unsigned int my1M=(my+1)*M;
    for(unsigned int i=0; i < threads; ++i) {
      unsigned int imy1M=i*my1M;
      u[i]=u1+imy1M;
      v[i]=v1+imy1M;
      Complex *wi=w1+imy1M;
      yconvolve->initpointers(W[i],wi);
    }
  }
    
  void deletepointers(Complex **&u, Complex **&v, Complex ***&W,
                      unsigned int threads) {
    for(unsigned int i=0; i < threads; ++i)
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
    unsigned int mu=2*mx*(my+1);
    for(unsigned int s=0; s < M; ++s) {
      unsigned int smu=s*mu;
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
    xfftpad=new fft0bipad(mx,my,my+1,u2);
    
    yconvolve=new ImplicitHTConvolution(my,u1,v1,w1,M);
    yconvolve->Threads(1);

    initpointers(u,v,W,threads);
    initpointers(U2,V2,W2,u2,v2,w2);
  }
  
  // u1, v1, and w1 are temporary arrays of size (my+1)*M*threads;
  // u2, v2, and w2 are temporary arrays of size 2mx*(my+1)*M.
  // M is the number of data blocks (each corresponding to a dot product term).
  // threads is the number of threads to use in the outer subconvolution loop.
  ImplicitHTConvolution2(unsigned int mx, unsigned int my,
                         Complex *u1, Complex *v1, Complex *w1, 
                         Complex *u2, Complex *v2, Complex *w2,
                         unsigned int M=1,
                         unsigned int threads=fftw::maxthreads) :
    ThreadBase(threads), mx(mx), my(my), u1(u1), v1(v1), w1(w1),
    u2(u2), v2(v2), w2(w2), M(M), allocated(false) {
    init();
  }
  
  ImplicitHTConvolution2(unsigned int mx, unsigned int my,
                         unsigned int M=1,
                         unsigned int threads=fftw::maxthreads) :
    ThreadBase(threads), mx(mx), my(my),
    u1(ComplexAlign((my+1)*M*threads)),
    v1(ComplexAlign((my+1)*M*threads)),
    w1(ComplexAlign((my+1)*M*threads)),
    u2(ComplexAlign(2*mx*(my+1)*M)),
    v2(ComplexAlign(2*mx*(my+1)*M)),
    w2(ComplexAlign(2*mx*(my+1)*M)),
    M(M), allocated(true) {
    init();
  }
  
  ~ImplicitHTConvolution2() {
    deletepointers(U2,V2,W2);
    deletepointers(u,v,W,threads);
    
    delete yconvolve;
    delete xfftpad;
    
    if(allocated) {
      deleteAlign(w2);
      deleteAlign(v2);
      deleteAlign(u2);
      deleteAlign(w1);
      deleteAlign(v1);
      deleteAlign(u1);
    }
  }
  
  void convolve(Complex **F, Complex **G, Complex **H, 
                Complex **u, Complex **v, Complex ***W, 
                Complex **U2, Complex **V2, Complex **W2,
                bool symmetrize=true, unsigned int offset=0) {
    Complex *u2=U2[0];
    Complex *v2=V2[0];
    Complex *w2=W2[0];
    
    unsigned int my1=my+1;
    unsigned int mu=2*mx*my1;
    
    for(unsigned int s=0; s < M; ++s) {
      Complex *f=F[s]+offset;
      if(symmetrize)
        HermitianSymmetrizeX(mx,my1,mx,f);
      xfftpad->backwards(f,u2+s*mu);
    }
    
    for(unsigned int s=0; s < M; ++s) {
      Complex *g=G[s]+offset;
      if(symmetrize)
        HermitianSymmetrizeX(mx,my1,mx,g);
      xfftpad->backwards(g,v2+s*mu);
    }
    
    for(unsigned int s=0; s < M; ++s) {
      Complex *h=H[s]+offset;
      if(symmetrize)
        HermitianSymmetrizeX(mx,my1,mx,h);
      xfftpad->backwards(h,w2+s*mu);
    }

#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif    
    for(unsigned int i=0; i < mu; i += my1) {
      unsigned int thread=get_thread_num();
      yconvolve->convolve(F,G,H,u[thread],v[thread],W[thread],i+offset);
    }
    
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif    
    for(unsigned int i=0; i < mu; i += my1) {
      unsigned int thread=get_thread_num();
      yconvolve->convolve(U2,V2,W2,u[thread],v[thread],W[thread],i+offset);
    }

    xfftpad->forwards(F[0]+offset,u2);
  }
  
  // F, G, and H are distinct pointers to M distinct data blocks each of size
  // 2mx*(my+1), shifted by offset (contents not preserved).
  // The output is returned in F[0].
  void convolve(Complex **F, Complex **G, Complex **H, bool symmetrize=true,
                unsigned int offset=0) {
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
  unsigned int mx,my;
  Complex *u1,*v1;
  Complex *u2,*v2;
  fft0bipad *xfftpad;
  ImplicitHFGGConvolution *yconvolve;
  bool allocated;
  Complex **u,**v;
public:  
  void initpointers(Complex **&u, Complex **&v, unsigned int threads) {
    u=new Complex *[threads];
    v=new Complex *[threads];
    unsigned int my1=my+1;
    for(unsigned int i=0; i < threads; ++i) {
      unsigned int imy1=i*my1;
      u[i]=u1+imy1;
      v[i]=v1+imy1;
    }
  }
    
  void deletepointers(Complex **&u, Complex **&v) {
    delete [] v;
    delete [] u;
  }
    
  void init() {
    xfftpad=new fft0bipad(mx,my,my+1,u2);
    
    yconvolve=new ImplicitHFGGConvolution(my,u1,v1);
    yconvolve->Threads(1);

    initpointers(u,v,threads);
  }
  
  // u1 and v1 are temporary arrays of size (my+1)*threads.
  // u2 and v2 are temporary arrays of size 2mx*(my+1).
  // threads is the number of threads to use in the outer subconvolution loop.
  ImplicitHFGGConvolution2(unsigned int mx, unsigned int my,
                           Complex *u1, Complex *v1,
                           Complex *u2, Complex *v2,
                           unsigned int threads=fftw::maxthreads) :
    ThreadBase(threads), mx(mx), my(my), u1(u1), v1(v1), u2(u2), v2(v2),
    allocated(false) {
    init();
  }
  
  ImplicitHFGGConvolution2(unsigned int mx, unsigned int my,
                           unsigned int threads=fftw::maxthreads) :
    ThreadBase(threads), mx(mx), my(my),
    u1(ComplexAlign((my+1)*threads)),
    v1(ComplexAlign((my+1)*threads)),
    u2(ComplexAlign(2*mx*(my+1))),
    v2(ComplexAlign(2*mx*(my+1))),
    allocated(true) {
    init();
  }
  
  ~ImplicitHFGGConvolution2() {
    deletepointers(u,v);
    
    delete yconvolve;
    delete xfftpad;
    
    if(allocated) {
      deleteAlign(v2);
      deleteAlign(u2);
      deleteAlign(v1);
      deleteAlign(u1);
    }
  }
  
  void convolve(Complex *f, Complex *g,
                Complex **u, Complex **v,
                Complex *u2, Complex *v2, bool symmetrize=true) {
    unsigned int my1=my+1;
    unsigned int mu=2*mx*my1;
    
    if(symmetrize)
      HermitianSymmetrizeX(mx,my1,mx,f);
    xfftpad->backwards(f,u2);
    
    if(symmetrize)
      HermitianSymmetrizeX(mx,my1,mx,g);
    xfftpad->backwards(g,v2);
    
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif    
    for(unsigned int i=0; i < mu; i += my1) {
      unsigned int thread=get_thread_num();
      yconvolve->convolve(f+i,g+i,u[thread],v[thread]);
    }
    
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif    
    for(unsigned int i=0; i < mu; i += my1) {
      unsigned int thread=get_thread_num();
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
  unsigned int mx,my;
  Complex *u1;
  Complex *u2;
  fft0bipad *xfftpad;
  ImplicitHFFFConvolution *yconvolve;
  bool allocated;
  Complex **u;
public:  
  void initpointers(Complex **&u, unsigned int threads) {
    u=new Complex *[threads];
    unsigned int my1=my+1;
    for(unsigned int i=0; i < threads; ++i)
      u[i]=u1+i*my1;
  }
    
  void deletepointers(Complex **&u) {
    delete [] u;
  }
    
  void init() {
    xfftpad=new fft0bipad(mx,my,my+1,u2);
    
    yconvolve=new ImplicitHFFFConvolution(my,u1);
    yconvolve->Threads(1);
    initpointers(u,threads);
  }
  
  // u1 is a temporary array of size (my+1)*threads.
  // u2 is a temporary array of size 2mx*(my+1).
  // threads is the number of threads to use in the outer subconvolution loop.
  ImplicitHFFFConvolution2(unsigned int mx, unsigned int my,
                           Complex *u1, Complex *u2,
                           unsigned int threads=fftw::maxthreads) :
    ThreadBase(threads), mx(mx), my(my),
    u1(u1), u2(u2), allocated(false) {
    init();
  }
  
  ImplicitHFFFConvolution2(unsigned int mx, unsigned int my,
                           unsigned int threads=fftw::maxthreads) :
    ThreadBase(threads), mx(mx), my(my),
    u1(ComplexAlign((my+1)*threads)),
    u2(ComplexAlign(2*mx*(my+1))),
    allocated(true) {
    init();
  }
  
  ~ImplicitHFFFConvolution2() {
    deletepointers(u);
    delete yconvolve;
    delete xfftpad;
    
    if(allocated) {
      deleteAlign(u2);
      deleteAlign(u1);
    }
  }
  
  void convolve(Complex *f, Complex **u, Complex *u2, bool symmetrize=true) {
    unsigned int my1=my+1;
    unsigned int mu=2*mx*my1;
    
    if(symmetrize)
      HermitianSymmetrizeX(mx,my1,mx,f);
    xfftpad->backwards(f,u2);
    
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif    
    for(unsigned int i=0; i < mu; i += my1)
      yconvolve->convolve(f+i,u[get_thread_num()]);
    
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif    
    for(unsigned int i=0; i < mu; i += my1)
      yconvolve->convolve(u2+i,u[get_thread_num()]);

    xfftpad->forwards(f,u2);
  }
  
  void convolve(Complex *f, bool symmetrize=true) {
    convolve(f,u,u2,symmetrize);
  }
};

} //end namespace fftwpp

#endif
