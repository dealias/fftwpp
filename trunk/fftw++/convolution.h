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

namespace fftwpp {

#ifndef __convolution_h__
#define __convolution_h__ 1

extern const double sqrt3;
extern const double hsqrt3;

extern const Complex hSqrt3;
extern const Complex mhsqrt3;
extern const Complex mhalf;
extern const Complex zeta3;

// Build the factored zeta tables.
unsigned int BuildZeta(unsigned int n, unsigned int m,
                       Complex *&ZetaH, Complex *&ZetaL, unsigned int threads=1);

class ThreadBase
{
protected:
  unsigned int threads;
public:  
  ThreadBase() {threads=fftw::maxthreads;}
  ThreadBase(unsigned int threads) : threads(threads) {}
  void Threads(unsigned int nthreads) {threads=nthreads;}
};
  

// In-place implicitly dealiased 1D complex convolution.
class ImplicitConvolution : public ThreadBase {
protected:
  unsigned int m;
  Complex *u,*v;
  unsigned int M;
  unsigned int s;
  fft1d *Backwards,*Forwards;
  Complex *ZetaH, *ZetaL;
  Complex **V;
  bool allocated;
public:  
  unsigned int getm() {return m;}
  unsigned int getM() {return M;}

  void initpointers(Complex **&V, Complex *v) {
    V=new Complex *[M];
    for(unsigned int s=0; s < M; ++s) 
      V[s]=v+s*m;
  }
  
  void deletepointers(Complex **&V) {
    delete [] V;
  }
  
  void init() {
    Backwards=new fft1d(m,1,u,v);
    Forwards=new fft1d(m,-1,u,v);
    
    threads=Forwards->Threads();

    s=BuildZeta(2*m,m,ZetaH,ZetaL,threads);
    
    initpointers(V,v);
  }
  
  // m is the number of Complex data values.
  // u and v are distinct temporary arrays each of size m*M.
  // M is the number of data blocks (each corresponding to a dot product term).
  ImplicitConvolution(unsigned int m, Complex *u, Complex *v, unsigned int M=1)
    : m(m), u(u), v(v), M(M), allocated(false) {
    init();
  }
  
  ImplicitConvolution(unsigned int m, unsigned int M=1)
    : m(m), u(ComplexAlign(m*M)), v(ComplexAlign(m*M)), M(M),
      allocated(true) {
    init();
  }
  
  ~ImplicitConvolution() {
    deletepointers(V);
    
    if(allocated) {
      deleteAlign(u);
      deleteAlign(v);
    }
    deleteAlign(ZetaL);
    deleteAlign(ZetaH);
    delete Forwards;
    delete Backwards;
  }
  
  void mult(Complex *a, Complex **B, unsigned int offset=0);
  
  void convolve(Complex **F, Complex **G, Complex *u, Complex **V,
                unsigned int offset=0);
  
  // F and G are distinct pointers to M distinct data blocks each of size m,
  // shifted by offset (contents not preserved).
  // The output is returned in F[0].
  void convolve(Complex **F, Complex **G, unsigned int offset=0) {
    convolve(F,G,u,V,offset);
  }
  
  // Constructor for special case M=1:
  void convolve(Complex *f, Complex *g) {
    convolve(&f,&g);
  }
};


// In-place implicitly dealiased 1D Hermitian convolution.
class ImplicitHConvolution : public ThreadBase {
protected:
  unsigned int m;
  unsigned int c;
  Complex *u,*v,*w;
  unsigned int M;
  bool odd;
  unsigned int stride;
public:
  unsigned int s;
  rcfft1d *rc,*rco;
  crfft1d *cr,*cro;
  Complex *ZetaH,*ZetaL;
  Complex **U;
  bool allocated;

  unsigned int getm() {return m;}
  unsigned int getM() {return M;}

  void initpointers(Complex **&U, Complex *u) {
    U=new Complex *[M];
    unsigned int stride=c+1;
    for(unsigned int s=0; s < M; ++s) 
      U[s]=u+s*stride;
  }
  
  void deletepointers(Complex **&U) {
    delete [] U;
  }
  
  void init() {
    odd=m % 2 == 1;
    stride=odd ? m+1 : m+2;
    
    rc=new rcfft1d(m,u);
    cr=new crfft1d(m,u);

    rco=new rcfft1d(m,(double *) u,v);
    cro=new crfft1d(m,v,(double *) u);
    
    threads=cro->Threads();
    
    s=BuildZeta(3*m,2*c == m ? c : c+1,ZetaH,ZetaL,threads);
    
    initpointers(U,u);
  }
  
  // m is the number of independent data values.
  // u and v are temporary arrays each of size (m/2+1)*M.
  // w is a temporary array of size 3*M.
  // M is the number of data blocks (each corresponding to a dot product term).
  ImplicitHConvolution(unsigned int m, Complex *u, Complex *v, Complex *w,
                       unsigned int M=1)
    : m(m), c(m/2), u(u), v(v), w(w), M(M), allocated(false) {
    init();
  }

  ImplicitHConvolution(unsigned int m, unsigned int M=1)
    : m(m), c(m/2), u(ComplexAlign(c*M+M)),v(ComplexAlign(c*M+M)),
      w(ComplexAlign(3*M)), M(M), allocated(true) {
    init();
  }
    
  virtual ~ImplicitHConvolution() {
    deletepointers(U);
    
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
  
  void conjreverse(Complex *f, unsigned int m);
  
  virtual void mult(double *a, double **B, unsigned int offset=0);
  
  void convolve(Complex **F, Complex **G, Complex **U, Complex *v,
                Complex *w, unsigned int offset=0);
  
  // F and G are distinct pointers to M distinct data blocks each of size m,
  // shifted by offset (contents not preserved).
  // The output is returned in F[0].
  void convolve(Complex **F, Complex **G, unsigned int offset=0) {
    convolve(F,G,U,v,w,offset);
  }
  
  // Constructor for special case M=1:
  void convolve(Complex *f, Complex *g) {
    convolve(&f,&g);
  }
};
  

// Compute the scrambled virtual m-padded complex Fourier transform of M complex
// vectors, each of length m.
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
  mfft1d *Backwards; 
  mfft1d *Forwards;
  Complex *ZetaH, *ZetaL;
  unsigned int threads;
public:  
  fftpad(unsigned int m, unsigned int M,
         unsigned int stride, Complex *u=NULL) : m(m), M(M), stride(stride) {
    Backwards=new mfft1d(m,1,M,stride,1,u);
    Forwards=new mfft1d(m,-1,M,stride,1,u);
    
    threads=Forwards->Threads();
    
    s=BuildZeta(2*m,m,ZetaH,ZetaL,threads);
  }
  
  ~fftpad() {
    deleteAlign(ZetaL);
    deleteAlign(ZetaH);
    delete Forwards;
    delete Backwards;
  }
  
  void backwards(Complex *f, Complex *u);
  void forwards(Complex *f, Complex *u);
};
  
// Compute the scrambled virtual m-padded complex Fourier transform of M complex
// vectors, each of length 2m-1 with the origin at index m-1
// (i.e. physical wavenumber k=-m+1 to k=m-1).
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
  unsigned int m;
  unsigned int M;
  unsigned int s;
  unsigned int stride;
  mfft1d *Backwards;
  mfft1d *Forwards;
  Complex *ZetaH, *ZetaL;
public:  
  fft0pad(unsigned int m, unsigned int M, unsigned int stride, Complex *u=NULL)
    : m(m), M(M), stride(stride) {
    Backwards=new mfft1d(m,1,M,stride,1,u);
    Forwards=new mfft1d(m,-1,M,stride,1,u);
    
    s=BuildZeta(3*m,m,ZetaH,ZetaL);
  }
  
  ~fft0pad() {
    deleteAlign(ZetaL);
    deleteAlign(ZetaH);
    delete Forwards;
    delete Backwards;
  }
  
  // Unscramble indices.
  inline unsigned findex(unsigned i) {
    return i < m-1 ? 3*i : 3*i+4-3*m;
  }

  inline unsigned uindex(unsigned i) {
    return i > 0 ? (i < m ? 3*i-1 : 3*m-3) : 3*m-1;
  }

  void backwards(Complex *f, Complex *u);
  void forwards(Complex *f, Complex *u);
};
  

// In-place implicitly dealiased 2D complex convolution.
class ImplicitConvolution2 : public ThreadBase {
protected:
  unsigned int mx,my;
  Complex *u1,*v1;
  Complex *u2,*v2;
  unsigned int M;
  fftpad *xfftpad;
  ImplicitConvolution *yconvolve;
  Complex **U2,**V2;
  bool allocated;
  Complex **u;
  Complex ***V;
public:  
  void initpointers(Complex **&u, Complex ***&V, unsigned int threads) {
    u=new Complex *[threads];
    V=new Complex **[threads];
    unsigned int stride=my*M;
    for(unsigned int i=0; i < threads; ++i) {
      unsigned int istride=i*stride;
      u[i]=u1+istride;
      Complex *vi=v1+istride;
      yconvolve->initpointers(V[i],vi);
    }
  }
    
  void deletepointers(Complex **&u, Complex ***&V, unsigned int threads) {
    for(unsigned int i=0; i < threads; ++i)
      yconvolve->deletepointers(V[i]);
    delete [] V;
    delete [] u;
  }
  
  void initpointers2(Complex **&U2, Complex **&V2, Complex *u2, Complex *v2,
                     unsigned int stride) {
    U2=new Complex *[M];
    V2=new Complex *[M];
    for(unsigned int s=0; s < M; ++s) {
      unsigned int sstride=s*stride;
      U2[s]=u2+sstride;
      V2[s]=v2+sstride;
    }
  }
  
  void deletepointers2(Complex **&U2, Complex **&V2) {
    delete [] V2;
    delete [] U2;
  }
  
  void init(unsigned int ny, unsigned int stride) {
    xfftpad=new fftpad(mx,ny,ny,u2);
    yconvolve=new ImplicitConvolution(my,u1,v1,M);
    yconvolve->Threads(1);
    
    initpointers(u,V,threads);
    initpointers2(U2,V2,u2,v2,stride);
  }
  
  unsigned int getmx() {return mx;}
  unsigned int getmy() {return my;}
  unsigned int getM() {return M;}

  void set(unsigned int& ny, unsigned int& stride) {
    if(ny == 0) {
      ny=my;
      stride=mx*my;
    }
  }
  
  // u1 and v1 are temporary arrays of size my*M*threads.
  // u2 and v2 are temporary arrays of size mx*my*M.
  // M is the number of data blocks (each corresponding to a dot product term).
  // threads is the number of threads to use in the outer subconvolution loop.
  // ny and stride are used by the MPI interface and should not be set directly.
  ImplicitConvolution2(unsigned int mx, unsigned int my,
                       Complex *u1, Complex *v1, Complex *u2, Complex *v2,
                       unsigned int M=1, unsigned int threads=fftw::maxthreads,
                       unsigned int ny=0, unsigned int stride=0) :
    ThreadBase(threads), mx(mx), my(my), u1(u1), v1(v1), u2(u2), v2(v2), M(M),
    allocated(false) {
    set(ny,stride);
    init(ny,stride);
  }
  
  ImplicitConvolution2(unsigned int mx, unsigned int my, unsigned int M=1,
                       unsigned int threads=fftw::maxthreads,
                       unsigned int ny=0, unsigned int stride=0) : 
    ThreadBase(threads), mx(mx), my(my), M(M), allocated(true) {
    unsigned int n1=my*M*threads;
    u1=ComplexAlign(n1);
    v1=ComplexAlign(n1);
    set(ny,stride);
    unsigned int n2=stride*M;
    u2=ComplexAlign(n2);
    v2=ComplexAlign(n2);
    allocated=true;
    init(ny,stride);
  }
  
  virtual ~ImplicitConvolution2() {
    deletepointers2(U2,V2);
    deletepointers(u,V,threads);
    
    delete yconvolve;
    delete xfftpad;
    
    if(allocated) {
      deleteAlign(v2);
      deleteAlign(u2);
      deleteAlign(v1);
      deleteAlign(u1);
    }
  }
  
  void backwards(Complex **F, Complex *u2,
                 unsigned int stride, unsigned int offset) {
    for(unsigned int s=0; s < M; ++s)
      xfftpad->backwards(F[s]+offset,u2+s*stride);
  }

  void subconvolution(Complex **F, Complex **G, Complex **u, Complex ***V,
                      unsigned int start, unsigned int stop) {
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif    
    for(unsigned int i=start; i < stop; i += my) {
      unsigned int thread=get_thread_num();
      yconvolve->convolve(F,G,u[thread],V[thread],i);
    }
  }
  
  void forwards(Complex *f, Complex *u2) {
    xfftpad->forwards(f,u2);
  }
  
  virtual void convolve(Complex **F, Complex **G, Complex **u, Complex ***V,
                        Complex **U2, Complex **V2, unsigned int offset=0) {
    Complex *u2=U2[0];
    Complex *v2=V2[0];
    unsigned int size=mx*my;
    
    backwards(F,u2,size,offset);
    backwards(G,v2,size,offset);
    
    subconvolution(F,G,u,V,offset,size+offset);
    subconvolution(U2,V2,u,V,0,size);
    
    forwards(F[0]+offset,u2);
  }
  
  // F and G are distinct pointers to M distinct data blocks each of size mx*my,
  // shifted by offset (contents not preserved).
  // The output is returned in F[0].
  void convolve(Complex **F, Complex **G, unsigned int offset=0) {
    convolve(F,G,u,V,U2,V2,offset);
  }

  // Convolution for special case M=1:
  void convolve(Complex *f, Complex *g) {
    convolve(&f,&g);
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

// Enforce 2D Hermiticity using specified (x >= 0,y=0) data.
inline void HermitianSymmetrizeX(unsigned int mx, unsigned int my,
                                 Complex *f)
{
  return HermitianSymmetrizeX(mx,my,mx-1,f);
}
                                 
// Enforce 3D Hermiticity using specified (x,y > 0,z=0) and (x >= 0,y=0,z=0)
// data.
inline void HermitianSymmetrizeXY(unsigned int mx, unsigned int my,
                                  unsigned int mz, Complex *f)
{
  int stride=(2*my-1)*mz;
  int mxstride=mx*stride;
  unsigned int myz=my*mz;
  unsigned int origin=(mx-1)*stride+myz-mz;
  
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
  Complex *u1,*v1,*w1;
  Complex *u2,*v2;
  unsigned int M;
  fft0pad *xfftpad;
  Complex **U2,**V2;
  bool allocated;
public:  
  
  void initpointers2(Complex **&U2, Complex **&V2, Complex *u2, Complex *v2,
                     unsigned int stride) 
  {
    U2=new Complex *[M];
    V2=new Complex *[M];
    for(unsigned int s=0; s < M; ++s) {
      unsigned int sstride=s*stride;
      U2[s]=u2+sstride;
      V2[s]=v2+sstride;
    }
  }
  
  void deletepointers2(Complex **&U2, Complex **&V2) {
    delete [] V2;
    delete [] U2;
  }
  
  void init(unsigned int ny, unsigned int stride) {
    xfftpad=new fft0pad(mx,ny,ny,u2);
    initpointers2(U2,V2,u2,v2,stride);
  }

  void set(unsigned int& ny, unsigned int& stride) {
    if(ny == 0) {
      ny=my;
      stride=(mx+1)*my;
    }
  }
  
  ImplicitHConvolution2Base(unsigned int mx, unsigned int my,
                            Complex *u1, Complex *v1, Complex *w1,
                            Complex *u2, Complex *v2,
                            unsigned int M=1, 
                            unsigned int threads=fftw::maxthreads,
                            unsigned int ny=0, unsigned int stride=0) :
    ThreadBase(threads), mx(mx), my(my), u1(u1), v1(v1), w1(w1), u2(u2), v2(v2),
    M(M), allocated(false) {
    set(ny,stride);
    init(ny,stride);
  }
  
  ImplicitHConvolution2Base(unsigned int mx, unsigned int my, unsigned int M=1,
                            unsigned int threads=fftw::maxthreads,
                            unsigned int ny=0, unsigned int stride=0) :
    ThreadBase(threads), mx(mx), my(my), M(M) {
    unsigned n1=(my/2+1)*M*threads;
    u1=ComplexAlign(n1);
    v1=ComplexAlign(n1);
    w1=ComplexAlign(3*M*threads);
    set(ny,stride);
    unsigned int n2=stride*M;
    u2=ComplexAlign(n2);
    v2=ComplexAlign(n2);
    allocated=true;
    init(ny,stride);
  }
  
  ~ImplicitHConvolution2Base() {
    deletepointers2(U2,V2);
    
    if(allocated) {
      deleteAlign(v2);
      deleteAlign(u2);
      deleteAlign(w1);
      deleteAlign(v1);
      deleteAlign(u1);
    }
    delete xfftpad;
  }
};
  
// In-place implicitly dealiased 2D Hermitian convolution.
class ImplicitHConvolution2 : public ImplicitHConvolution2Base {
protected:
  ImplicitHConvolution *yconvolve;
  Complex ***U;
  Complex **w,**v;
public:
  void initpointers(Complex ***&U, Complex **&v, Complex **&w,
                    unsigned int threads) {
    U=new Complex **[threads];
    v=new Complex *[threads];
    w=new Complex *[threads];
    int cy1=my/2+1;
    unsigned int stride=cy1*M;
    unsigned int M3=3*M;
    for(unsigned int i=0; i < threads; ++i) {
      unsigned int istride=i*stride;
      Complex *ui=u1+istride;
      yconvolve->initpointers(U[i],ui);
      v[i]=v1+istride;
      w[i]=w1+i*M3;
    }
  }
    
  void deletepointers(Complex ***&U, Complex **&v, Complex **&w,
                      unsigned int threads) {
    for(unsigned int i=0; i < threads; ++i)
      yconvolve->deletepointers(U[i]);
      
    delete [] w;
    delete [] v;
    delete [] U;
  }
  
  void initconvolve() {
    yconvolve=new ImplicitHConvolution(my,u1,v1,w1,M);
    yconvolve->Threads(1);
    initpointers(U,v,w,threads);
  }
    
  unsigned int getmx() {return mx;}
  unsigned int getmy() {return my;}
  unsigned int getM() {return M;}

  // u1 and v1 are temporary arrays of size (my/2+1)*M*threads.
  // w1 is a temporary array of size 3*M*threads.
  // u2 and v2 are temporary arrays of size (mx+1)*my*M;
  // M is the number of data blocks (each corresponding to a dot product term).
  // ny and stride are used by the MPI interface and should not be set directly.
  ImplicitHConvolution2(unsigned int mx, unsigned int my,
                        Complex *u1, Complex *v1, Complex *w1,
                        Complex *u2, Complex *v2, unsigned int M=1,
                        unsigned int threads=fftw::maxthreads,
                        unsigned int ny=0, unsigned int stride=0) :
    ImplicitHConvolution2Base(mx,my,u1,v1,w1,u2,v2,M,threads,ny,stride) {
    initconvolve();
  }
  
  ImplicitHConvolution2(unsigned int mx, unsigned int my, unsigned int M=1,
                        unsigned int threads=fftw::maxthreads,
                        unsigned int ny=0, unsigned int stride=0) :
    ImplicitHConvolution2Base(mx,my,M,threads,ny,stride) {
    initconvolve();
  }
  
  virtual ~ImplicitHConvolution2() {
    deletepointers(U,v,w,threads);
    delete yconvolve;
  }
  
  void backwards(Complex **F, Complex *u2, unsigned int ny,
                 unsigned int stride, bool symmetrize, unsigned int offset) {
    for(unsigned int s=0; s < M; ++s) {
      Complex *f=F[s]+offset;
      if(symmetrize)
        HermitianSymmetrizeX(mx,ny,f);
      xfftpad->backwards(f,u2+s*stride);
    }
  }

  void subconvolution(Complex **F, Complex **G,
                      Complex ***U, Complex **v, Complex **w,
                      unsigned int start, unsigned int stop) {
    unsigned int stride=my;
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif    
    for(unsigned int i=start; i < stop; i += stride) {
      unsigned int thread=get_thread_num();
      yconvolve->convolve(F,G,U[thread],v[thread],w[thread],i);
    }
  }  
  
  void forwards(Complex *f, Complex *u2) {
    xfftpad->forwards(f,u2);
  }
  
  // F and G are distinct pointers to M distinct data blocks each of size 
  // (2mx-1)*my, shifted by offset (contents not preserved).
  // The output is returned in F[0].
  virtual void convolve(Complex **F, Complex **G, Complex ***U, Complex **v,
                        Complex **w, Complex **U2, Complex **V2,
                        bool symmetrize=true, unsigned int offset=0) {
    Complex *u2=U2[0];
    Complex *v2=V2[0];
    
    unsigned int usize=(mx+1)*my;
    
    backwards(F,u2,my,usize,symmetrize,offset);
    backwards(G,v2,my,usize,symmetrize,offset);
    
    subconvolution(F,G,U,v,w,offset,(2*mx-1)*my+offset);
    subconvolution(U2,V2,U,v,w,0,usize);
    
    forwards(F[0]+offset,u2);
  }
  
  void convolve(Complex **F, Complex **G, bool symmetrize=true,
                unsigned int offset=0) {
    convolve(F,G,U,v,w,U2,V2,symmetrize,offset);
  }
  
  // Constructor for special case M=1:
  void convolve(Complex *f, Complex *g, bool symmetrize=true) {
    convolve(&f,&g,symmetrize);
  }
};


// In-place implicitly dealiased 3D complex convolution.
class ImplicitConvolution3 : public ThreadBase {
protected:
  unsigned int mx,my,mz;
  Complex *u1,*v1;
  Complex *u2,*v2;
  Complex *u3,*v3;
  unsigned int M;
  fftpad *xfftpad;
  ImplicitConvolution2 *yzconvolve;
  Complex **U3,**V3;
  bool allocated;
  Complex **u;
  Complex ***V;
  Complex ***U2,***V2;
public:  
  void initpointers2(Complex ***&U2, Complex ***&V2, unsigned int threads,
                     unsigned int stride2) {
    U2=new Complex **[threads];
    V2=new Complex **[threads];
    unsigned int stride=stride2*M;
    for(unsigned int i=0; i < threads; ++i) {
      unsigned int istride=i*stride;
      Complex *u2i=u2+istride;
      Complex *v2i=v2+istride;
      yzconvolve->initpointers2(U2[i],V2[i],u2i,v2i,stride2);
    }
  }
    
  void deletepointers2(Complex ***&U2, Complex ***&V2, unsigned int threads) {
    for(unsigned int i=0; i < threads; ++i)
      yzconvolve->deletepointers2(U2[i],V2[i]);
    
    delete [] V2;
    delete [] U2;
  }
  
  void initpointers3(Complex **&U3, Complex **&V3, Complex *u3, Complex *v3,
                     unsigned int stride) {
    U3=new Complex *[M];
    V3=new Complex *[M];
    for(unsigned int s=0; s < M; ++s) {
      unsigned int sstride=s*stride;
      U3[s]=u3+sstride;
      V3[s]=v3+sstride;
    }
  }
  
  void deletepointers3(Complex **&U3, Complex **&V3) {
    delete [] V3;
    delete [] U3;
  }
  
  void init(unsigned int ny, unsigned int nz, unsigned int stride2,
            unsigned int stride3) {
    unsigned int nyz=ny*nz;
    xfftpad=new fftpad(mx,nyz,nyz,u3);
    
    if(nz == mz) {
      yzconvolve=new ImplicitConvolution2(my,mz,u1,v1,u2,v2,M,1);
      yzconvolve->initpointers(u,V,threads);
      initpointers2(U2,V2,threads,stride2);
      initpointers3(U3,V3,u3,v3,stride3);
    } else yzconvolve=NULL;
  }
  
  unsigned int getmx() {return mx;}
  unsigned int getmy() {return my;}
  unsigned int getmz() {return mz;}
  unsigned int getM() {return M;}

  void set(unsigned int& ny,unsigned int& nz,
           unsigned int& stride2, unsigned int& stride3) {
    if(ny == 0) {
      ny=my;
      nz=mz;
      stride2=my*mz;
      stride3=mx*my*mz;
    }
  }
  
  // u1 and v1 are temporary arrays of size mz*M*threads.
  // u2 and v2 are temporary arrays of size my*mz*M*threads.
  // u3 and v3 are temporary arrays of size mx*my*mz*M.
  // M is the number of data blocks (each corresponding to a dot product term).
  // threads is the number of threads to use in the outer subconvolution loop.
  // ny, nz, stride2, and stride3 are used by the MPI interface
  // and should not be set directly.
  ImplicitConvolution3(unsigned int mx, unsigned int my, unsigned int mz,
                       Complex *u1, Complex *v1,
                       Complex *u2, Complex *v2,
                       Complex *u3, Complex *v3,
                       unsigned int M=1, unsigned int threads=fftw::maxthreads,
                       unsigned int ny=0, unsigned int nz=0,
                       unsigned int stride2=0, unsigned int stride3=0) :
    ThreadBase(threads), mx(mx), my(my), mz(mz), u1(u1), v1(v1), u2(u2), v2(v2),
    u3(u3), v3(v3), M(M), allocated(false) {
    set(ny,nz,stride2,stride3);
    init(ny,nz,stride2,stride3);
  }

  // innerthreads, ny, nz, stride2, and stride3 are used by the MPI interface
  // and should not be set directly.
  ImplicitConvolution3(unsigned int mx, unsigned int my, unsigned int mz,
                       unsigned int M=1, unsigned int threads=fftw::maxthreads,
                       unsigned int innerthreads=1,
                       unsigned int ny=0, unsigned int nz=0,
                       unsigned int stride2=0, unsigned int stride3=0) :
    ThreadBase(threads), mx(mx), my(my), mz(mz), M(M), allocated(true) {
    unsigned int n1=mz*M*threads*innerthreads;
    u1=ComplexAlign(n1);
    v1=ComplexAlign(n1);
    set(ny,nz,stride2,stride3);
    unsigned int n2=stride2*M*threads;
    unsigned int n3=stride3*M;
    u2=ComplexAlign(n2);
    v2=ComplexAlign(n2);
    u3=ComplexAlign(n3);
    v3=ComplexAlign(n3);
    init(ny,nz,stride2,stride3);
  }
  
  virtual ~ImplicitConvolution3() {
    if(yzconvolve) {
      deletepointers3(U3,V3);
      deletepointers2(U2,V2,threads);
      yzconvolve->deletepointers(u,V,threads);
      delete yzconvolve;
    }
    delete xfftpad;
    
    if(allocated) {
      deleteAlign(v3);
      deleteAlign(u3);
      deleteAlign(v2);
      deleteAlign(u2);
      deleteAlign(v1);
      deleteAlign(u1);
    }
  }
  
  void backwards(Complex **F, Complex *u3,
                 unsigned int stride, unsigned int offset) {
    for(unsigned int s=0; s < M; ++s)
      xfftpad->backwards(F[s]+offset,u3+s*stride);
  }

  void subconvolution(Complex **F, Complex **G, Complex **u, Complex ***V,
                      Complex ***U2, Complex ***V2,
                      unsigned int start, unsigned int stop,
                      unsigned int stride) {
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif    
    for(unsigned int i=start; i < stop; i += stride) {
      unsigned int thread=get_thread_num();
      yzconvolve->convolve(F,G,u+thread,V+thread,U2[thread],V2[thread],i);
    }
  }

  void forwards(Complex *f, Complex *u3) {
    xfftpad->forwards(f,u3);
  }
  
  // F and G are distinct pointers to M distinct data blocks each of size
  // mx*my*mz, shifted by offset (contents not preserved).
  // The output is returned in F[0].
  virtual void convolve(Complex **F, Complex **G, Complex **u, Complex ***V,
                        Complex ***U2, Complex ***V2, Complex **U3, Complex **V3,
                        unsigned int offset=0) {
    Complex *u3=U3[0];
    Complex *v3=V3[0];
    unsigned int myz=my*mz;
    unsigned int size=mx*myz;
    
    backwards(F,u3,size,offset);
    backwards(G,v3,size,offset);
    
    subconvolution(F,G,u,V,U2,V2,offset,size+offset,myz);
    subconvolution(U3,V3,u,V,U2,V2,0,size,myz);
    
    forwards(F[0]+offset,u3);
  }
  
  void convolve(Complex **F, Complex **G, unsigned int offset=0) {
    convolve(F,G,u,V,U2,V2,U3,V3,offset);
  }
  
  // Constructor for special case M=1:
  void convolve(Complex *f, Complex *g) {
    convolve(&f,&g);
  }
};


// In-place implicitly dealiased 3D Hermitian convolution.
class ImplicitHConvolution3 : public ThreadBase {
protected:
  unsigned int mx,my,mz;
  Complex *u1,*v1,*w1;
  Complex *u2,*v2;
  Complex *u3,*v3;
  unsigned int M;
  fft0pad *xfftpad;
  ImplicitHConvolution2 *yzconvolve;
  Complex **U3,**V3;
  bool allocated;
  Complex ***U;
  Complex **v,**w;
  Complex ***U2,***V2;
public:  
  
  void initpointers2(Complex ***&U2, Complex ***&V2, unsigned int threads,
                     unsigned int stride2) {
    U2=new Complex **[threads];
    V2=new Complex **[threads];
    unsigned int stride=stride2*M;
    for(unsigned int i=0; i < threads; ++i) {
      unsigned int istride=i*stride;
      Complex *u2i=u2+istride;
      Complex *v2i=v2+istride;
      yzconvolve->initpointers2(U2[i],V2[i],u2i,v2i,stride2);
    }
  }
    
  void deletepointers2(Complex ***&U2, Complex ***&V2, unsigned int threads) {
    for(unsigned int i=0; i < threads; ++i)
      yzconvolve->deletepointers2(U2[i],V2[i]);
    delete [] V2;
    delete [] U2;
  }
    
  void initpointers3(Complex **&U3, Complex **&V3, Complex *u3, Complex *v3,
                     unsigned int stride) {
    U3=new Complex *[M];
    V3=new Complex *[M];
    for(unsigned int s=0; s < M; ++s) {
      unsigned int sstride=s*stride;
      U3[s]=u3+sstride;
      V3[s]=v3+sstride;
    }
  }
  
  void deletepointers3(Complex **&U3, Complex **&V3) {
    delete [] V3;
    delete [] U3;
  }
    
  void init(unsigned int ny, unsigned int nz, unsigned int stride2,
            unsigned int stride3) {
    unsigned int nyz=ny*nz;
    xfftpad=new fft0pad(mx,nyz,nyz,u3);

    if(nz == mz) {
      yzconvolve=new ImplicitHConvolution2(my,mz,u1,v1,w1,u2,v2,M,1);
      yzconvolve->initpointers(U,v,w,threads);
      initpointers2(U2,V2,threads,stride2);
      initpointers3(U3,V3,u3,v3,stride3);
    } else yzconvolve=NULL;
  }
  
  unsigned int getmx() {return mx;}
  unsigned int getmy() {return my;}
  unsigned int getmz() {return mz;}
  unsigned int getM() {return M;}

  void set(unsigned int& ny,unsigned int& nz,
           unsigned int& stride2, unsigned int& stride3) {
    if(ny == 0) {
      ny=2*my-1;
      nz=mz;
      stride2=(my+1)*mz;
      stride3=(mx+1)*ny*mz;
    }
  }
  
  // u1 and v1 are temporary arrays of size (mz/2+1)*M*threads.
  // w1 is a temporary array of size 3*M*threads.
  // u2 and v2 are temporary array of size (my+1)*mz*M*threads.
  // u3 and v3 are temporary arrays of size (mx+1)*(2my-1)*mz*M.
  // M is the number of data blocks (each corresponding to a dot product term).
  // threads is the number of threads to use in the outer subconvolution loop.
  // innerthreads, ny, nz, stride2, and stride3 are used by the MPI interface
  // and should not be set directly.
  ImplicitHConvolution3(unsigned int mx, unsigned int my, unsigned int mz,
                        Complex *u1, Complex *v1, Complex *w1,
                        Complex *u2, Complex *v2,
                        Complex *u3, Complex *v3,
                        unsigned int M=1, unsigned int threads=fftw::maxthreads,
                        unsigned int ny=0, unsigned int nz=0,
                        unsigned int stride2=0, unsigned int stride3=0) :
    ThreadBase(threads), mx(mx), my(my), mz(mz), u1(u1), v1(v1), w1(w1),
    u2(u2), v2(v2), u3(u3), v3(v3), M(M), allocated(false) {
    set(ny,nz,stride2,stride3);
    init(ny,nz,stride2,stride3);
  }
  
  ImplicitHConvolution3(unsigned int mx, unsigned int my, unsigned int mz,
                        unsigned int M=1, unsigned int threads=fftw::maxthreads,
                        unsigned int innerthreads=1,
                        unsigned int ny=0, unsigned int nz=0,
                        unsigned int stride2=0, unsigned int stride3=0) :
    ThreadBase(threads), mx(mx), my(my), mz(mz), M(M), allocated(true) {
    unsigned int n1=(mz/2+1)*M*threads*innerthreads;
    u1=ComplexAlign(n1);
    v1=ComplexAlign(n1);
    w1=ComplexAlign(3*M*threads*innerthreads); 
    set(ny,nz,stride2,stride3);
    unsigned int n2=stride2*M*threads;
    unsigned int n3=stride3*M;
    u2=ComplexAlign(n2);
    v2=ComplexAlign(n2);
    u3=ComplexAlign(n3);
    v3=ComplexAlign(n3);
    init(ny,nz,stride2,stride3);
  }
  
  ~ImplicitHConvolution3() {
    if(yzconvolve) {
      deletepointers3(U3,V3);
      deletepointers2(U2,V2,threads);
      yzconvolve->deletepointers(U,v,w,threads);
      delete yzconvolve;
    }
    delete xfftpad;
    
    if(allocated) {
      deleteAlign(v3);
      deleteAlign(u3);
      deleteAlign(v2);
      deleteAlign(u2);
      deleteAlign(w1);
      deleteAlign(v1);
      deleteAlign(u1);
    }
  }
  
  virtual void HermitianSymmetrize(Complex *f, Complex *u) {
    HermitianSymmetrizeXY(mx,my,mz,f);
  }
  
  void backwards(Complex **F, Complex *u3, unsigned int stride,
                 bool symmetrize, unsigned int offset) {
    for(unsigned int s=0; s < M; ++s) {
      Complex *f=F[s]+offset;
      Complex *u=u3+s*stride;
      if(symmetrize)
        HermitianSymmetrize(f,u);
      xfftpad->backwards(f,u);
    }
  }

  void subconvolution(Complex **F, Complex **G, Complex ***U, Complex **v,
                      Complex **w, Complex ***U2, Complex ***V2,
                      unsigned int start, unsigned int stop,
                      unsigned int stride) {
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif    
    for(unsigned int i=start; i < stop; i += stride) {
      unsigned int thread=get_thread_num();
      yzconvolve->convolve(F,G,U+thread,v+thread,w+thread,
                           U2[thread],V2[thread],false,i);
    }
  }

  void forwards(Complex *f, Complex *u3) {
    xfftpad->forwards(f,u3);
  }
  
  // F and G are distinct pointers to M distinct data blocks each of size 
  // (2mx-1)*(2my-1)*mz, shifted by offset (contents not preserved).
  // The output is returned in F[0].
  virtual void convolve(Complex **F, Complex **G, Complex ***U, Complex **v,
                        Complex **w, Complex ***U2, Complex ***V2,
                        Complex **U3, Complex **V3, bool symmetrize=true,
                        unsigned int offset=0) {
    Complex *u3=U3[0];
    Complex *v3=V3[0];
    unsigned int nymz=(2*my-1)*mz;
    unsigned int usize=(mx+1)*nymz;
    
    backwards(F,u3,usize,symmetrize,offset);
    backwards(G,v3,usize,symmetrize,offset);
    
    subconvolution(F,G,U,v,w,U2,V2,offset,(2*mx-1)*nymz+offset,nymz);
    subconvolution(U3,V3,U,v,w,U2,V2,0,usize,nymz);
    
    forwards(F[0]+offset,u3);
  }
    
  void convolve(Complex **F, Complex **G, bool symmetrize=true,
                unsigned int offset=0) {
    convolve(F,G,U,v,w,U2,V2,U3,V3,symmetrize,offset);
  }
  
  // Constructor for special case M=1:
  void convolve(Complex *f, Complex *g, bool symmetrize=true) {
    convolve(&f,&g,symmetrize);
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
    
    threads=cro->Threads();
    
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
    
    threads=cro->Threads();
    
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
    
    threads=cr->Threads();
    
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
  
// Compute the scrambled virtual 2m-padded complex Fourier transform of M
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
    
    threads=Forwards->Threads();
    
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
  // u2 is a temporary arrays of size 2mx*(my+1).
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

}

#endif
