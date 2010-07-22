/* Implicitly and explicitly dealiased convolution routines.
   Copyright (C) 2010 John C. Bowman and Malcolm Roberts, University of Alberta

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
                       Complex *&ZetaH, Complex *&ZetaL);

// In-place explicitly dealiased 1D complex convolution.
class ExplicitConvolution {
protected:
  unsigned int n,m;
  fft1d *Backwards,*Forwards;
public:  
  
  // u is a temporary array of size n.
  ExplicitConvolution(unsigned int n, unsigned int m, Complex *u) :
    n(n), m(m) {
    Backwards=new fft1d(n,1,u);
    Forwards=new fft1d(n,-1,u);
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

class CDot {
protected:
  unsigned int m;
  unsigned int M;
  
public:
  CDot(unsigned int m, unsigned int M) : 
    m(m), M(M) {}
  
  // a[0][k]=sum_i a[i][k]*b[i][k]
  void mult(Complex *a, Complex **B, unsigned int offset=0) {
    if(M == 1) { // a[k]=a[k]*b[k]
      Complex *B0=B[0]+offset;
      for(unsigned int k=0; k < m; ++k) {
        Complex *p=a+k;
#ifdef __SSE2__      
        STORE(p,ZMULT(LOAD(p),LOAD(B0+k)));
#else
        Complex ak=*p;
        Complex bk=*(B0+k);
        p->re=ak.re*bk.re-ak.im*bk.im;
        p->im=ak.re*bk.im+ak.im*bk.re;
#endif      
      }
    } else if(M == 2) {
      Complex *a1=a+m;
      Complex *B0=B[0]+offset;
      Complex *B1=B[1]+offset;
      for(unsigned int k=0; k < m; ++k) {
        Complex *p=a+k;
#ifdef __SSE2__
        STORE(p,ZMULT(LOAD(p),LOAD(B0+k))+ZMULT(LOAD(a1+k),LOAD(B1+k)));
#else
        Complex ak=*p;
        Complex bk=B0[k];
        double re=ak.re*bk.re-ak.im*bk.im;
        double im=ak.re*bk.im+ak.im*bk.re;
        ak=a1[k];
        bk=B1[k];
        re += ak.re*bk.re-ak.im*bk.im;
        im += ak.re*bk.im+ak.im*bk.re; 
        p->re=re;
        p->im=im;
#endif      
      }
    } else if(M == 3) {
      Complex *a1=a+m;
      Complex *a2=a+2*m;
      Complex *B0=B[0]+offset;
      Complex *B1=B[1]+offset;
      Complex *B2=B[2]+offset;
      for(unsigned int k=0; k < m; ++k) {
        Complex *p=a+k;
#ifdef __SSE2__
        STORE(p,ZMULT(LOAD(p),LOAD(B0+k))+ZMULT(LOAD(a1+k),LOAD(B1+k))
              +ZMULT(LOAD(a2+k),LOAD(B2+k)));
#else
        Complex ak=*p;
        Complex bk=B0[k];
        double re=ak.re*bk.re-ak.im*bk.im;
        double im=ak.re*bk.im+ak.im*bk.re;
        ak=a1[k];
        bk=B1[k];
        re += ak.re*bk.re-ak.im*bk.im;
        im += ak.re*bk.im+ak.im*bk.re; 
        ak=a2[k];
        bk=B2[k];
        re += ak.re*bk.re-ak.im*bk.im;
        im += ak.re*bk.im+ak.im*bk.re; 
        p->re=re;
        p->im=im;
#endif      
      }
    } else {
      Complex *A=a-offset;
      Complex *B0=B[0];
      unsigned int stop=offset+m;
      for(unsigned int k=offset; k < stop; ++k) {
        Complex *p=A+k;
#ifdef __SSE2__      
        Vec sum=ZMULT(LOAD(p),LOAD(B0+k));
        for(unsigned int i=1; i < M; ++i)
          sum += ZMULT(LOAD(p+m*i),LOAD(B[i]+k));
        STORE(p,sum);
#else
        Complex ak=*p;
        Complex bk=B0[k];
        double re=ak.re*bk.re-ak.im*bk.im;
        double im=ak.re*bk.im+ak.im*bk.re;
        for(unsigned int i=1; i < M; ++i) {
          Complex ak=p[m*i];
          Complex bk=B[i][k];
          re += ak.re*bk.re-ak.im*bk.im;
          im += ak.re*bk.im+ak.im*bk.re; 
        }
        p->re=re;
        p->im=im;
#endif      
      }
    }
  }
};

class Dot {
protected:
  unsigned int m;
  unsigned int M;
  bool odd;
  unsigned int stride;
public:
  Dot(unsigned int m, unsigned int M) : 
    m(m), M(M), odd(m % 2 == 1), stride(odd ? m+1 : m+2) {}
  
  // a[0][k]=sum_i a[i][k]*b[i][k]
  void mult(double *a, double **B, unsigned int offset=0) {
    unsigned int m1=m-1;
    if(M == 1) { // a[k]=a[k]*b[k]
      double *B0=B[0]+offset;
#ifdef __SSE2__        
      for(unsigned int k=0; k < m1; k += 2)
        STORE(a+k,LOAD(a+k)*LOAD(B0+k));
      if(odd)
        STORE(a+m1,LOAD(a+m1)*LOAD(B0+m1));
#else        
      for(unsigned int k=0; k < m; ++k)
        a[k] *= B0[k];
#endif        
    } else if(M == 2) {
      double *a1=a+stride;
      double *B0=B[0]+offset;
      double *B1=B[1]+offset;
#ifdef __SSE2__        
      for(unsigned int k=0; k < m1; k += 2)
        STORE(a+k,LOAD(a+k)*LOAD(B0+k)+LOAD(a1+k)*LOAD(B1+k));
      if(odd)
        STORE(a+m1,LOAD(a+m1)*LOAD(B0+m1)+LOAD(a1+m1)*LOAD(B1+m1));
#else        
      for(unsigned int k=0; k < m; ++k)
        a[k]=a[k]*B0[k]+a1[k]*B1[k];
#endif        
    } else if(M == 3) {
      double *a1=a+stride;
      double *a2=a1+stride;
      double *B0=B[0]+offset;
      double *B1=B[1]+offset;
      double *B2=B[2]+offset;
#ifdef __SSE2__        
      for(unsigned int k=0; k < m1; k += 2)
        STORE(a+k,LOAD(a+k)*LOAD(B0+k)+LOAD(a1+k)*LOAD(B1+k)+
              LOAD(a2+k)*LOAD(B2+k));
      if(odd)
        STORE(a+m1,LOAD(a+m1)*LOAD(B0+m1)+LOAD(a1+m1)*LOAD(B1+m1)+
              LOAD(a2+m1)*LOAD(B2+m1));
#else        
      for(unsigned int k=0; k < m; ++k)
        a[k]=a[k]*B0[k]+a1[k]*B1[k]+a2[k]*B2[k];
#endif        
    } else {
      double *A=a-offset;
      double *B0=B[0];
      unsigned int stop=m1+offset;
#ifdef __SSE2__        
      for(unsigned int k=offset; k < stop; k += 2) {
        double *p=A+k;
        Vec sum=LOAD(p)*LOAD(B0+k);
        for(unsigned int i=1; i < M; ++i)
          sum += LOAD(p+i*stride)*LOAD(B[i]+k);
        STORE(p,sum);
      }
      if(odd) {
        double *p=A+stop;
        Vec sum=LOAD(p)*LOAD(B0+stop);
        for(unsigned int i=1; i < M; ++i)
          sum += LOAD(p+i*stride)*LOAD(B[i]+stop);
        STORE(p,sum);
      }
#else        
      for(unsigned int k=offset; k <= stop; ++k) {
        double *p=A+k;
        double sum=(*p)*B0[k];
        for(unsigned int i=1; i < M; ++i)
          sum += p[i*stride]*B[i][k];
        *p=sum;
      }
#endif        
    }
  }
};
  
class Dot3 {
protected:
  unsigned int m;
  unsigned int M;
  unsigned int twom;
  unsigned int stride;
  
public:
  Dot3(unsigned int m, unsigned int M) : 
    m(m), M(M), twom(2*m), stride(twom+2) {}
  
  // a[0][k]=sum_i a[i][k]*b[i][k]*c[i][k]
  void mult(double *a, double *b, double **C, unsigned int offset=0) {
    unsigned int twom=2*m;
    if(M == 1) { // a[k]=a[k]*b[k]*c[k]
      double *C0=C[0]+offset;
#ifdef __SSE2__        
      for(unsigned int k=0; k < twom; k += 2)
        STORE(a+k,LOAD(a+k)*LOAD(b+k)*LOAD(C0+k));
#else        
      for(unsigned int k=0; k < twom; ++k)
        a[k] *= b[k]*C0[k];
#endif        
    } else if(M == 2) {
      double *a1=a+stride;
      double *b1=b+stride;
      double *C0=C[0]+offset;
      double *C1=C[1]+offset;
#ifdef __SSE2__        
      for(unsigned int k=0; k < twom; k += 2)
        STORE(a+k,LOAD(a+k)*LOAD(b+k)*LOAD(C0+k)+
              LOAD(a1+k)*LOAD(b1+k)*LOAD(C1+k));
#else        
      for(unsigned int k=0; k < twom; ++k)
        a[k]=a[k]*b[k]*C0[k]+a1[k]*b1[k]*C1[k];
#endif        
    } else if(M == 3) {
      double *a1=a+stride;
      double *a2=a1+stride;
      double *b1=b+stride;
      double *b2=b1+stride;
      double *C0=C[0]+offset;
      double *C1=C[1]+offset;
      double *C2=C[2]+offset;
#ifdef __SSE2__        
      for(unsigned int k=0; k < twom; k += 2)
        STORE(a+k,LOAD(a+k)*LOAD(b+k)*LOAD(C0+k)+
              LOAD(a1+k)*LOAD(b1+k)*LOAD(C1+k)+
              LOAD(a2+k)*LOAD(b2+k)*LOAD(C2+k));
#else        
      for(unsigned int k=0; k < twom; ++k)
        a[k]=a[k]*b[k]*C0[k]+a1[k]*b1[k]*C1[k]+a2[k]*b2[k]*C2[k];
#endif        
    } else {
      double *A=a-offset;
      double *B=b-offset;
      double *C0=C[0];
      unsigned int stop=twom+offset;
#ifdef __SSE2__        
      for(unsigned int k=offset; k < stop; k += 2) {
        double *p=A+k;
        double *q=B+k;
        Vec sum=LOAD(p)*LOAD(q)*LOAD(C0+k);
        for(unsigned int i=1; i < M; ++i) {
          unsigned int istride=i*stride;
          sum += LOAD(p+istride)*LOAD(q+istride)*LOAD(C[i]+k);
        }
        STORE(p,sum);
      }
#else        
      for(unsigned int k=offset; k < stop; ++k) {
        double *p=A+k;
        double *q=B+k;
        double sum=(*p)*(*q)*C0[k];
        for(unsigned int i=1; i < M; ++i) {
          unsigned int istride=i*stride;
          sum += p[istride]*q[istride]*C[i][k];
        }
        *p=sum;
      }
#endif        
    }
  }
};
  
// In-place implicitly dealiased 1D complex convolution.
class ImplicitConvolution : public CDot {
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
  
  void init() {
    Backwards=new fft1d(m,1,u,v);
    Forwards=new fft1d(m,-1,u,v);
    
    s=BuildZeta(2*m,m,ZetaH,ZetaL);
    
    V=new Complex *[M];
    for(unsigned int s=0; s < M; ++s)
      V[s]=v+s*m;
  }
  
  // m is the number of Complex data values.
  // u and v are distinct temporary arrays each of size m*M.
  // M is the number of data blocks (each corresponding to a dot product term).
  ImplicitConvolution(unsigned int m, Complex *u, Complex *v, unsigned int M=1)
    : CDot(m,M), m(m), u(u), v(v), M(M), allocated(false) {
    init();
  }
  
  ImplicitConvolution(unsigned int m, unsigned int M=1)
    : CDot(m,M), m(m), u(ComplexAlign(m*M)), v(ComplexAlign(m*M)), M(M),
      allocated(true) {
    init();
  }
  
  ~ImplicitConvolution() {
    delete [] V;
    
    if(allocated) {
      deleteAlign(u);
      deleteAlign(v);
    }
    deleteAlign(ZetaL);
    deleteAlign(ZetaH);
    delete Forwards;
    delete Backwards;
  }
  
  // F and G are pointers to M distinct data blocks each of size m,
  // shifted by offset (contents not preserved).
  // The output is returned in F[0].
  void convolve(Complex **F, Complex **G, unsigned int offset=0);
  
  // Constructor for special case M=1:
  void convolve(Complex *f, Complex *g) {
    convolve(&f,&g);
  }
};

// Out-of-place direct 1D complex convolution.
class DirectConvolution {
protected:
  unsigned int m;
public:  
  DirectConvolution(unsigned int m) : m(m) {}
  
  void convolve(Complex *h, Complex *f, Complex *g);
};

// In-place explicitly dealiased 1D Hermitian convolution.
class ExplicitHConvolution {
protected:
  unsigned int n,m;
  rcfft1d *rc;
  crfft1d *cr;
public:
  // u is a temporary array of size n.
  ExplicitHConvolution(unsigned int n, unsigned int m, Complex *u) :
    n(n), m(m) {
    rc=new rcfft1d(n,u);
    cr=new crfft1d(n,u);
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

// In-place implicitly dealiased 1D Hermitian convolution.
class ImplicitHConvolution : public Dot {
protected:
  unsigned int m;
  unsigned int c;
  Complex *u,*v,*w;
  unsigned int M;
public:
  unsigned int s;
  rcfft1d *rc,*rco;
  crfft1d *cr,*cro;
  Complex *ZetaH,*ZetaL;
  Complex **U;
  bool allocated;
  
  void init() {
    U=new Complex *[M];
    unsigned int cp1=c+1;
    for(unsigned int i=0; i < M; ++i)
      U[i]=u+i*cp1;
    
    rc=new rcfft1d(m,u);
    cr=new crfft1d(m,u);

    rco=new rcfft1d(m,(double *) u,v);
    cro=new crfft1d(m,v,(double *) u);
    
    s=BuildZeta(3*m,2*c == m ? c : cp1,ZetaH,ZetaL);
  }
  
  // m is the number of independent data values.
  // u and v are temporary arrays each of size (m/2+1)*M.
  // w is a temporary array of size 3*M.
  // M is the number of data blocks (each corresponding to a dot product term).
  ImplicitHConvolution(unsigned int m, Complex *u, Complex *v, Complex *w,
                       unsigned int M=1)
    : Dot(m,M), m(m), c(m/2), u(u), v(v), w(w), M(M), allocated(false) {
    init();
  }

  ImplicitHConvolution(unsigned int m, unsigned int M=1)
    : Dot(m,M), m(m), c(m/2), u(ComplexAlign(c*M+M)),v(ComplexAlign(c*M+M)),
      w(ComplexAlign(3*M)), M(M), allocated(true) {
    init();
  }
    
  ~ImplicitHConvolution() {
    delete [] U;
    
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
  
  // F and G are pointers to M distinct data blocks each of size m,
  // shifted by offset (contents not preserved).
  // The output is returned in F[0].
  void convolve(Complex **F, Complex **G, unsigned int offset=0);
  
  // Constructor for special case M=1:
  void convolve(Complex *f, Complex *g) {
    convolve(&f,&g);
  }
};
  
// Out-of-place direct 1D Hermitian convolution.
class DirectHConvolution {
protected:
  unsigned int m;
public:  
  DirectHConvolution(unsigned int m) : m(m) {}
  
// Compute h= f (*) g via direct convolution, where f and g contain the m
// non-negative Fourier components of real functions (contents
// preserved). The output of m complex values is returned in the array h,
// which must be distinct from f and g.
  void convolve(Complex *h, Complex *f, Complex *g);
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
public:  
  fftpad(unsigned int m, unsigned int M,
         unsigned int stride, Complex *f) : m(m), M(M), stride(stride) {
    Backwards=new mfft1d(m,1,M,stride,1,f);
    Forwards=new mfft1d(m,-1,M,stride,1,f);
    
    s=BuildZeta(2*m,m,ZetaH,ZetaL);
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
//   fft0pad fft(m,M,stride);
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
  fft0pad(unsigned int m, unsigned int M, unsigned int stride, Complex *u)
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
  
  void backwards(Complex *f, Complex *u);
  void forwards(Complex *f, Complex *u);
};
  
// In-place explicitly dealiased 2D complex convolution.
class ExplicitConvolution2 {
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
      xBackwards=new mfft1d(nx,1,my,ny,1,f);
      yBackwards=new mfft1d(ny,1,nx,1,ny,f);
      yForwards=new mfft1d(ny,-1,nx,1,ny,f);
      xForwards=new mfft1d(nx,-1,my,ny,1,f);
    } else {
      Backwards=new fft2d(nx,ny,1,f);
      Forwards=new fft2d(nx,ny,-1,f);
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

// In-place implicitly dealiased 2D complex convolution.
class ImplicitConvolution2 {
protected:
  unsigned int mx,my;
  Complex *u1,*v1;
  Complex *u2,*v2;
  unsigned int M;
  fftpad *xfftpad;
  ImplicitConvolution *yconvolve;
  Complex **U2,**V2;
  bool allocated;
public:  
  void init() {
    xfftpad=new fftpad(mx,my,my,u2);
    yconvolve=new ImplicitConvolution(my,u1,v1,M);
    
    U2=new Complex *[M];
    V2=new Complex *[M];
    unsigned int mxy=mx*my;
    for(unsigned int s=0; s < M; ++s) {
      unsigned int smxy=s*mxy;
      U2[s]=u2+smxy;
      V2[s]=v2+smxy;
    }
  }
  
  // u1 and v1 are temporary arrays of size my*M.
  // u2 and v2 are temporary arrays of size mx*my*M.
  // M is the number of data blocks (each corresponding to a dot product term).
  ImplicitConvolution2(unsigned int mx, unsigned int my,
                       Complex *u1, Complex *v1, Complex *u2, Complex *v2,
                       unsigned int M=1) : 
    mx(mx), my(my), u1(u1), v1(v1), u2(u2), v2(v2), M(M), allocated(false) {
    init();
  }
  
  ImplicitConvolution2(unsigned int mx, unsigned int my, unsigned int M=1) :
    mx(mx), my(my), u1(ComplexAlign(my*M)), v1(ComplexAlign(my*M)),
    u2(ComplexAlign(mx*my*M)), v2(ComplexAlign(mx*my*M)),
    M(M), allocated(true) {
    init();
  }
  
  ~ImplicitConvolution2() {
    delete [] V2;
    delete [] U2;
    
    delete yconvolve;
    delete xfftpad;
    
    if(allocated) {
      deleteAlign(v2);
      deleteAlign(u2);
      deleteAlign(v1);
      deleteAlign(u1);
    }
  }
  
  // F and G are pointers to M distinct data blocks each of size mx*my,
  // shifted by offset (contents not preserved).
  // The output is returned in F[0].
  void convolve(Complex **F, Complex **G, unsigned int offset=0) {
    unsigned int mxy=mx*my;
    for(unsigned int s=0; s < M; ++s)
      xfftpad->backwards(F[s]+offset,u2+s*mxy);
    for(unsigned int s=0; s < M; ++s)
      xfftpad->backwards(G[s]+offset,v2+s*mxy);
    
    for(unsigned int i=0; i < mxy; i += my)
      yconvolve->convolve(F,G,i+offset);
    for(unsigned int i=0; i < mxy; i += my)
      yconvolve->convolve(U2,V2,i);
        
    xfftpad->forwards(F[0]+offset,u2);
  }
  
  // Constructor for special case M=1:
  void convolve(Complex *f, Complex *g) {
    convolve(&f,&g);
  }
};

// Out-of-place direct 2D complex convolution.
class DirectConvolution2 {
protected:  
  unsigned int mx,my;
public:
  DirectConvolution2(unsigned int mx, unsigned int my) : mx(mx), my(my) {}
  
  void convolve(Complex *h, Complex *f, Complex *g);
};

// Enforce Hermiticity by symmetrizing 2D data on the X axis.
inline void HermitianSymmetrizeX(unsigned int mx, unsigned int my,
                                 unsigned int xorigin, Complex *f)
{
  unsigned int offset=xorigin*my;
  unsigned int stop=mx*my;
  f[offset].im=0.0;
  for(unsigned int i=my; i < stop; i += my)
    f[offset-i]=conj(f[offset+i]);
}

// Enforce Hermiticity by symmetrizing 3D data on the X axis.
inline void HermitianSymmetrizeX(unsigned int mx, unsigned int my,
                                 unsigned int mz, unsigned int ny,
                                 unsigned int xorigin, unsigned int yorigin, 
                                 Complex *f)
{
  unsigned int nymz=ny*mz;
  unsigned int offset=xorigin*nymz+yorigin*mz;
  unsigned int mxnymz=mx*nymz;
  f[offset].im=0.0;
  for(unsigned int i=nymz; i < mxnymz; i += nymz)
    f[offset-i]=conj(f[offset+i]);
}

// Enforce Hermiticity by symmetrizing 2D data on the XY plane.
inline void HermitianSymmetrizeXY(unsigned int mx, unsigned int my,
                                  unsigned int mz, unsigned int ny,
                                  unsigned int xorigin, unsigned int yorigin, 
                                  Complex *f)
{
  HermitianSymmetrizeX(mx,my,mz,ny,xorigin,yorigin,f);
  
  unsigned int mymz=my*mz;
  unsigned int nymz=ny*mz;
  unsigned int mxnymz=mx*nymz;
  unsigned int offset=xorigin*nymz+yorigin*mz;
  for(unsigned int i=0; i < mxnymz; i += nymz) {
    unsigned int offsetm=offset-i;
    unsigned int offsetp=offset+i;
    for(unsigned int j=mz; j < mymz; j += mz) {
      f[offsetm-j]=conj(f[offsetp+j]);
      f[offsetp-j]=conj(f[offsetm+j]);
    }
  }
}

// In-place explicitly dealiased 2D Hermitian convolution.
class ExplicitHConvolution2 {
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
public:  
  ExplicitHConvolution2(unsigned int nx, unsigned int ny, 
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
      yBackwards=new mcrfft1d(ny,nx,1,nyp,f);
      yForwards=new mrcfft1d(ny,nx,1,nyp,f);
      xForwards=new mfft1d(nx,-1,My,nyp,1,f);
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
  void convolve(Complex *f, Complex *g, bool symmetrize=true);
};

// In-place implicitly dealiased 2D Hermitian convolution.
class ImplicitHConvolution2 {
protected:
  unsigned int mx,my;
  Complex *u1,*v1,*w1;
  Complex *u2,*v2;
  unsigned int M;
  fft0pad *xfftpad;
  ImplicitHConvolution *yconvolve;
  Complex **U2,**V2;
  bool allocated;
public:  
  
  void init() {
    xfftpad=new fft0pad(mx,my,my,u2);
    yconvolve=new ImplicitHConvolution(my,u1,v1,w1,M);
    
    U2=new Complex *[M];
    V2=new Complex *[M];
    unsigned int mu=(mx+1)*my;
    for(unsigned int s=0; s < M; ++s) {
      unsigned int smu=s*mu;
      U2[s]=u2+smu;
      V2[s]=v2+smu;
    }
  }

  // u1 and v1 are temporary arrays of size (my/2+1)*M.
  // w1 is a temporary array of size 3*M.
  // u2 and v2 are temporary arrays of size (mx+1)*my*M;
  // M is the number of data blocks (each corresponding to a dot product term).
  ImplicitHConvolution2(unsigned int mx, unsigned int my,
                        Complex *u1, Complex *v1, Complex *w1,
                        Complex *u2, Complex *v2, unsigned int M=1) :
    mx(mx), my(my), u1(u1), v1(v1), w1(w1), u2(u2), v2(v2),
    M(M), allocated(false) {
    init();
  }
  
  ImplicitHConvolution2(unsigned int mx, unsigned int my, unsigned int M=1) :
    mx(mx), my(my), u1(ComplexAlign((my/2+1)*M)), v1(ComplexAlign((my/2+1)*M)),
    w1(ComplexAlign(3*M)),
    u2(ComplexAlign((mx+1)*my*M)), v2(ComplexAlign((mx+1)*my*M)),
    M(M), allocated(true) {
    init();
  }
  
  ~ImplicitHConvolution2() {
    delete [] V2;
    delete [] U2;
    
    if(allocated) {
      deleteAlign(v2);
      deleteAlign(u2);
      deleteAlign(w1);
      deleteAlign(v1);
      deleteAlign(u1);
    }
    delete yconvolve;
    delete xfftpad;
  }
  
  // F and G are pointers to M distinct data blocks each of size (2mx-1)*my,
  // shifted by offset (contents not preserved).
  // The output is returned in F[0].
  void convolve(Complex **F, Complex **G, bool symmetrize=true,
                unsigned int offset=0) {
    unsigned int xorigin=mx-1;
    unsigned int nx=2*mx-1;
    unsigned int mu=(mx+1)*my;
    
    for(unsigned int s=0; s < M; ++s) {
      Complex *f=F[s]+offset;
      if(symmetrize)
        HermitianSymmetrizeX(mx,my,xorigin,f);
      xfftpad->backwards(f,u2+s*mu);
    }
    
    for(unsigned int s=0; s < M; ++s) {
      Complex *g=G[s]+offset;
      if(symmetrize)
        HermitianSymmetrizeX(mx,my,xorigin,g);
      xfftpad->backwards(g,v2+s*mu);
    }
    
    unsigned int mf=nx*my;
    for(unsigned int i=0; i < mf; i += my)
      yconvolve->convolve(F,G,i+offset);
    for(unsigned int i=0; i < mu; i += my)
      yconvolve->convolve(U2,V2,i);
    
    xfftpad->forwards(F[0]+offset,u2);
  }
  
  // Constructor for special case M=1:
  void convolve(Complex *f, Complex *g, bool symmetrize=true) {
    convolve(&f,&g,symmetrize);
  }
};

// Out-of-place direct 2D Hermitian convolution.
class DirectHConvolution2 {
protected:  
  unsigned int mx,my;
public:
  DirectHConvolution2(unsigned int mx, unsigned int my) : mx(mx), my(my) {}
  
  void convolve(Complex *h, Complex *f, Complex *g, bool symmetrize=true);
};

// In-place explicitly dealiased 3D complex convolution.
class ExplicitConvolution3 {
protected:
  unsigned int nx,ny,nz;
  unsigned int mx,my,mz;
  bool prune; // Skip Fourier transforming rows containing all zeroes?
  mfft1d *xBackwards, *xForwards;
  mfft1d *yBackwards, *yForwards;
  mfft1d *zBackwards, *zForwards;
  fft3d *Backwards, *Forwards;
public:
  ExplicitConvolution3(unsigned int nx, unsigned int ny, unsigned int nz,
                       unsigned int mx, unsigned int my, unsigned int mz,
                       Complex *f, bool prune=false) :
    nx(nx), ny(ny), nz(nz), mx(mx), my(my), mz(mz), prune(prune) {
    unsigned int nxy=nx*ny;
    unsigned int nyz=ny*nz;
    if(prune) {
      xBackwards=new mfft1d(nx,1,nyz,nyz,1,f);
      yBackwards=new mfft1d(ny,1,mz,nz,1,f);
      zBackwards=new mfft1d(nz,1,nxy,1,nz,f);
      zForwards=new mfft1d(nz,-1,nxy,1,nz,f);
      yForwards=new mfft1d(ny,-1,mz,nz,1,f);
      xForwards=new mfft1d(nx,-1,nyz,nyz,1,f);
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

// In-place implicitly dealiased 3D complex convolution.
class ImplicitConvolution3 {
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
public:  
  void init() {
    unsigned int myz=my*mz;
    xfftpad=new fftpad(mx,myz,myz,u3);
    yzconvolve=new ImplicitConvolution2(my,mz,u1,v1,u2,v2,M);
    
    U3=new Complex *[M];
    V3=new Complex *[M];
    unsigned int mxyz=mx*myz;
    for(unsigned int s=0; s < M; ++s) {
      unsigned int smxy=s*mxyz;
      U3[s]=u3+smxy;
      V3[s]=v3+smxy;
    }
  }
  
  // u1 and v1 are temporary arrays of size mz*M.
  // u2 and v2 are temporary arrays of size my*mz*M.
  // u3 and v3 are temporary arrays of size mx*my*mz*M.
  // M is the number of data blocks (each corresponding to a dot product term).
  ImplicitConvolution3(unsigned int mx, unsigned int my, unsigned int mz,
                       Complex *u1, Complex *v1,
                       Complex *u2, Complex *v2,
                       Complex *u3, Complex *v3, unsigned int M=1) :
    mx(mx), my(my), mz(mz), u1(u1), v1(v1), u2(u2), v2(v2),
    u3(u3), v3(v3), M(M), allocated(false) {
    init();
  }

  ImplicitConvolution3(unsigned int mx, unsigned int my, unsigned int mz,
                       unsigned int M=1) :
    mx(mx), my(my), mz(mz), u1(ComplexAlign(mz*M)), v1(ComplexAlign(mz*M)),
    u2(ComplexAlign(my*mz*M)), v2(ComplexAlign(my*mz*M)),
    u3(ComplexAlign(mx*my*mz*M)), v3(ComplexAlign(mx*my*mz*M)),
    M(M), allocated(true) {
    init();
  }
  
  ~ImplicitConvolution3() {
    delete [] V3;
    delete [] U3;
    
    if(allocated) {
      deleteAlign(v3);
      deleteAlign(u3);
      deleteAlign(v2);
      deleteAlign(u2);
      deleteAlign(v1);
      deleteAlign(u1);
    }
    
    delete yzconvolve;
    delete xfftpad;
  }
  
  // F and G are pointers to M distinct data blocks each of size mx*my*mz,
  // shifted by offset (contents not preserved).
  // The output is returned in F[0].
  void convolve(Complex **F, Complex **G, unsigned int offset=0) {
    unsigned int myz=my*mz;
    unsigned int mxyz=mx*myz;
    
    for(unsigned int s=0; s < M; ++s)
      xfftpad->backwards(F[s]+offset,u3+s*mxyz);
    for(unsigned int s=0; s < M; ++s)
      xfftpad->backwards(G[s]+offset,v3+s*mxyz);

    for(unsigned int i=0; i < mxyz; i += myz)
      yzconvolve->convolve(F,G,i+offset);
    for(unsigned int i=0; i < mxyz; i += myz)
      yzconvolve->convolve(U3,V3,i);
    
    xfftpad->forwards(F[0]+offset,u3);
  }
  
  // Constructor for special case M=1:
  void convolve(Complex *f, Complex *g) {
    convolve(&f,&g);
  }
};

// Out-of-place direct 3D complex convolution.
class DirectConvolution3 {
protected:  
  unsigned int mx,my,mz;
  unsigned int myz;
public:
  DirectConvolution3(unsigned int mx, unsigned int my, unsigned int mz) : 
    mx(mx), my(my), mz(mz), myz(my*mz) {}
  
  void convolve(Complex *h, Complex *f, Complex *g);
};

// In-place implicitly dealiased 3D Hermitian convolution.
class ImplicitHConvolution3 {
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
public:  
  void init() {
    unsigned int nymz=(2*my-1)*mz;
    xfftpad=new fft0pad(mx,nymz,nymz,u3);
    yzconvolve=new ImplicitHConvolution2(my,mz,u1,v1,w1,u2,v2,M);
    
    U3=new Complex *[M];
    V3=new Complex *[M];
    unsigned int mu=(mx+1)*(2*my-1)*mz;
    for(unsigned int s=0; s < M; ++s) {
      unsigned int smu=s*mu;
      U3[s]=u3+smu;
      V3[s]=v3+smu;
    }
  }
  
  // u1 and v1 are temporary arrays of size (mz/2+1)*M.
  // w1 is a temporary array of size 3*M.
  // u2 and v2 are temporary array of size (my+1)*mz*M.
  // u3 and v3 are temporary arrays of size (mx+1)*(2my-1)*mz*M.
  // M is the number of data blocks (each corresponding to a dot product term).
  ImplicitHConvolution3(unsigned int mx, unsigned int my, unsigned int mz,
                        Complex *u1, Complex *v1, Complex *w1,
                        Complex *u2, Complex *v2,
                        Complex *u3, Complex *v3,
                        unsigned int M=1) :
    mx(mx), my(my), mz(mz), u1(u1), v1(v1), w1(w1), u2(u2), v2(v2),
    u3(u3), v3(v3), M(M), allocated(false) {
    init();
  }
  
  ImplicitHConvolution3(unsigned int mx, unsigned int my, unsigned int mz,
                        unsigned int M=1) :
    mx(mx), my(my), mz(mz), u1(ComplexAlign((mz/2+1)*M)),
    v1(ComplexAlign((mz/2+1)*M)), w1(ComplexAlign(3*M)), 
    u2(ComplexAlign((my+1)*mz*M)), v2(ComplexAlign((my+1)*mz*M)),
    u3(ComplexAlign((mx+1)*(2*my-1)*mz*M)),
    v3(ComplexAlign((mx+1)*(2*my-1)*mz*M)), M(M), allocated(true) {
    init();
  }
  
  ~ImplicitHConvolution3() {
    delete [] V3;
    delete [] U3;
    
    if(allocated) {
      deleteAlign(v3);
      deleteAlign(u3);
      deleteAlign(v2);
      deleteAlign(u2);
      deleteAlign(w1);
      deleteAlign(v1);
      deleteAlign(u1);
    }
    
    delete yzconvolve;
    delete xfftpad;
  }
  
  // F and G are pointers to M distinct data blocks each of size 
  // (2mx-1)*(2my-1)*mz,   // shifted by offset (contents not preserved).
  // The output is returned in F[0].
  void convolve(Complex **F, Complex **G, bool symmetrize=true,
                unsigned int offset=0) {
    unsigned int xorigin=mx-1;
    unsigned int yorigin=my-1;
    unsigned int nx=xorigin+mx;
    unsigned int ny=yorigin+my;
    unsigned int mf=nx*ny*mz;
    unsigned int mxp1=mx+1;
    unsigned int mu=mxp1*ny*mz;
    
    for(unsigned int s=0; s < M; ++s) {
      Complex *f=F[s]+offset;
      if(symmetrize)
        HermitianSymmetrizeXY(mx,my,mz,ny,xorigin,yorigin,f);
      xfftpad->backwards(f,u3+s*mu);
    }
    
    for(unsigned int s=0; s < M; ++s) {
      Complex *g=G[s]+offset;
      if(symmetrize)
        HermitianSymmetrizeXY(mx,my,mz,ny,xorigin,yorigin,g);
      xfftpad->backwards(g,v3+s*mu);
    }
        
    unsigned int nymz=ny*mz;
    for(unsigned int i=0; i < mf; i += nymz)
      yzconvolve->convolve(F,G,false,i+offset);
    for(unsigned int i=0; i < mu; i += nymz)
      yzconvolve->convolve(U3,V3,false,i);
    
    xfftpad->forwards(F[0]+offset,u3);
  }
  
  // Constructor for special case M=1:
  void convolve(Complex *f, Complex *g, bool symmetrize=true) {
    convolve(&f,&g,symmetrize);
  }
};

// Out-of-place direct 3D Hermitian convolution.
class DirectHConvolution3 {
protected:  
  unsigned int mx,my,mz;
public:
  DirectHConvolution3(unsigned int mx, unsigned int my, unsigned int mz) : 
    mx(mx), my(my), mz(mz) {}
  
  void convolve(Complex *h, Complex *f, Complex *g, bool symmetrize=true);
};

// In-place explicitly dealiased Hermitian ternary convolution.
class ExplicitHTConvolution {
protected:
  unsigned int n;
  unsigned int m;
  rcfft1d *rc;
  crfft1d *cr;
public:
  // u is a temporary array of size n.
  ExplicitHTConvolution(unsigned int n, unsigned int m, Complex *u) :
    n(n), m(m) {
    rc=new rcfft1d(n,u);
    cr=new crfft1d(n,u);
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

// In-place implicitly dealiased Hermitian ternary convolution.
class ImplicitHTConvolution : public Dot3 {
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
public:  
  
  void init() {
    unsigned int twom=2*m;
    
    W=new Complex *[M];
    unsigned int m1=m+1;
    for(unsigned int i=0; i < M; ++i)
      W[i]=w+i*m1;
    
    rc=new rcfft1d(twom,u);
    cr=new crfft1d(twom,u);
    
    rco=new rcfft1d(twom,(double *) u,v);
    cro=new crfft1d(twom,v,(double *) u);
    
    s=BuildZeta(4*m,m,ZetaH,ZetaL);
  }
  
  // u, v, and w are distinct temporary arrays each of size m+1.
  ImplicitHTConvolution(unsigned int m, Complex *u, Complex *v,
                              Complex *w, unsigned int M=1) :
    Dot3(m,M), m(m), u(u), v(v), w(w), M(M), allocated(false) {
    init();
  }
  
  // u, v, and w are distinct temporary arrays each of size m+1.
  ImplicitHTConvolution(unsigned int m, unsigned int M=1) : 
    Dot3(m,M), m(m), u(ComplexAlign(m*M+M)), v(ComplexAlign(m*M+M)),
    w(ComplexAlign(m*M+M)), M(M), allocated(true) {
    init();
  }
  
  ~ImplicitHTConvolution() {
    delete [] W;
    
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
  
  // In-place implicitly dealiased convolution.
  // The input arrays f, g, and h are each of size m+1 (contents not preserved).
  // The output is returned in f.
  // u, v, and w are temporary arrays each of size m+1.
  void convolve(Complex **F, Complex **G, Complex **H, unsigned int offset=0);
  
  // Constructor for special case M=1:
  void convolve(Complex *f, Complex *g, Complex *h) {
    convolve(&f,&g,&h);
  }
};

// Out-of-place direct 1D Hermitian ternary convolution.
class DirectHTConvolution {
protected:  
  unsigned int m;
public:
  DirectHTConvolution(unsigned int m) : m(m) {}
  
  void convolve(Complex *h, Complex *e, Complex *f, Complex *g);
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
public:  
  fft0bipad(unsigned int m, unsigned int M, unsigned int stride,
            Complex *f) : m(m), M(M), stride(stride) {
    unsigned int twom=2*m;
    Backwards=new mfft1d(twom,1,M,stride,1,f);
    Forwards=new mfft1d(twom,-1,M,stride,1,f);
    
    s=BuildZeta(4*m,twom,ZetaH,ZetaL);
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
  
// In-place explicitly dealiased 2D Hermitian ternary convolution.
class ExplicitHTConvolution2 {
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
      yBackwards=new mcrfft1d(ny,nx,1,nyp,f);
      yForwards=new mrcfft1d(ny,nx,1,nyp,f);
      xForwards=new mfft1d(nx,-1,My,nyp,1,f);
    } else {
      Backwards=new crfft2d(nx,ny,f);
      Forwards=new rcfft2d(nx,ny,f);
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

// In-place implicitly dealiased 2D Hermitian ternary convolution.
class ImplicitHTConvolution2 {
protected:
  unsigned int mx,my;
  Complex *u1,*v1,*w1;
  Complex *u2,*v2,*w2;
  unsigned int M;
  fft0bipad *xfftpad;
  ImplicitHTConvolution *yconvolve;
  Complex **U2,**V2,**W2;
  bool allocated;
public:  
  void init() {
    xfftpad=new fft0bipad(mx,my,my+1,u2);
    yconvolve=new ImplicitHTConvolution(my,u1,v1,w1,M);
    
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
  
  // u1, v1, and w1 are temporary arrays of size (my+1)*M;
  // u2, v2, and w2 are temporary arrays of size 2mx*(my+1)*M.
  // M is the number of data blocks (each corresponding to a dot product term).
  ImplicitHTConvolution2(unsigned int mx, unsigned int my,
                          Complex *u1, Complex *v1, Complex *w1, 
                          Complex *u2, Complex *v2, Complex *w2,
                          unsigned int M=1) :
    mx(mx), my(my), u1(u1), v1(v1), w1(w1), u2(u2), v2(v2), w2(w2),
    M(M), allocated(false) {
    init();
  }
  
  ImplicitHTConvolution2(unsigned int mx, unsigned int my,
                          unsigned int M=1) :
    mx(mx), my(my), u1(ComplexAlign(my*M+M)), v1(ComplexAlign(my*M+M)),
    w1(ComplexAlign(my*M+M)), u2(ComplexAlign(2*mx*(my*M+M))),
    v2(ComplexAlign(2*mx*(my*M+M))), w2(ComplexAlign(2*mx*(my*M+M))),
    M(M), allocated(true) {
    init();
  }
  
  ~ImplicitHTConvolution2() {
    delete [] W2;
    delete [] V2;
    delete [] U2;
    
    if(allocated) {
      deleteAlign(w2);
      deleteAlign(v2);
      deleteAlign(u2);
      deleteAlign(w1);
      deleteAlign(v1);
      deleteAlign(u1);
    }
    delete yconvolve;
    delete xfftpad;
  }
  
  // F, G, and H are pointers to M distinct data blocks each of size 2mx*(my+1),
  // shifted by offset (contents not preserved).
  // The output is returned in F[0].
  void convolve(Complex **F, Complex **G, Complex **H, bool symmetrize=true,
                unsigned int offset=0) {
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
    
    for(unsigned int i=0; i < mu; i += my1)
      yconvolve->convolve(F,G,H,i+offset);
    for(unsigned int i=0; i < mu; i += my1)
      yconvolve->convolve(U2,V2,W2,i);
    
    xfftpad->forwards(F[0]+offset,u2);
  }
  
  // Constructor for special case M=1:
  void convolve(Complex *f, Complex *g, Complex *h, bool symmetrize=true) {
    convolve(&f,&g,&h,symmetrize);
  }
};

// Out-of-place direct 2D Hermitian ternary convolution.
class DirectHTConvolution2 {
protected:  
  unsigned int mx,my;
public:
  DirectHTConvolution2(unsigned int mx, unsigned int my) : mx(mx), my(my)
  {}
  
  void convolve(Complex *h, Complex *e, Complex *f, Complex *g,
                bool symmetrize=true);
};

}

#endif
