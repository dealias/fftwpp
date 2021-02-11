/* General implicitly dealiased convolution routines.
   Copyright (C) 2021 John C. Bowman and Noel Murasko, Univ. of Alberta

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

// TODO:
// Implement built-in shift for p > 2 centered case
// Optimize shift when M=2L for p=1
// Implement 3D convolutions
// Precompute best D and inline options for each m value
// Only check m <= M/2 and m=M; how many surplus sizes to check?
// Use experience or heuristics (sparse distribution?) to determine best m value
// Use power of P values for m when L,M,M-L are powers of P?
// Multithread
// Port to MPI

#include <cfloat>
#include <climits>

#include "Complex.h"
#include "fftw++.h"
#include "utils.h"
#include "Array.h"

namespace fftwpp {

#ifndef __convolve_h__
#define __convolve_h__ 1

extern const double twopi;

// Constants used for initialization and testing.
const Complex I(0.0,1.0);

extern unsigned int threads;

extern unsigned int mOption;
extern unsigned int DOption;

extern int IOption;

// Temporary
extern unsigned int A; // number of inputs
extern unsigned int B; // number of outputs
extern unsigned int C; // number of copies
extern unsigned int L;
extern unsigned int M;

extern unsigned int surplusFFTsizes;

#ifndef _GNU_SOURCE
inline void sincos(const double x, double *sinx, double *cosx)
{
  *sinx=sin(x); *cosx=cos(x);
}
#endif

inline Complex expi(double phase)
{
  double cosy,siny;
  sincos(phase,&siny,&cosy);
  return Complex(cosy,siny);
}

template<class T>
T pow(T x, unsigned int y)
{
  if(y == 0) return 1;
  if(x == 0) return 0;

  unsigned int r = 1;
  while(true) {
    if(y & 1) r *= x;
    if((y >>= 1) == 0) return r;
    x *= x;
  }
}

unsigned int nextfftsize(unsigned int m);

class fftBase;

typedef void (fftBase::*FFTcall)(Complex *f, Complex *F, unsigned int r, Complex *W);
typedef void (fftBase::*FFTPad)(Complex *W);

class Application {
public:
  Application() {};
  virtual void init(fftBase &fft)=0;
  virtual void clear()=0;
  virtual double time(fftBase &fft, unsigned int K)=0;
};

class fftBase {
public:
  unsigned int L; // number of unpadded Complex data values
  unsigned int M; // minimum number of padded Complex data values
  unsigned int C; // number of FFTs to compute in parallel
  unsigned int m;
  unsigned int p;
  unsigned int q;
  unsigned int n;
  unsigned int Q;
  unsigned int D;
  unsigned int Cm;
  Complex *W0; // Temporary work memory for testing accuracy

  FFTcall Forward,Backward;
  FFTPad Pad;

protected:
  Complex *Zetaqp;
  Complex *Zetaq;
  Complex *Zetaqm;
  Complex *Zetaqm2;
  bool inplace;
public:

  void common();

  void initZetaq() {
    p=1;
    Q=n=q;
    Zetaq=utils::ComplexAlign(q);
    double twopibyq=twopi/q;
    for(unsigned int r=1; r < q; ++r)
      Zetaq[r]=expi(r*twopibyq);
  }

  void initZetaqm() {
    double twopibyM=twopi/M;
    Zetaqm=utils::ComplexAlign((q-1)*m)-m;
    for(unsigned int r=1; r < q; ++r) {
      Zetaqm[m*r]=1.0;
      for(unsigned int s=1; s < m; ++s)
        Zetaqm[m*r+s]=expi(r*s*twopibyM);
    }
  }

  class OptBase {
  public:
    unsigned int m,q,D;
    double T;

    virtual double time(unsigned int L, unsigned int M, unsigned int C,
                        unsigned int m, unsigned int q,unsigned int D,
                        Application &app)=0;

    void check(unsigned int L, unsigned int M,
               Application& app, unsigned int C, unsigned int m,
               bool fixed=false, bool mForced=false);

    // Determine optimal m,q values for padding L data values to
    // size >= M
    // If fixed=true then an FFT of size M is enforced.
    void scan(unsigned int L, unsigned int M, Application& app,
              unsigned int C, bool Explicit=false, bool fixed=false);
  };

  fftBase(unsigned int L, unsigned int M, unsigned int C,
          unsigned int m, unsigned int q, unsigned int D) :
    L(L), M(M), C(C), m(m), p(utils::ceilquotient(L,m)), q(q), D(D) {}

  fftBase(unsigned int L, unsigned int M, Application& app,
          unsigned int C=1, bool Explicit=false, bool fixed=false) :
    L(L), M(M), C(C) {}

  ~fftBase();

  void padNone(Complex *W) {}

  virtual void padSingle(Complex *W) {}
  virtual void padMany(Complex *W) {}

  virtual void forwardShifted(Complex *f, Complex *F, unsigned int r, Complex *W) {}
  virtual void backwardShifted(Complex *F, Complex *f, unsigned int r, Complex *W) {}

  virtual void forward(Complex *f, Complex *F)=0;
  virtual void backward(Complex *F, Complex *f)=0;

  virtual void forwardExplicit(Complex *f, Complex *F, unsigned int, Complex *W)=0;
  virtual void forwardExplicitMany(Complex *f, Complex *F, unsigned int, Complex *W)=0;

  virtual void backwardExplicit(Complex *F, Complex *f, unsigned int, Complex *W)=0;
  virtual void backwardExplicitMany(Complex *F, Complex *f, unsigned int, Complex *W)=0;

  virtual void forward(Complex *f, Complex *F0, unsigned int r0, Complex *W) {}
  virtual void forwardMany(Complex *f, Complex *F, unsigned int r, Complex *W) {}

  virtual void forward2(Complex *f, Complex *F0, unsigned int r0, Complex *W) {}
  virtual void forward2Many(Complex *f, Complex *F, unsigned int r, Complex *W) {}

  virtual void forwardInner(Complex *f, Complex *F0, unsigned int r0, Complex *W) {}
  virtual void forwardInnerMany(Complex *f, Complex *F, unsigned int r, Complex *W) {}

  virtual void backward(Complex *F0, Complex *f, unsigned int r0, Complex *W) {}
  virtual void backwardMany(Complex *F, Complex *f, unsigned int r, Complex *W) {}

  virtual void backward2(Complex *F0, Complex *f, unsigned int r0, Complex *W) {}
  virtual void backward2Many(Complex *F, Complex *f, unsigned int r, Complex *W) {}

  virtual void backwardInner(Complex *F0, Complex *f, unsigned int r0, Complex *W) {}
  virtual void backwardInnerMany(Complex *F, Complex *f, unsigned int r, Complex *W) {}

  // input array length
  unsigned int bufferLength() {
    return C*m*p;
  }

  // FFT input length
  virtual unsigned int length() {
    return m*p;
  }

  // FFT output length
  unsigned int Length() {
    return q == 1 ? M : m*p;
  }

  unsigned int size() {
    return M;
//    return q == 1 ? M : m*q; // TODO: m*q  Can we make M=m*q in all cases?
  }

  virtual unsigned int worksizeF() {
    return C*(q == 1 ? M : m*p*D);
  }

  bool loop2() {
    return D < Q && 2*D >= Q && A > B;
  }

  unsigned int worksizeV() {
    return q == 1 || D >= Q || loop2() ? 0 : length();
  }

  virtual unsigned int worksizeW() {
    return q == 1 || inplace ? 0 : worksizeF();
  }

  unsigned int padding() {
    return !inplace && L < p*m;
  }

  void initialize(Complex *f, Complex *g);

  double meantime(Application& app, double *Stdev=NULL);
  double report(Application& app);
};

class fftPad : public fftBase {
protected:
  mfft1d *fftm,*fftm2;
  mfft1d *ifftm,*ifftm2;
  mfft1d *fftp;
  mfft1d *ifftp;
public:
  FFTcall Forward,Backward;

  class Opt : public OptBase {
  public:
    Opt(unsigned int L, unsigned int M, Application& app,
        unsigned int C, bool Explicit=false, bool fixed=false) {
      scan(L,M,app,C,Explicit,fixed);
    }

    double time(unsigned int L, unsigned int M, unsigned int C,
                unsigned int m, unsigned int q,unsigned int D,
                Application &app) {
      fftPad fft(L,M,C,m,q,D);
      return fft.meantime(app);
    }
  };

  // Compute an fft padded to N=m*q >= M >= L
  fftPad(unsigned int L, unsigned int M, unsigned int C,
         unsigned int m, unsigned int q,unsigned int D) :
    fftBase(L,M,C,m,q,D) {
    init();
  }

  // Normal entry point.
  // Compute C ffts of length L and distance 1 padded to at least M
  // (or exactly M if fixed=true)
  fftPad(unsigned int L, unsigned int M, Application& app,
         unsigned int C=1, bool Explicit=false, bool fixed=false) :
    fftBase(L,M,app,C,Explicit,fixed) {
    Opt opt=Opt(L,M,app,C,Explicit,fixed);
    m=opt.m;
    if(Explicit)
      M=m;
    q=opt.q;
    D=opt.D;
    init();
  }

  ~fftPad();

  void init();

  // Explicitly pad to m.
  void padSingle(Complex *W);

  // Explicitly pad C FFTs to m.
  void padMany(Complex *W);

  void forward(Complex *f, Complex *F);
  void backward(Complex *F, Complex *f);

  void forwardExplicit(Complex *f, Complex *F, unsigned int, Complex *W);
  void forwardExplicitMany(Complex *f, Complex *F, unsigned int, Complex *W);

  void backwardExplicit(Complex *F, Complex *f, unsigned int, Complex *W);
  void backwardExplicitMany(Complex *F, Complex *f, unsigned int, Complex *W);

  // p=1 && C=1
  void forward(Complex *f, Complex *F0, unsigned int r0, Complex *W);

  void forwardMany(Complex *f, Complex *F, unsigned int r, Complex *W);

  // p=2 && q odd
  void forward2(Complex *f, Complex *F0, unsigned int r0, Complex *W);

  void forward2Many(Complex *f, Complex *F, unsigned int r, Complex *W);

  void forwardInner(Complex *f, Complex *F0, unsigned int r0, Complex *W);

  void forwardInnerMany(Complex *f, Complex *F, unsigned int r, Complex *W);

// Compute an inverse fft of length N=m*q unpadded back
// to size m*p >= L.
// input and output arrays must be distinct
// Input F destroyed
  void backward(Complex *F0, Complex *f, unsigned int r0, Complex *W);

  void backwardMany(Complex *F, Complex *f, unsigned int r, Complex *W);

  void backward2(Complex *F0, Complex *f, unsigned int r0, Complex *W);

  void backward2Many(Complex *F, Complex *f, unsigned int r, Complex *W);

  void backwardInner(Complex *F0, Complex *f, unsigned int r0, Complex *W);
  void backwardInnerMany(Complex *F, Complex *f, unsigned int r, Complex *W);
};

class fftPadCentered : public fftPad {
  Complex *ZetaShift;
public:
  FFTcall Forward,Backward;
  FFTcall ForwardShifted,BackwardShifted;

  class Opt : public OptBase {
  public:
    Opt(unsigned int L, unsigned int M, Application& app,
        unsigned int C, bool Explicit=false, bool fixed=false) {
      scan(L,M,app,C,Explicit,fixed);
    }

    double time(unsigned int L, unsigned int M, unsigned int C,
                unsigned int m, unsigned int q,unsigned int D,
                Application &app) {
      fftPad fft(L,M,C,m,q,D);
      return fft.meantime(app);
    }
  };

  // Compute an fft padded to N=m*q >= M >= L
  fftPadCentered(unsigned int L, unsigned int M, unsigned int C,
                 unsigned int m, unsigned int q,unsigned int D) :
    fftPad(L,M,C,m,q,D) {
    init();
  }

  // Normal entry point.
  // Compute C ffts of length L and distance 1 padded to at least M
  // (or exactly M if fixed=true)
  fftPadCentered(unsigned int L, unsigned int M, Application& app,
                 unsigned int C=1, bool Explicit=false, bool fixed=false) :
    fftPad(L,M,app,C,Explicit,fixed) {
    init();
  }

  ~fftPadCentered() {
    if(ZetaShift)
      utils::deleteAlign(ZetaShift);
  }

  void init();
  void initShift();

  void forwardShifted(Complex *f, Complex *F, unsigned int r, Complex *W);
  void backwardShifted(Complex *F, Complex *f, unsigned int r, Complex *W);

  void forwardShift(Complex *F, unsigned int r0);
  void backwardShift(Complex *F, unsigned int r0);

  // p=2 && q odd
  void forward2(Complex *f, Complex *F0, unsigned int r0, Complex *W);
  void forward2Many(Complex *f, Complex *F, unsigned int r, Complex *W);

  void backward2(Complex *F0, Complex *f, unsigned int r0, Complex *W);
  void backward2Many(Complex *F, Complex *f, unsigned int r, Complex *W);
};

class fftPadHermitian : public fftBase {
  unsigned int e;
  mcrfft1d *crfftm,*crfftm2;
  mrcfft1d *rcfftm,*rcfftm2;
public:
  FFTcall Forward,Backward;

  class Opt : public OptBase {
  public:
    Opt(unsigned int L, unsigned int M, Application& app,
        unsigned int C, bool Explicit=false, bool fixed=false) {
      scan(L,M,app,C,Explicit,fixed);
    }

    double time(unsigned int L, unsigned int M, unsigned int C,
                unsigned int m, unsigned int q,unsigned int D,
                Application &app) {
      fftPadHermitian fft(L,M,C,m,q,D);
      return fft.meantime(app);
    }
  };

  fftPadHermitian(unsigned int L, unsigned int M, unsigned int C,
                  unsigned int m, unsigned int q, unsigned int D) :
    fftBase(L,M,C,m,q,D) {
    init();
  }

  fftPadHermitian(unsigned int L, unsigned int M, Application& app,
                  unsigned int C=1, bool Explicit=false, bool fixed=false) :
    fftBase(L,M,app,C,Explicit,fixed) {
    Opt opt=Opt(L,M,app,C,Explicit,fixed);
    m=opt.m;
    if(Explicit)
      M=m;
    q=opt.q;
    D=opt.D;
    init();
  }

  ~fftPadHermitian();

  void init();

  void forward(Complex *f, Complex *F);
  void backward(Complex *F, Complex *f);

  void forwardExplicit(Complex *f, Complex *F, unsigned int, Complex *W);
  void forwardExplicitMany(Complex *f, Complex *F, unsigned int, Complex *W);

  void backwardExplicit(Complex *F, Complex *f, unsigned int, Complex *W);
  void backwardExplicitMany(Complex *F, Complex *f, unsigned int, Complex *W);

  void forward2(Complex *f, Complex *F0, unsigned int r0, Complex *W);
  void forward2Many(Complex *f, Complex *F, unsigned int r, Complex *W);

  void backward2(Complex *F0, Complex *f, unsigned int r0, Complex *W);
  void backward2Many(Complex *F, Complex *f, unsigned int r, Complex *W);

  // FFT input length
  unsigned int length() { // Only for p=2;
    return e+1;
  }

  unsigned int worksizeF() {
    return C*(q == 1 ? M : (e+1)*D);
  }
};

class ForwardBackward : public Application {
protected:
  unsigned int A;
  unsigned int B;
  FFTcall Forward,Backward;
  unsigned int C;
  unsigned int D;
  unsigned int Q;
  Complex **f;
  Complex **F;
  Complex **h;
  Complex *W;

public:
  ForwardBackward(unsigned int A=2, unsigned int B=1) :
    A(A), B(B), f(NULL), F(NULL), h(NULL), W(NULL) {
  }

  ~ForwardBackward() {
    clear();
  }

  void init(fftBase &fft);

  double time(fftBase &fft, unsigned int K);

  void clear();

};

typedef void multiplier(Complex **, unsigned int e, unsigned int threads);

// Multiplication routine for binary convolutions and taking two inputs of size e.
void multbinary(Complex **F, unsigned int e, unsigned int threads);


class Convolution {
public:
  fftPad *fft;
  unsigned int A;
  unsigned int B;
  unsigned int L;
private:
  unsigned int Q;
  unsigned int D;
  unsigned int c;
  Complex **F,**Fp;
  Complex **V;
  Complex *W;
  Complex *H;
  Complex *W0;
  double scale;
  bool allocateU;
  bool allocateV;
  bool allocateW;
  bool loop2;

  FFTcall Forward,Backward;
  FFTPad Pad;

public:
  // A is the number of inputs.
  // B is the number of outputs.
  // F is an optional work array of size std::max(A,B)*fft->worksizeF(),
  // V is an optional work array of size B*fft->worksizeV() (for inplace usage)
  // W is an optional work array of size fft->worksizeW();
  //   if changed between calls to convolve(), be sure to call pad()
  // TODO: add inplace flag to avoid allocating W.
  Convolution(fftPad &fft, unsigned int A=2, unsigned int B=1,
              Complex *F=NULL, Complex *V=NULL, Complex *W=NULL) :
    fft(&fft), A(A), B(B), W(W), allocateU(false) {
    init(F,V);
  }

  void init(Complex *F, Complex *V);

  void initV() {
    allocateV=true;
    V=new Complex*[B];
    unsigned int size=fft->worksizeV();
    for(unsigned int i=0; i < B; ++i)
      V[i]=utils::ComplexAlign(size);
  }

  ~Convolution();

  // f is an input array of A pointers to distinct data blocks each of size
  // fft->length()
  // h is an output array of B pointers to distinct data blocks each of size
  // fft->length(), which may coincide with f.
  // offset is applied to each input and output component

  void convolve0(Complex **f, Complex **h, multiplier *mult,
                 unsigned int offset=0);

  void convolve(Complex **f, Complex **h, multiplier *mult,
                unsigned int offset=0) {
    convolve0(f,h,mult,offset);

    unsigned int H=L/2;
    for(unsigned int b=0; b < B; ++b) {
      Complex *hb=h[b]+offset;
      for(unsigned int i=0; i <= H; ++i)
        hb[i] *= scale;
    }
  }
};

// class HermitianConvolution : public Convolution {
class HermitianConvolution {
public:
  fftPadHermitian *fft;
  unsigned int A;
  unsigned int B;
  unsigned int L;
private:
  unsigned int Q;
  unsigned int D;
  unsigned int c;
  Complex **F,**Fp;
  Complex **V;
  Complex *W;
  Complex *H;
  Complex *W0;
  double scale;
  bool allocateU;
  bool allocateV;
  bool allocateW;
  bool loop2;

  FFTcall Forward,Backward;
  FFTPad Pad;

public:
  // A is the number of inputs.
  // B is the number of outputs.
  // F is an optional work array of size std::max(A,B)*fft->worksizeF(),
  // V is an optional work array of size B*fft->worksizeV() (for inplace usage)
  // W is an optional work array of size fft->worksizeW();
  //   if changed between calls to convolve(), be sure to call pad()
  // TODO: add inplace flag to avoid allocating W.
  HermitianConvolution(fftPadHermitian &fft, unsigned int A=2,
                       unsigned int B=1, Complex *F=NULL, Complex *V=NULL,
                       Complex *W=NULL) :
    fft(&fft), A(A), B(B), W(W), allocateU(false) {
    init(F,V);
  }

  void init(Complex *F, Complex *V);

  void initV() {
    allocateV=true;
    V=new Complex*[B];
    unsigned int size=fft->worksizeV();
    for(unsigned int i=0; i < B; ++i)
      V[i]=utils::ComplexAlign(size);
  }

  ~HermitianConvolution();

  // f is an input array of A pointers to distinct data blocks each of size
  // fft->length()
  // h is an output array of B pointers to distinct data blocks each of size
  // fft->length(), which may coincide with f.
  // offset is applied to each input and output component

  void convolve0(Complex **f, Complex **h, multiplier *mult,
                 unsigned int offset=0);

  void convolve(Complex **f, Complex **h, multiplier *mult,
                unsigned int offset=0) {
    convolve0(f,h,mult,offset);

    for(unsigned int b=0; b < B; ++b) {
      Complex *hb=h[b]+offset;
      for(unsigned int i=0; i < L; ++i)
        hb[i] *= scale;
    }
  }
};

class Convolution2 {
  fftPad *fftx;
  Convolution *convolvey;
  unsigned int Sx; // x dimension of Ux buffer
  unsigned int Lx,Ly; // x,y dimensions of input arrays
  unsigned int A;
  unsigned int B;
  unsigned int Q;
  unsigned int D;
  unsigned int qx,Qx;
  Complex **Fx;
  Complex *Fy;
  Complex *Vy;
  Complex *Wy;
  bool allocateUx;
  double scale;

  FFTcall Forward,Backward;

public:
  // Fx is an optional work array of size max(A,B)*fftx->worksizeF(),
  Convolution2(fftPad &fftx, Convolution &convolvey, Complex *Fx=NULL) :
    fftx(&fftx), convolvey(&convolvey), allocateUx(false) {

    Forward=fftx.Forward;
    Backward=fftx.Backward;

    A=convolvey.A;
    B=convolvey.B;

    qx=fftx.q;
    Qx=fftx.Q;
    Sx=fftx.Length();
    scale=1.0/(fftx.size()*convolvey.fft->size());

    unsigned int c=fftx.worksizeF();
    unsigned int N=std::max(A,B);
    this->Fx=new Complex*[N];
    if(Fx) {
      for(unsigned int i=0; i < N; ++i)
        this->Fx[i]=Fx+i*c;
    } else {
      allocateUx=true;
      for(unsigned int i=0; i < N; ++i)
        this->Fx[i]=utils::ComplexAlign(c);
    }

    Lx=fftx.L;
    Ly=convolvey.L;
  }

  // A is the number of inputs.
  // B is the number of outputs.
  // Fy is an optional work array of size max(A,B)*ffty->worksizeF(),
  // Vy is an optional work array of size B*ffty->worksizeV() (inplace usage)
  // Wy is an optional work array of size ffty->worksizeW();
  //   if changed between calls to convolve(), be sure to call pad()
  /*
    Convolution2(unsigned int Lx, unsigned int Ly,
    unsigned int Mx, unsigned int My,
    unsigned int A, unsigned int B, Complex *Fx=NULL,
    Complex *Fy=NULL, Complex *Vy=NULL, Complex *Wy=NULL): Fy(Fy), Vy(Vy), Wy(Wy) {}
  */

  ~Convolution2() {
    unsigned int N=std::max(A,B);
    if(allocateUx) {
      for(unsigned int i=0; i < N; ++i)
        utils::deleteAlign(Fx[i]);
    }
    delete [] Fx;
  }

  void forward(Complex **f, Complex **F, unsigned int rx) {
    for(unsigned int a=0; a < A; ++a)
      (fftx->*Forward)(f[a],F[a],rx,NULL); // C=Ly <= my py, Dx=1
  }

  void subconvolution(Complex **f, multiplier *mult,
                      unsigned int C, unsigned int stride,
                      unsigned int offset=0) {
    for(unsigned int i=0; i < C; ++i)
      convolvey->convolve0(f,f,mult,offset+i*stride);
  }

  void backward(Complex **F, Complex **f, unsigned int rx) {
    // TODO: Support out-of-place
    for(unsigned int b=0; b < B; ++b)
      (fftx->*Backward)(F[b],f[b],rx,NULL);
  }

// f is a pointer to A distinct data blocks each of size Lx*Ly,
// shifted by offset (contents not preserved).
  virtual void convolve(Complex **f, Complex **h, multiplier *mult,
                        unsigned int offset=0) {
    for(unsigned int rx=0; rx < Qx; ++rx) {
      forward(f,Fx,rx);
      subconvolution(Fx,mult,Sx,Ly,offset);
      backward(Fx,h,rx);
    }
    for(unsigned int b=0; b < B; ++b) {
      Complex *hb=h[b];
      for(unsigned int i=0; i < Lx; ++i)
        for(unsigned int j=0; j < Ly; ++j)
          hb[Ly*i+j] *= scale;
    }
  }
};

extern void optionsHybrid(int argc, char* argv[]);

} //end namespace fftwpp

#endif
