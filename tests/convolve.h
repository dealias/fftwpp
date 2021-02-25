using namespace std; // Temporary

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
// Abort timing when best time exceeded
// Support out-of-place in fftPadCentered
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
  unsigned int b; // Block size
  Complex *W0; // Temporary work memory for testing accuracy
  bool inplace;

  FFTcall Forward,Backward;
  FFTPad Pad;

protected:
  Complex *Zetaqp;
  Complex *Zetaq;
  Complex *Zetaqm;
  Complex *Zetaqm2;
public:

  void common();

  void initZetaq() {
    Q=n=q;
    Zetaq=utils::ComplexAlign(q);
    double twopibyq=twopi/q;
    for(unsigned int r=1; r < q; ++r)
      Zetaq[r]=expi(r*twopibyq);
  }

  void initZetaqm(unsigned int q) {
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

  unsigned int normalization() {
    return M;
  }

  virtual unsigned int outputSize() {
    return b*D;
  }

  virtual unsigned int fullOutputSize() {
    return b*Q;
  }

  unsigned int outputs() {
    return C*M;
  }

  bool loop2() {
    return D < Q && 2*D >= Q && A > B;
  }

  unsigned int workSizeV() {
    return q == 1 || D >= Q || loop2() ? 0 : C*L;
  }

  virtual unsigned int workSizeW() {
    return q == 1 || inplace ? 0 : outputSize();
  }

  unsigned int repad() {
    return !inplace && L < m;
  }

  virtual unsigned int conjugates(unsigned int r) {
    return 1;
  }

  virtual unsigned int blockOffset(unsigned int r) {
    return b*r;
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
      D=1; // D > 1 is not yet implemented
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

  unsigned int outputSize() {
    return 2*C*(e+1)*D;
  }

  unsigned int fullOutputSize() {
    return C*(e+1)*q; // FIXME for inner
  }

  unsigned int conjugates(unsigned int r) {
    return r == 0 ? (q % 2 ? 1 : 2) : 2;
  }

  unsigned int blockOffset(unsigned int r) {
    return r == 0 ? 0 : b*(conjugates(0)+(r-1)*conjugates(1));
//    return r == 0 ? 0 : b*(2*r-q % 2);
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
void realmultbinary(Complex **F, unsigned int e, unsigned int threads);


class Convolution {
public:
  fftBase *fft;
  unsigned int A;
  unsigned int B;
  unsigned int L;
protected:
  unsigned int q;
  unsigned int Q;
  unsigned int D;
  unsigned int b;
  Complex **F,**Fp;
  Complex *FpB;
  Complex **V;
  Complex *W;
  Complex *H;
  Complex *W0;
  double scale;
  bool allocate;
  bool allocateV;
  bool allocateW;
  bool loop2;
  unsigned int noutputs;

  FFTcall Forward,Backward;
  FFTPad Pad;

public:
  // A is the number of inputs.
  // B is the number of outputs.
  // F is an optional work array of size max(A,B)*fft->outputSize(),
  // V is an optional work array of size B*fft->workSizeV() (for inplace usage)
  // W is an optional work array of size fft->workSizeW();
  //   if changed between calls to convolve(), be sure to call pad()
  // TODO: add inplace flag to avoid allocating W.
  Convolution(unsigned int A=2, unsigned int B=1,
              Complex *F=NULL, Complex *V=NULL, Complex *W=NULL) :
    A(A), B(B), W(W), allocate(false) {}

  Convolution(fftBase &fft, unsigned int A=2, unsigned int B=1,
              Complex *F=NULL, Complex *V=NULL, Complex *W=NULL) :
    fft(&fft), A(A), B(B), W(W), allocate(false) {
    init(F,V);
    noutputs=C*L;
  }

  void init(Complex *F, Complex *V);

  void initV() {
    allocateV=true;
    V=new Complex*[B];
    unsigned int size=fft->workSizeV();
    for(unsigned int i=0; i < B; ++i)
      V[i]=utils::ComplexAlign(size);
  }

  ~Convolution();

  void normalize(Complex **h, unsigned int offset=0) {
    for(unsigned int b=0; b < B; ++b) {
      Complex *hb=h[b]+offset;
      for(unsigned int i=0; i < noutputs; ++i)
        hb[i] *= scale;
    }
  }

  void convolve0(Complex **f, Complex **h, multiplier *mult,
                 unsigned int offset=0);

  void convolve(Complex **f, Complex **h, multiplier *mult,
                unsigned int offset=0) {
    convolve0(f,h,mult,offset);
    normalize(h,offset);
  }
};

class ConvolutionHermitian : public Convolution {
public:
  // A is the number of inputs.
  // B is the number of outputs.
  // F is an optional work array of size max(A,B)*fft->outputSize(),
  // V is an optional work array of size B*fft->workSizeV() (for inplace usage)
  // W is an optional work array of size fft->workSizeW();
  //   if changed between calls to convolve(), be sure to call pad()
  // TODO: add inplace flag to avoid allocating W.
  ConvolutionHermitian(fftPadHermitian &fft, unsigned int A=2,
                       unsigned int B=1, Complex *F=NULL, Complex *V=NULL,
                       Complex *W=NULL) : Convolution(A,B,F,V,W) {
    this->fft=&fft;
    init(F,V);
    b=q == 1 ? fft.Cm : 2*fft.b;
    noutputs=C*utils::ceilquotient(L,2);
  }
};

class Convolution2 {
protected:
  fftBase *fftx;
  Convolution *convolvey;
  unsigned int Sx;    // x dimension of Fx buffer
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
  bool allocate;
  double scale;

  FFTcall Forward,Backward;

public:
  Convolution2() {}

  // Fx is an optional work array of size max(A,B)*fftx->outputSize(),
  Convolution2(fftPad &fftx, Convolution &convolvey, Complex *Fx=NULL) :
    fftx(&fftx), convolvey(&convolvey), allocate(false) {
    init(Fx);
  }

  void init(Complex *Fx) {
    Forward=fftx->Forward;
    Backward=fftx->Backward;

    A=convolvey->A;
    B=convolvey->B;

    unsigned int c=fftx->outputSize();

    qx=fftx->q;
    Qx=fftx->Q;
    Sx=c/fftx->C; // Improve
    scale=1.0/(fftx->normalization()*convolvey->fft->normalization());

    unsigned int N=std::max(A,B);
    this->Fx=new Complex*[N];
    if(Fx) {
      for(unsigned int i=0; i < N; ++i)
        this->Fx[i]=Fx+i*c;
    } else {
      allocate=true;
      for(unsigned int i=0; i < N; ++i)
        this->Fx[i]=utils::ComplexAlign(c);
    }

    Lx=fftx->L;
    Ly=convolvey->L;
  }

  // A is the number of inputs.
  // B is the number of outputs.
  // Fy is an optional work array of size max(A,B)*ffty->outputSize(),
  // Vy is an optional work array of size B*ffty->workSizeV() (inplace usage)
  // Wy is an optional work array of size ffty->workSizeW();
  //   if changed between calls to convolve(), be sure to call pad()
  /*
    Convolution2(unsigned int Lx, unsigned int Ly,
    unsigned int Mx, unsigned int My,
    unsigned int A, unsigned int B, Complex *Fx=NULL,
    Complex *Fy=NULL, Complex *Vy=NULL, Complex *Wy=NULL): Fy(Fy), Vy(Vy), Wy(Wy) {}
  */

  ~Convolution2() {
    unsigned int N=std::max(A,B);
    if(allocate) {
      for(unsigned int i=0; i < N; ++i)
        utils::deleteAlign(Fx[i]);
    }
    delete [] Fx;
  }

  void forward(Complex **f, Complex **F, unsigned int rx) {
    for(unsigned int a=0; a < A; ++a)
      (fftx->*Forward)(f[a],F[a],rx,NULL); // C=Ly <= my py, Dx=1
  }

  void subconvolution(Complex **f, multiplier *mult, unsigned int C,
                      unsigned int stride, unsigned int offset=0) {
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
      for(unsigned int i=0; i < Lx; ++i) {
        Complex *hbLy=hb+Ly*i;
        for(unsigned int j=0; j < Ly; ++j)
          hbLy[j] *= scale;
      }
    }
  }
};

class ConvolutionHermitian2 : public Convolution2 {
public:
  // A is the number of inputs.
  // B is the number of outputs.
  // F is an optional work array of size max(A,B)*fft->outputSize(),
  // V is an optional work array of size B*fft->workSizeV() (for inplace usage)
  // W is an optional work array of size fft->workSizeW();
  //   if changed between calls to convolve(), be sure to call pad()
  // TODO: add inplace flag to avoid allocating W.
  ConvolutionHermitian2(fftPadCentered &fftx, ConvolutionHermitian &convolvey,
                        Complex *Fx=NULL) {
    this->fftx=&fftx;
    this->convolvey=&convolvey;
    init(Fx);
    Ly=utils::ceilquotient(convolvey.L,2);
  }
};


extern void optionsHybrid(int argc, char* argv[]);

} //end namespace fftwpp

#endif
