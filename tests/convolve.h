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

#include <cfloat>
#include <climits>

#include "Complex.h"
#include "fftw++.h"
#include "utils.h"
#include "Array.h"

#ifndef __convolve_h__
#define __convolve_h__ 1

namespace utils {
extern void optionsHybrid(int argc, char* argv[]);
}

namespace fftwpp {

extern const double twopi;

// Constants used for initialization and testing.
const Complex I(0.0,1.0);

extern unsigned int mOption;
extern unsigned int DOption;

extern int IOption;

extern unsigned int surplusFFTsizes;

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
typedef void (fftBase::*FFTCall)(Complex *f, Complex *F, unsigned int r, Complex *W, double scale);
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
  unsigned int Q; // number of residues
  unsigned int R; // number of residue blocks
  unsigned int dr; // r increment
  unsigned int D; // number of residues stored in F at a time
  unsigned int D0; // Remainder
  unsigned int Cm;
  unsigned int b; // Block size
  Complex *W0; // Temporary work memory for testing accuracy
  bool centered;
  bool inplace;
  bool overwrite;
  FFTcall Forward;
  FFTCall Backward;
  FFTPad Pad;
protected:
  Complex *Zetaqp;
  Complex *Zetaqp0;
  Complex *Zetaqm;
  Complex *ZetaqmS;
  Complex *Zetaqm0;
  Complex *ZetaqmS0;
public:

  void common();

  void initZetaqm(unsigned int q, unsigned int m);

  class OptBase {
  public:
    unsigned int m,q,D;
    double T;

    virtual double time(unsigned int L, unsigned int M, unsigned int C,
                        unsigned int m, unsigned int q,unsigned int D,
                        Application &app)=0;

    virtual bool valid(unsigned int D, unsigned int p) {return true;}

    void check(unsigned int L, unsigned int M,
               Application& app, unsigned int C, unsigned int m,
               bool fixed=false, bool mForced=false, bool centered=false);

    // Determine optimal m,q values for padding L data values to
    // size >= M
    // If fixed=true then an FFT of size M is enforced.
    void scan(unsigned int L, unsigned int M, Application& app,
              unsigned int C, bool Explicit=false, bool fixed=false,
              bool centered=false);
  };

  fftBase(unsigned int L, unsigned int M, unsigned int C,
          bool centered=false) :
    L(L), M(M), C(C), centered(centered) {}

  fftBase(unsigned int L, unsigned int M, unsigned int C,
          unsigned int m, unsigned int q, unsigned int D,
          bool centered=false) :
    L(L), M(M), C(C), m(m), p(utils::ceilquotient(L,m)), q(q), D(D),
    centered(centered) {}

  fftBase(unsigned int L, unsigned int M, Application& app,
          unsigned int C=1, bool Explicit=false, bool fixed=false,
          bool centered=false) :
    L(L), M(M), C(C), centered(centered) {}

  virtual ~fftBase();

  void padNone(Complex *W) {}

  virtual void padSingle(Complex *W) {}
  virtual void padMany(Complex *W) {}

  virtual void forwardShifted(Complex *f, Complex *F, unsigned int r, Complex *W) {}
  virtual void backwardShifted(Complex *F, Complex *f, unsigned int r, Complex *W, double scale) {}

  virtual void forwardExplicit(Complex *f, Complex *F, unsigned int, Complex *W)=0;
  virtual void forwardExplicitMany(Complex *f, Complex *F, unsigned int, Complex *W)=0;

  virtual void backwardExplicit(Complex *F, Complex *f, unsigned int, Complex *W, double scale)=0;
  virtual void backwardExplicitMany(Complex *F, Complex *f, unsigned int, Complex *W, double scale)=0;

  virtual void forward(Complex *f, Complex *F0, unsigned int r0, Complex *W) {}
  virtual void forwardMany(Complex *f, Complex *F, unsigned int r, Complex *W) {}

  virtual void forward2(Complex *f, Complex *F0, unsigned int r0, Complex *W) {}
  virtual void forward2Many(Complex *f, Complex *F, unsigned int r, Complex *W) {}
  virtual void forward2C(Complex *f, Complex *F0, unsigned int r0, Complex *W) {}
  virtual void forward2CMany(Complex *f, Complex *F, unsigned int r, Complex *W) {}

  virtual void forwardInner(Complex *f, Complex *F0, unsigned int r0, Complex *W) {}
  virtual void forwardInnerMany(Complex *f, Complex *F, unsigned int r, Complex *W) {}
  virtual void forwardInnerC(Complex *f, Complex *F0, unsigned int r0, Complex *W) {}
//  virtual void forwardInnerCMany(Complex *f, Complex *F, unsigned int r, Complex *W) {}

  virtual void backward(Complex *F0, Complex *f, unsigned int r0, Complex *W, double scale) {}
  virtual void backwardMany(Complex *F, Complex *f, unsigned int r, Complex *W, double scale) {}

  virtual void backward2(Complex *F0, Complex *f, unsigned int r0, Complex *W, double scale) {}
  virtual void backward2Many(Complex *F, Complex *f, unsigned int r, Complex *W, double scale) {}
  virtual void backward2C(Complex *F0, Complex *f, unsigned int r0, Complex *W, double scale) {}
  virtual void backward2CMany(Complex *F, Complex *f, unsigned int r, Complex *W, double scale) {}

  virtual void backwardInner(Complex *F0, Complex *f, unsigned int r0, Complex *W, double scale) {}
  virtual void backwardInnerMany(Complex *F, Complex *f, unsigned int r, Complex *W, double scale) {}
  virtual void backwardInnerC(Complex *F0, Complex *f, unsigned int r0, Complex *W, double scale) {}
//  virtual void backwardInnerCMany(Complex *F, Complex *f, unsigned int r, Complex *W, double scale) {}

  unsigned int normalization() {
    return M;
  }

  virtual unsigned int outputSize() {
    return b*D;
  }

  virtual unsigned int fullOutputSize() {
    return b*D*Q; // Check!
  }

  unsigned int outputs() {
    return C*M;
  }

  virtual bool conjugates() {
    return D > 1 && (p <= 2 || (centered && p % 2 == 0));
  }

  unsigned int residueBlocks() {
    return conjugates() ? utils::ceilquotient(Q,2) : Q;
  }

  unsigned int Dr() {return conjugates() ? D/2 : D;}

  unsigned int increment(unsigned int r) {
    return r > 0 ? dr : (conjugates() ? utils::ceilquotient(D0,2) : D0);
  }

  unsigned int blockOffset(unsigned int r) {
    if(conjugates())
      return r > 0 ? b*(D0+2*(r-utils::ceilquotient(D0,2))) : 0;
    else return b*r;
  }

  unsigned int nloops() {
    unsigned int count=0;
    for(unsigned int r=0; r < R; r += increment(r))
      ++count;
    return count;
  }

  bool loop2(unsigned int A, unsigned int B) {
    return nloops() == 2 && A > B;
  }

  virtual unsigned int ninputs() {
    return C*L;
  }

  unsigned int workSizeV(unsigned int A, unsigned int B) {
    return nloops() == 1 || loop2(A,B) ? 0 : ninputs();
  }

  virtual unsigned int workSizeW() {
    return q == 1 || inplace ? 0 : outputSize();
  }

  unsigned int repad() {
    return !inplace && L < m;
  }

  void initialize(Complex *f, Complex *g);

  double meantime(Application& app, double *Stdev=NULL);
  double report(Application& app);

  unsigned int residue(unsigned int r, unsigned int q);
};

class fftPad : public fftBase {
protected:
  mfft1d *fftm,*fftm2;
  mfft1d *ifftm,*ifftm2;
  mfft1d *fftp;
  mfft1d *ifftp;
public:

  class Opt : public OptBase {
  public:
    Opt(unsigned int L, unsigned int M, Application& app,
        unsigned int C, bool Explicit=false, bool fixed=false) {
      scan(L,M,app,C,Explicit,fixed);
    }

    double time(unsigned int L, unsigned int M, unsigned int C,
                unsigned int m, unsigned int q,unsigned int D,
                Application &app) {
      fftPad fft(L,M,C,m,q,D,false);
      return fft.meantime(app);
    }
  };

  fftPad(unsigned int L, unsigned int M, unsigned int C,
         bool centered) : fftBase(L,M,C,centered) {};

  // Compute an fft padded to N=m*q >= M >= L
  fftPad(unsigned int L, unsigned int M, unsigned int C,
         unsigned int m, unsigned int q,unsigned int D, bool centered=false) :
    fftBase(L,M,C,m,q,D,centered) {
    init();
  }

  // Normal entry point.
  // Compute C ffts of length L and distance 1 padded to at least M
  // (or exactly M if fixed=true)
  fftPad(unsigned int L, unsigned int M, Application& app,
         unsigned int C=1, bool Explicit=false, bool fixed=false,
         bool centered=false) :
    fftBase(L,M,app,C,Explicit,fixed,centered) {
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

  void forward(Complex *f, Complex *F) {
    (this->*Pad)(W0);
    for(unsigned int r=0; r < R; r += increment(r))
      (this->*Forward)(f,F+blockOffset(r),r,W0);
  }

  void backward(Complex *F, Complex *f, double scale=1.0) {
    for(unsigned int r=0; r < R; r += increment(r))
      (this->*Backward)(F+blockOffset(r),f,r,W0,scale);
  }

  void forwardExplicit(Complex *f, Complex *F, unsigned int, Complex *W);
  void forwardExplicitMany(Complex *f, Complex *F, unsigned int, Complex *W);

  void backwardExplicit(Complex *F, Complex *f, unsigned int, Complex *W, double scale);
  void backwardExplicitMany(Complex *F, Complex *f, unsigned int, Complex *W, double scale);

  // p=1 && C=1
  void forward(Complex *f, Complex *F0, unsigned int r0, Complex *W);

  void forwardMany(Complex *f, Complex *F, unsigned int r, Complex *W);

  void forward2(Complex *f, Complex *F0, unsigned int r0, Complex *W);

  void forward2Many(Complex *f, Complex *F, unsigned int r, Complex *W);

  void forwardInner(Complex *f, Complex *F0, unsigned int r0, Complex *W);

  void forwardInnerMany(Complex *f, Complex *F, unsigned int r, Complex *W);

// Compute an inverse fft of length N=m*q unpadded back
// to size m*p >= L.
// input and output arrays must be distinct
// Input F destroyed
  void backward(Complex *F0, Complex *f, unsigned int r0, Complex *W, double scale);

  void backwardMany(Complex *F, Complex *f, unsigned int r, Complex *W, double scale);

  void backward2(Complex *F0, Complex *f, unsigned int r0, Complex *W, double scale);

  void backward2Many(Complex *F, Complex *f, unsigned int r, Complex *W, double scale);

  void backwardInner(Complex *F0, Complex *f, unsigned int r0, Complex *W, double scale);
  void backwardInnerMany(Complex *F, Complex *f, unsigned int r, Complex *W, double scale);
};

class fftPadCentered : public fftPad {
  Complex *ZetaShift;
  FFTcall fftBaseForward;
  FFTCall fftBaseBackward;
public:

  class Opt : public OptBase {
  public:
    virtual bool valid(unsigned int D, unsigned int p) {
      return D > 1 || p == 2;
    }

    Opt(unsigned int L, unsigned int M, Application& app,
        unsigned int C, bool Explicit=false, bool fixed=false) {
      scan(L,M,app,C,Explicit,fixed,true);
    }

    double time(unsigned int L, unsigned int M, unsigned int C,
                unsigned int m, unsigned int q,unsigned int D,
                Application &app) {
      fftPadCentered fft(L,M,C,m,q,D);
      return fft.meantime(app);
    }
  };

  // Compute an fft padded to N=m*q >= M >= L
  fftPadCentered(unsigned int L, unsigned int M, unsigned int C,
                 unsigned int m, unsigned int q,unsigned int D) :
    fftPad(L,M,C,m,q,D,true) {
    init();
  }

  // Normal entry point.
  // Compute C ffts of length L and distance 1 padded to at least M
  // (or exactly M if fixed=true)
  fftPadCentered(unsigned int L, unsigned int M, Application& app,
                 unsigned int C=1, bool Explicit=false, bool fixed=false) :
    fftPad(L,M,C,true) {
    Opt opt=Opt(L,M,app,C,Explicit,fixed);
    m=opt.m;
    if(Explicit)
      M=m;
    q=opt.q;
    D=opt.D;
    fftPad::init();
    init();
  }

  ~fftPadCentered() {
    if(ZetaShift)
      utils::deleteAlign(ZetaShift);
  }

  bool conjugates() {return D > 1 && (p == 1 || p % 2 == 0);}

  void init();
  void initShift();

  void forwardShifted(Complex *f, Complex *F, unsigned int r, Complex *W);
  void backwardShifted(Complex *F, Complex *f, unsigned int r, Complex *W, double scale);

  void forwardShift(Complex *F, unsigned int r0);
  void backwardShift(Complex *F, unsigned int r0);

  void forward2C(Complex *f, Complex *F0, unsigned int r0, Complex *W);
  void forward2CMany(Complex *f, Complex *F, unsigned int r, Complex *W);

  void backward2C(Complex *F0, Complex *f, unsigned int r0, Complex *W, double scale);
  void backward2CMany(Complex *F, Complex *f, unsigned int r, Complex *W, double scale);

  void forwardInnerC(Complex *f, Complex *F0, unsigned int r0, Complex *W);
  void backwardInnerC(Complex *F0, Complex *f, unsigned int r0, Complex *W, double scale);
};

class fftPadHermitian : public fftBase {
  unsigned int e;
  mcrfft1d *crfftm;
  mrcfft1d *rcfftm;
public:

  class Opt : public OptBase {
  public:
    virtual bool valid(unsigned int D, unsigned int p) {return D == 2;}

    Opt(unsigned int L, unsigned int M, Application& app,
        unsigned int C, bool Explicit=false, bool fixed=false) {
      scan(L,M,app,C,Explicit,fixed);
    }

    double time(unsigned int L, unsigned int M, unsigned int C,
                unsigned int m, unsigned int q, unsigned int D,
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

  void forward(Complex *f, Complex *F) {
    for(unsigned int r=0; r < R; r += increment(r))
      (this->*Forward)(f,F+blockOffset(r),r,W0);
  }

  void backward(Complex *F, Complex *f, double scale=1.0) {
    for(unsigned int r=0; r < R; r += increment(r))
      (this->*Backward)(F+blockOffset(r),f,r,W0,scale);
  }

  void forwardExplicit(Complex *f, Complex *F, unsigned int, Complex *W);
  void forwardExplicitMany(Complex *f, Complex *F, unsigned int, Complex *W);

  void backwardExplicit(Complex *F, Complex *f, unsigned int, Complex *W, double scale);
  void backwardExplicitMany(Complex *F, Complex *f, unsigned int, Complex *W, double scale);

  void forward2(Complex *f, Complex *F0, unsigned int r0, Complex *W);
  void forward2Many(Complex *f, Complex *F, unsigned int r, Complex *W);

  void backward2(Complex *F0, Complex *f, unsigned int r0, Complex *W, double scale);
  void backward2Many(Complex *F, Complex *f, unsigned int r, Complex *W, double scale);

  void forwardInnerC(Complex *f, Complex *F0, unsigned int r0, Complex *W);
  void backwardInnerC(Complex *F0, Complex *f, unsigned int r0, Complex *W, double scale);

  unsigned int outputSize() {
    return C*(e+1)*D;
  }

  unsigned int ninputs() {
    return C*utils::ceilquotient(L,2);
  }

  unsigned int fullOutputSize() {
    return C*(e+1)*q; // FIXME for inner
  }
};

class ForwardBackward : public Application {
protected:
  unsigned int A;
  unsigned int B;
  unsigned int C;
  unsigned int D;
  unsigned int D0;
  unsigned int dr;
  unsigned int Q;
  Complex **f;
  Complex **F;
  Complex **h;
  Complex *W;
  FFTcall Forward;
  FFTCall Backward;
public:
  ForwardBackward(unsigned int A, unsigned int B) :
    A(A), B(B), f(NULL), F(NULL), h(NULL), W(NULL) {
  }

  virtual ~ForwardBackward() {
    clear();
  }

  void init(fftBase &fft);

  double time(fftBase &fft, unsigned int K);

  void clear();

};

typedef void multiplier(Complex **, unsigned int n, unsigned int threads);

// Multiplication routine for binary convolutions and taking two inputs of size e.
void multbinary(Complex **F, unsigned int n, unsigned int threads);
void realmultbinary(Complex **F, unsigned int n, unsigned int threads);

class Convolution {
public:
  fftBase *fft;
  unsigned int A;
  unsigned int B;
  unsigned int N; // max(A,B)
  unsigned int L;
protected:
  unsigned int q;
  unsigned int Q;
  unsigned int D;
  unsigned int R;
  unsigned int D0;
  unsigned int r;
  unsigned int b;
  Complex **F,**Fp;
  Complex *FpB;
  Complex **G; // For overwrite optimization
  Complex **V;
  Complex *W;
  Complex *H;
  Complex *W0;
  double scale;
  bool allocate;
  bool allocateV;
  bool allocateW;
  unsigned int nloops;
  bool loop2;
  unsigned int overwrite;
  unsigned int noutputs;
  FFTcall Forward;
  FFTCall Backward;
  FFTPad Pad;
public:
  // A is the number of inputs.
  // B is the number of outputs.
  // F is an optional work array of size max(A,B)*fft->outputSize(),
  // W is an optional work array of size fft->workSizeW();
  //    call pad() if changed between calls to convolve()
  // V is an optional work array of size B*fft->workSizeV(A,B)
  //   (only needed for inplace usage)
  Convolution(unsigned int A, unsigned int B,
              Complex *F=NULL, Complex *W=NULL, Complex *V=NULL) :
    A(A), B(B), W(W), allocate(false) {}

  Convolution(fftBase &fft, unsigned int A, unsigned int B,
              Complex *F=NULL, Complex *W=NULL, Complex *V=NULL) :
    fft(&fft), A(A), B(B), W(W), allocate(false) {
    init(F,V);
  }

  void init(Complex *F, Complex *V);

  void initV() {
    allocateV=true;
    V=new Complex*[B];
    unsigned int size=fft->workSizeV(A,B);
    for(unsigned int i=0; i < B; ++i)
      V[i]=utils::ComplexAlign(size);
  }

  unsigned int increment(unsigned int r) {
    return fft->increment(r);
  }

  ~Convolution();

  void normalize(Complex **f, unsigned int offset=0) {
    for(unsigned int b=0; b < B; ++b) {
      Complex *fb=f[b]+offset;
      for(unsigned int i=0; i < noutputs; ++i)
        fb[i] *= scale;
    }
  }

  void convolve0(Complex **f, multiplier *mult, unsigned int offset=0);

  void convolve(Complex **f, multiplier *mult, unsigned int offset=0) {
    convolve0(f,mult,offset);
    // TODO:
    // if(fft->overwrite) return;
    normalize(f,offset);
  }
};

class ConvolutionHermitian : public Convolution {
public:
  // A is the number of inputs.
  // B is the number of outputs.
  // F is an optional work array of size max(A,B)*fft->outputSize(),
  // W is an optional work array of size fft->workSizeW();
  //   if changed between calls to convolve(), be sure to call pad()
  // V is an optional work array of size B*fft->workSizeV(A,B)
  //   (for inplace usage)
  ConvolutionHermitian(fftPadHermitian &fft, unsigned int A,
                       unsigned int B, Complex *F=NULL, Complex *W=NULL,
                       Complex *V=NULL) : Convolution(A,B,F,W,V) {
    this->fft=&fft;
    init(F,V);
    b=q == 1 ? fft.Cm : 2*fft.b;
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

  // Zero out Nyquist modes in noncompact case
  if(xorigin == mx) {
    unsigned int Nyquist=offset-stop;
    for(unsigned int j=0; j < my; ++j)
      f[Nyquist+j]=0.0;
  }
}

class Convolution2 {
protected:
  fftBase *fftx;
  fftBase *ffty;
  Convolution *convolvey;
  ForwardBackward *FB;
  unsigned int Nx;    // x dimension of Fx buffer
  unsigned int Lx,Ly; // x,y dimensions of input data
  unsigned int A;
  unsigned int B;
  unsigned int Q;
  unsigned int qx;
  unsigned int Qx; // number of residues
  unsigned int Rx; // number of residue blocks
  unsigned int r;
  Complex **F,**Fp;
  Complex *FpB;
  Complex **V;
  Complex *W;
  Complex *W0;
  bool allocate;
  bool allocateV;
  bool allocateW;
  bool loop2;
  double scale;
  FFTcall Forward;
  FFTCall Backward;
  unsigned int nloops;
public:
  Convolution2() : W(NULL), allocate(false), allocateW(false) {}

  // Lx,Ly: x,y dimensions of input data
  // Mx,My: x,y dimensions of transformed data, including padding
  // A: number of inputs
  // B: number of outputs
  Convolution2(unsigned int Lx, unsigned int Ly,
               unsigned int Mx, unsigned int My,
               unsigned int A, unsigned int B) :
    W(NULL), allocate(false), allocateW(false) {
    FB=new ForwardBackward(A,B);
    fftx=new fftPad(Lx,Mx,*FB,Ly);
    ffty=new fftPad(Ly,My,*FB);
    convolvey=new Convolution(*ffty,A,B);
    init();
  }

  // F is an optional work array of size max(A,B)*fftx->outputSize(),
  // W is an optional work array of size fftx->workSizeW(),
  //   call fftx->pad() if changed between calls to convolve(),
  // V is an optional work array of size B*fftx->workSizeV(A,B)
  //    (only needed for inplace usage)
  Convolution2(fftPad &fftx, Convolution &convolvey,
               Complex *F=NULL, Complex *W=NULL, Complex *V=NULL) :
    fftx(&fftx), ffty(NULL), convolvey(&convolvey), W(W), allocate(false),
    allocateW(false) {
    init(F,V);
  }

  void init(Complex *F=NULL, Complex *V=NULL) {
    Forward=fftx->Forward;
    Backward=fftx->Backward;

    if(!fftx->inplace && !W) {
      allocateW=true;
      W=utils::ComplexAlign(fftx->workSizeW());
    }

    A=convolvey->A;
    B=convolvey->B;

    unsigned int c=fftx->outputSize();

    qx=fftx->q;
    Qx=fftx->Q;
    Rx=fftx->R;
    Nx=fftx->b/fftx->C; // Improve
    scale=1.0/(fftx->normalization()*convolvey->fft->normalization());

    unsigned int N=std::max(A,B);
    this->F=new Complex*[N];
    if(F) {
      for(unsigned int i=0; i < N; ++i)
        this->F[i]=F+i*c;
    } else {
      allocate=true;
      for(unsigned int i=0; i < N; ++i)
        this->F[i]=utils::ComplexAlign(c);
    }

    Lx=fftx->L;
    Ly=convolvey->L;

    nloops=fftx->nloops();
    loop2=fftx->loop2(A,B);
    int extra;
    if(loop2) {
      r=fftx->increment(0);
      Fp=new Complex*[A];
      Fp[0]=this->F[A-1];
      for(unsigned int a=1; a < A; ++a)
        Fp[a]=this->F[a-1];
      FpB=fftx->inplace ? NULL : Fp[B];
      extra=1;
    } else {
      extra=0;
    }

    if(A > B+extra && !fftx->inplace) {
      W0=this->F[B];
    } else
      W0=this->W;

    allocateV=false;

    if(V) {
      this->V=new Complex*[B];
      unsigned int size=fftx->workSizeV(A,B);
      for(unsigned int i=0; i < B; ++i)
        this->V[i]=V+i*size;
    } else
      this->V=NULL;
  }

  void initV() {
    allocateV=true;
    V=new Complex*[B];
    unsigned int size=fftx->workSizeV(A,B);
    for(unsigned int i=0; i < B; ++i)
      V[i]=utils::ComplexAlign(size);
  }

  ~Convolution2() {
    if(allocateW)
      utils::deleteAlign(W);
    unsigned int N=std::max(A,B);
    if(allocate) {
      for(unsigned int i=0; i < N; ++i)
        utils::deleteAlign(F[i]);
    }
    delete [] F;

    if(allocateV) {
      for(unsigned int i=0; i < B; ++i)
        utils::deleteAlign(V[i]);
    }
    if(V)
      delete [] V;
    if(ffty) {
      delete convolvey;
      delete FB;
      delete ffty;
      delete fftx;
    }
  }

  void forward(Complex **f, Complex **F, unsigned int rx,
               unsigned int offset=0) {
    for(unsigned int a=0; a < A; ++a)
      (fftx->*Forward)(f[a]+offset,F[a],rx,W); // C=Ly <= my py, Dx=1
  }

  void subconvolution(Complex **f, multiplier *mult, unsigned int C,
                      unsigned int stride, unsigned int offset=0) {
    for(unsigned int i=0; i < C; ++i)
      convolvey->convolve0(f,mult,offset+i*stride);
  }

  void backward(Complex **F, Complex **f, unsigned int rx,
                unsigned int offset=0) {
    for(unsigned int b=0; b < B; ++b)
      (fftx->*Backward)(F[b],f[b]+offset,rx,W,scale);
  }

  void normalize(Complex **h, unsigned int offset=0) {
    for(unsigned int b=0; b < B; ++b) {
      Complex *hb=h[b]+offset;
      for(unsigned int i=0; i < Lx; ++i) {
        Complex *hbi=hb+Ly*i;
        for(unsigned int j=0; j < Ly; ++j)
          hbi[j] *= scale;
      }
    }
  }

// f is a pointer to A distinct data blocks each of size Lx*Ly,
// shifted by offset.
  void convolve0(Complex **f, multiplier *mult, unsigned int offset=0) {
    if(fftx->overwrite) {
      forward(f,F,Rx,offset);
      subconvolution(f,mult,fftx->p*Nx,Ly,offset);
      subconvolution(F,mult,Nx,Ly,offset);
      backward(F,f,Rx,offset);
    } else {
      if(loop2) {
        for(unsigned int a=0; a < A; ++a)
          (fftx->*Forward)(f[a]+offset,F[a],0,W);
        subconvolution(F,mult,fftx->D0*Nx,Ly,offset);

        for(unsigned int b=0; b < B; ++b) {
          (fftx->*Forward)(f[b]+offset,Fp[b],r,W);
          (fftx->*Backward)(F[b],f[b]+offset,0,W0,1.0);
        }
        for(unsigned int a=B; a < A; ++a)
          (fftx->*Forward)(f[a]+offset,Fp[a],r,W);
        subconvolution(Fp,mult,fftx->D*Nx,Ly,offset);
        for(unsigned int b=0; b < B; ++b)
          (fftx->*Backward)(Fp[b],f[b]+offset,r,FpB,1.0);
      } else {
        unsigned int Offset;
        bool useV=nloops > 1;
        Complex **h0;
        if(useV) {
          if(!V) initV();
          h0=V;
          Offset=0;
        } else {
          Offset=offset;
          h0=f;
        }

        for(unsigned int rx=0; rx < Rx; rx += fftx->increment(rx)) {
          forward(f,F,rx,offset);
          subconvolution(F,mult,(rx == 0 ? fftx->D0 : fftx->D)*Nx,Ly,offset);
          backward(F,h0,rx,Offset);
        }

        if(useV) {
          for(unsigned int b=0; b < B; ++b) {
            Complex *fb=f[b]+offset;
            Complex *hb=h0[b];
            for(unsigned int i=0; i < Lx; ++i) {
              unsigned int Lyi=Ly*i;
              Complex *fbi=fb+Lyi;
              Complex *hbi=hb+Lyi;
              for(unsigned int j=0; j < Ly; ++j)
                fbi[j]=hbi[j];
            }
          }
        }
      }
    }
  }

  void convolve(Complex **f, multiplier *mult, unsigned int offset=0) {
    convolve0(f,mult,offset);
    if(fftx->overwrite) return;
    normalize(f,offset);
  }
};

class ConvolutionHermitian2 : public Convolution2 {
public:
  // F is an optional work array of size max(A,B)*fftx->outputSize(),
  // W is an optional work array of size fftx->workSizeW(),
  //    call fftx->pad() if changed between calls to convolve()
  // V is an optional work array of size B*fftx->workSizeV(A,B)
  //    (only needed for inplace usage)
  ConvolutionHermitian2(fftPadCentered &fftx, ConvolutionHermitian &convolvey,
                        Complex *F=NULL, Complex *W=NULL, Complex *V=NULL) {
    this->fftx=&fftx;
    this->convolvey=&convolvey;
    init(F,V);
    Ly=utils::ceilquotient(convolvey.L,2);
  }

  ConvolutionHermitian2(unsigned int Lx, unsigned int Ly,
                        unsigned int Mx, unsigned int My,
                        unsigned int A, unsigned int B) {
    unsigned int Hy=utils::ceilquotient(Ly,2);
    FB=new ForwardBackward(A,B);
    fftx=new fftPadCentered(Lx,Mx,*FB,Hy);
    unsigned int q=utils::ceilquotient(My,Hy);
    fftPadHermitian *ffty=new fftPadHermitian(Ly,My,1,Hy,q,2);
    convolvey=new ConvolutionHermitian(*ffty,A,B);
    this->ffty=ffty;
    init();
    this->Ly=Hy;
  }
};


} //end namespace fftwpp

#endif
