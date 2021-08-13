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
unsigned int prevfftsize(unsigned int M, bool mixed);

class fftBase;

typedef void (fftBase::*FFTcall)(Complex *f, Complex *F, unsigned int r, Complex *W);
typedef void (fftBase::*FFTPad)(Complex *W);

class Application : public ThreadBase
{
public:
  Application(unsigned int threads=fftw::maxthreads) : ThreadBase(threads) {
    cout << "Using " << threads << " threads." << endl << endl;
  };
  virtual void init(fftBase &fft)=0;
  virtual void clear()=0;
  virtual double time(fftBase &fft, unsigned int K)=0;
};

class fftBase : public ThreadBase {
public:
  unsigned int L; // number of unpadded Complex data values
  unsigned int M; // minimum number of padded Complex data values
  unsigned int C; // number of FFTs to compute in parallel
  unsigned int S; // stride between successive elements
  unsigned int m;
  unsigned int p;
  unsigned int q;
  unsigned int n;
  unsigned int Q; // number of residues
  unsigned int R; // number of residue blocks
  unsigned int dr; // r increment
  unsigned int D; // number of residues stored in F at a time
  unsigned int D0; // Remainder
  unsigned int Cm,Sm;
  unsigned int b; // Total block size, including stride
  unsigned int z; // Block size of a single copy
  bool centered;
  bool inplace;
  bool overwrite;
  FFTcall Forward,Backward;
  FFTcall ForwardAll,BackwardAll;
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

    virtual double time(unsigned int L, unsigned int M, unsigned int C, unsigned int S,
                        unsigned int m, unsigned int q,unsigned int D,
                        Application &app)=0;

    virtual bool valid(unsigned int D, unsigned int p, unsigned int S) {
      return D == 1 || S == 1;
    }

    void check(unsigned int L, unsigned int M,
               Application& app, unsigned int C, unsigned int S, unsigned int m,
               bool fixed=false, bool mForced=false, bool centered=false);

    // Determine optimal m,q values for padding L data values to
    // size >= M
    // If fixed=true then an FFT of size M is enforced.
    void scan(unsigned int L, unsigned int M, Application& app,
              unsigned int C, unsigned int S, bool Explicit=false, bool fixed=false,
              bool centered=false);
  };

  void invalid () {
    cerr << "Invalid parameters: " << endl
         << " D=" << D << " p=" << p << " C=" << C << endl;
    exit(-1);
  }

  fftBase(unsigned int L, unsigned int M, unsigned int C, unsigned int S,
          unsigned int threads=fftw::maxthreads, bool centered=false) :
    ThreadBase(threads), L(L), M(M), C(C), S(S), centered(centered) {}

  fftBase(unsigned int L, unsigned int M, unsigned int C, unsigned int S,
          unsigned int m, unsigned int q, unsigned int D,
          unsigned int threads=fftw::maxthreads, bool centered=false) :
    ThreadBase(threads), L(L), M(M), C(C), S(S), m(m),
    p(utils::ceilquotient(L,m)), q(q), D(D), centered(centered) {}

  fftBase(unsigned int L, unsigned int M, Application& app,
          unsigned int C=1, unsigned int S=0, bool Explicit=false, bool fixed=false,
          bool centered=false) :
    ThreadBase(app.Threads()), L(L), M(M), C(C), S(S == 0 ? C : S), centered(centered) {}

  virtual ~fftBase();

  void padNone(Complex *W) {}

  virtual void padSingle(Complex *W) {}
  virtual void padMany(Complex *W) {}

  void pad(Complex *W=NULL) {
    if(W)
      (this->*Pad)(W);
  }

  void forward(Complex *f, Complex *F, unsigned int r=0, Complex *W=NULL) {
    (this->*Forward)(f,F,r,W);
  }

  void backward(Complex *f, Complex *F, unsigned int r=0, Complex *W=NULL) {
    (this->*Backward)(f,F,r,W);
  }

  virtual void forwardExplicit(Complex *f, Complex *F, unsigned int, Complex *W)=0;
  virtual void forwardExplicitMany(Complex *f, Complex *F, unsigned int, Complex *W)=0;
  virtual void forwardExplicitFast(Complex *f, Complex *F, unsigned int, Complex *W) {}
  virtual void forwardExplicitManyFast(Complex *f, Complex *F, unsigned int, Complex *W) {}
  virtual void forwardExplicitSlow(Complex *f, Complex *F, unsigned int r, Complex *W) {}
  virtual void forwardExplicitManySlow(Complex *f, Complex *F, unsigned int r, Complex *W) {}

  virtual void backwardExplicit(Complex *F, Complex *f, unsigned int, Complex *W)=0;
  virtual void backwardExplicitMany(Complex *F, Complex *f, unsigned int, Complex *W)=0;
  virtual void backwardExplicitFast(Complex *F, Complex *f, unsigned int, Complex *W) {}
  virtual void backwardExplicitManyFast(Complex *F, Complex *f, unsigned int, Complex *W) {}
  virtual void backwardExplicitSlow(Complex *F, Complex *f, unsigned int r, Complex *W) {}
  virtual void backwardExplicitManySlow(Complex *F, Complex *f, unsigned int r, Complex *W) {}

  virtual unsigned int indexExplicit(unsigned int r, unsigned int i) {
    return i;
  }

  // Return spatial index for residue r at position i
  unsigned int index(unsigned int r, unsigned int I) {
    if(q == 1) return I;
    unsigned int i=I/S;
    unsigned int s=i%m;
    unsigned int u;
    unsigned int P;
    if(D > 1 && ((centered && p % 2 == 0) || p <= 2)) {
      P=utils::ceilquotient(p,2);
      u=(i/m)%P;
      unsigned int offset=r == 0 && i >= P*m && D0 % 2 == 1 ? 1 : 0;
      double incr=(i+P*m*offset)/(2*P*m);
      r += incr;
      if(i/(P*m)-2*incr+offset == 1) {
        if((!centered && p == 2) || (r > 0 && u == 0))
          s=s > 0 ? s-1 : m-1;
        if(r == 0)
          r=n/2;
        else {
          r=n-r;
          u=u > 0 ? u-1 : P-1;
        }
      }
    } else {
      u=(i/m)%p;
      r += i/(p*m);
    }
    return S*(q*s+n*u+r-i)+I;
  }

  virtual void forward1(Complex *f, Complex *F0, unsigned int r0, Complex *W) {
  }
  virtual void forward1All(Complex *f, Complex *F0, unsigned int r0, Complex *W) {}
  virtual void forward1Many(Complex *f, Complex *F, unsigned int r, Complex *W) {}
  virtual void forward1ManyAll(Complex *f, Complex *F, unsigned int r, Complex *W) {}
  virtual void forward2(Complex *f, Complex *F0, unsigned int r0, Complex *W) {}
  virtual void forward2All(Complex *f, Complex *F0, unsigned int r0, Complex *W) {}
  virtual void forward2Many(Complex *f, Complex *F, unsigned int r, Complex *W) {}
  virtual void forward2ManyAll(Complex *f, Complex *F, unsigned int r, Complex *W) {}
  virtual void forwardInner(Complex *f, Complex *F0, unsigned int r0, Complex *W) {}
  virtual void forwardInnerAll(Complex *f, Complex *F0, unsigned int r0, Complex *W) {}
  virtual void forwardInnerMany(Complex *f, Complex *F, unsigned int r, Complex *W) {}
  virtual void forwardInnerManyAll(Complex *f, Complex *F, unsigned int r, Complex *W) {}

  virtual void backward1(Complex *F0, Complex *f, unsigned int r0, Complex *W) {}
  virtual void backward1All(Complex *F0, Complex *f, unsigned int r0, Complex *W) {}
  virtual void backward1Many(Complex *F, Complex *f, unsigned int r, Complex *W) {}
  virtual void backward1ManyAll(Complex *F, Complex *f, unsigned int r, Complex *W) {}
  virtual void backward2(Complex *F0, Complex *f, unsigned int r0, Complex *W) {}
  virtual void backward2All(Complex *F0, Complex *f, unsigned int r0, Complex *W) {}
  virtual void backward2Many(Complex *F, Complex *f, unsigned int r, Complex *W) {}
  virtual void backward2ManyAll(Complex *F, Complex *f, unsigned int r, Complex *W) {}
  virtual void backwardInner(Complex *F0, Complex *f, unsigned int r0, Complex *W) {}
  virtual void backwardInnerAll(Complex *F0, Complex *f, unsigned int r0, Complex *W) {}
  virtual void backwardInnerMany(Complex *F, Complex *f, unsigned int r, Complex *W) {}
  virtual void backwardInnerManyAll(Complex *F, Complex *f, unsigned int r, Complex *W) {}

  unsigned int normalization() {
    return M;
  }

  virtual unsigned int outputSize() {
    return b*D;
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

  unsigned int nloops() {
    unsigned int count=0;
    for(unsigned int r=0; r < R; r += increment(r))
      ++count;
    return count;
  }

  bool loop2(unsigned int A, unsigned int B) {
    return false;
    return nloops() == 2 && A > B;
  }

  virtual unsigned int inputSize() {
    return S*L;
  }

  // Number of complex outputs per residue per copy
  virtual unsigned int noutputs() {
    return z;
  }

  // Number of complex outputs per copy per iteration
  unsigned int complexOutputs(unsigned int r) {
    return z*(r == 0 ? D0 : D);
  }

  // Number of outputs per iteration
  virtual unsigned int noutputs(unsigned int r) {
    return complexOutputs(r);
  }

  unsigned int workSizeV(unsigned int A, unsigned int B) {
    return nloops() == 1 || loop2(A,B) ? 0 : inputSize();
  }

  virtual unsigned int workSizeW() {
    return q == 1 || inplace ? 0 : outputSize();
  }

  unsigned int repad() {
    return !inplace && L < m;
  }

  bool Overwrite(unsigned int A, unsigned int B) {
    return overwrite && A >= B;
  }

  double meantime(Application& app, double *Stdev=NULL);
  double report(Application& app);
};

class fftPad : public fftBase {
protected:
  mfft1d *fftm,*fftm0;
  mfft1d *ifftm,*ifftm0;
  mfft1d *fftp;
  mfft1d *ifftp;
public:

  class Opt : public OptBase {
  public:
    Opt() {}

    Opt(unsigned int L, unsigned int M, Application& app,
        unsigned int C, unsigned int S, bool Explicit=false, bool fixed=false) {
      scan(L,M,app,C,S,Explicit,fixed);
    }

    double time(unsigned int L, unsigned int M, unsigned int C, unsigned int S,
                unsigned int m, unsigned int q,unsigned int D,
                Application &app) {
      fftPad fft(L,M,C,S,m,q,D,app.Threads(),false);
      return fft.meantime(app);
    }
  };



  fftPad(unsigned int L, unsigned int M, unsigned int C, unsigned int S,
         unsigned int threads=fftw::maxthreads, bool centered=false) :
    fftBase(L,M,C,S,threads,centered) {}

  // Compute an fft padded to N=m*q >= M >= L
  fftPad(unsigned int L, unsigned int M, unsigned int C, unsigned int S,
         unsigned int m, unsigned int q, unsigned int D,
         unsigned int threads=fftw::maxthreads, bool centered=false) :
    fftBase(L,M,C,S,m,q,D,threads,centered) {
    Opt opt;
    if(q > 1 && !opt.valid(D,p,S)) invalid();
    init();
  }

  // Normal entry point.
  // Compute C ffts of length L with stride S >= C and distance 1
  // padded to at least M (or exactly M if fixed=true)
  fftPad(unsigned int L, unsigned int M, Application& app,
         unsigned int C=1, unsigned int S=0, bool Explicit=false,
         bool fixed=false, bool centered=false) :
    fftBase(L,M,app,C,S,Explicit,fixed,centered) {
    Opt opt=Opt(L,M,app,C,this->S,Explicit,fixed);
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

  void forwardExplicit(Complex *f, Complex *F, unsigned int, Complex *W);
  void forwardExplicitMany(Complex *f, Complex *F, unsigned int, Complex *W);

  void backwardExplicit(Complex *F, Complex *f, unsigned int, Complex *W);
  void backwardExplicitMany(Complex *F, Complex *f, unsigned int, Complex *W);

  // p=1 && C=1
  void forward1(Complex *f, Complex *F0, unsigned int r0, Complex *W);
  void forward1All(Complex *f, Complex *F0, unsigned int r0, Complex *W);
  void forward1Many(Complex *f, Complex *F, unsigned int r, Complex *W);
  void forward1ManyAll(Complex *f, Complex *F, unsigned int r, Complex *W);
  void forward2(Complex *f, Complex *F0, unsigned int r0, Complex *W);
  void forward2All(Complex *f, Complex *F0, unsigned int r0, Complex *W);
  void forward2Many(Complex *f, Complex *F, unsigned int r, Complex *W);
  void forward2ManyAll(Complex *f, Complex *F, unsigned int r, Complex *W);
  void forwardInner(Complex *f, Complex *F0, unsigned int r0, Complex *W);
  void forwardInnerMany(Complex *f, Complex *F, unsigned int r, Complex *W);

// Compute an inverse fft of length N=m*q unpadded back
// to size m*p >= L.
// input and output arrays must be distinct
// Input F destroyed
  void backward1(Complex *F0, Complex *f, unsigned int r0, Complex *W);
  void backward1All(Complex *F0, Complex *f, unsigned int r0, Complex *W);
  void backward1Many(Complex *F, Complex *f, unsigned int r, Complex *W);
  void backward1ManyAll(Complex *F, Complex *f, unsigned int r, Complex *W);
  void backward2(Complex *F0, Complex *f, unsigned int r0, Complex *W);
  void backward2All(Complex *F0, Complex *f, unsigned int r0, Complex *W);
  void backward2Many(Complex *F, Complex *f, unsigned int r, Complex *W);
  void backward2ManyAll(Complex *F, Complex *f, unsigned int r, Complex *W);
  void backwardInner(Complex *F0, Complex *f, unsigned int r0, Complex *W);
  void backwardInnerMany(Complex *F, Complex *f, unsigned int r, Complex *W);
};

class fftPadCentered : public fftPad {
  Complex *ZetaShift;
  FFTcall fftBaseForward,fftBaseBackward;
public:

  class Opt : public OptBase {
  public:
    Opt() {}

    Opt(unsigned int L, unsigned int M, Application& app,
        unsigned int C, unsigned int S, bool Explicit=false, bool fixed=false) {
      scan(L,M,app,C,S,Explicit,fixed,true);
    }

    bool valid(unsigned int D, unsigned int p, unsigned int S) {
      return p%2 == 0 && (D == 1 || S == 1);
    }

    double time(unsigned int L, unsigned int M, unsigned int C, unsigned int S,
                unsigned int m, unsigned int q, unsigned int D,
                Application &app) {
      fftPadCentered fft(L,M,C,S,m,q,D,app.Threads());
      return fft.meantime(app);
    }
  };

  // Compute an fft padded to N=m*q >= M >= L
  fftPadCentered(unsigned int L, unsigned int M, unsigned int C, unsigned int S,
                 unsigned int m, unsigned int q, unsigned int D,
                 unsigned int threads=fftw::maxthreads, bool fast=true) :
    fftPad(L,M,C,S,m,q,D,threads,true) {
    Opt opt;
    if(q > 1 && !opt.valid(D,p,S)) invalid();
    init(fast);
  }

  // Normal entry point.
  // Compute C ffts of length L and distance 1 padded to at least M
  // (or exactly M if fixed=true)
  fftPadCentered(unsigned int L, unsigned int M, Application& app,
                 unsigned int C=1, unsigned int S=0, bool Explicit=false, bool fixed=false,
                 bool fast=true) :
    fftPad(L,M,C,S,app.Threads(),true) {
    Opt opt=Opt(L,M,app,C,this->S,Explicit,fixed);
    m=opt.m;
    if(Explicit)
      M=m;
    q=opt.q;
    D=opt.D;
    fftPad::init();
    init(fast);
  }

  ~fftPadCentered() {
    if(ZetaShift)
      utils::deleteAlign(ZetaShift);
  }

  bool conjugates() {return D > 1 && (p == 1 || p % 2 == 0);}

  void init(bool fast);

  void forwardExplicitFast(Complex *f, Complex *F, unsigned int r, Complex *W);
  void forwardExplicitManyFast(Complex *f, Complex *F, unsigned int r, Complex *W);
  void forwardExplicitSlow(Complex *f, Complex *F, unsigned int r, Complex *W);
  void forwardExplicitManySlow(Complex *f, Complex *F, unsigned int r, Complex *W);

  void backwardExplicitFast(Complex *F, Complex *f, unsigned int r, Complex *W);
  void backwardExplicitManyFast(Complex *F, Complex *f, unsigned int r, Complex *W);
  void backwardExplicitSlow(Complex *F, Complex *f, unsigned int r, Complex *W);
  void backwardExplicitManySlow(Complex *F, Complex *f, unsigned int r, Complex *W);

  void initShift();

  void forward2(Complex *f, Complex *F0, unsigned int r0, Complex *W);
  void forward2All(Complex *f, Complex *F0, unsigned int r0, Complex *W);
  void forward2Many(Complex *f, Complex *F, unsigned int r, Complex *W);
  void forward2ManyAll(Complex *f, Complex *F, unsigned int r, Complex *W);
  void forwardInner(Complex *f, Complex *F0, unsigned int r0, Complex *W);
  void forwardInnerAll(Complex *f, Complex *F0, unsigned int r0, Complex *W);
  void forwardInnerMany(Complex *f, Complex *F0, unsigned int r0, Complex *W);
  void forwardInnerManyAll(Complex *f, Complex *F0, unsigned int r0, Complex *W);

  void backward2(Complex *F0, Complex *f, unsigned int r0, Complex *W);
  void backward2All(Complex *F0, Complex *f, unsigned int r0, Complex *W);
  void backward2Many(Complex *F, Complex *f, unsigned int r, Complex *W);
  void backward2ManyAll(Complex *F, Complex *f, unsigned int r, Complex *W);
  void backwardInner(Complex *F0, Complex *f, unsigned int r0, Complex *W);
  void backwardInnerAll(Complex *F0, Complex *f, unsigned int r0, Complex *W);
  void backwardInnerMany(Complex *F0, Complex *f, unsigned int r0, Complex *W);
  void backwardInnerManyAll(Complex *F0, Complex *f, unsigned int r0, Complex *W);
};

class fftPadHermitian : public fftBase {
  unsigned int e;
  unsigned int B; // Work block size
  mcrfft1d *crfftm;
  mrcfft1d *rcfftm;
  mfft1d *fftp;
  mfft1d *ifftp;
public:

  class Opt : public OptBase {
  public:
    Opt() {}

    Opt(unsigned int L, unsigned int M, Application& app,
        unsigned int C, unsigned int S, bool Explicit=false, bool fixed=false) {
      scan(L,M,app,C,S,Explicit,fixed,true);
    }

    bool valid(unsigned int D, unsigned int p, unsigned int S) {
      return D == 2 && p%2 == 0 && (p == 2 || S == 1);
    }

    double time(unsigned int L, unsigned int M, unsigned int C, unsigned int S,
                unsigned int m, unsigned int q, unsigned int D,
                Application &app) {
      fftPadHermitian fft(L,M,C,S,m,q,D,app.Threads());
      return fft.meantime(app);
    }
  };

  fftPadHermitian(unsigned int L, unsigned int M, unsigned int C,
                  unsigned int S, unsigned int m, unsigned int q,
                  unsigned int D, unsigned int threads=fftw::maxthreads) :
    fftBase(L,M,C,S,m,q,D,threads,true) {
    Opt opt;
    if(q > 1 && !opt.valid(D,p,S)) invalid();
    init();
  }

  fftPadHermitian(unsigned int L, unsigned int M, Application& app,
                  unsigned int C=1, unsigned int S=0, bool Explicit=false, bool fixed=false) :
    fftBase(L,M,app,C,S,Explicit,fixed,true) {
    Opt opt=Opt(L,M,app,C,this->S,Explicit,fixed);
    m=opt.m;
    if(Explicit)
      M=m;
    q=opt.q;
    D=opt.D;
    init();
  }

  ~fftPadHermitian();

  void init();

  void forwardExplicit(Complex *f, Complex *F, unsigned int, Complex *W);
  void forwardExplicitMany(Complex *f, Complex *F, unsigned int, Complex *W);
  void forward2(Complex *f, Complex *F0, unsigned int r0, Complex *W);
  void forward2Many(Complex *f, Complex *F, unsigned int r, Complex *W);
  void forwardInner(Complex *f, Complex *F0, unsigned int r0, Complex *W);

  void backwardExplicit(Complex *F, Complex *f, unsigned int, Complex *W);
  void backwardExplicitMany(Complex *F, Complex *f, unsigned int, Complex *W);
  void backward2(Complex *F0, Complex *f, unsigned int r0, Complex *W);
  void backward2Many(Complex *F, Complex *f, unsigned int r, Complex *W);
  void backwardInner(Complex *F0, Complex *f, unsigned int r0, Complex *W);

  // Number of real outputs per residue
  unsigned int noutputs() {
    return m*(q == 1 ? 1 : p/2);
  }

  unsigned int noutputs(unsigned int r) {
    cerr << "For Hermitian transforms, use noutputs() instead of noutputs(unsigned int r)." << endl;
    exit(-1);
  }

  unsigned int workSizeW() {
    return q == 1 || inplace ? 0 : B*D;
  }

  unsigned int inputSize() {
    return S*utils::ceilquotient(L,2);
  }
};

class ForwardBackward : public Application {
protected:
  unsigned int A;
  unsigned int B;
  unsigned int R;
  Complex **f;
  Complex **F;
  Complex **h;
  Complex *W;
  FFTcall Forward,Backward;
public:
  ForwardBackward(unsigned int A, unsigned int B,
                  unsigned int threads=fftw::maxthreads) :
    Application(threads), A(A), B(B), f(NULL), F(NULL), h(NULL), W(NULL) {
  }

  virtual ~ForwardBackward() {
    clear();
  }

  void init(fftBase &fft);

  double time(fftBase &fft, unsigned int K);

  void clear();

};

typedef void multiplier(Complex **, unsigned int offset, unsigned int n,
                        unsigned int threads);

// Multiplication routine for binary convolutions and taking two inputs of size e.
void multbinary(Complex **F, unsigned int n, unsigned int offset, unsigned int threads);
void realmultbinary(Complex **F, unsigned int n, unsigned int offset, unsigned int threads);

class Convolution : public ThreadBase {
public:
  fftBase *fft;
  unsigned int L;
  unsigned int A;
  unsigned int B;
  double scale;
protected:
  unsigned int N; // max(A,B)
  unsigned int q;
  unsigned int Q;
  unsigned int R;
  unsigned int r;
  unsigned int blocksize;
  Complex **F,**Fp;
  Complex *FpB;
  Complex **V;
  Complex *W;
  Complex *H;
  Complex *W0;
  bool allocate;
  bool allocateF;
  bool allocateV;
  bool allocateW;
  unsigned int nloops;
  bool loop2;
  unsigned int inputSize;
  FFTcall Forward,Backward;
  FFTPad Pad;
public:
  Convolution(unsigned int threads=fftw::maxthreads) :
    ThreadBase(threads), W(NULL), allocate(false) {}
  // L, dimension of input data
  // M, dimension of transformed data, including padding
  // A is the number of inputs.
  // B is the number of outputs.
  Convolution(unsigned int L, unsigned int M,
              unsigned int A, unsigned int B,
              unsigned int threads=fftw::maxthreads) :
    ThreadBase(threads), A(A), B(B), W(NULL), allocate(true) {
    ForwardBackward FB(A,B,threads);
    fft=new fftPad(L,M,FB);
    init();
  }

  // fft, precomputed fftPad
  // A is the number of inputs.
  // B is the number of outputs.
  // F is an optional work array of size max(A,B)*fft->outputSize(),
  // W is an optional work array of size fft->workSizeW();
  //    call pad() if changed between calls to convolve()
  // V is an optional work array of size B*fft->workSizeV(A,B)
  //   (only needed for inplace usage)

  Convolution(fftBase &fft, unsigned int A, unsigned int B,
              unsigned int threads=fftw::maxthreads,
              Complex *F=NULL, Complex *W=NULL, Complex *V=NULL) :
    ThreadBase(threads), fft(&fft), A(A), B(B), W(W), allocate(false) {
    init(F,V);
  }

  void init(Complex *F=NULL, Complex *V=NULL) {
    if(fft->Overwrite(A,B)) {
      Forward=fft->ForwardAll;
      Backward=fft->BackwardAll;
    } else {
      Forward=fft->Forward;
      Backward=fft->Backward;
    }

    L=fft->L;
    q=fft->q;
    Q=fft->Q;
    blocksize=fft->noutputs();
    R=fft->residueBlocks();

    scale=1.0/fft->normalization();
    unsigned int outputSize=fft->outputSize();
    unsigned int workSizeW=fft->workSizeW();
    inputSize=fft->inputSize();

    N=max(A,B);
    this->F=new Complex*[N];
    if(F) {
      allocateF=false;
      for(unsigned int i=0; i < N; ++i)
        this->F[i]=F+i*outputSize;
    } else {
      allocateF=true;
      for(unsigned int i=0; i < N; ++i) // CHECK performance vs. single block
        this->F[i]=utils::ComplexAlign(outputSize);
    }

    if(q > 1) {
      allocateV=false;
      if(V) {
        this->V=new Complex*[B];
        unsigned int size=fft->workSizeV(A,B);
        for(unsigned int i=0; i < B; ++i)
          this->V[i]=V+i*size;
      } else
        this->V=NULL;

      allocateW=!this->W && !fft->inplace;
      this->W=allocateW ? utils::ComplexAlign(workSizeW) : NULL;

      Pad=fft->Pad;
      (fft->*Pad)(this->W);

      nloops=fft->nloops();
      loop2=fft->loop2(A,B);
      int extra;
      if(loop2 && !fft->overwrite) {
        r=fft->increment(0);
        Fp=new Complex*[A];
        Fp[0]=this->F[A-1];
        for(unsigned int a=1; a < A; ++a)
          Fp[a]=this->F[a-1];
        extra=1;
      } else {
        Fp=NULL;
        extra=0;
      }

      if(A > B+extra && !fft->inplace && workSizeW <= outputSize) {
        W0=this->F[B];
        Pad=&fftBase::padNone;
      } else
        W0=this->W;
    }
  }

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
      for(unsigned int i=0; i < inputSize; ++i)
        fb[i] *= scale;
    }
  }

  void convolveRaw(Complex **f, multiplier *mult, unsigned int offset=0);

  void convolve(Complex **f, multiplier *mult, unsigned int offset=0) {
    convolveRaw(f,mult,offset);
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
  ConvolutionHermitian(unsigned int L, unsigned int M,
                       unsigned int A, unsigned int B,
                       unsigned int threads=fftw::maxthreads) :
    Convolution(threads) {
    this->A=A;
    this->B=B;
    ForwardBackward FB(A,B,threads);
    fft=new fftPadHermitian(L,M,FB);
    init();
  }

  ConvolutionHermitian(fftPadHermitian &fft,
                       unsigned int A, unsigned int B,
                       unsigned int threads=fftw::maxthreads,
                       Complex *F=NULL, Complex *W=NULL, Complex *V=NULL) :
    Convolution(fft,A,B,threads,F,W,V) {}
};

inline void HermitianSymmetrizeX(unsigned int mx, unsigned int stride,
                                 unsigned int xorigin, Complex *f)
{
  Complex *F=f+xorigin*stride;
  F[0].im=0.0;
  for(unsigned int i=1; i < mx; ++i) {
    unsigned int istride=i*stride;
    *(F-istride)=conj(F[istride]);
  }

  // Zero out Nyquist modes in noncompact case
  if(xorigin == mx) {
    Complex *F=f+(xorigin-mx)*stride;
    for(unsigned int j=0; j < stride; ++j)
      F[j]=0.0;
  }
}

class Convolution2 : public ThreadBase {
public:
  fftBase *fftx;
  Convolution *convolvey;
  unsigned int Lx,Ly; // x,y dimensions of input data
  unsigned int Sy; // y stride
  unsigned int A;
  unsigned int B;
  double scale;
protected:
  ForwardBackward *FB;
  unsigned int zx;    // x dimension of Fx buffer
  unsigned int Q;
  unsigned int qx;
  unsigned int Qx; // number of residues
  unsigned int Rx; // number of residue blocks
  unsigned int r;
  Complex **F,**Fp;
  Complex **V;
  Complex *W;
  Complex *W0;
  bool allocate;
  bool allocateF;
  bool allocateV;
  bool allocateW;
  bool loop2;
  FFTcall Forward,Backward;
  unsigned int nloops;
public:
  Convolution2(unsigned int threads=fftw::maxthreads) :
    ThreadBase(threads), W(NULL), allocate(false), allocateF(false),
    allocateW(false) {}

  // Lx,Ly: x,y dimensions of input data
  // Mx,My: x,y dimensions of transformed data, including padding
  // A: number of inputs
  // B: number of outputs
  Convolution2(unsigned int Lx, unsigned int Ly,
               unsigned int Mx, unsigned int My,
               unsigned int A, unsigned int B, unsigned int Sy=0,
               unsigned int threads=fftw::maxthreads) :
    ThreadBase(threads), W(NULL), allocate(true), allocateF(false),
    allocateW(false) {
    if(Sy == 0) Sy=Ly;
    ForwardBackward FB(A,B,threads);
    fftx=new fftPad(Lx,Mx,FB,Ly,Sy);
    fftPad *ffty=new fftPad(Ly,My,FB);
    convolvey=new Convolution(*ffty,A,B);
    init();
  }

  // F is an optional work array of size max(A,B)*fftx->outputSize(),
  // W is an optional work array of size fftx->workSizeW(),
  //   call fftx->pad() if changed between calls to convolve(),
  // V is an optional work array of size B*fftx->workSizeV(A,B)
  Convolution2(fftPad &fftx, Convolution &convolvey,
               unsigned int threads=fftw::maxthreads,
               Complex *F=NULL, Complex *W=NULL, Complex *V=NULL) :
    ThreadBase(threads), fftx(&fftx), convolvey(&convolvey), W(W),
    allocate(false), allocateF(false), allocateW(false) {
    init(F,V);
  }

  void init(Complex *F=NULL, Complex *V=NULL) {
    A=convolvey->A;
    B=convolvey->B;

    if(fftx->Overwrite(A,B)) {
      Forward=fftx->ForwardAll;
      Backward=fftx->BackwardAll;
    } else {
      Forward=fftx->Forward;
      Backward=fftx->Backward;
    }

    if(!fftx->inplace && !W) {
      allocateW=true;
      W=utils::ComplexAlign(fftx->workSizeW());
    }

    unsigned int c=fftx->outputSize();

    qx=fftx->q;
    Qx=fftx->Q;
    Rx=fftx->R;
    zx=fftx->z;
    scale=1.0/(fftx->normalization()*convolvey->fft->normalization());

    unsigned int N=std::max(A,B);
    this->F=new Complex*[N];
    if(F) {
      for(unsigned int i=0; i < N; ++i)
        this->F[i]=F+i*c;
    } else {
      allocateF=true;
      for(unsigned int i=0; i < N; ++i)
        this->F[i]=utils::ComplexAlign(c);
    }

    Lx=fftx->L;
    Ly=fftx->C;
    Sy=fftx->S;

    nloops=fftx->nloops();
    loop2=fftx->loop2(A,B);
    int extra;
    if(loop2) {
      r=fftx->increment(0);
      Fp=new Complex*[A];
      Fp[0]=this->F[A-1];
      for(unsigned int a=1; a < A; ++a)
        Fp[a]=this->F[a-1];
      extra=1;
    } else
      extra=0;

    if(A > B+extra && !fftx->inplace &&
       fftx->workSizeW() <= fftx->outputSize()) {
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
    if(allocateF) {
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
    if(allocate) {
      delete convolvey;
      delete fftx;
    }
  }

  void forward(Complex **f, Complex **F, unsigned int rx,
               unsigned int offset=0) {
    for(unsigned int a=0; a < A; ++a)
      (fftx->*Forward)(f[a]+offset,F[a],rx,W);
  }

  void subconvolution(Complex **f, multiplier *mult, unsigned int C,
                      unsigned int stride, unsigned int offset=0) {
    for(unsigned int i=0; i < C; ++i)
      convolvey->convolveRaw(f,mult,offset+i*stride);
  }

  void backward(Complex **F, Complex **f, unsigned int rx,
                unsigned int offset=0) {
    for(unsigned int b=0; b < B; ++b)
      (fftx->*Backward)(F[b],f[b]+offset,rx,W);
  }

  void normalize(Complex **h, unsigned int offset=0) {
    for(unsigned int b=0; b < B; ++b) {
      Complex *hb=h[b]+offset;
      for(unsigned int i=0; i < Lx; ++i) {
        Complex *hbi=hb+Sy*i;
        for(unsigned int j=0; j < Ly; ++j)
          hbi[j] *= scale;
      }
    }
  }

// f is a pointer to A distinct data blocks each of size Lx*Ly,
// shifted by offset.
  void convolveRaw(Complex **f, multiplier *mult, unsigned int offset=0) {
    if(fftx->Overwrite(A,B)) {
      forward(f,F,0,offset);
      subconvolution(f,mult,(fftx->n-1)*zx,Sy,offset);
      subconvolution(F,mult,zx,Sy,offset);
      backward(F,f,0,offset);
    } else {
      if(loop2) {
        for(unsigned int a=0; a < A; ++a)
          (fftx->*Forward)(f[a]+offset,F[a],0,W);
        subconvolution(F,mult,fftx->D0*zx,Sy,offset);

        for(unsigned int b=0; b < B; ++b) {
          (fftx->*Forward)(f[b]+offset,Fp[b],r,W);
          (fftx->*Backward)(F[b],f[b]+offset,0,W0);
        }
        for(unsigned int a=B; a < A; ++a)
          (fftx->*Forward)(f[a]+offset,Fp[a],r,W);
        subconvolution(Fp,mult,fftx->D*zx,Sy,offset);
        for(unsigned int b=0; b < B; ++b)
          (fftx->*Backward)(Fp[b],f[b]+offset,r,W0);
      } else {
        unsigned int Offset;
        Complex **h0;
        if(nloops > 1) {
          if(!V) initV();
          h0=V;
          Offset=0;
        } else {
          Offset=offset;
          h0=f;
        }

        for(unsigned int rx=0; rx < Rx; rx += fftx->increment(rx)) {
          forward(f,F,rx,offset);
          subconvolution(F,mult,(rx == 0 ? fftx->D0 : fftx->D)*zx,Sy,offset);
          backward(F,h0,rx,Offset);
        }

        if(nloops > 1) {
          for(unsigned int b=0; b < B; ++b) {
            Complex *fb=f[b]+offset;
            Complex *hb=h0[b];
            for(unsigned int i=0; i < Lx; ++i) {
              unsigned int Syi=Sy*i;
              Complex *fbi=fb+Syi;
              Complex *hbi=hb+Syi;
              for(unsigned int j=0; j < Ly; ++j)
                fbi[j]=hbi[j];
            }
          }
        }
      }
    }
  }

  void convolve(Complex **f, multiplier *mult, unsigned int offset=0) {
    convolveRaw(f,mult,offset);
    normalize(f,offset);
  }
};

class ConvolutionHermitian2 : public Convolution2 {
public:
  ConvolutionHermitian2(unsigned int Lx, unsigned int Ly,
                        unsigned int Mx, unsigned int My,
                        unsigned int A, unsigned int B, unsigned int Sy=0,
                        unsigned int threads=fftw::maxthreads) :
    Convolution2(threads) {
    allocate=true;
    unsigned int Hy=utils::ceilquotient(Ly,2);
    if(Sy == 0) Sy=Hy;
    ForwardBackward FB(A,B,threads);
    fftx=new fftPadCentered(Lx,Mx,FB,Hy,Sy);
   fftPadHermitian *ffty=new fftPadHermitian(Ly,My,FB);
   convolvey=new ConvolutionHermitian(*ffty,A,B);
    init();
  }

  // F is an optional work array of size max(A,B)*fftx->outputSize(),
  // W is an optional work array of size fftx->workSizeW(),
  //    call fftx->pad() if changed between calls to convolve()
  // V is an optional work array of size B*fftx->workSizeV(A,B)
  ConvolutionHermitian2(fftPadCentered &fftx,
                        ConvolutionHermitian &convolvey,
                        unsigned int threads=fftw::maxthreads,
                        Complex *F=NULL, Complex *W=NULL, Complex *V=NULL) :
    Convolution2(fftx,convolvey,threads,F,W,V) {
  }
};


} //end namespace fftwpp

#endif
