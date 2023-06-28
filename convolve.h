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
#include <list>

#include "Complex.h"
#include "fftw++.h"
#include "utils.h"
#include "Array.h"

#ifndef __convolve_h__
#define __convolve_h__ 1

namespace fftwpp {

extern const double twopi;

// Constants used for initialization and testing.
const Complex I(0.0,1.0);

extern size_t L,Lx,Ly,Lz; // input data lengths
extern size_t M,Mx,My,Mz; // minimum padded lengths

extern size_t mx,my,mz; // internal FFT sizes
extern size_t Dx,Dy,Dz; // numbers of residues computed at a time
extern size_t Sx,Sy;    // strides
extern ptrdiff_t Ix,Iy,Iz;          // inplace flags

extern bool Output;
extern bool testError;
extern bool showOptTimes;
extern bool Centered;
extern bool normalized;
extern bool Tforced;
extern bool showRoutines;

size_t nextfftsize(size_t m);

class fftBase;

typedef void (fftBase::*FFTcall)(Complex *f, Complex *F, size_t r,
                                 Complex *W);
typedef void (fftBase::*FFTPad)(Complex *W);

class Indices
{
public:
  fftBase *fft;
  size_t *index;
  size_t size,maxsize;
  bool allocated;
  size_t r;

  Indices() : index(NULL), maxsize(0) {}

  void copy(Indices *indices, size_t size0) {
    size=indices ? indices->size : size0;
    if(size > maxsize) {
      if(maxsize > 0)
        delete [] index;
      index=new size_t[size];
      maxsize=size;
    }
    if(indices)
      for(size_t d=1; d < size; ++d)
        index[d]=indices->index[d];
  }

  ~Indices() {
    if(maxsize > 0)
      delete [] index;
  }
};

typedef void multiplier(Complex **F, size_t n,
                        Indices *indices, size_t threads);

// Multiplication routines for binary convolutions that take two inputs.
multiplier multNone,multbinary,realmultbinary,multcorrelation;

class Application : public ThreadBase
{
public:
  size_t A;
  size_t B;
  multiplier *mult;
  size_t m;
  size_t D;
  ptrdiff_t I;
  size_t maxthreads;

  Application(size_t A, size_t B, multiplier *mult,
              size_t threads=fftw::maxthreads,
              size_t m=0, size_t D=0, ptrdiff_t I=-1) :
    ThreadBase(threads), A(A), B(B), mult(mult), m(m), D(D), I(I)
  {
    maxthreads=threads;
  }

  Application(size_t A, size_t B, multiplier *mult, Application &parent,
              size_t m=0, size_t D=0, ptrdiff_t I=-1) :
    ThreadBase(1), A(A), B(B), mult(mult), m(m), D(D), I(I)
  {
    maxthreads=parent.maxthreads;
  }
};

class fftBase : public ThreadBase {
public:
  size_t L; // number of unpadded Complex data values
  size_t M; // minimum number of padded Complex data values
  size_t C; // number of FFTs to compute in parallel
  size_t S; // stride between successive elements
  size_t m;
  size_t p;
  size_t q;
  size_t n;  // number of residues
  size_t R;  // number of residue blocks
  size_t dr; // r increment
  size_t D;  // number of residues stored in F at a time
  size_t D0; // remainder
  size_t Cm,Sm;
  size_t l;  // block size of a single FFT
  size_t b;  // total block size, including stride
  bool inplace;
  Application app;
  bool centered;
  bool overwrite;
  FFTcall Forward,Backward;
  FFTPad Pad;
protected:
  Complex *Zetaqp;
  Complex *Zetaqp0;
  Complex *Zetaqm;
  Complex *ZetaqmS;
  Complex *Zetaqm0;
  Complex *ZetaqmS0;
public:

  void checkParameters();

  void common();

  void initZetaqm(size_t q, size_t m);

  class OptBase {
  public:
    size_t counter;
    size_t m,q,D;
    bool inplace;
    bool mForced;
    bool DForced;
    typedef std::list<size_t> mList;
    mList mlist;
    double threshold;
    double T;

    virtual double time(size_t L, size_t M, size_t C,
                        size_t S, size_t m, size_t q,
                        size_t D, bool inplace, Application &app)=0;

    virtual bool valid(size_t m, size_t p, size_t q, size_t n, size_t D, size_t S)=0;

    virtual size_t maxD(size_t n)=0;

    // Called by the optimizer to record the time to complete an application
    // for a given value of m.
    void check(size_t L, size_t M,
               size_t C, size_t S, size_t m,
               size_t p, size_t q, size_t n, size_t D,
               bool inplace,  Application& app, bool useTimer);

    // Determine the optimal m value for padding L data values to
    // size >= M for an application app.
    // If Explicit=true, we only consider m >= M.
    // centered must be true for all centered and Hermitian routines.
    void scan(size_t L, size_t M, Application& app,
              size_t C, size_t S, bool Explicit=false,
              bool centered=false);

    // A function called by opt to iterate over m and D values
    // and call check.
    // Inner is true if p > 2.
    // If inner is false, ubound is the maximum number maximum number of
    // m values that are checked.
    // If inner is true, ubound is the maximum size of m values that are
    // checked.
    void optloop(size_t& m0, size_t L, size_t M,
                 Application& app, size_t C, size_t S,
                 bool centered, size_t ubound, bool useTimer,
                 bool Explicit, bool inner=false);

    // The default optimizer routine. Used by scan to iterate and check
    // different m values for a given geometry and application.
    // 'minInner' is the minimum size of FFT we consider for the inner routines.
    // 'itmax' is the maximum number of iterations done by optloop
    // (when p <= 2).
    void opt(size_t L, size_t M, Application& app,
             size_t C, size_t S, size_t minInner,
             size_t itmax, bool Explicit, bool centered,
             bool useTimer=true);
  };

  void invalid () {
    std::cerr << "Invalid parameters: " << std::endl
              << "m=" << m << " p=" << p << " q=" << q
              << " n=" << n << " " << " D=" << D << " S=" << S
              << std::endl;
    exit(-1);
  }

  fftBase(size_t L, size_t M, size_t C, size_t S,
          Application &app, bool centered=false) :
    ThreadBase(app.threads), L(L), M(M), C(C),  S(S == 0 ? C : S),
    app(app), centered(centered) {checkParameters();}

  fftBase(size_t L, size_t M, size_t C, size_t S,
          size_t m, size_t q, size_t D, bool inplace,
          Application &app, bool centered=false) :
    ThreadBase(app.threads), L(L), M(M), C(C),  S(S == 0 ? C : S), m(m),
    p(utils::ceilquotient(L,m)), q(q), D(D), inplace(inplace),
    app(app), centered(centered) {
    checkParameters();
    this->app.D=D;
  }

  fftBase(size_t L, size_t M, Application& app,
          size_t C=1, size_t S=0, bool Explicit=false,
          bool centered=false) :
    ThreadBase(app.threads), L(L), M(M), C(C), S(S == 0 ? C : S),
    app(app), centered(centered) {checkParameters();}

  virtual ~fftBase();

  void padNone(Complex *W) {}

  virtual void padSingle(Complex *W) {}
  virtual void padMany(Complex *W) {}

  void pad(Complex *W) {
    if(W)
      (this->*Pad)(W);
  }

  void forward(Complex *f, Complex *F, size_t r=0, Complex *W=NULL) {
    (this->*Forward)(f,F,r,W);
  }

  void backward(Complex *f, Complex *F, size_t r=0, Complex *W=NULL) {
    (this->*Backward)(f,F,r,W);
  }

  virtual void forwardExplicit(Complex *f, Complex *F, size_t,
                               Complex *W) {};
  virtual void forwardExplicitMany(Complex *f, Complex *F, size_t,
                                   Complex *W) {};
  virtual void forwardExplicitFast(Complex *f, Complex *F, size_t,
                                   Complex *W) {}
  virtual void forwardExplicitManyFast(Complex *f, Complex *F, size_t,
                                       Complex *W) {}
  virtual void forwardExplicitSlow(Complex *f, Complex *F, size_t r,
                                   Complex *W) {}
  virtual void forwardExplicitManySlow(Complex *f, Complex *F, size_t r,
                                       Complex *W) {}

  virtual void backwardExplicit(Complex *F, Complex *f, size_t,
                                Complex *W) {};
  virtual void backwardExplicitMany(Complex *F, Complex *f, size_t,
                                    Complex *W) {};
  virtual void backwardExplicitFast(Complex *F, Complex *f, size_t,
                                    Complex *W) {}
  virtual void backwardExplicitManyFast(Complex *F, Complex *f, size_t,
                                        Complex *W) {}
  virtual void backwardExplicitSlow(Complex *F, Complex *f, size_t r,
                                    Complex *W) {}
  virtual void backwardExplicitManySlow(Complex *F, Complex *f, size_t r,
                                        Complex *W) {}

  // Return transformed index for residue r at position I
  size_t index(size_t r, size_t i) {
    if(q == 1) return i;
    size_t s=i%m;
    size_t u;
    size_t P;
    if(D > 1 && ((centered && p % 2 == 0) || p <= 2)) {
      P=utils::ceilquotient(p,2);
      u=(i/m)%P;
      size_t offset=r == 0 && i >= P*m && D0 % 2 == 1 ? 1 : 0;
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
    return q*s+n*u+r;
  }

  size_t Index(size_t r, size_t I) {
    size_t i=I/S;
    return S*(index(r,i)-i)+I;
  }

  virtual void forward1(Complex *f, Complex *F0, size_t r0, Complex *W)
  {}
  virtual void forward1All(Complex *f, Complex *F0, size_t r0,
                           Complex *W) {}
  virtual void forward1Many(Complex *f, Complex *F, size_t r,
                            Complex *W) {}
  virtual void forward1ManyAll(Complex *f, Complex *F, size_t r,
                               Complex *W) {}
  virtual void forward2(Complex *f, Complex *F0, size_t r0, Complex *W)
  {}
  virtual void forward2All(Complex *f, Complex *F0, size_t r0,
                           Complex *W) {}
  virtual void forward2Many(Complex *f, Complex *F, size_t r,
                            Complex *W) {}
  virtual void forward2ManyAll(Complex *f, Complex *F, size_t r,
                               Complex *W) {}
  virtual void forwardInner(Complex *f, Complex *F0, size_t r0,
                            Complex *W) {}
  virtual void forwardInnerAll(Complex *f, Complex *F0, size_t r0,
                               Complex *W) {}
  virtual void forwardInnerMany(Complex *f, Complex *F, size_t r,
                                Complex *W) {}
  virtual void forwardInnerManyAll(Complex *f, Complex *F, size_t r,
                                   Complex *W) {}

  virtual void backward1(Complex *F0, Complex *f, size_t r0, Complex *W)
  {}
  virtual void backward1All(Complex *F0, Complex *f, size_t r0,
                            Complex *W) {}
  virtual void backward1Many(Complex *F, Complex *f, size_t r,
                             Complex *W) {}
  virtual void backward1ManyAll(Complex *F, Complex *f, size_t r,
                                Complex *W) {}
  virtual void backward2(Complex *F0, Complex *f, size_t r0,
                         Complex *W) {}
  virtual void backward2All(Complex *F0, Complex *f, size_t r0,
                            Complex *W) {}
  virtual void backward2Many(Complex *F, Complex *f, size_t r,
                             Complex *W) {}
  virtual void backward2ManyAll(Complex *F, Complex *f, size_t r,
                                Complex *W) {}
  virtual void backwardInner(Complex *F0, Complex *f, size_t r0,
                             Complex *W) {}
  virtual void backwardInnerAll(Complex *F0, Complex *f, size_t r0,
                                Complex *W) {}
  virtual void backwardInnerMany(Complex *F, Complex *f, size_t r,
                                 Complex *W) {}
  virtual void backwardInnerManyAll(Complex *F, Complex *f, size_t r,
                                    Complex *W) {}

  size_t normalization() {
    return M;
  }

  virtual size_t paddedSize() {
    return m*q;
  }

  virtual size_t outputSize() {
    return b*D;
  }

  virtual size_t qReduced(size_t p, size_t q) {
    return p == 2 ? q : q/p;
  }

  virtual bool conjugates() {
    return D > 1 && (p <= 2 || (centered && p % 2 == 0));
  }

  virtual size_t residueBlocks() {
    return conjugates() ? utils::ceilquotient(n,2) : n;
  }

  size_t Dr() {return conjugates() ? D/2 : D;}

  virtual size_t increment(size_t r) {
    return r > 0 ? dr : (conjugates() ? utils::ceilquotient(D0,2) : D0);
  }

  size_t nloops() {
    size_t count=0;
    for(size_t r=0; r < R; r += increment(r))
      ++count;
    return count;
  }

  bool loop2() {
    return nloops() == 2 && app.A > app.B && !Overwrite();
  }

  virtual size_t inputSize() {
    return S*L;
  }

  // Number of complex outputs per residue per copy
  virtual size_t noutputs() {
    return l;
  }

  // Number of complex outputs per residue per copy
  virtual size_t blocksize(size_t) {
    return noutputs();
  }

  // Number of outputs per iteration per copy
  virtual size_t noutputs(size_t r) {
    return l*(r == 0 ? D0 : D);
  }

  // Number of complex outputs per iteration
  virtual size_t complexOutputs(size_t r) {
    return S*noutputs(r);
  }

  size_t workSizeV() {
    return nloops() == 1 || loop2() ? 0 : inputSize();
  }

  virtual size_t workSizeW() {
    return inplace ? 0 : outputSize();
  }

  size_t repad() {
    return !inplace && L < m;
  }

  bool Overwrite() {
    return overwrite && app.A >= app.B;
  }

  virtual double time()=0;

  double report() {
    double median=time()*1.0e-9;
    std::cout << "median=" << median << std::endl;
    return median;
  }
};

typedef double timer(fftBase *fft, double& threshold);
timer timePad;

class fftPad : public fftBase {
protected:
  fft1d *fftm1;
  fft1d *ifftm1;
  mfft1d *fftm,*fftm0;
  mfft1d *ifftm,*ifftm0;
  mfft1d *fftp;
  mfft1d *ifftp;
public:


  static bool valid(size_t m, size_t p, size_t q , size_t n, size_t D, size_t S) {
    if(q == 1) return D == 1;
    return D == 1 || (S == 1 && ((D < n && D % 2 == 0) || D == n));
  }

  class Opt : public OptBase {
  public:
    Opt() {}

    Opt(size_t L, size_t M, Application& app,
        size_t C, size_t S, bool Explicit=false) {
      scan(L,M,app,C,S,Explicit);
    }

    bool valid(size_t m, size_t p, size_t q, size_t n, size_t D, size_t S) {
      return fftPad::valid(m,p,q,n,D,S);
    }

    size_t maxD(size_t n) {
      return n;
    }

    double time(size_t L, size_t M, size_t C, size_t S,
                size_t m, size_t q,size_t D, bool inplace, Application &app) {
      fftPad fft(L,M,C,S,m,q,D,inplace,app);
      double threshold=DBL_MAX;
      return timePad(&fft,threshold);
    }
  };

  fftPad(size_t L, size_t M, size_t C, size_t S,
         Application &app, bool centered) :
    fftBase(L,M,C,S,app,centered) {}

  fftPad(size_t L, size_t M, size_t C, size_t S,
         size_t m, size_t q, size_t D, bool inplace,
         Application &app, bool centered) :
    fftBase(L,M,C,S,m,q,D,inplace,app) {}

  // Compute an fft padded to N=m*q >= M >= L
  fftPad(size_t L, size_t M, size_t C, size_t S,
         size_t m, size_t q, size_t D, bool inplace,
         Application &app) :
    fftBase(L,M,C,S,m,q,D,inplace,app) {
    Opt opt;
    p=utils::ceilquotient(L,m);
    n=qReduced(p,q);
    if(q > 1 && !opt.valid(m,p,q,n,D,this->S)) invalid();
    init();
  }

  // Normal entry point.
  // Compute C ffts of length L with stride S >= C and distance 1
  // padded to at least M
  fftPad(size_t L, size_t M, Application& app,
         size_t C=1, size_t S=0, bool Explicit=false, bool centered=false) :
    fftBase(L,M,app,C,S,Explicit,centered) {
    Opt opt=Opt(L,M,app,C,this->S,Explicit);
    m=opt.m;
    if(Explicit)
      M=m;
    q=opt.q;
    D=opt.D;
    inplace=opt.inplace;
//    n=opt.n;
    p=utils::ceilquotient(L,m);
    n=qReduced(p,q);
    init();
  }

  ~fftPad();

  void init();

  double time() {
    double threshold=DBL_MAX;
    return timePad(this,threshold);
  }

  // Explicitly pad to m.
  void padSingle(Complex *W);

  // Explicitly pad C FFTs to m.
  void padMany(Complex *W);

  void forwardExplicit(Complex *f, Complex *F, size_t, Complex *W);
  void forwardExplicitMany(Complex *f, Complex *F, size_t, Complex *W);

  void backwardExplicit(Complex *F, Complex *f, size_t, Complex *W);
  void backwardExplicitMany(Complex *F, Complex *f, size_t, Complex *W);

  // p=1 && C=1
  void forward1(Complex *f, Complex *F0, size_t r0, Complex *W);
  void forward1All(Complex *f, Complex *F0, size_t r0, Complex *W);
  void forward1Many(Complex *f, Complex *F, size_t r, Complex *W);
  void forward1ManyAll(Complex *f, Complex *F, size_t r, Complex *W);
  void forward2(Complex *f, Complex *F0, size_t r0, Complex *W);
  void forward2All(Complex *f, Complex *F0, size_t r0, Complex *W);
  void forward2Many(Complex *f, Complex *F, size_t r, Complex *W);
  void forward2ManyAll(Complex *f, Complex *F, size_t r, Complex *W);
  void forwardInner(Complex *f, Complex *F0, size_t r0, Complex *W);
  void forwardInnerMany(Complex *f, Complex *F, size_t r, Complex *W);

// Compute an inverse fft of length N=m*q unpadded back
// to size m*p >= L.
// input and output arrays must be distinct
// Input F destroyed
  void backward1(Complex *F0, Complex *f, size_t r0, Complex *W);
  void backward1All(Complex *F0, Complex *f, size_t r0, Complex *W);
  void backward1Many(Complex *F, Complex *f, size_t r, Complex *W);
  void backward1ManyAll(Complex *F, Complex *f, size_t r, Complex *W);
  void backward2(Complex *F0, Complex *f, size_t r0, Complex *W);
  void backward2All(Complex *F0, Complex *f, size_t r0, Complex *W);
  void backward2Many(Complex *F, Complex *f, size_t r, Complex *W);
  void backward2ManyAll(Complex *F, Complex *f, size_t r, Complex *W);
  void backwardInner(Complex *F0, Complex *f, size_t r0, Complex *W);
  void backwardInnerMany(Complex *F, Complex *f, size_t r, Complex *W);
};

class fftPadCentered : public fftPad {
  Complex *ZetaShift;
public:

  size_t qReduced(size_t p, size_t q) {
    size_t p2=p/2;
    return p == 2*p2 ? q/p2 : q/p;
  }

  class Opt : public OptBase {
  public:
    Opt() {}

    Opt(size_t L, size_t M, Application& app,
        size_t C, size_t S, bool Explicit=false) {
      scan(L,M,app,C,S,Explicit,true);
    }

    bool valid(size_t m, size_t p, size_t q , size_t n, size_t D, size_t S) {
      return (q == 1 || p%2 == 0) && fftPad::valid(m,p,q,n,D,S);
    }

    size_t maxD(size_t n) {
      return n;
    }

    double time(size_t L, size_t M, size_t C, size_t S,
                size_t m, size_t q, size_t D, bool inplace, Application &app) {
      fftPadCentered fft(L,M,C,S,m,q,D,inplace,app);
      double threshold=DBL_MAX;
      return timePad(&fft,threshold);
    }
  };

  // Compute an fft padded to N=m*q >= M >= L
  fftPadCentered(size_t L, size_t M, size_t C,
                 size_t S, size_t m, size_t q,
                 size_t D, bool inplace, Application &app) :
    fftPad(L,M,C,S,m,q,D,inplace,app,true) {
    Opt opt;
    p=utils::ceilquotient(L,m);
    n=qReduced(p,q);
    if(q > 1 && !opt.valid(m,p,q,n,D,this->S)) invalid();
    fftPad::init();
    init();
  }

  // Normal entry point.
  // Compute C ffts of length L and distance 1 padded to at least M
  fftPadCentered(size_t L, size_t M, Application& app,
                 size_t C=1, size_t S=0, bool Explicit=false) :
    fftPad(L,M,C,S,app,true) {
    Opt opt=Opt(L,M,app,C,this->S,Explicit);
    m=opt.m;
    if(Explicit)
      M=m;
    q=opt.q;
    D=opt.D;
    inplace=opt.inplace;
    p=utils::ceilquotient(L,m);
    n=qReduced(p,q);
    fftPad::init();
    init();
  }

  ~fftPadCentered() {
    if(ZetaShift)
      utils::deleteAlign(ZetaShift);
  }

  bool conjugates() {return D > 1 && (p == 1 || p % 2 == 0);}

  void init();

  double time() {
    double threshold=DBL_MAX;
    return timePad(this,threshold);
  }

  void forwardExplicitFast(Complex *f, Complex *F, size_t r, Complex *W);
  void forwardExplicitManyFast(Complex *f, Complex *F, size_t r,
                               Complex *W);
  void forwardExplicitSlow(Complex *f, Complex *F, size_t r, Complex *W);
  void forwardExplicitManySlow(Complex *f, Complex *F, size_t r,
                               Complex *W);

  void backwardExplicitFast(Complex *F, Complex *f, size_t r,
                            Complex *W);
  void backwardExplicitManyFast(Complex *F, Complex *f, size_t r,
                                Complex *W);
  void backwardExplicitSlow(Complex *F, Complex *f, size_t r,
                            Complex *W);
  void backwardExplicitManySlow(Complex *F, Complex *f, size_t r,
                                Complex *W);

  void initShift();

  void forward2(Complex *f, Complex *F0, size_t r0, Complex *W);
  void forward2All(Complex *f, Complex *F0, size_t r0, Complex *W);
  void forward2Many(Complex *f, Complex *F, size_t r, Complex *W);
  void forward2ManyAll(Complex *f, Complex *F, size_t r, Complex *W);
  void forwardInner(Complex *f, Complex *F0, size_t r0, Complex *W);
  void forwardInnerAll(Complex *f, Complex *F0, size_t r0, Complex *W);
  void forwardInnerMany(Complex *f, Complex *F0, size_t r0, Complex *W);
  void forwardInnerManyAll(Complex *f, Complex *F0, size_t r0,
                           Complex *W);

  void backward2(Complex *F0, Complex *f, size_t r0, Complex *W);
  void backward2All(Complex *F0, Complex *f, size_t r0, Complex *W);
  void backward2Many(Complex *F, Complex *f, size_t r, Complex *W);
  void backward2ManyAll(Complex *F, Complex *f, size_t r, Complex *W);
  void backwardInner(Complex *F0, Complex *f, size_t r0, Complex *W);
  void backwardInnerAll(Complex *F0, Complex *f, size_t r0, Complex *W);
  void backwardInnerMany(Complex *F0, Complex *f, size_t r0, Complex *W);
  void backwardInnerManyAll(Complex *F0, Complex *f, size_t r0,
                            Complex *W);
};

class fftPadHermitian : public fftBase {
  size_t e;
  size_t B; // Work block size
  crfft1d *crfftm1;
  rcfft1d *rcfftm1;
  mcrfft1d *crfftm;
  mrcfft1d *rcfftm;
  mfft1d *fftp;
  mfft1d *ifftp;
public:

  size_t qReduced(size_t p, size_t q) {
    if(p == 1) return 1;
    size_t p2=p/2;
    return q/p2;
  }

  class Opt : public OptBase {
  public:
    Opt() {}

    Opt(size_t L, size_t M, Application& app,
        size_t C, size_t, bool Explicit=false) {
      scan(L,M,app,C,C,Explicit,true);
    }

    bool valid(size_t m, size_t p, size_t q , size_t n, size_t D, size_t C) {
      return (D == 1 && q == 1) || (D == 2 && p%2 == 0 && (p == 2 || C == 1));
    }

    size_t maxD(size_t n) {
      return n;
    }

    double time(size_t L, size_t M, size_t C, size_t,
                size_t m, size_t q, size_t D, bool inplace, Application &app) {
      fftPadHermitian fft(L,M,C,m,q,D,inplace,app);
      return timePad(&fft,threshold);
    }
  };

  fftPadHermitian(size_t L, size_t M, size_t C,
                  size_t m, size_t q, size_t D,
                  bool inplace, Application &app) :
    fftBase(L,M,C,C,m,q,D,inplace,app,true) {
    Opt opt;
    p=utils::ceilquotient(L,m);
    n=qReduced(p,q);
    if(q > 1 && !opt.valid(m,p,q,n,D,C)) invalid();
    init();
  }

  fftPadHermitian(size_t L, size_t M, Application& app,
                  size_t C=1, bool Explicit=false) :
    fftBase(L,M,app,C,C,Explicit,true) {
    Opt opt=Opt(L,M,app,C,C,Explicit);
    m=opt.m;
    if(Explicit)
      M=m;
    q=opt.q;
    D=opt.D;
    inplace=opt.inplace;
    p=utils::ceilquotient(L,m);
    n=qReduced(p,q);
    init();
  }

  ~fftPadHermitian();

  void init();

  double time() {
    double threshold=DBL_MAX;
    return timePad(this,threshold);
  }

  void forwardExplicit(Complex *f, Complex *F, size_t, Complex *W);
  void forwardExplicitMany(Complex *f, Complex *F, size_t, Complex *W);
  void forward2(Complex *f, Complex *F0, size_t r0, Complex *W);
  void forward2Many(Complex *f, Complex *F, size_t r, Complex *W);
  void forwardInner(Complex *f, Complex *F0, size_t r0, Complex *W);

  void backwardExplicit(Complex *F, Complex *f, size_t, Complex *W);
  void backwardExplicitMany(Complex *F, Complex *f, size_t, Complex *W);
  void backward2(Complex *F0, Complex *f, size_t r0, Complex *W);
  void backward2Many(Complex *F, Complex *f, size_t r, Complex *W);
  void backwardInner(Complex *F0, Complex *f, size_t r0, Complex *W);

  // Number of real outputs per residue per copy
  size_t noutputs() {
    return m*(q == 1 ? 1 : p/2);
  }

  size_t noutputs(size_t r) {
    std::cerr << "For Hermitian transforms, use noutputs() instead of noutputs(size_t r)." << std::endl;
    exit(-1);
  }

  // Number of complex outputs per iteration
  size_t complexOutputs(size_t r) {
    return b*(r == 0 ? D0 : D);
  }

  size_t workSizeW() {
    return inplace ? 0 : B*D;
  }

  size_t inputSize() {
    return C*utils::ceilquotient(L,2);
  }
};

class fftPadReal : public fftBase {
  size_t e;
  rcfft1d *rcfftm1;
  crfft1d *crfftm1;
  mrcfft1d *rcfftm2;// TODO: deallocate in destructor!
  mcrfft1d *crfftm2;
  //crfft1d *crfftm1;
  //fft1d *fftm1;
  //fft1d *ifftm1;
  mfft1d *fftm,*fftm0,*fftmp2m1,*fftmp2;
  mfft1d *ifftm,*ifftm0,*ifftmp2m1,*ifftmp2;
  fft1d *ffte;
  fft1d *iffte;
  mrcfft1d *rcfftp; // TODO: deallocate in destructor!
  mcrfft1d *crfftp;
  mfft1d *fftp;
  mfft1d *ifftp;
  mfft1d *fftp2;
  mfft1d *ifftp2;
public:

  class Opt : public OptBase {
  public:
    Opt() {}

    Opt(size_t L, size_t M, Application& app,
        size_t C, size_t S, bool Explicit=false) {
      scan(L,M,app,C,S,Explicit);
    }

    bool valid(size_t m, size_t p, size_t q, size_t n, size_t D, size_t S) {
      return (n%2 == 1 || (p%2 == 0 || p <= 2)) && (q%2 == 1 || m%2 == 0) &&
        (D == 1 || (S == 1 && ((D < (n-1)/2 && D % 2 == 0) || D == (n-1)/2)));
    }

    size_t maxD(size_t n) {
      return n > 2 ? (n-1)/2 : 1;
    }

    double time(size_t L, size_t M, size_t C, size_t S,
                size_t m, size_t q,size_t D, bool inplace, Application &app) {
      fftPadReal fft(L,M,C,S,m,q,D,inplace,app);
      double threshold=DBL_MAX;
      return timePad(&fft,threshold);
    }
  };

  fftPadReal(size_t L, size_t M, size_t C, size_t S, Application &app) :
    fftBase(L,M,C,S,app) {}

  // Compute an fft padded to N=m*q >= M >= L
  fftPadReal(size_t L, size_t M, size_t C, size_t S,
         size_t m, size_t q, size_t D, bool inplace,
         Application &app) :
    fftBase(L,M,C,S,m,q,D,inplace,app) {
    Opt opt;
    p=utils::ceilquotient(L,m);
    n=qReduced(p,q);
    if(q > 1 && !opt.valid(m,p,q,n,D,this->S)) invalid();
    init();
  }

  // Normal entry point.
  // Compute C ffts of length L with stride S >= C and distance 1
  // padded to at least M
  fftPadReal(size_t L, size_t M, Application& app,
         size_t C=1, size_t S=0, bool Explicit=false) :
    fftBase(L,M,app,C,S,Explicit) {
    Opt opt=Opt(L,M,app,C,this->S,Explicit);
    m=opt.m;
    if(Explicit)
      M=m;
    q=opt.q;
    D=opt.D;
    inplace=opt.inplace;
    p=utils::ceilquotient(L,m);
    n=qReduced(p,q);
    init();
  }

  ~fftPadReal();

  void init();

  double time() {
    double threshold=DBL_MAX;
    return timePad(this,threshold);
  }

  // Explicitly pad to m.
  void padSingle(Complex *W);

  /*
  // Explicitly pad C FFTs to m.
  void padMany(Complex *W) {}
  */

  void forwardExplicit(Complex *f, Complex *F, size_t, Complex *W);
  void backwardExplicit(Complex *F, Complex *f, size_t, Complex *W);

  // p=1 && C=1
  void forward1(Complex *f, Complex *F, size_t r0, Complex *W);
  void backward1(Complex *F, Complex *f, size_t r0, Complex *W);

  // p=2 && C=1
  void forward2(Complex *f, Complex *F, size_t r0, Complex *W);
  void backward2(Complex *F, Complex *f, size_t r0, Complex *W);

  void forwardInner(Complex *f, Complex *F, size_t r0, Complex *W);
  void backwardInner(Complex *F, Complex *f, size_t r0, Complex *W);

  size_t inputSize() {
    return S*utils::ceilquotient(L,2);
  }

  virtual size_t outputSize() {
    return q == 2 ? e : b*D;
  }

  bool conjugates() {
    return false;
  }

  size_t residueBlocks() {
    return utils::ceilquotient(n+1,2);
  }

  size_t increment(size_t r) {
    return r > 1 ? D : r == 1 ? D0 : 1;
  }

  size_t blocksize(size_t r) {
    if(r == 0) return p > 2 ? (p/2+1)*m : e;
    if(2*r == q) return e-1;
    return l;
  }

  // Number of outputs per iteration per copy
  size_t noutputs(size_t r) {
   return blocksize(r)*((r == 0 || 2*r == n) ? 1 : r == 1 ? D0 : D);
  }
};

class Convolution : public ThreadBase {
public:
  fftBase *fft;
  size_t L;
  size_t A;
  size_t B;
  multiplier *mult;
  double scale;
protected:
  size_t q;
  size_t n;
  size_t R;
  size_t r;
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
  size_t nloops;
  bool loop2;
  FFTcall Forward,Backward;
  FFTPad Pad;
public:
  Indices indices;

  // fft: precomputed fftPad
  // F: optional array of max(A,B) work arrays of size fft->outputSize()
  // W: optional work array of size fft->workSizeW();
  //    call pad() if changed between calls to convolve()
  // V: optional work array of size B*fft->workSizeV()
  //   (only needed for inplace usage)
  Convolution(fftBase *fft, Complex **F=NULL, Complex *W=NULL, Complex *V=NULL)
    : ThreadBase(fft->Threads()), fft(fft), A(fft->app.A), B(fft->app.B),
      mult(fft->app.mult), W(W), allocate(false) {
    init(F,V);
  }

  double normalization() {
    return fft->normalization();
  }

  void init(Complex **F=NULL, Complex *V=NULL) {
    for(unsigned int t=0; t < threads; ++t)
      indices.copy(NULL,0);
    indices.fft=fft;

    L=fft->L;
    q=fft->q;
    n=fft->n;

    Forward=fft->Forward;
    Backward=fft->Backward;

    R=fft->residueBlocks();

    scale=1.0/normalization();
    size_t outputSize=fft->outputSize();
    size_t workSizeW=fft->workSizeW();

    size_t N=std::max(A,B);
    allocateF=!F;
    this->F=allocateF ? utils::ComplexAlign(N,outputSize) : F;

    allocateW=!W && !fft->inplace;
    W=allocateW ? utils::ComplexAlign(workSizeW) : NULL;

    if(q > 1) {
      allocateV=false;
      if(V) {
        this->V=new Complex*[B];
        size_t size=fft->workSizeV();
        for(size_t i=0; i < B; ++i)
          this->V[i]=V+i*size;
      } else
        this->V=NULL;

      Pad=fft->Pad;
      (fft->*Pad)(W);

      nloops=fft->nloops();
      loop2=fft->loop2();
      size_t extra;
      if(loop2) {
        r=fft->increment(0);
        Fp=new Complex*[A];
        size_t C=A-B;

        for(size_t c=0; c < C; c++)
          Fp[c]=this->F[B+c];

        for(size_t b=0; b < B; b += C)
          for(size_t c=b; c < B; c++)
            Fp[C+c]=this->F[c];

        extra=1;
      } else
        extra=0;

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
    size_t size=fft->workSizeV();
    for(size_t i=0; i < B; ++i)
      V[i]=utils::ComplexAlign(size);
  }

  size_t increment(size_t r) {
    return fft->increment(r);
  }

  ~Convolution();

  void normalize(Complex **f, size_t offset=0) {
    size_t inputSize=fft->inputSize();
    for(size_t b=0; b < B; ++b) {
      Complex *fb=f[b]+offset;
      for(size_t i=0; i < inputSize; ++i)
        fb[i] *= scale;
    }
  }

  void forward(Complex **f, Complex **F, size_t r,
               size_t start, size_t stop) {
    for(size_t a=start; a < stop; ++a)
      (fft->*Forward)(f[a],F[a],r,W);
  }

  void operate(Complex **F, size_t r, Indices *indices) {
    size_t incr=fft->b;
    size_t stop=fft->complexOutputs(r);
    indices->r=r;
    size_t blocksize=fft->blocksize(r);
    for(size_t d=0; d < stop; d += incr) {
      Complex *G[A];
      for(size_t a=0; a < A; ++a)
        G[a]=F[a]+d;
      (*mult)(G,blocksize,indices,threads);
      ++indices->r;
    }
  }

  void backward(Complex **F, Complex **f, size_t r,
                size_t start, size_t stop,
                Complex *W0=NULL) {
    for(size_t b=start; b < stop; ++b)
      (fft->*Backward)(F[b],f[b],r,W0);
  }

  void backwardPad(Complex **F, Complex **f, size_t r,
                   size_t start, size_t stop,
                   Complex *W0=NULL) {
    backward(F,f,r,start,stop,W0);
    if(W && W == W0) (fft->*Pad)(W0);
  }

  void convolveRaw(Complex **f);
  void convolveRaw(Complex **f, Indices *indices);

  void convolveRaw(Complex **f, size_t offset);
  void convolveRaw(Complex **f, size_t offset, Indices *indices);

  void convolve(Complex **f) {
    convolveRaw(f);
    normalize(f);
  }
  void convolve(Complex **f, size_t offset) {
    convolveRaw(f,offset);
    normalize(f,offset);
  }

};

// Enforce 2D Hermiticity using specified (x >= 0,y=0) data.
inline void HermitianSymmetrizeX(size_t Hx, size_t Hy,
                                 size_t x0, Complex *f,
                                 size_t Sx, size_t threads=fftw::maxthreads)
{
  Complex *F=f+x0*Sx;
  size_t stop=Hx*Sx;
  PARALLELIF(
    Hx > threshold,
  for(size_t i=Sx; i < stop; i += Sx)
    *(F-i)=conj(F[i]);
    );

  F[0].im=0.0;

  // Zero out Nyquist modes
  if(x0 == Hx) {
    PARALLELIF(
      Hy > threshold,
    for(size_t j=0; j < Hy; ++j)
      f[j]=0.0;
      );
  }
}

inline void HermitianSymmetrizeX(size_t Hx, size_t Hy,
                                 size_t x0, Complex *f)
{
  HermitianSymmetrizeX(Hx,Hy,x0,f,Hy);
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
  PARALLELIF(
    Hx > threshold,
  for(size_t i=Sx; i < stop; i += Sx)
    *(F-i)=conj(F[i]);
    );

  F[0].im=0.0;

  PARALLELIF(
    2*Hx*Hy > threshold,
    for(ptrdiff_t i=(-Hx+1)*Sx; i < (ptrdiff_t) stop; i += Sx) {
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
                                  Complex *f)
{
  size_t Ly=y0+Hy;
  HermitianSymmetrizeXY(Hx,Hy,Hz,x0,y0,f,Ly*Hz,Hz);
}

class Convolution2 : public ThreadBase {
public:
  fftBase *fftx,*ffty;
  Convolution **convolvey;
  size_t Lx,Ly; // x,y dimensions of input data
  size_t Sx; // x stride
  size_t A;
  size_t B;
  multiplier *mult;
  double scale;
protected:
  size_t lx; // x dimension of Fx buffer
  size_t qx;
  size_t nx; // number of residues
  size_t Rx; // number of residue blocks
  size_t r;
  Complex **F,**Fp;
  Complex **V;
  Complex *W;
  Complex *W0;
  bool allocateF;
  bool allocateV;
  bool allocateW;
  bool loop2;
  FFTcall Forward,Backward;
  FFTPad Pad;
  size_t nloops;
public:
  Indices indices;

  // F: optional array of max(A,B) work arrays of size fftx->outputSize()
  // W: optional work array of size fftx->workSizeW();
  //    call fftx->pad() if W changed between calls to convolve()
  // V: optional work array of size B*fftx->workSizeV()
  Convolution2(fftBase *fftx, fftBase *ffty,
               Complex **F=NULL, Complex *W=NULL, Complex *V=NULL) :
    ThreadBase(fftx->Threads()), fftx(fftx), ffty(ffty),
    A(fftx->app.A), B(fftx->app.B), mult(fftx->app.mult),
    W(W), allocateF(false), allocateW(false) {

    if(fftx->l < threads) {
      ffty->Threads(threads);
      threads=1;
    }

    this->convolvey=new Convolution*[threads];
    for(size_t t=0; t < threads; ++t)
      this->convolvey[t]=new Convolution(ffty);

    init(F,V);
  }

  double normalization() {
    return fftx->normalization()*convolvey[0]->normalization();
  }

  void init(Complex **F=NULL, Complex *V=NULL) {
    Forward=fftx->Forward;
    Backward=fftx->Backward;

    size_t outputSize=fftx->outputSize();
    size_t workSizeW=fftx->workSizeW();

    size_t N=std::max(A,B);
    allocateF=!F;
    this->F=allocateF ? utils::ComplexAlign(N,outputSize) : F;

    allocateW=!W && !fftx->inplace;
    W=allocateW ? utils::ComplexAlign(workSizeW) : NULL;

    qx=fftx->q;
    nx=fftx->n;
    Rx=fftx->R;
    lx=fftx->l;
    scale=1.0/normalization();

    Lx=fftx->L;
    Ly=fftx->C;

    Sx=fftx->S;

    Pad=fftx->Pad;
    (fftx->*Pad)(W);

    nloops=fftx->nloops();
    loop2=fftx->loop2();
    size_t extra;
    if(loop2) {
      r=fftx->increment(0);
      Fp=new Complex*[A];
      Fp[0]=this->F[A-1];
      for(size_t a=1; a < A; ++a)
        Fp[a]=this->F[a-1];
      extra=1;
    } else
      extra=0;

    if(A > B+extra && !fftx->inplace && workSizeW <= outputSize) {
      W0=this->F[B];
    } else
      W0=this->W;

    allocateV=false;

    if(V) {
      this->V=new Complex*[B];
      size_t size=fftx->workSizeV();
      for(size_t i=0; i < B; ++i)
        this->V[i]=V+i*size;
    } else
      this->V=NULL;
  }

  void initV() {
    allocateV=true;
    V=new Complex*[B];
    size_t size=fftx->workSizeV();
    for(size_t i=0; i < B; ++i)
      V[i]=utils::ComplexAlign(size);
  }

  ~Convolution2() {
    if(allocateF) {
      utils::deleteAlign(F[0]);
      delete [] F;
    }

    if(allocateW)
      utils::deleteAlign(W);

    if(allocateV) {
      for(size_t i=0; i < B; ++i)
        utils::deleteAlign(V[i]);
    }

    if(V)
      delete [] V;

    if(loop2)
      delete [] Fp;

    for(size_t t=0; t < threads; ++t)
      delete convolvey[t];
    delete [] convolvey;
  }

  void forward(Complex **f, Complex **F, size_t rx,
               size_t start, size_t stop,
               size_t offset=0) {
    for(size_t a=start; a < stop; ++a)
      (fftx->*Forward)(f[a]+offset,F[a],rx,W);
  }

  void subconvolution(Complex **F, size_t rx, size_t offset=0) {
    size_t D=rx == 0 ? fftx->D0 : fftx->D;
    PARALLEL(
      for(size_t i=0; i < lx; ++i) {
        size_t t=parallel::get_thread_num(threads);
        Convolution *cy=convolvey[t];
        for(size_t d=0; d < D; ++d) {
          cy->indices.index[0]=fftx->index(rx+d,i);
          cy->convolveRaw(F,offset+(D*i+d)*Sx,&cy->indices);
        }
      });
  }

  void backward(Complex **F, Complex **f, size_t rx,
                size_t start, size_t stop,
                size_t offset=0, Complex *W0=NULL) {
    for(size_t b=start; b < stop; ++b)
      (fftx->*Backward)(F[b],f[b]+offset,rx,W0);
    if(W && W == W0) (fftx->*Pad)(W0);
  }

  void normalize(Complex **h, size_t offset=0) {
    for(size_t b=0; b < B; ++b) {
      Complex *hb=h[b]+offset;
      for(size_t i=0; i < Lx; ++i) {
        Complex *hbi=hb+Sx*i;
        for(size_t j=0; j < Ly; ++j)
          hbi[j] *= scale;
      }
    }
  }

// f is a pointer to A distinct data blocks each of size Lx*Sx,
// shifted by offset.
  void convolveRaw(Complex **f, size_t offset=0, Indices *indices=NULL) {
    for(size_t t=0; t < threads; ++t)
      convolvey[t]->indices.copy(indices,1);

    if(fftx->Overwrite()) {
      forward(f,F,0,0,A,offset);
      size_t final=fftx->n-1;
      for(size_t r=0; r < final; ++r)
        subconvolution(f,r,offset+Sx*r*lx);
      subconvolution(F,final);
      backward(F,f,0,0,B,offset,W);
    } else {
      if(loop2) {
        forward(f,F,0,0,A,offset);
        subconvolution(F,0);
        size_t C=A-B;
        size_t a=0;
        for(; a+C <= B; a += C) {
          forward(f,Fp,r,a,a+C,offset);
          backward(F,f,0,a,a+C,offset,W0);
        }
        forward(f,Fp,r,a,A,offset);
        subconvolution(Fp,r);
        backward(Fp,f,r,0,B,offset,W0);
      } else {
        size_t Offset;
        Complex **h0;
        if(nloops > 1) {
          if(!V) initV();
          h0=V;
          Offset=0;
        } else {
          Offset=offset;
          h0=f;
        }

        for(size_t rx=0; rx < Rx; rx += fftx->increment(rx)) {
          forward(f,F,rx,0,A,offset);
          subconvolution(F,rx);
          backward(F,h0,rx,0,B,Offset,W);
        }

        if(nloops > 1) {
          for(size_t b=0; b < B; ++b) {
            Complex *fb=f[b]+offset;
            Complex *hb=h0[b];
            for(size_t i=0; i < Lx; ++i) {
              size_t Sxi=Sx*i;
              Complex *fbi=fb+Sxi;
              Complex *hbi=hb+Sxi;
              for(size_t j=0; j < Ly; ++j)
                fbi[j]=hbi[j];
            }
          }
        }
      }
    }
  }

  void convolve(Complex **f, size_t offset=0) {
    convolveRaw(f,offset);
    normalize(f,offset);
  }
};

class Convolution3 : public ThreadBase {
public:
  fftBase *fftx,*ffty,*fftz;
  Convolution **convolvez;
  Convolution2 **convolveyz;
  size_t Lx,Ly,Lz; // x,y,z dimensions of input data
  size_t Sx,Sy; // x stride, y stride
  size_t A;
  size_t B;
  multiplier *mult;
  double scale;
protected:
  size_t lx;    // x dimension of Fx buffer
  size_t qx;
  size_t nx; // number of residues
  size_t Rx; // number of residue blocks
  size_t r;
  Complex **F,**Fp;
  Complex **V;
  Complex *W;
  Complex *W0;
  bool allocateF;
  bool allocateV;
  bool allocateW;
  bool loop2;
  FFTcall Forward,Backward;
  FFTPad Pad;
  size_t nloops;
public:
  Indices indices;

  // F: optional array of max(A,B) work arrays of size fftx->outputSize()
  // W: optional work array of size fftx->workSizeW();
  //    call fftx->pad() if W changed between calls to convolve()
  // V: optional work array of size B*fftx->workSizeV()
  Convolution3(fftBase *fftx, fftBase *ffty, fftBase *fftz,
               Complex **F=NULL, Complex *W=NULL, Complex *V=NULL) :
    ThreadBase(fftx->Threads()), fftx(fftx), ffty(ffty), fftz(fftz),
    A(fftx->app.A), B(fftx->app.B), mult(fftx->app.mult),
    W(W), allocateF(false), allocateW(false) {

    if(fftx->l < threads) {
      ffty->Threads(threads);
      threads=1;
    }

    this->convolveyz=new Convolution2*[threads];
    for(size_t t=0; t < threads; ++t)
      this->convolveyz[t]=new Convolution2(ffty,fftz);

    init(F,V);
  }

  double normalization() {
    return fftx->normalization()*convolveyz[0]->normalization();
  }

  void init(Complex **F=NULL, Complex *V=NULL) {
    Forward=fftx->Forward;
    Backward=fftx->Backward;

    size_t outputSize=fftx->outputSize();
    size_t workSizeW=fftx->workSizeW();

    size_t N=std::max(A,B);
    allocateF=!F;
    this->F=allocateF ? utils::ComplexAlign(N,outputSize) : F;

    allocateW=!W && !fftx->inplace;
    W=allocateW ? utils::ComplexAlign(workSizeW) : NULL;

    qx=fftx->q;
    nx=fftx->n;
    Rx=fftx->R;
    lx=fftx->l;
    scale=1.0/normalization();

    Lx=fftx->L;
    Ly=ffty->L;
    Lz=ffty->C;

    Sx=fftx->S;
    Sy=ffty->S;

    if(Sx < Ly*Sy) {
      std::cerr << "Sx cannot be less than Ly*Sy" << std::endl;
      exit(-1);
    }

    if(fftx->C != (Sy == Lz ? Ly*Lz : Lz)) {
      std::cerr << "fftx->C is invalid" << std::endl;
      exit(-1);
    }

    Pad=fftx->Pad;
    (fftx->*Pad)(W);

    nloops=fftx->nloops();
    loop2=fftx->loop2();
    size_t extra;
    if(loop2) {
      r=fftx->increment(0);
      Fp=new Complex*[A];
      Fp[0]=this->F[A-1];
      for(size_t a=1; a < A; ++a)
        Fp[a]=this->F[a-1];
      extra=1;
    } else
      extra=0;

    if(A > B+extra && !fftx->inplace && workSizeW <= outputSize) {
      W0=this->F[B];
    } else
      W0=this->W;

    allocateV=false;

    if(V) {
      this->V=new Complex*[B];
      size_t size=fftx->workSizeV();
      for(size_t i=0; i < B; ++i)
        this->V[i]=V+i*size;
    } else
      this->V=NULL;
  }

  void initV() {
    allocateV=true;
    V=new Complex*[B];
    size_t size=fftx->workSizeV();
    for(size_t i=0; i < B; ++i)
      V[i]=utils::ComplexAlign(size);
  }

  ~Convolution3() {
    if(allocateF) {
      utils::deleteAlign(F[0]);
      delete [] F;
    }

    if(allocateW)
      utils::deleteAlign(W);

    if(allocateV) {
      for(size_t i=0; i < B; ++i)
        utils::deleteAlign(V[i]);
    }

    if(V)
      delete [] V;

    if(loop2)
      delete [] Fp;

    for(size_t t=0; t < threads; ++t)
      delete convolveyz[t];
    delete [] convolveyz;
  }

  void forward(Complex **f, Complex **F, size_t rx,
               size_t start, size_t stop,
               size_t offset=0) {
    for(size_t a=start; a < stop; ++a) {
      if(Sy == Lz)
        (fftx->*Forward)(f[a]+offset,F[a],rx,W);
      else {
        Complex *fa=f[a]+offset;
        Complex *Fa=F[a];
        for(size_t j=0; j < Ly; ++j) {
          size_t Syj=Sy*j;
          (fftx->*Forward)(fa+Syj,Fa+Syj,rx,W);
        }
      }
    }
  }

  void subconvolution(Complex **F, size_t rx, size_t offset=0) {
    size_t D=rx == 0 ? fftx->D0 : fftx->D;
    PARALLEL(
      for(size_t i=0; i < lx; ++i) {
        size_t t=parallel::get_thread_num(threads);
        Convolution2 *cyz=convolveyz[t];
        for(size_t d=0; d < D; ++d) {
          cyz->indices.index[1]=fftx->index(rx+d,i);
          cyz->convolveRaw(F,offset+(D*i+d)*Sx,&cyz->indices);
        }
      });
  }

  void backward(Complex **F, Complex **f, size_t rx,
                size_t start, size_t stop,
                size_t offset=0, Complex *W0=NULL) {
    for(size_t b=start; b < stop; ++b) {
      if(Sy == Lz)
        (fftx->*Backward)(F[b],f[b]+offset,rx,W0);
      else {
        Complex *Fb=F[b];
        Complex *fb=f[b]+offset;
        for(size_t j=0; j < Ly; ++j) {
          size_t Syj=Sy*j;
          (fftx->*Backward)(Fb+Syj,fb+Syj,rx,W0);
        }
      }
    }
    if(W && W == W0) (fftx->*Pad)(W0);
  }

  void normalize(Complex **h, size_t offset=0) {
    for(size_t b=0; b < B; ++b) {
      Complex *hb=h[b]+offset;
      for(size_t i=0; i < Lx; ++i) {
        Complex *hbi=hb+Sx*i;
        for(size_t j=0; j < Ly; ++j) {
          Complex *hbij=hbi+Sy*j;
          for(size_t k=0; k < Lz; ++k)
            hbij[k] *= scale;
        }
      }
    }
  }

// f is a pointer to A distinct data blocks each of size Lx*Sx,
// shifted by offset.
  void convolveRaw(Complex **f, size_t offset=0, Indices *indices=NULL) {
    for(size_t t=0; t < threads; ++t)
      convolveyz[t]->indices.copy(indices,2);

    if(fftx->Overwrite()) {
      forward(f,F,0,0,A,offset);
      size_t final=fftx->n-1;
      for(size_t r=0; r < final; ++r)
        subconvolution(f,r,offset+Sx*r*lx);
      subconvolution(F,final);
      backward(F,f,0,0,B,offset,W);
    } else {
      if(loop2) {
        forward(f,F,0,0,A,offset);
        subconvolution(F,0);
        size_t C=A-B;
        size_t a=0;
        for(; a+C <= B; a += C) {
          forward(f,Fp,r,a,a+C,offset);
          backward(F,f,0,a,a+C,offset,W0);
        }
        forward(f,Fp,r,a,A,offset);
        subconvolution(Fp,r);
        backward(Fp,f,r,0,B,offset,W0);
      } else {
        size_t Offset;
        Complex **h0;
        if(nloops > 1) {
          if(!V) initV();
          h0=V;
          Offset=0;
        } else {
          Offset=offset;
          h0=f;
        }

        for(size_t rx=0; rx < Rx; rx += fftx->increment(rx)) {
          forward(f,F,rx,0,A,offset);
          subconvolution(F,rx);
          backward(F,h0,rx,0,B,Offset,W);
        }

        if(nloops > 1) {
          for(size_t b=0; b < B; ++b) {
            Complex *fb=f[b]+offset;
            Complex *hb=h0[b];
            for(size_t i=0; i < Lx; ++i) {
              size_t Sxi=Sx*i;
              Complex *fbi=fb+Sxi;
              Complex *hbi=hb+Sxi;
              for(size_t j=0; j < Ly; ++j) {
                size_t Syj=Sy*j;
                Complex *fbij=fbi+Syj;
                Complex *hbij=hbi+Syj;
                for(size_t k=0; k < Lz; ++k)
                  fbij[k]=hbij[k];
              }
            }
          }
        }
      }
    }
  }

  void convolve(Complex **f, size_t offset=0) {
    convolveRaw(f,offset);
    normalize(f,offset);
  }
};

} //end namespace fftwpp

#endif
