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

extern unsigned int L,Lx,Ly,Lz; // input data lengths
extern unsigned int M,Mx,My,Mz; // minimum padded lengths

extern unsigned int mx,my,mz; // internal FFT sizes
extern unsigned int Dx,Dy,Dz; // numbers of residues computed at a time
extern unsigned int Sx,Sy;    // strides
extern int Ix,Iy,Iz;          // inplace flags

extern bool Output;
extern bool testError;
extern bool showOptTimes;
extern bool Centered;
extern bool normalized;
extern bool Tforced;
extern bool showRoutines;

unsigned int nextfftsize(unsigned int m);

class fftBase;

typedef void (fftBase::*FFTcall)(Complex *f, Complex *F, unsigned int r,
                                 Complex *W);
typedef void (fftBase::*FFTPad)(Complex *W);

class Indices
{
public:
  fftBase *fft;
  unsigned int *index;
  size_t size,maxsize;
  bool allocated;
  unsigned int r;

  Indices() : index(NULL), maxsize(0) {}

  void copy(Indices *indices, unsigned int size0) {
    size=indices ? indices->size : size0;
    if(size > maxsize) {
      if(maxsize > 0)
        delete [] index;
      index=new unsigned int[size];
      maxsize=size;
    }
    if(indices)
      for(unsigned int d=1; d < size; ++d)
        index[d]=indices->index[d];
  }

  ~Indices() {
    if(maxsize > 0)
      delete [] index;
  }
};

typedef void multiplier(Complex **F, unsigned int n,
                        Indices *indices, unsigned int threads);

// Multiplication routines for binary convolutions that take two inputs.
multiplier multNone,multbinary,realmultbinary;

class Application : public ThreadBase
{
public:
  unsigned int A;
  unsigned int B;
  multiplier *mult;
  unsigned int m;
  unsigned int D;
  int I;

  Application(unsigned int A, unsigned int B, multiplier *mult=multNone,
              unsigned int threads=fftw::maxthreads, unsigned int n=0,
              unsigned int m=0, unsigned int D=0, int I=-1) :
    ThreadBase(threads), A(A), B(B), mult(mult), m(m), D(D), I(I) {
    if(n == 0)
      multithread(threads);
    else {
      multithread(n);
      this->threads=innerthreads;
    }
    std::cout << "Requesting " << this->threads << " threads." << std::endl;
    std::cout << std::endl;
  };
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
  unsigned int Q;  // number of residues
  unsigned int R;  // number of residue blocks
  unsigned int dr; // r increment
  unsigned int D;  // number of residues stored in F at a time
  unsigned int D0; // remainder
  size_t Cm,Sm;
  unsigned int b;  // total block size, including stride
  unsigned int l;  // block size of a single FFT
  bool inplace;
  multiplier *mult;
  bool centered;
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

  void checkParameters();

  void common();

  void initZetaqm(unsigned int q, unsigned int m);

  class OptBase {
  public:
    unsigned int counter=0;
    unsigned int m,q,D;
    bool inplace;
    unsigned int threads;
    bool mForced;
    typedef std::list<unsigned int> mList;
    mList mlist;
    double threshold;
    double T;

    virtual double time(unsigned int L, unsigned int M, unsigned int C,
                        unsigned int S, unsigned int m, unsigned int q,
                        unsigned int D, bool inplace, unsigned int threads,
                        Application &app)=0;

    virtual bool valid(unsigned int D, unsigned int p, unsigned int S) {
      return D == 1 || S == 1;
    }

    // Called by the optimizer to record the time to complete an application
    // for a given value of m.
    void check(unsigned int L, unsigned int M,
               unsigned int C, unsigned int S, unsigned int m,
               unsigned int p, unsigned int q, unsigned int D,
               bool inplace, unsigned int threads, Application& app,
               bool useTimer);

    // Determine the optimal m value for padding L data values to
    // size >= M for an application app.
    // If Explicit=true, we only consider m >= M.
    // centered must be true for all centered and Hermitian routines.
    void scan(unsigned int L, unsigned int M, Application& app,
              unsigned int C, unsigned int S, bool Explicit=false,
              bool centered=false);

    // A function called by opt to iterate over m and D values
    // and call check.
    // Inner is true if p > 2.
    // If inner is false, ubound is the maximum number maximum number of
    // m values that are checked.
    // If inner is true, ubound is the maximum size of m values that are
    // checked.
    void optloop(unsigned int& m0, unsigned int L, unsigned int M,
                 Application& app, unsigned int C, unsigned int S,
                 bool centered, unsigned int ubound, bool useTimer,
                 bool Explicit, bool inner=false);

    // The default optimizer routine. Used by scan to iterate and check
    // different m values for a given geometry and application.
    // 'minInner' is the minimum size of FFT we consider for the inner routines.
    // 'itmax' is the maximum number of iterations done by optloop
    // (when p <= 2).
    void opt(unsigned int L, unsigned int M, Application& app,
             unsigned int C, unsigned int S, unsigned int minInner,
             unsigned int itmax, bool Explicit, bool centered,
             bool useTimer=true);
  };

  void invalid () {
    std::cerr << "Invalid parameters: " << std::endl
         << " D=" << D << " p=" << p << " C=" << C << std::endl;
    exit(-1);
  }

  fftBase(unsigned int L, unsigned int M, unsigned int C, unsigned int S,
          multiplier *mult, unsigned int threads=fftw::maxthreads,
          bool centered=false) :
    ThreadBase(threads), L(L), M(M), C(C),  S(S == 0 ? C : S),
    mult(mult), centered(centered) {checkParameters();}

  fftBase(unsigned int L, unsigned int M, unsigned int C, unsigned int S,
          unsigned int m, unsigned int q, unsigned int D, bool inplace,
          multiplier *mult, unsigned int threads=fftw::maxthreads,
          bool centered=false) :
    ThreadBase(threads), L(L), M(M), C(C),  S(S == 0 ? C : S), m(m),
    p(utils::ceilquotient(L,m)), q(q), D(D), inplace(inplace), mult(mult),
    centered(centered) {checkParameters();}

  fftBase(unsigned int L, unsigned int M, Application& app,
          unsigned int C=1, unsigned int S=0, bool Explicit=false,
          bool centered=false) :
    ThreadBase(app.threads), L(L), M(M), C(C), S(S == 0 ? C : S),
    mult(app.mult), centered(centered) {checkParameters();}

  virtual ~fftBase();

  void padNone(Complex *W) {}

  virtual void padSingle(Complex *W) {}
  virtual void padMany(Complex *W) {}

  void pad(Complex *W) {
    if(W)
      (this->*Pad)(W);
  }

  void forward(Complex *f, Complex *F, unsigned int r=0, Complex *W=NULL) {
    (this->*Forward)(f,F,r,W);
  }

  void backward(Complex *f, Complex *F, unsigned int r=0, Complex *W=NULL) {
    (this->*Backward)(f,F,r,W);
  }

  virtual void forwardExplicit(Complex *f, Complex *F, unsigned int,
                               Complex *W)=0;
  virtual void forwardExplicitMany(Complex *f, Complex *F, unsigned int,
                                   Complex *W)=0;
  virtual void forwardExplicitFast(Complex *f, Complex *F, unsigned int,
                                   Complex *W) {}
  virtual void forwardExplicitManyFast(Complex *f, Complex *F, unsigned int,
                                       Complex *W) {}
  virtual void forwardExplicitSlow(Complex *f, Complex *F, unsigned int r,
                                   Complex *W) {}
  virtual void forwardExplicitManySlow(Complex *f, Complex *F, unsigned int r,
                                       Complex *W) {}

  virtual void backwardExplicit(Complex *F, Complex *f, unsigned int,
                                Complex *W)=0;
  virtual void backwardExplicitMany(Complex *F, Complex *f, unsigned int,
                                    Complex *W)=0;
  virtual void backwardExplicitFast(Complex *F, Complex *f, unsigned int,
                                    Complex *W) {}
  virtual void backwardExplicitManyFast(Complex *F, Complex *f, unsigned int,
                                        Complex *W) {}
  virtual void backwardExplicitSlow(Complex *F, Complex *f, unsigned int r,
                                    Complex *W) {}
  virtual void backwardExplicitManySlow(Complex *F, Complex *f, unsigned int r,
                                        Complex *W) {}

  // Return transformed index for residue r at position I
  unsigned int index(unsigned int r, unsigned int i) {
    if(q == 1) return i;
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
    return q*s+n*u+r;
  }

  unsigned int Index(unsigned int r, unsigned int I) {
    unsigned int i=I/S;
    return S*(index(r,i)-i)+I;
  }

  virtual void forward1(Complex *f, Complex *F0, unsigned int r0, Complex *W)
  {}
  virtual void forward1All(Complex *f, Complex *F0, unsigned int r0,
                           Complex *W) {}
  virtual void forward1Many(Complex *f, Complex *F, unsigned int r,
                            Complex *W) {}
  virtual void forward1ManyAll(Complex *f, Complex *F, unsigned int r,
                               Complex *W) {}
  virtual void forward2(Complex *f, Complex *F0, unsigned int r0, Complex *W)
  {}
  virtual void forward2All(Complex *f, Complex *F0, unsigned int r0,
                           Complex *W) {}
  virtual void forward2Many(Complex *f, Complex *F, unsigned int r,
                            Complex *W) {}
  virtual void forward2ManyAll(Complex *f, Complex *F, unsigned int r,
                               Complex *W) {}
  virtual void forwardInner(Complex *f, Complex *F0, unsigned int r0,
                            Complex *W) {}
  virtual void forwardInnerAll(Complex *f, Complex *F0, unsigned int r0,
                               Complex *W) {}
  virtual void forwardInnerMany(Complex *f, Complex *F, unsigned int r,
                                Complex *W) {}
  virtual void forwardInnerManyAll(Complex *f, Complex *F, unsigned int r,
                                   Complex *W) {}

  virtual void backward1(Complex *F0, Complex *f, unsigned int r0, Complex *W)
  {}
  virtual void backward1All(Complex *F0, Complex *f, unsigned int r0,
                            Complex *W) {}
  virtual void backward1Many(Complex *F, Complex *f, unsigned int r,
                             Complex *W) {}
  virtual void backward1ManyAll(Complex *F, Complex *f, unsigned int r,
                                Complex *W) {}
  virtual void backward2(Complex *F0, Complex *f, unsigned int r0,
                         Complex *W) {}
  virtual void backward2All(Complex *F0, Complex *f, unsigned int r0,
                            Complex *W) {}
  virtual void backward2Many(Complex *F, Complex *f, unsigned int r,
                             Complex *W) {}
  virtual void backward2ManyAll(Complex *F, Complex *f, unsigned int r,
                                Complex *W) {}
  virtual void backwardInner(Complex *F0, Complex *f, unsigned int r0,
                             Complex *W) {}
  virtual void backwardInnerAll(Complex *F0, Complex *f, unsigned int r0,
                                Complex *W) {}
  virtual void backwardInnerMany(Complex *F, Complex *f, unsigned int r,
                                 Complex *W) {}
  virtual void backwardInnerManyAll(Complex *F, Complex *f, unsigned int r,
                                    Complex *W) {}

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
    return nloops() == 2 && A > B && !Overwrite(A,B);
  }

  virtual unsigned int inputSize() {
    return S*L;
  }

  // Number of complex outputs per residue per copy
  virtual unsigned int noutputs() {
    return l;
  }

  // Number of outputs per iteration per copy
  virtual unsigned int noutputs(unsigned int r) {
    return l*(r == 0 ? D0 : D);
  }

  // Number of complex outputs per iteration
  unsigned int complexOutputs(unsigned int r) {
    return b*(r == 0 ? D0 : D);
  }

  unsigned int workSizeV(unsigned int A, unsigned int B) {
    return nloops() == 1 || loop2(A,B) ? 0 : inputSize();
  }

  virtual unsigned int workSizeW() {
    return inplace ? 0 : outputSize();
  }

  // Allow input data to be embedded within output buffer.
  virtual bool embed() {
    return q == 1 || (p == 1 && nloops() == 1);
  }

  unsigned int repad() {
    return !inplace && L < m;
  }

  bool Overwrite(unsigned int A, unsigned int B) {
    return overwrite && A >= B;
  }

  virtual double time(Application& app)=0;

  double report(Application& app) {
    double median=time(app)*1.0e-9;
    std::cout << "median=" << median << std::endl;
    return median;
  }
};

typedef double timer(fftBase *fft, Application &app, double& threshold);
timer timePad,timePadHermitian;

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
        unsigned int C, unsigned int S, bool Explicit=false) {
      scan(L,M,app,C,S,Explicit);
    }

    double time(unsigned int L, unsigned int M, unsigned int C, unsigned int S,
                unsigned int m, unsigned int q,unsigned int D,
                bool inplace, unsigned int threads, Application &app) {
      fftPad fft(L,M,C,S,m,q,D,inplace,app.mult,threads);
      return timePad(&fft,app,threshold);
    }
  };

  fftPad(unsigned int L, unsigned int M, unsigned int C, unsigned int S,
         multiplier *mult, unsigned int threads=fftw::maxthreads,
         bool centered=false) :
    fftBase(L,M,C,S,mult,threads,centered) {}

  // Compute an fft padded to N=m*q >= M >= L
  fftPad(unsigned int L, unsigned int M, unsigned int C, unsigned int S,
         unsigned int m, unsigned int q, unsigned int D, bool inplace,
         multiplier *mult, unsigned int threads=fftw::maxthreads,
         bool centered=false) :
    fftBase(L,M,C,S,m,q,D,inplace,mult,threads,centered) {
    Opt opt;
    if(q > 1 && !opt.valid(D,p,this->S)) invalid();
    init();
  }

  // Normal entry point.
  // Compute C ffts of length L with stride S >= C and distance 1
  // padded to at least M
  fftPad(unsigned int L, unsigned int M, Application& app,
         unsigned int C=1, unsigned int S=0, bool Explicit=false,
         bool centered=false) :
    fftBase(L,M,app,C,S,Explicit,centered) {
    Opt opt=Opt(L,M,app,C,this->S,Explicit);
    m=opt.m;
    if(Explicit)
      M=m;
    q=opt.q;
    D=opt.D;
    inplace=opt.inplace;
    threads=opt.threads;
    init();
  }

  ~fftPad();

  void init();

  double time(Application& app) {
    double threshold=DBL_MAX;
    return timePad(this,app,threshold);
  }

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
public:

  class Opt : public OptBase {
  public:
    Opt() {}

    Opt(unsigned int L, unsigned int M, Application& app,
        unsigned int C, unsigned int S, bool Explicit=false) {
      scan(L,M,app,C,S,Explicit,true);
    }

    bool valid(unsigned int D, unsigned int p, unsigned int S) {
      return p%2 == 0 && (D == 1 || S == 1);
    }

    double time(unsigned int L, unsigned int M, unsigned int C, unsigned int S,
                unsigned int m, unsigned int q, unsigned int D,
                bool inplace, unsigned int threads, Application &app) {
      fftPadCentered fft(L,M,C,S,m,q,D,inplace,app.mult,threads);
      return timePad(&fft,app,threshold);
    }
  };

  // Compute an fft padded to N=m*q >= M >= L
  fftPadCentered(unsigned int L, unsigned int M, unsigned int C,
                 unsigned int S, unsigned int m, unsigned int q,
                 unsigned int D, bool inplace, multiplier *mult,
                 unsigned int threads=fftw::maxthreads, bool fast=true) :
    fftPad(L,M,C,S,m,q,D,inplace,mult,threads,true) {
    Opt opt;
    if(q > 1 && !opt.valid(D,p,this->S)) invalid();
    init(fast);
  }

  // Normal entry point.
  // Compute C ffts of length L and distance 1 padded to at least M
  fftPadCentered(unsigned int L, unsigned int M, Application& app,
                 unsigned int C=1, unsigned int S=0, bool Explicit=false,
                 bool fast=true) :
    fftPad(L,M,C,S,app.mult,app.threads,true) {
    Opt opt=Opt(L,M,app,C,this->S,Explicit);
    m=opt.m;
    if(Explicit)
      M=m;
    q=opt.q;
    D=opt.D;
    inplace=opt.inplace;
    threads=opt.threads;
    fftPad::init();
    init(fast);
  }

  bool embed() {
    return false;
  }

  ~fftPadCentered() {
    if(ZetaShift)
      utils::deleteAlign(ZetaShift);
  }

  bool conjugates() {return D > 1 && (p == 1 || p % 2 == 0);}

  void init(bool fast);

  double time(Application& app) {
    double threshold=DBL_MAX;
    return timePad(this,app,threshold);
  }

  void forwardExplicitFast(Complex *f, Complex *F, unsigned int r, Complex *W);
  void forwardExplicitManyFast(Complex *f, Complex *F, unsigned int r,
                               Complex *W);
  void forwardExplicitSlow(Complex *f, Complex *F, unsigned int r, Complex *W);
  void forwardExplicitManySlow(Complex *f, Complex *F, unsigned int r,
                               Complex *W);

  void backwardExplicitFast(Complex *F, Complex *f, unsigned int r,
                            Complex *W);
  void backwardExplicitManyFast(Complex *F, Complex *f, unsigned int r,
                                Complex *W);
  void backwardExplicitSlow(Complex *F, Complex *f, unsigned int r,
                            Complex *W);
  void backwardExplicitManySlow(Complex *F, Complex *f, unsigned int r,
                                Complex *W);

  void initShift();

  void forward2(Complex *f, Complex *F0, unsigned int r0, Complex *W);
  void forward2All(Complex *f, Complex *F0, unsigned int r0, Complex *W);
  void forward2Many(Complex *f, Complex *F, unsigned int r, Complex *W);
  void forward2ManyAll(Complex *f, Complex *F, unsigned int r, Complex *W);
  void forwardInner(Complex *f, Complex *F0, unsigned int r0, Complex *W);
  void forwardInnerAll(Complex *f, Complex *F0, unsigned int r0, Complex *W);
  void forwardInnerMany(Complex *f, Complex *F0, unsigned int r0, Complex *W);
  void forwardInnerManyAll(Complex *f, Complex *F0, unsigned int r0,
                           Complex *W);

  void backward2(Complex *F0, Complex *f, unsigned int r0, Complex *W);
  void backward2All(Complex *F0, Complex *f, unsigned int r0, Complex *W);
  void backward2Many(Complex *F, Complex *f, unsigned int r, Complex *W);
  void backward2ManyAll(Complex *F, Complex *f, unsigned int r, Complex *W);
  void backwardInner(Complex *F0, Complex *f, unsigned int r0, Complex *W);
  void backwardInnerAll(Complex *F0, Complex *f, unsigned int r0, Complex *W);
  void backwardInnerMany(Complex *F0, Complex *f, unsigned int r0, Complex *W);
  void backwardInnerManyAll(Complex *F0, Complex *f, unsigned int r0,
                            Complex *W);
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
        unsigned int C, unsigned int, bool Explicit=false) {
      scan(L,M,app,C,C,Explicit,true);
    }

    bool valid(unsigned int D, unsigned int p, unsigned int C) {
      return D == 2 && p%2 == 0 && (p == 2 || C == 1);
    }

    double time(unsigned int L, unsigned int M, unsigned int C, unsigned int,
                unsigned int m, unsigned int q, unsigned int D,
                bool inplace, unsigned int threads, Application &app) {
      fftPadHermitian fft(L,M,C,m,q,D,inplace,app.mult,threads);
      return timePadHermitian(&fft,app,threshold);
    }
  };

  fftPadHermitian(unsigned int L, unsigned int M, unsigned int C,
                  unsigned int m, unsigned int q, unsigned int D,
                  bool inplace, multiplier *mult,
                  unsigned int threads=fftw::maxthreads) :
    fftBase(L,M,C,C,m,q,D,inplace,mult,threads,true) {
    Opt opt;
    if(q > 1 && !opt.valid(D,p,C)) invalid();
    init();
  }

  fftPadHermitian(unsigned int L, unsigned int M, Application& app,
                  unsigned int C=1, bool Explicit=false) :
    fftBase(L,M,app,C,C,Explicit,true) {
    Opt opt=Opt(L,M,app,C,C,Explicit);
    m=opt.m;
    if(Explicit)
      M=m;
    q=opt.q;
    D=opt.D;
    inplace=opt.inplace;
    threads=opt.threads;
    init();
  }

  ~fftPadHermitian();

  void init();

  double time(Application& app) {
    double threshold=DBL_MAX;
    return timePadHermitian(this,app,threshold);
  }

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

  // Number of real outputs per residue per copy
  unsigned int noutputs() {
    return m*(q == 1 ? 1 : p/2);
  }

  unsigned int noutputs(unsigned int r) {
    std::cerr << "For Hermitian transforms, use noutputs() instead of noutputs(unsigned int r)." << std::endl;
    exit(-1);
  }

  unsigned int workSizeW() {
    return inplace ? 0 : B*D;
  }

  unsigned int inputSize() {
    return C*utils::ceilquotient(L,2);
  }
};

class Convolution : public ThreadBase {
public:
  fftBase *fft;
  unsigned int L;
  unsigned int A;
  unsigned int B;
  multiplier *mult;
  double scale;
protected:
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
  bool overwrite;
  unsigned int inputSize;
  FFTcall Forward,Backward;
  FFTPad Pad;
public:
  Indices indices;

  Convolution() :
    ThreadBase(), q(1), W(NULL), allocate(false),
    allocateF(false), allocateW(false), loop2(false) {}

  Convolution(Application &app) :
    ThreadBase(app.threads), A(app.A), B(app.B), mult(app.mult),
    q(1), W(NULL), allocate(false), allocateF(false),
    allocateW(false) {}

  Convolution(unsigned int L, unsigned int M, Application &app) :
    ThreadBase(app.threads), A(app.A), B(app.B), mult(app.mult), W(NULL), allocate(true) {
    fft=new fftPad(L,M,app);
    init();
  }

  // L: dimension of input data
  // M: dimension of transformed data, including padding
  // A: number of inputs.
  // B: number of outputs.
  Convolution(unsigned int L, unsigned int M,
              unsigned int A, unsigned int B, multiplier *mult,
              unsigned int threads=fftw::maxthreads) :
    ThreadBase(threads), A(A), B(B), mult(mult), W(NULL), allocate(true) {
    Application app(A,B,mult,threads);
    fft=new fftPad(L,M,app);
    init();
  }

  // fft: precomputed fftPad
  // A: number of inputs
  // B: number of outputs
  // F: optional array of max(A,B) work arrays of size fft->outputSize()
  // W: optional work array of size fft->workSizeW();
  //    call pad() if changed between calls to convolve()
  // V: optional work array of size B*fft->workSizeV(A,B)
  //   (only needed for inplace usage)
  Convolution(fftBase *fft, unsigned int A, unsigned int B,
              Complex **F=NULL, Complex *W=NULL,
              Complex *V=NULL) : ThreadBase(fft->Threads()), fft(fft),
                                 A(A), B(B), mult(fft->mult), W(W),
                                 allocate(false) {
    init(F,V);
  }

  double normalization() {
    return fft->normalization();
  }

  void init(Complex **F=NULL, Complex *V=NULL) {
    L=fft->L;
    q=fft->q;
    Q=fft->Q;

    overwrite=fft->Overwrite(A,B);
    if(overwrite) {
      Forward=fft->ForwardAll;
      Backward=fft->BackwardAll;
    } else {
      Forward=fft->Forward;
      Backward=fft->Backward;
    }

    blocksize=fft->noutputs();
    R=fft->residueBlocks();

    scale=1.0/normalization();
    unsigned int outputSize=fft->outputSize();
    unsigned int workSizeW=fft->workSizeW();
    inputSize=fft->inputSize();

    unsigned int N=std::max(A,B);
    allocateF=!F;
    this->F=allocateF ? utils::ComplexAlign(N,outputSize) : F;

    allocateW=!W && !fft->inplace;
    W=allocateW ? utils::ComplexAlign(workSizeW) : NULL;

    if(q > 1) {
      allocateV=false;
      if(V) {
        this->V=new Complex*[B];
        unsigned int size=fft->workSizeV(A,B);
        for(unsigned int i=0; i < B; ++i)
          this->V[i]=V+i*size;
      } else
        this->V=NULL;

      Pad=fft->Pad;
      (fft->*Pad)(W);

      nloops=fft->nloops();
      loop2=fft->loop2(A,B);
      int extra;
      if(loop2) {
        r=fft->increment(0);
        Fp=new Complex*[A];
        unsigned int C=A-B;

        for(unsigned int c=0; c < C; c++)
          Fp[c]=this->F[B+c];

        for(unsigned int b=0; b < B; b += C)
          for(unsigned int c=b; c < B; c++)
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

  void forward(Complex **f, Complex **F, unsigned int r,
               unsigned int start, unsigned int stop) {
    for(unsigned int a=start; a < stop; ++a)
      (fft->*Forward)(f[a],F[a],r,W);
  }

  void operate(Complex **F, unsigned int r, Indices *indices) {
    unsigned int incr=fft->b;
    unsigned int stop=fft->complexOutputs(r);
    indices->r=r;
    for(unsigned int d=0; d < stop; d += incr) {
      Complex *G[A];
      for(unsigned int a=0; a < A; ++a)
        G[a]=F[a]+d;
      (*mult)(G,blocksize,indices,threads);
      ++indices->r;
    }
  }

  void backward(Complex **F, Complex **f, unsigned int r,
                unsigned int start, unsigned int stop,
                Complex *W0=NULL) {
    for(unsigned int b=start; b < stop; ++b)
      (fft->*Backward)(F[b],f[b],r,W0);
  }

  void backwardPad(Complex **F, Complex **f, unsigned int r,
                   unsigned int start, unsigned int stop,
                   Complex *W0=NULL) {
    backward(F,f,r,start,stop,W0);
    if(W && W == W0) (fft->*Pad)(W0);
  }

  void convolveRaw(Complex **f, unsigned int offset=0, Indices *indices=NULL);

  void convolve(Complex **f, unsigned int offset=0) {
    convolveRaw(f,offset);
    normalize(f,offset);
  }

};

class ConvolutionHermitian : public Convolution {
public:
  ConvolutionHermitian(unsigned int L, unsigned int M, Application &app) :
    Convolution() {
    threads=app.threads;
    A=app.A;
    B=app.B;
    mult=app.mult;
    fft=new fftPadHermitian(L,M,app);
    init();
  }

  // A: number of inputs.
  // B: number of outputs.
  // F: optional array of max(A,B) work arrays of size fft->outputSize()
  // W: optional work array of size fft->workSizeW();
  //    if changed between calls to convolve(), be sure to call pad()
  // V: optional work array of size B*fft->workSizeV(A,B)
  //    (for inplace usage)
  ConvolutionHermitian(unsigned int L, unsigned int M,
                       unsigned int A, unsigned int B, multiplier *mult,
                       unsigned int threads=fftw::maxthreads) :
    Convolution() {
    this->threads=threads;
    this->A=A;
    this->B=B;
    this->mult=mult;
    Application app(A,B,mult,threads);
    fft=new fftPadHermitian(L,M,app);
    init();
  }

  ConvolutionHermitian(fftBase *fft, unsigned int A, unsigned B,
                       Complex **F=NULL, Complex *W=NULL, Complex *V=NULL) :
    Convolution(fft,A,B,F,W,V) {}
};

// Enforce 2D Hermiticity using specified (x >= 0,y=0) data.
inline void HermitianSymmetrizeX(unsigned int Hx, unsigned int Hy,
                                 unsigned int x0, Complex *f,
                                 unsigned int Sx)
{
  Complex *F=f+x0*Sx;
  unsigned int stop=Hx*Sx;
  for(unsigned int i=Sx; i < stop; i += Sx)
    *(F-i)=conj(F[i]);

  F[0].im=0.0;

  // Zero out Nyquist modes
  if(x0 == Hx) {
    for(unsigned int j=0; j < Hy; ++j)
      f[j]=0.0;
  }
}

inline void HermitianSymmetrizeX(unsigned int Hx, unsigned int Hy,
                                 unsigned int x0, Complex *f)
{
  HermitianSymmetrizeX(Hx,Hy,x0,f,Hy);
}

// Enforce 3D Hermiticity using specified (x >= 0,y=0,z=0) and (x,y > 0,z=0).
// data.
inline void HermitianSymmetrizeXY(unsigned int Hx, unsigned int Hy,
                                  unsigned int Hz,
                                  unsigned int x0, unsigned int y0,
                                  Complex *f,
                                  unsigned int Sx, unsigned int Sy,
                                  unsigned int threads=fftw::maxthreads)
{
  unsigned int origin=x0*Sx+y0*Sy;
  Complex *F=f+origin;
  unsigned int stop=Hx*Sx;
  for(unsigned int i=Sx; i < stop; i += Sx)
    *(F-i)=conj(F[i]);

  F[0].im=0.0;

  PARALLEL(
    for(int i=(-Hx+1)*Sx; i < (int) stop; i += Sx) {
      unsigned int m=origin-i;
      unsigned int p=origin+i;
      unsigned int Stop=Sy*Hy;
      for(unsigned int j=Sy; j < Stop; j += Sy) {
        f[m-j]=conj(f[p+j]);
      }
    }
    )

    // Zero out Nyquist modes
    if(x0 == Hx) {
      unsigned int Ly=y0+Hy;
      for(unsigned int j=0; j < Ly; ++j) {
        for(unsigned int k=0; k < Hz; ++k) {
          f[Sy*j+k]=0.0;
        }
      }
    }

  if(y0 == Hy) {
    unsigned int Lx=x0+Hx;
    for(unsigned int i=0; i < Lx; ++i) {
      for(unsigned int k=0; k < Hz; ++k) {
        f[Sx*i+k]=0.0;
      }
    }
  }
}

inline void HermitianSymmetrizeXY(unsigned int Hx, unsigned int Hy,
                                  unsigned int Hz,
                                  unsigned int x0, unsigned int y0,
                                  Complex *f,
                                  unsigned int threads=fftw::maxthreads)
{
  unsigned int Ly=y0+Hy;
  HermitianSymmetrizeXY(Hx,Hy,Hz,x0,y0,f,Ly*Hz,Hz,threads);
}

class Convolution2 : public ThreadBase {
public:
  fftBase *fftx,*ffty;
  Convolution **convolvey;
  unsigned int Lx,Ly; // x,y dimensions of input data
  unsigned int Sx; // x stride
  unsigned int A;
  unsigned int B;
  multiplier *mult;
  double scale;
protected:
  unsigned int lx; // x dimension of Fx buffer
  unsigned int Q;
  unsigned int qx;
  unsigned int Qx; // number of residues
  unsigned int Rx; // number of residue blocks
  unsigned int r;
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
  unsigned int nloops;
  bool overwrite;
public:
  Indices indices;

  Convolution2(unsigned int threads=fftw::maxthreads) :
    ThreadBase(threads), ffty(NULL), convolvey(NULL), V(NULL),
    W(NULL), allocateF(false), allocateV(false), allocateW(false),
    loop2(false) {}

  // Lx,Ly: x,y dimensions of input data
  // Mx,My: x,y dimensions of transformed data, including padding
  // A: number of inputs
  // B: number of outputs
  Convolution2(unsigned int Lx, unsigned int Mx,
               unsigned int Ly, unsigned int My,
               unsigned int A, unsigned int B, multiplier *mult,
               unsigned int threads=fftw::maxthreads) :
    ThreadBase(threads), mult(mult), W(NULL) {
    Application appx(A,B,multNone,threads);
    fftx=new fftPad(Lx,Mx,appx,Ly);
    Application appy(A,B,mult,threads,fftx->l);
    ffty=new fftPad(Ly,My,appy);
    convolvey=new Convolution*[threads];
    for(unsigned int t=0; t < threads; ++t)
      convolvey[t]=new Convolution(ffty,A,B);
    init();
  }

  // F: optional array of max(A,B) work arrays of size fftx->outputSize()
  // W: optional work array of size fftx->workSizeW();
  //    call fftx->pad() if W changed between calls to convolve()
  // V: optional work array of size B*fftx->workSizeV(A,B)
  Convolution2(fftBase *fftx, Convolution *convolvey,
               Complex **F=NULL, Complex *W=NULL, Complex *V=NULL) :
    ThreadBase(fftx->Threads()), fftx(fftx), ffty(NULL),
    mult(fftx->mult), W(W), allocateF(false), allocateW(false) {
    multithread(fftx->l);
    this->convolvey=new Convolution*[threads];
    this->convolvey[0]=convolvey;
    for(unsigned int t=1; t < threads; ++t)
      this->convolvey[t]=new Convolution(convolvey->fft,
                                         convolvey->A,convolvey->B);
    init(F,V);
  }

  double normalization() {
    return fftx->normalization()*convolvey[0]->normalization();
  }

  void init(Complex **F=NULL, Complex *V=NULL) {
    A=convolvey[0]->A;
    B=convolvey[0]->B;

    overwrite=fftx->Overwrite(A,B);
    if(overwrite) {
      Forward=fftx->ForwardAll;
      Backward=fftx->BackwardAll;
    } else {
      Forward=fftx->Forward;
      Backward=fftx->Backward;
    }

    unsigned int outputSize=fftx->outputSize();
    unsigned int workSizeW=fftx->workSizeW();

    unsigned int N=std::max(A,B);
    allocateF=!F;
    this->F=allocateF ? utils::ComplexAlign(N,outputSize) : F;

    allocateW=!W && !fftx->inplace;
    W=allocateW ? utils::ComplexAlign(workSizeW) : NULL;

    qx=fftx->q;
    Qx=fftx->Q;
    Rx=fftx->R;
    lx=fftx->l;
    scale=1.0/normalization();

    Lx=fftx->L;
    Ly=fftx->C;

    Sx=fftx->S;

    Pad=fftx->Pad;
    (fftx->*Pad)(W);

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

    if(A > B+extra && !fftx->inplace && workSizeW <= outputSize) {
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
    if(allocateF) {
      utils::deleteAlign(F[0]);
      delete [] F;
    }

    if(allocateW)
      utils::deleteAlign(W);

    if(allocateV) {
      for(unsigned int i=0; i < B; ++i)
        utils::deleteAlign(V[i]);
    }

    if(V)
      delete [] V;

    if(loop2)
      delete [] Fp;

    if(ffty) {
      delete convolvey[0];
      delete ffty;
      delete fftx;
    }

    if(convolvey) {
      for(unsigned int t=1; t < threads; ++t)
        delete convolvey[t];
      delete [] convolvey;
    }
  }

  void forward(Complex **f, Complex **F, unsigned int rx,
               unsigned int start, unsigned int stop,
               unsigned int offset=0) {
    for(unsigned int a=start; a < stop; ++a)
      (fftx->*Forward)(f[a]+offset,F[a],rx,W);
  }

  void subconvolution(Complex **F, unsigned int C,
                      unsigned int stride, unsigned int rx,
                      unsigned int offset=0) {
    unsigned int D=rx == 0 ? fftx->D0 : fftx->D;
    PARALLEL(
      for(unsigned int i=0; i < C; ++i) {
        unsigned int t=ThreadBase::get_thread_num0();
        Convolution *cy=convolvey[t];
        for(unsigned int d=0; d < D; ++d) {
          cy->indices.index[0]=fftx->index(rx+d,i);
          cy->convolveRaw(F,offset+(D*i+d)*stride,&cy->indices);
        }
      }
      );
  }

  void backward(Complex **F, Complex **f, unsigned int rx,
                unsigned int start, unsigned int stop,
                unsigned int offset=0, Complex *W0=NULL) {
    for(unsigned int b=start; b < stop; ++b)
      (fftx->*Backward)(F[b],f[b]+offset,rx,W0);
    if(W && W == W0) (fftx->*Pad)(W0);
  }

  void normalize(Complex **h, unsigned int offset=0) {
    for(unsigned int b=0; b < B; ++b) {
      Complex *hb=h[b]+offset;
      for(unsigned int i=0; i < Lx; ++i) {
        Complex *hbi=hb+Sx*i;
        for(unsigned int j=0; j < Ly; ++j)
          hbi[j] *= scale;
      }
    }
  }

// f is a pointer to A distinct data blocks each of size Lx*Sx,
// shifted by offset.
  void convolveRaw(Complex **f, unsigned int offset=0, Indices *indices=NULL) {
    for(unsigned int t=0; t < threads; ++t)
      convolvey[t]->indices.copy(indices,1);

    if(overwrite) {
      forward(f,F,0,0,A,offset);
      unsigned int final=fftx->n-1;
      for(unsigned int r=0; r < final; ++r)
        subconvolution(f,lx,Sx,r,offset+Sx*r*lx);
      subconvolution(F,lx,Sx,final);
      backward(F,f,0,0,B,offset,W);
    } else {
      if(loop2) {
        forward(f,F,0,0,A,offset);
        subconvolution(F,lx,Sx,0);
        unsigned int C=A-B;
        unsigned int a=0;
        for(; a+C <= B; a += C) {
          forward(f,Fp,r,a,a+C,offset);
          backward(F,f,0,a,a+C,offset,W0);
        }
        forward(f,Fp,r,a,A,offset);
        subconvolution(Fp,lx,Sx,r);
        backward(Fp,f,r,0,B,offset,W0);
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
          forward(f,F,rx,0,A,offset);
          subconvolution(F,lx,Sx,rx);
          backward(F,h0,rx,0,B,Offset,W);
        }

        if(nloops > 1) {
          for(unsigned int b=0; b < B; ++b) {
            Complex *fb=f[b]+offset;
            Complex *hb=h0[b];
            for(unsigned int i=0; i < Lx; ++i) {
              unsigned int Sxi=Sx*i;
              Complex *fbi=fb+Sxi;
              Complex *hbi=hb+Sxi;
              for(unsigned int j=0; j < Ly; ++j)
                fbi[j]=hbi[j];
            }
          }
        }
      }
    }
  }

  void convolve(Complex **f, unsigned int offset=0) {
    convolveRaw(f,offset);
    normalize(f,offset);
  }
};

class ConvolutionHermitian2 : public Convolution2 {
public:
  ConvolutionHermitian2(unsigned int Lx, unsigned int Mx,
                        unsigned int Ly, unsigned int My,
                        unsigned int A, unsigned int B, multiplier *mult,
                        unsigned int threads=fftw::maxthreads) :
    Convolution2(threads) {
    this->mult=mult;
    unsigned int Hy=utils::ceilquotient(Ly,2);
    Application appx(A,B,multNone,threads);
    fftx=new fftPadCentered(Lx,Mx,appx,Hy);
    Application appy(A,B,mult,threads,fftx->l);
    ffty=new fftPadHermitian(Ly,My,appy);
    convolvey=new Convolution*[threads];
    for(unsigned int t=0; t < threads; ++t)
      convolvey[t]=new ConvolutionHermitian(ffty,A,B);
    init();
  }

  // F: optional array of max(A,B) work arrays of size fftx->outputSize()
  // W is an optional work array of size fftx->workSizeW(),
  //    call fftx->pad() if changed between calls to convolve()
  // V is an optional work array of size B*fftx->workSizeV(A,B)
  ConvolutionHermitian2(fftPadCentered *fftx,
                        ConvolutionHermitian *convolvey,
                        Complex **F=NULL, Complex *W=NULL, Complex *V=NULL) :
    Convolution2(fftx->Threads()) {
    this->fftx=fftx;
    this->mult=fftx->mult;
    multithread(fftx->l);
    this->convolvey=new Convolution*[threads];
    this->convolvey[0]=convolvey;
    for(unsigned int t=1; t < threads; ++t)
      this->convolvey[t]=new ConvolutionHermitian(convolvey->fft,
                                                  convolvey->A,convolvey->B);
    this->W=W;
    init(F,V);
  }

  ConvolutionHermitian2(fftBase *fftx,  Convolution *convolvey,
                        Complex **F=NULL, Complex *W=NULL, Complex *V=NULL) :
    Convolution2(fftx,convolvey,F,W,V) {}
};

class Convolution3 : public ThreadBase {
public:
  fftBase *fftx,*ffty,*fftz;
  Convolution **convolvez;
  Convolution2 **convolveyz;
  unsigned int Lx,Ly,Lz; // x,y,z dimensions of input data
  unsigned int Sx,Sy; // x stride, y stride
  unsigned int A;
  unsigned int B;
  multiplier *mult;
  double scale;
protected:
  unsigned int lx;    // x dimension of Fx buffer
  unsigned int Q;
  unsigned int qx;
  unsigned int Qx; // number of residues
  unsigned int Rx; // number of residue blocks
  unsigned int r;
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
  unsigned int nloops;
  bool overwrite;
public:
  Indices indices;

  Convolution3(unsigned int threads=fftw::maxthreads) :
    ThreadBase(threads), fftz(NULL), convolvez(NULL), convolveyz(NULL),
    W(NULL), allocateF(false), allocateV(false), allocateW(false),
    loop2(false) {}

  // Lx,Ly,Lz: x,y,z dimensions of input data
  // Mx,My,Mz: x,y,z dimensions of transformed data, including padding
  // A: number of inputs
  // B: number of outputs
  Convolution3(unsigned int Lx, unsigned int Mx,
               unsigned int Ly, unsigned int My,
               unsigned int Lz, unsigned int Mz,
               unsigned int A, unsigned int B, multiplier *mult,
               unsigned int threads=fftw::maxthreads) :
    ThreadBase(threads), mult(mult), W(NULL), allocateW(false) {
    Application appx(A,B,multNone,threads);
    fftx=new fftPad(Lx,Mx,appx,Ly*Lz);
    Application appy(A,B,multNone,appx.Threads(),fftx->l);
    ffty=new fftPad(Ly,My,appy,Lz);
    Application appz(A,B,mult,appy.Threads(),ffty->l);
    fftz=new fftPad(Lz,Mz,appz);
    convolvez=new Convolution*[threads];
    for(unsigned int t=0; t < threads; ++t)
      convolvez[t]=new Convolution(fftz,A,B);
    convolveyz=new Convolution2*[threads];
    for(unsigned int t=0; t < threads; ++t)
      convolveyz[t]=new Convolution2(ffty,convolvez[t]);
    init();
  }

  // F: optional array of max(A,B) work arrays of size fftx->outputSize()
  // W: optional work array of size fftx->workSizeW();
  //    call fftx->pad() if W changed between calls to convolve()
  // V: optional work array of size B*fftx->workSizeV(A,B)
  Convolution3(fftBase *fftx, Convolution2 *convolveyz,
               Complex **F=NULL, Complex *W=NULL, Complex *V=NULL) :
    ThreadBase(fftx->Threads()), fftx(fftx), fftz(NULL), mult(fftx->mult),
    W(W), allocateW(false) {
    multithread(fftx->l);

    fftBase *fftz=convolveyz->convolvey[0]->fft;
    convolvez=new Convolution*[threads];
    convolvez[0]=convolveyz->convolvey[0];
    for(unsigned int t=1; t < threads; ++t)
      convolvez[t]=new Convolution(fftz,convolveyz->A,convolveyz->B);

    ffty=convolveyz->fftx;
    this->convolveyz=new Convolution2*[threads];
    this->convolveyz[0]=convolveyz;
    for(unsigned int t=1; t < threads; ++t)
      this->convolveyz[t]=new Convolution2(ffty,convolvez[t]);

    init(F,V);
  }

  double normalization() {
    return fftx->normalization()*convolveyz[0]->normalization();
  }

  void init(Complex **F=NULL, Complex *V=NULL) {
    A=convolveyz[0]->A;
    B=convolveyz[0]->B;

    overwrite=fftx->Overwrite(A,B);
    if(overwrite) {
      Forward=fftx->ForwardAll;
      Backward=fftx->BackwardAll;
    } else {
      Forward=fftx->Forward;
      Backward=fftx->Backward;
    }

    unsigned int outputSize=fftx->outputSize();
    unsigned int workSizeW=fftx->workSizeW();

    unsigned int N=std::max(A,B);
    allocateF=!F;
    this->F=allocateF ? utils::ComplexAlign(N,outputSize) : F;

    allocateW=!W && !fftx->inplace;
    W=allocateW ? utils::ComplexAlign(workSizeW) : NULL;

    qx=fftx->q;
    Qx=fftx->Q;
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

    if(A > B+extra && !fftx->inplace && workSizeW <= outputSize) {
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

  ~Convolution3() {
    if(allocateF) {
      utils::deleteAlign(F[0]);
      delete [] F;
    }

    if(allocateW)
      utils::deleteAlign(W);

    if(allocateV) {
      for(unsigned int i=0; i < B; ++i)
        utils::deleteAlign(V[i]);
    }

    if(V)
      delete [] V;

    if(loop2)
      delete [] Fp;

    if(fftz) {
      delete convolveyz[0];
      delete convolvez[0];
      delete fftz;
      delete ffty;
      delete fftx;
    }

    if(convolveyz) {
      for(unsigned int t=1; t < threads; ++t)
        delete convolveyz[t];
      delete [] convolveyz;
    }

    if(convolvez) {
      for(unsigned int t=1; t < threads; ++t)
        delete convolvez[t];
      delete [] convolvez;
    }
  }

  void forward(Complex **f, Complex **F, unsigned int rx,
               unsigned int start, unsigned int stop,
               unsigned int offset=0) {
    for(unsigned int a=start; a < stop; ++a) {
      if(Sy == Lz)
        (fftx->*Forward)(f[a]+offset,F[a],rx,W);
      else {
        Complex *fa=f[a]+offset;
        Complex *Fa=F[a];
        for(unsigned int j=0; j < Ly; ++j) {
          unsigned int Syj=Sy*j;
          (fftx->*Forward)(fa+Syj,Fa+Syj,rx,W);
        }
      }
    }
  }

  void subconvolution(Complex **F, unsigned int C,
                      unsigned int stride, unsigned int rx,
                      unsigned int offset=0) {
    unsigned int D=rx == 0 ? fftx->D0 : fftx->D;
    PARALLEL(
      for(unsigned int i=0; i < C; ++i) {
        unsigned int t=ThreadBase::get_thread_num0();
        Convolution2 *cyz=convolveyz[t];
        for(unsigned int d=0; d < D; ++d) {
          cyz->indices.index[1]=fftx->index(rx+d,i);
          cyz->convolveRaw(F,offset+(D*i+d)*stride,&cyz->indices);
        }
      }
      );
  }

  void backward(Complex **F, Complex **f, unsigned int rx,
                unsigned int start, unsigned int stop,
                unsigned int offset=0, Complex *W0=NULL) {
    for(unsigned int b=start; b < stop; ++b) {
      if(Sy == Lz)
        (fftx->*Backward)(F[b],f[b]+offset,rx,W0);
      else {
        Complex *Fb=F[b];
        Complex *fb=f[b]+offset;
        for(unsigned int j=0; j < Ly; ++j) {
          unsigned int Syj=Sy*j;
          (fftx->*Backward)(Fb+Syj,fb+Syj,rx,W0);
        }
      }
    }
    if(W && W == W0) (fftx->*Pad)(W0);
  }

  void normalize(Complex **h, unsigned int offset=0) {
    for(unsigned int b=0; b < B; ++b) {
      Complex *hb=h[b]+offset;
      for(unsigned int i=0; i < Lx; ++i) {
        Complex *hbi=hb+Sx*i;
        for(unsigned int j=0; j < Ly; ++j) {
          Complex *hbij=hbi+Sy*j;
          for(unsigned int k=0; k < Lz; ++k)
            hbij[k] *= scale;
        }
      }
    }
  }

// f is a pointer to A distinct data blocks each of size Lx*Sx,
// shifted by offset.
  void convolveRaw(Complex **f, unsigned int offset=0, Indices *indices=NULL) {
    for(unsigned int t=0; t < threads; ++t)
      convolveyz[t]->indices.copy(indices,2);

    if(overwrite) {
      forward(f,F,0,0,A,offset);
      unsigned int final=fftx->n-1;
      for(unsigned int r=0; r < final; ++r)
        subconvolution(f,lx,Sx,r,offset+Sx*r*lx);
      subconvolution(F,lx,Sx,final);
      backward(F,f,0,0,B,offset,W);
    } else {
      if(loop2) {
        forward(f,F,0,0,A,offset);
        subconvolution(F,lx,Sx,0);
        unsigned int C=A-B;
        unsigned int a=0;
        for(; a+C <= B; a += C) {
          forward(f,Fp,r,a,a+C,offset);
          backward(F,f,0,a,a+C,offset,W0);
        }
        forward(f,Fp,r,a,A,offset);
        subconvolution(Fp,lx,Sx,r);
        backward(Fp,f,r,0,B,offset,W0);
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
          forward(f,F,rx,0,A,offset);
          subconvolution(F,lx,Sx,rx);
          backward(F,h0,rx,0,B,Offset,W);
        }

        if(nloops > 1) {
          for(unsigned int b=0; b < B; ++b) {
            Complex *fb=f[b]+offset;
            Complex *hb=h0[b];
            for(unsigned int i=0; i < Lx; ++i) {
              unsigned int Sxi=Sx*i;
              Complex *fbi=fb+Sxi;
              Complex *hbi=hb+Sxi;
              for(unsigned int j=0; j < Ly; ++j) {
                unsigned int Syj=Sy*j;
                Complex *fbij=fbi+Syj;
                Complex *hbij=hbi+Syj;
                for(unsigned int k=0; k < Lz; ++k)
                  fbij[k]=hbij[k];
              }
            }
          }
        }
      }
    }
  }

  void convolve(Complex **f, unsigned int offset=0) {
    convolveRaw(f,offset);
    normalize(f,offset);
  }
};

class ConvolutionHermitian3 : public Convolution3 {
public:
  ConvolutionHermitian3(unsigned int Lx, unsigned int Mx,
                        unsigned int Ly, unsigned int My,
                        unsigned int Lz, unsigned int Mz,
                        unsigned int A, unsigned int B, multiplier *mult,
                        unsigned int threads=fftw::maxthreads) :
    Convolution3(threads) {
    unsigned int Hz=utils::ceilquotient(Lz,2);
    this->mult=mult;

    Application appx(A,B,multNone,threads);
    fftx=new fftPadCentered(Lx,Mx,appx,Ly*Hz);
    Application appy(A,B,multNone,appx.Threads(),fftx->l);
    ffty=new fftPadCentered(Ly,My,appy,Hz);
    Application appz(A,B,mult,appy.Threads(),ffty->l);
    fftz=new fftPadHermitian(Lz,Mz,appz);
    convolvez=new Convolution*[threads];
    for(unsigned int t=0; t < threads; ++t)
      convolvez[t]=new ConvolutionHermitian(fftz,A,B);
    convolveyz=new Convolution2*[threads];
    for(unsigned int t=0; t < threads; ++t)
      convolveyz[t]=new ConvolutionHermitian2(ffty,convolvez[t]);
    init();
  }

  // F: optional array of max(A,B) work arrays of size fftx->outputSize()
  // W: optional work array of size fftx->workSizeW(),
  //    call fftx->pad() if changed between calls to convolve()
  // V: optional work array of size B*fftx->workSizeV(A,B)
  ConvolutionHermitian3(fftPadCentered *fftx,
                        ConvolutionHermitian2 *convolveyz,
                        Complex **F=NULL, Complex *W=NULL, Complex *V=NULL) :
    Convolution3(fftx->Threads()) {
    this->fftx=fftx;
    this->mult=fftx->mult;
    multithread(fftx->l);

    fftBase *fftz=convolveyz->convolvey[0]->fft;
    convolvez=new Convolution*[threads];
    convolvez[0]=convolveyz->convolvey[0];
    for(unsigned int t=1; t < threads; ++t)
      convolvez[t]=new ConvolutionHermitian(fftz,convolveyz->A,convolveyz->B);

    ffty=convolveyz->fftx;
    this->convolveyz=new Convolution2*[threads];
    this->convolveyz[0]=convolveyz;
    for(unsigned int t=1; t < threads; ++t)
      this->convolveyz[t]=new ConvolutionHermitian2(convolveyz->fftx,
                                                    convolvez[t]);
    this->W=W;
    init(F,V);
  }

  ConvolutionHermitian3(fftBase *fftx, Convolution2 *convolveyz,
                        Complex **F=NULL, Complex *W=NULL, Complex *V=NULL) :
    Convolution3(fftx,convolveyz,F,W,V) {}
};

} //end namespace fftwpp

#endif
