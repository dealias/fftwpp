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

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

namespace fftwpp {

extern const double twopi;

// Constants used for initialization and testing.
const Complex I(0.0,1.0);

bool Test=false;

unsigned int mOption=0;
unsigned int DOption=0;

int IOption=-1;

unsigned int C=1;

unsigned int A=2; // number of inputs
unsigned int B=1; // number of outputs
bool centered=false;

unsigned int surplusFFTsizes=25;

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

unsigned int nextfftsize(unsigned int m)
{
  unsigned int N=-1;
  unsigned int ni=1;
  for(unsigned int i=0; ni < 7*m; ni=pow(7,i), ++i) {
    unsigned int nj=ni;
    for(unsigned int j=0; nj < 5*m; nj=ni*pow(5,j), ++j) {
      unsigned int nk=nj;
      for(unsigned int k=0; nk < 3*m; nk=nj*pow(3,k), ++k) {
        N=min(N,nk*ceilpow2(ceilquotient(m,nk)));
      }
    }
  }
  return N;
}

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
  unsigned int L;
  unsigned int M;
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
  Complex *ZetaHalf;
  bool inplace;
public:

  void common() {
    if(C > 1) D=1;
    inplace=IOption == -1 ? C > 1 : IOption;

    Cm=C*m;
    p=ceilquotient(L,m);
    n=q/p;
    M=q*m;
    Pad=&fftBase::padNone;
  }

  void initZetaq () {
    p=1;
    Q=n=q;
    Zetaq=ComplexAlign(q);
    double twopibyq=twopi/q;
    for(unsigned int r=1; r < q; ++r)
      Zetaq[r]=expi(r*twopibyq);
  }

  void initZetaqm() {
    unsigned int N=m*q;
    double twopibyN=twopi/N;
    Zetaqm=ComplexAlign((q-1)*m)-m;
    for(unsigned int r=1; r < q; ++r) {
      Zetaqm[m*r]=1.0;
      for(unsigned int s=1; s < m; ++s)
        Zetaqm[m*r+s]=expi(r*s*twopibyN);
    }
  }

  class OptBase {
  public:
    unsigned int m,q,D;
    double T;

    virtual double time(unsigned int L, unsigned int M, unsigned int C,
                        unsigned int m, unsigned int q,unsigned int D,
                        Application &app)=0;

    virtual void check(unsigned int L, unsigned int M,
               Application& app, unsigned int C, unsigned int m,
               bool fixed=false, bool mForced=false) {
//    cout << "m=" << m << endl;
      unsigned int q=ceilquotient(M,m);
      unsigned int p=ceilquotient(L,m);

      if(p > 2) return; // Temporary ***********************************

      if(p == q && p > 1 && !mForced) return;

      if(!fixed) {
        unsigned int n=ceilquotient(M,m*p);
        unsigned int q2=p*n;
        if(q2 != q) {
          unsigned int start=DOption > 0 ? min(DOption,n) : 1;
          unsigned int stop=DOption > 0 ? min(DOption,n) : n;
          if(fixed || C > 1) start=stop=1;
          for(unsigned int D=start; D <= stop; D *= 2) {
            if(2*D > stop) D=stop;
//            cout << "q2=" << q2 << endl;
//            cout << "D2=" << D << endl;
            double t=time(L,M,C,m,q2,D,app);
            if(t < T) {
              this->m=m;
              this->q=q2;
              this->D=D;
              T=t;
            }
          }
        }
      }

//      if(p % 2 == 0) q=ceilquotient(2*M,p*m);
      if(p != 2 && q % p != 0) return;

      unsigned int start=DOption > 0 ? min(DOption,q) : 1;
      unsigned int stop=DOption > 0 ? min(DOption,q) : q;
      if(fixed || C > 1) start=stop=1;
      for(unsigned int D=start; D <= stop; D *= 2) {
        if(2*D > stop) D=stop;
//        cout << "q=" << q << endl;
//        cout << "D=" << D << endl;
        double t=time(L,M,C,m,q,D,app);

        if(t < T) {
          this->m=m;
          this->q=q;
          this->D=D;
          T=t;
        }
      }
    }

    OptBase() {}

    // Determine optimal m,q values for padding L data values to
    // size >= M
    // If fixed=true then an FFT of size M is enforced.
    void scan(unsigned int L, unsigned int M, Application& app,
              unsigned int C, bool Explicit=false, bool fixed=false) {
      if(L > M) {
        cerr << "L=" << L << " is greater than M=" << M << "." << endl;
        exit(-1);
      }
      m=M;
      q=1;
      D=1;

      if(Explicit && fixed)
        return;

      T=DBL_MAX;
      unsigned int i=0;

      unsigned int stop=M-1;
      for(unsigned int k=0; k < surplusFFTsizes; ++k)
        stop=nextfftsize(stop+1);

      unsigned int m0=1;

      if(mOption >= 1 && !Explicit)
        check(L,M,app,C,mOption,fixed,true);
      else
        while(true) {
          m0=nextfftsize(m0+1);
          if(Explicit) {
            if(m0 > stop) break;
            if(m0 < M) {++i; continue;}
            M=m0;
          } else if(m0 > stop) break;
//        } else if(m0 > L) break;
          if(!fixed || Explicit || M % m0 == 0)
            check(L,M,app,C,m0,fixed || Explicit);
          ++i;
        }

      unsigned int p=ceilquotient(L,m);
      cout << endl;
      cout << "Optimal values:" << endl;
      cout << "m=" << m << endl;
      cout << "p=" << p << endl;
      cout << "q=" << q << endl;
      cout << "C=" << C << endl;
      cout << "D=" << D << endl;
      cout << "Padding:" << m*p-L << endl;
    }
  };

  fftBase(unsigned int L, unsigned int M, unsigned int C,
         unsigned int m, unsigned int q, unsigned int D) :
    L(L), M(M), C(C), m(m), p(ceilquotient(L,m)), q(q), D(D) {}

  fftBase(unsigned int L, unsigned int M, Application& app,
         unsigned int C=1, bool Explicit=false, bool fixed=false) :
    L(L), M(M), C(C) {}

  void padNone(Complex *W) {}

  virtual void padSingle(Complex *W) {}
  virtual void padMany(Complex *W) {}

  virtual void forwardShifted(Complex *f, Complex *F, unsigned int r, Complex *W) {}
  virtual void backwardShifted(Complex *F, Complex *f, unsigned int r, Complex *W) {}

  virtual void forward(Complex *f, Complex *F)=0;
  virtual void backward(Complex *F, Complex *f)=0;

  virtual void forwardExplicit(Complex *f, Complex *F, unsigned int, Complex *W=NULL)=0;
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

  // FFT input length
  unsigned int length() {
    return max(L,m*p);
  }

  // FFT output length
  unsigned int Length() {
    return q == 1 ? M : m*p;
  }

  unsigned int size() {
    return q == 1 ? M : m*q; // TODO: m*q  Can we make M=m*q in all cases?
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

  void initialize(Complex *f, Complex *g) {
    for(unsigned int j=0; j < L; ++j) {
      Complex f0(j,j+1);
      Complex g0(j,2*j+1);
      unsigned int Cj=C*j;
      Complex *fj=f+Cj;
      Complex *gj=g+Cj;
      for(unsigned int c=0; c < C; ++c) {
        fj[c]=f0;
        gj[c]=g0;
      }
    }
  }

  double meantime(Application& app, double *Stdev=NULL) {
    unsigned int K=1;
    double eps=0.1;

    utils::statistics Stats;
    app.init(*this);
    while(true) {
      Stats.add(app.time(*this,K));
      double mean=Stats.mean();
      double stdev=Stats.stdev();
      if(Stats.count() < 7) continue;
      int threshold=5000;
      if(mean*CLOCKS_PER_SEC < threshold || eps*mean < stdev) {
        K *= 2;
        Stats.clear();
      } else {
        if(Stdev) *Stdev=stdev/K;
        app.clear();
        return mean/K;
      }
    }
    return 0.0;
  }

  double report(Application& app) {
    double stdev;
    cout << endl;

    double mean=meantime(app,&stdev);

    cout << "mean=" << mean << " +/- " << stdev << endl;

    return mean;
  }
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

  ~fftPad() {
    if(q == 1) { // Simplify
      delete fftm;
      delete ifftm;
    } else {
      if(Zetaq) {
        deleteAlign(Zetaq);
      } else if(p > 1) {
        deleteAlign(Zetaqp+p);
        delete fftp;
        delete ifftp;
      }
      deleteAlign(Zetaqm+m);
      delete fftm;
      delete ifftm;
      if(Q % D > 0) {
        delete fftm2;
        delete ifftm2;
      }
    }
  }

  void init() {
    common();
//    if(m > M) M=m;

    if(q == 1) {
      if(C == 1) {
        Forward=&fftBase::forwardExplicit;
        Backward=&fftBase::backwardExplicit;
      } else {
        Forward=&fftBase::forwardExplicitMany;
        Backward=&fftBase::backwardExplicitMany;
      }
      Complex *G;
      G=ComplexAlign(Cm);
      fftm=new mfft1d(m,1,C, C,1, G);
      ifftm=new mfft1d(m,-1,C, C,1, G);
      deleteAlign(G);
      Q=1;
    } else {
      unsigned int N=m*q;
      double twopibyN=twopi/N;
      double twopibyq=twopi/q;

      bool twop=p == 2 && 2*n != q;

      if(twop)
        initZetaq();
      else Zetaq=NULL;

      Complex *G,*H;

      unsigned int d=p*D*C;
      G=ComplexAlign(m*d);
      H=inplace ? G : ComplexAlign(m*d);

      if(twop) {
        unsigned Lm=L-m;
        Zetaqm2=ComplexAlign((q-1)*Lm)-L;
        for(unsigned int r=1; r < q; ++r) {
          for(unsigned int s=m; s < L; ++s)
            Zetaqm2[Lm*r+s]=expi(r*s*twopibyN);
        }

        if(C == 1) {
          Forward=&fftBase::forward2;
          Backward=&fftBase::backward2;
        } else {
          Forward=&fftBase::forward2Many;
          Backward=&fftBase::backward2Many;
        }
      } else if(p > 1) { // Implies L > m
        if(C == 1) {
          Forward=&fftBase::forwardInner;
          Backward=&fftBase::backwardInner;
        } else {
          Forward=&fftBase::forwardInnerMany;
          Backward=&fftBase::backwardInnerMany;
        }
        Q=n;
        Zetaqp=ComplexAlign((n-1)*(p-1))-p;
        for(unsigned int r=1; r < n; ++r)
          for(unsigned int t=1; t < p; ++t)
            Zetaqp[p*r-r+t]=expi(r*t*twopibyq);

        // L'=p, M'=q, m'=p, p'=1, q'=n
        fftp=new mfft1d(p,1,Cm, Cm,1, G);
        ifftp=new mfft1d(p,-1,Cm, Cm,1, G);
      } else { // p == 1
        if(C == 1) {
          Forward=&fftBase::forward;
          Backward=&fftBase::backward;
          if(padding())
            Pad=&fftBase::padSingle;
        } else {
          Forward=&fftBase::forwardMany;
          Backward=&fftBase::backwardMany;
          if(padding())
            Pad=&fftBase::padMany;
        }
        Q=q;
      }

      if(C == 1) {
        fftm=new mfft1d(m,1,d, 1,m, G,H);
        ifftm=new mfft1d(m,-1,d, 1,m, G,H);
      } else {
        fftm=new mfft1d(m,1,C, C,1, G,H);
        ifftm=new mfft1d(m,-1,C, C,1, G,H);
      }

      unsigned int extra=Q % D;
      if(extra > 0) {
        d=p*extra;
        fftm2=new mfft1d(m,1,d, 1,m, G,H);
        ifftm2=new mfft1d(m,-1,d, 1,m, G,H);
      }

      if(!inplace)
        deleteAlign(H);
      deleteAlign(G);

      initZetaqm();
    }

    fftBase::Forward=Forward;
    fftBase::Backward=Backward;
  }

  // Explicitly pad to m.
  void padSingle(Complex *W) {
    unsigned int mp=p*m;
    for(unsigned int d=0; d < D; ++d) {
      Complex *F=W+m*d;
      for(unsigned int s=L; s < mp; ++s)
        F[s]=0.0;
    }
  }

  // Explicitly pad C FFTs to m.
  void padMany(Complex *W) {
    unsigned int mp=p*m;
    for(unsigned int s=L; s < mp; ++s) {
      Complex *F=W+C*s;
      for(unsigned int c=0; c < C; ++c)
        F[c]=0.0;
    }
  }

  void forward(Complex *f, Complex *F) {
    unsigned int b=Cm*p;
    (this->*Pad)(W0);
    for(unsigned int r=0; r < Q; r += D)
      (this->*Forward)(f,F+b*r,r,W0);
  }

  void backward(Complex *F, Complex *f) {
    unsigned int b=Cm*p;
    for(unsigned int r=0; r < Q; r += D)
      (this->*Backward)(F+b*r,f,r,W0);
  }

  void forwardExplicit(Complex *f, Complex *F, unsigned int, Complex *W=NULL)
  {
    for(unsigned int s=0; s < L; ++s)
      F[s]=f[s];
    for(unsigned int s=L; s < M; ++s)
      F[s]=0.0;
    fftm->fft(F);
  }

  void forwardExplicitMany(Complex *f, Complex *F, unsigned int, Complex *W) {
    for(unsigned int s=0; s < L; ++s) {
      Complex *Fs=F+C*s;
      Complex *fs=f+C*s;
      for(unsigned int c=0; c < C; ++c)
        Fs[c]=fs[c];
    }
    padMany(F);
    fftm->fft(F);
  }

  void backwardExplicit(Complex *F, Complex *f, unsigned int, Complex *W) {
    ifftm->fft(F);
    for(unsigned int s=0; s < L; ++s)
      f[s]=F[s];
  }

  void backwardExplicitMany(Complex *F, Complex *f, unsigned int, Complex *W) {
    ifftm->fft(F);
    for(unsigned int s=0; s < L; ++s) {
      Complex *fs=f+C*s;
      Complex *Fs=F+C*s;
      for(unsigned int c=0; c < C; ++c)
        fs[c]=Fs[c];
    }
  }

    // p=1 && C=1
  void forward(Complex *f, Complex *F0, unsigned int r0, Complex *W) {
    if(W == NULL) W=F0;

    unsigned int D0=Q-r0;
    if(D0 > D) D0=D;

    if(W == F0) {
      for(unsigned int d=0; d < D0; ++d) {
        Complex *F=W+m*d;
        for(unsigned int s=L; s < m; ++s)
          F[s]=0.0;
      }
    }

    unsigned int first=r0 == 0;
    if(first) {
      for(unsigned int s=0; s < L; ++s)
        W[s]=f[s];
    }
    for(unsigned int d=first; d < D0; ++d) {
      Complex *F=W+m*d;
      unsigned int r=r0+d;
      F[0]=f[0];
      Complex *Zetar=Zetaqm+m*r;
      for(unsigned int s=1; s < L; ++s)
        F[s]=Zetar[s]*f[s];
    }
    (D0 == D ? fftm : fftm2)->fft(W,F0);
  }

  void forwardMany(Complex *f, Complex *F, unsigned int r, Complex *W) {
    if(W == NULL) W=F;

    if(W == F) {
      for(unsigned int s=L; s < m; ++s) {
        Complex *Fs=W+C*s;
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=0.0;
      }
    }

    if(r == 0) {
      for(unsigned int s=0; s < L; ++s) {
        unsigned Cs=C*s;
        Complex *Fs=W+Cs;
        Complex *fs=f+Cs;
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=fs[c];
      }
    } else {
      for(unsigned int c=0; c < C; ++c)
        W[c]=f[c];
      Complex *Zetar=Zetaqm+m*r;
      for(unsigned int s=1; s < L; ++s) {
        unsigned Cs=C*s;
        Complex *Fs=W+Cs;
        Complex *fs=f+Cs;
        Complex Zetars=Zetar[s];
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=Zetars*fs[c];
      }
    }
    fftm->fft(W,F);
  }

    // p=2 && q odd
  void forward2(Complex *f, Complex *F0, unsigned int r0, Complex *W) {
    if(W == NULL) W=F0;

    unsigned int D0=Q-r0;
    if(D0 > D) D0=D;

    unsigned Lm=L-m;
    unsigned int first=r0 == 0;
    if(first) {
      for(unsigned int s=0; s < Lm; ++s)
        W[s]=f[s]+f[m+s];
      for(unsigned int s=Lm; s < m; ++s)
        W[s]=f[s];
    }
    for(unsigned int d=first; d < D0; ++d) {
      Complex *F=W+m*d;
      unsigned int r=r0+d;
      Complex Zetaqr=Zetaq[r];
      F[0]=f[0]+Zetaqr*f[m];
      Complex *Zetar=Zetaqm+m*r;
      for(unsigned int s=1; s < Lm; ++s)
        F[s]=Zetar[s]*(f[s]+Zetaqr*f[m+s]);
      for(unsigned int s=Lm; s < m; ++s)
        F[s]=Zetar[s]*f[s];
    }
    (D0 == D ? fftm : fftm2)->fft(W,F0);
  }

  void forward2Many(Complex *f, Complex *F, unsigned int r, Complex *W) {
    if(W == NULL) W=F;

    unsigned Lm=L-m;
    if(r == 0) {
      for(unsigned int s=0; s < Lm; ++s) {
        unsigned int Cs=C*s;
        Complex *Fs=W+Cs;
        Complex *fs=f+Cs;
        Complex *fms=f+Cm+Cs;
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=fs[c]+fms[c];
      }
      for(unsigned int s=Lm; s < m; ++s) {
        unsigned int Cs=C*s;
        Complex *Fs=W+Cs;
        Complex *fs=f+Cs;
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=fs[c];
      }
    } else {
      Complex Zetaqr=Zetaq[r];
      Complex *fm=f+Cm;
      for(unsigned int c=0; c < C; ++c)
        W[c]=f[c]+Zetaqr*fm[c];
      Complex *Zetar=Zetaqm+m*r;
      for(unsigned int s=1; s < Lm; ++s) {
        unsigned int Cs=C*s;
        Complex *Fs=W+Cs;
        Complex *fs=f+Cs;
        Complex *fms=f+Cm+Cs;
        Complex Zetars=Zetar[s];
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=Zetars*(fs[c]+Zetaqr*fms[c]);
      }
      for(unsigned int s=Lm; s < m; ++s) {
        unsigned int Cs=C*s;
        Complex *Fs=W+Cs;
        Complex *fs=f+Cs;
        Complex Zetars=Zetar[s];
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=Zetars*fs[c];
      }
    }
    fftm->fft(W,F);
  }

  void forwardInner(Complex *f, Complex *F0, unsigned int r0, Complex *W) {
    if(W == NULL) W=F0;

    unsigned int D0=Q-r0;
    if(D0 > D) D0=D;

    unsigned int first=r0 == 0;
    unsigned int pm1=p-1;
    unsigned int stop=L-m*pm1;

    if(first) {
      for(unsigned int t=0; t < pm1; ++t) {
        unsigned int mt=m*t;
        Complex *Ft=W+mt;
        Complex *ft=f+mt;
        for(unsigned int s=0; s < m; ++s)
          Ft[s]=ft[s];
      }

      unsigned int mt=m*pm1;
      Complex *Ft=W+mt;
      Complex *ft=f+mt;
      for(unsigned int s=0; s < stop; ++s)
        Ft[s]=ft[s];
      for(unsigned int s=stop; s < m; ++s)
        Ft[s]=0.0;

      fftp->fft(W);
      for(unsigned int t=1; t < p; ++t) {
        unsigned int R=n*t;
        Complex *Ft=W+m*t;
        Complex *Zetar=Zetaqm+m*R;
        for(unsigned int s=1; s < m; ++s)
          Ft[s] *= Zetar[s];
      }
    }

    unsigned int b=m*p;
    for(unsigned int d=first; d < D0; ++d) {
      Complex *F=W+b*d;
      unsigned int r=r0+d;
      for(unsigned int s=0; s < m; ++s)
        F[s]=f[s];
      Complex *Zetaqr=Zetaqp+pm1*r;
      for(unsigned int t=1; t < pm1; ++t) {
        unsigned int mt=m*t;
        Complex *Ft=F+mt;
        Complex *ft=f+mt;
        Complex Zeta=Zetaqr[t];
        for(unsigned int s=0; s < m; ++s)
          Ft[s]=Zeta*ft[s];
      }
      unsigned int mt=m*pm1;
      Complex *Ft=F+mt;
      Complex *ft=f+mt;
      Complex Zeta=Zetaqr[pm1];
      for(unsigned int s=0; s < stop; ++s)
        Ft[s]=Zeta*ft[s];
      for(unsigned int s=stop; s < m; ++s)
        Ft[s]=0.0;

      fftp->fft(F);
      for(unsigned int t=0; t < p; ++t) {
        unsigned int R=n*t+r;
        Complex *Ft=F+m*t;
        Complex *Zetar=Zetaqm+m*R;
        for(unsigned int s=1; s < m; ++s)
          Ft[s] *= Zetar[s];
      }
    }
    (D0 == D ? fftm : fftm2)->fft(W,F0);
  }

  void forwardInnerMany(Complex *f, Complex *F, unsigned int r, Complex *W) {
    if(W == NULL) W=F;

    unsigned int pm1=p-1;
    unsigned int stop=L-m*pm1;

    if(r == 0) {
      for(unsigned int t=0; t < pm1; ++t) {
        unsigned int Cmt=Cm*t;
        Complex *Ft=W+Cmt;
        Complex *ft=f+Cmt;
        for(unsigned int s=0; s < m; ++s) {
          unsigned int Cs=C*s;
          Complex *Fts=Ft+Cs;
          Complex *fts=ft+Cs;
          for(unsigned int c=0; c < C; ++c)
            Fts[c]=fts[c];
        }
      }
      unsigned int Cmt=Cm*pm1;
      Complex *Ft=W+Cmt;
      Complex *ft=f+Cmt;
      for(unsigned int s=0; s < stop; ++s) {
        unsigned int Cs=C*s;
        Complex *Fts=Ft+Cs;
        Complex *fts=ft+Cs;
        for(unsigned int c=0; c < C; ++c)
          Fts[c]=fts[c];
      }
      for(unsigned int s=stop; s < m; ++s) {
        Complex *Fts=Ft+C*s;
        for(unsigned int c=0; c < C; ++c)
          Fts[c]=0.0;
      }

      fftp->fft(W);
      for(unsigned int t=1; t < p; ++t) {
        unsigned int R=n*t;
        Complex *Ft=W+Cm*t;
        Complex *Zetar=Zetaqm+m*R;
        for(unsigned int s=1; s < m; ++s) {
          Complex *Fts=Ft+C*s;
          Complex Zetars=Zetar[s];
          for(unsigned int c=0; c < C; ++c)
            Fts[c] *= Zetars;
        }
      }
    } else {
      for(unsigned int s=0; s < m; ++s) {
        unsigned int Cs=C*s;
        Complex *Fs=W+Cs;
        Complex *fs=f+Cs;
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=fs[c];
      }
      Complex *Zetaqr=Zetaqp+pm1*r;
      for(unsigned int t=1; t < pm1; ++t) {
        unsigned int Cmt=Cm*t;
        Complex *Ft=W+Cmt;
        Complex *ft=f+Cmt;
        Complex Zeta=Zetaqr[t];
        for(unsigned int s=0; s < m; ++s) {
          unsigned int Cs=C*s;
          Complex *Fts=Ft+Cs;
          Complex *fts=ft+Cs;
          for(unsigned int c=0; c < C; ++c)
            Fts[c]=Zeta*fts[c];
        }
      }
      Complex *Ft=W+Cm*pm1;
      Complex *ft=f+Cm*pm1;
      Complex Zeta=Zetaqr[pm1];
      for(unsigned int s=0; s < stop; ++s) {
        unsigned int Cs=C*s;
        Complex *Fts=Ft+Cs;
        Complex *fts=ft+Cs;
        for(unsigned int c=0; c < C; ++c)
          Fts[c]=Zeta*fts[c];
      }
      for(unsigned int s=stop; s < m; ++s) {
        Complex *Fts=Ft+C*s;
        for(unsigned int c=0; c < C; ++c)
          Fts[c]=0.0;
      }

      fftp->fft(W);
      for(unsigned int t=0; t < p; ++t) {
        unsigned int R=n*t+r;
        Complex *Ft=W+Cm*t;
        Complex *Zetar=Zetaqm+m*R;
        for(unsigned int s=1; s < m; ++s) {
          Complex *Fts=Ft+C*s;
          Complex Zetars=Zetar[s];
          for(unsigned int c=0; c < C; ++c)
            Fts[c] *= Zetars;
        }
      }
    }
    for(unsigned int t=0; t < p; ++t) {
      unsigned int Cmt=Cm*t;
      fftm->fft(W+Cmt,F+Cmt);
    }
  }

// Compute an inverse fft of length N=m*q unpadded back
// to size m*p >= L.
// input and output arrays must be distinct
// Input F destroyed
  void backward(Complex *F0, Complex *f, unsigned int r0, Complex *W) {
    if(W == NULL) W=F0;

    unsigned int D0=Q-r0;
    if(D0 > D) D0=D;

    (D0 == D ? ifftm : ifftm2)->fft(F0,W);

    unsigned int first=r0 == 0;
    if(first) {
      for(unsigned int s=0; s < L; ++s)
        f[s]=W[s];
    }
    for(unsigned int d=first; d < D0; ++d) {
      Complex *F=W+m*d;
      unsigned int r=r0+d;
      f[0] += F[0];
      Complex *Zetamr=Zetaqm+m*r;
      for(unsigned int s=1; s < L; ++s)
        f[s] += conj(Zetamr[s])*F[s];
    }
  }

  void backwardMany(Complex *F, Complex *f, unsigned int r, Complex *W) {
    if(W == NULL) W=F;

    ifftm->fft(F,W);

    if(r == 0) {
      for(unsigned int s=0; s < L; ++s) {
        unsigned int Cs=C*s;
        Complex *fs=f+Cs;
        Complex *Fs=W+Cs;
        for(unsigned int c=0; c < C; ++c)
          fs[c]=Fs[c];
      }
    } else {
      for(unsigned int c=0; c < C; ++c)
        f[c] += W[c];
      Complex *Zetamr=Zetaqm+m*r;
      for(unsigned int s=1; s < L; ++s) {
        unsigned int Cs=C*s;
        Complex *fs=f+Cs;
        Complex *Fs=W+Cs;
        Complex Zetamrs=Zetamr[s];
        for(unsigned int c=0; c < C; ++c)
          fs[c] += conj(Zetamrs)*Fs[c];
      }
    }
  }

  void backward2(Complex *F0, Complex *f, unsigned int r0, Complex *W)
  {
    if(W == NULL) W=F0;

    unsigned int D0=Q-r0;
    if(D0 > D) D0=D;

    (D0 == D ? ifftm : ifftm2)->fft(F0,W);

    unsigned int first=r0 == 0;
    if(first) {
      for(unsigned int s=0; s < m; ++s)
        f[s]=W[s];
      Complex *Wm=W-m;
      for(unsigned int s=m; s < L; ++s)
        f[s]=Wm[s];
    }
    unsigned int Lm=L-m;
    for(unsigned int d=first; d < D0; ++d) {
      Complex *F=W+m*d;
      unsigned int r=r0+d;
      f[0] += F[0];
      Complex *Zetamr=Zetaqm+m*r;
      for(unsigned int s=1; s < m; ++s)
        f[s] += conj(Zetamr[s])*F[s];
      Complex *Zetamr2=Zetaqm2+Lm*r;
      Complex *Fm=F-m;
      for(unsigned int s=m; s < L; ++s)
        f[s] += conj(Zetamr2[s])*Fm[s];  // Use a separate table for this case?
    }
  }

  void backward2Many(Complex *F, Complex *f, unsigned int r, Complex *W) {
    if(W == NULL) W=F;

    ifftm->fft(F,W);

    if(r == 0) {
      for(unsigned int s=0; s < m; ++s) {
        unsigned int Cs=C*s;
        Complex *fs=f+Cs;
        Complex *Fs=W+Cs;
        for(unsigned int c=0; c < C; ++c)
          fs[c]=Fs[c];
      }
      Complex *WCm=W-Cm;
      for(unsigned int s=m; s < L; ++s) {
        unsigned int Cs=C*s;
        Complex *fs=f+Cs;
        Complex *Fs=WCm+Cs;
        for(unsigned int c=0; c < C; ++c)
          fs[c]=Fs[c];
      }
    } else {
      unsigned int Lm=L-m;
      for(unsigned int c=0; c < C; ++c)
        f[c] += W[c];
      Complex *Zetamr=Zetaqm+m*r;
      for(unsigned int s=1; s < m; ++s) {
        unsigned int Cs=C*s;
        Complex *fs=f+Cs;
        Complex *Fs=W+Cs;
        Complex Zetamrs=conj(Zetamr[s]);
        for(unsigned int c=0; c < C; ++c)
          fs[c] += Zetamrs*Fs[c];
      }
      Complex *Zetamr2=Zetaqm2+Lm*r;
      Complex *WCm=W-Cm;
      for(unsigned int s=m; s < L; ++s) {
        unsigned int Cs=C*s;
        Complex *fs=f+Cs;
        Complex *Fs=WCm+Cs;
        Complex Zetamrs2=conj(Zetamr2[s]);
        for(unsigned int c=0; c < C; ++c)
          fs[c] += Zetamrs2*Fs[c];
      }
    }
  }

  void backwardInner(Complex *F0, Complex *f, unsigned int r0, Complex *W) {
    if(W == NULL) W=F0;

    unsigned int D0=Q-r0;
    if(D0 > D) D0=D;

    (D0 == D ? ifftm : ifftm2)->fft(F0,W);

    unsigned int first=r0 == 0;
    unsigned int pm1=p-1;
    unsigned int stop=L-m*pm1;

    if(first) {
      for(unsigned int t=1; t < p; ++t) {
        unsigned int R=n*t;
        Complex *Ft=W+m*t;
        Complex *Zetar=Zetaqm+m*R;
        for(unsigned int s=1; s < m; ++s)
          Ft[s] *= conj(Zetar[s]);
      }
      ifftp->fft(W);
      for(unsigned int t=0; t < pm1; ++t) {
        unsigned int mt=m*t;
        Complex *ft=f+mt;
        Complex *Ft=W+mt;
        for(unsigned int s=0; s < m; ++s)
          ft[s]=Ft[s];
      }
      unsigned int mt=m*pm1;
      Complex *ft=f+mt;
      Complex *Ft=W+mt;
      for(unsigned int s=0; s < stop; ++s)
        ft[s]=Ft[s];
    }

    unsigned int b=m*p;
    for(unsigned int d=first; d < D0; ++d) {
      Complex *F=W+b*d;
      unsigned int r=r0+d;
      for(unsigned int t=0; t < p; ++t) {
        unsigned int R=n*t+r;
        Complex *Ft=F+m*t;
        Complex *Zetar=Zetaqm+m*R;
        for(unsigned int s=1; s < m; ++s)
          Ft[s] *= conj(Zetar[s]);
      }
      ifftp->fft(F);
      for(unsigned int s=0; s < m; ++s)
        f[s] += F[s];
      Complex *Zetaqr=Zetaqp+pm1*r;
      for(unsigned int t=1; t < pm1; ++t) {
        unsigned int mt=m*t;
        Complex *ft=f+mt;
        Complex *Ft=F+mt;
        Complex Zeta=conj(Zetaqr[t]);
        for(unsigned int s=0; s < m; ++s)
          ft[s] += Zeta*Ft[s];
      }
      unsigned int mt=m*pm1;
      Complex *Ft=F+mt;
      Complex *ft=f+mt;
      Complex Zeta=conj(Zetaqr[pm1]);
      for(unsigned int s=0; s < stop; ++s)
        ft[s] += Zeta*Ft[s];
    }
  }

  void backwardInnerMany(Complex *F, Complex *f, unsigned int r, Complex *W) {
    if(W == NULL) W=F;

    for(unsigned int t=0; t < p; ++t) {
      unsigned int Cmt=Cm*t;
      ifftm->fft(F+Cmt,W+Cmt);
    }
    unsigned int pm1=p-1;
    unsigned int stop=L-m*pm1;

    if(r == 0) {
      for(unsigned int t=1; t < p; ++t) {
        unsigned int R=n*t;
        Complex *Ft=W+Cm*t;
        Complex *Zetar=Zetaqm+m*R;
        for(unsigned int s=1; s < m; ++s) {
          Complex *Fts=Ft+C*s;
          Complex Zetars=Zetar[s];
          for(unsigned int c=0; c < C; ++c)
            Fts[c] *= conj(Zetar[s]);
        }
      }
      ifftp->fft(W);
      for(unsigned int t=0; t < pm1; ++t) {
        unsigned int Cmt=Cm*t;
        Complex *ft=f+Cmt;
        Complex *Ft=W+Cmt;
        for(unsigned int s=0; s < m; ++s) {
          unsigned int Cs=C*s;
          Complex *fts=ft+Cs;
          Complex *Fts=Ft+Cs;
          for(unsigned int c=0; c < C; ++c)
            fts[c]=Fts[c];
        }
      }
      unsigned int Cmt=Cm*pm1;
      Complex *ft=f+Cmt;
      Complex *Ft=W+Cmt;
      for(unsigned int s=0; s < stop; ++s) {
        unsigned int Cs=C*s;
        Complex *fts=ft+Cs;
        Complex *Fts=Ft+Cs;
        for(unsigned int c=0; c < C; ++c)
          fts[c]=Fts[c];
      }
    } else {
      for(unsigned int t=0; t < p; ++t) {
        unsigned int R=n*t+r;
        Complex *Ft=W+Cm*t;
        Complex *Zetar=Zetaqm+m*R;
        for(unsigned int s=1; s < m; ++s) {
          Complex *Fts=Ft+C*s;
          Complex Zetars=conj(Zetar[s]);
          for(unsigned int c=0; c < C; ++c)
            Fts[c] *= Zetars;
        }
      }
      ifftp->fft(W);
      for(unsigned int s=0; s < m; ++s) {
        unsigned int Cs=C*s;
        Complex *fs=f+Cs;
        Complex *Fs=W+Cs;
        for(unsigned int c=0; c < C; ++c)
          fs[c] += Fs[c];
      }
      Complex *Zetaqr=Zetaqp+pm1*r;
      for(unsigned int t=1; t < pm1; ++t) {
        unsigned int Cmt=Cm*t;
        Complex *ft=f+Cmt;
        Complex *Ft=W+Cmt;
        Complex Zeta=conj(Zetaqr[t]);
        for(unsigned int s=0; s < m; ++s) {
          unsigned int Cs=C*s;
          Complex *fts=ft+Cs;
          Complex *Fts=Ft+Cs;
          for(unsigned int c=0; c < C; ++c)
            fts[c] += Zeta*Fts[c];
        }
      }
      Complex *ft=f+Cm*pm1;
      Complex *Ft=W+Cm*pm1;
      Complex Zeta=conj(Zetaqr[pm1]);
      for(unsigned int s=0; s < stop; ++s) {
        unsigned int Cs=C*s;
        Complex *fts=ft+Cs;
        Complex *Fts=Ft+Cs;
        for(unsigned int c=0; c < C; ++c)
          fts[c] += Zeta*Fts[c];
      }
    }
  }
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
      deleteAlign(ZetaShift);
  }

  void init() {
    if(Zetaq && q > 1) {
      if(C == 1) {
        Forward=&fftBase::forward2;
        Backward=&fftBase::backward2;
      } else {
        Forward=&fftBase::forward2Many;
        Backward=&fftBase::backward2Many;
      }
      fftBase::Forward=Forward;
      fftBase::Backward=Backward;
    } else {
      initShift();
      Forward=fftBase::Forward;
      Backward=fftBase::Backward;

      fftBase::Forward=&fftBase::forwardShifted;
      fftBase::Backward=&fftBase::backwardShifted;
    }
  }

  void initShift() {
    ZetaShift=ComplexAlign(M);
    double factor=L/2*twopi/M;
    for(unsigned int r=0; r < q; ++r) {
      Complex *Zetar=ZetaShift+r;
      for(unsigned int s=0; s < m; ++s) {
        Zetar[q*s]=expi(factor*(q*s+r));
      }
    }
  }

  void forwardShifted(Complex *f, Complex *F, unsigned int r, Complex *W) {
    (this->*Forward)(f,F,r,W);
    forwardShift(F,r);
  }

  void backwardShifted(Complex *F, Complex *f, unsigned int r, Complex *W) {
    backwardShift(F,r);
    (this->*Backward)(F,f,r,W);
  }

  void forwardShift(Complex *F, unsigned int r0) {
    unsigned int D0=Q-r0;
    if(D0 > D) D0=D;
    for(unsigned int d=0; d < D0; ++d) {
      Complex *W=F+m*d;
      unsigned int r=r0+d;
      for(unsigned int t=0; t < p; ++t) {
        Complex *Zetar=ZetaShift+n*t+r;
        Complex *Wt=W+Cm*t;
        for(unsigned int s=0; s < m; ++s) {
          Complex Zeta=conj(Zetar[q*s]);
          for(unsigned int c=0; c < C; ++c)
            Wt[C*s+c] *= Zeta;
        }
      }
    }
  }

  void backwardShift(Complex *F, unsigned int r0) {
    unsigned int D0=Q-r0;
    if(D0 > D) D0=D;
    for(unsigned int d=0; d < D0; ++d) {
      Complex *W=F+m*d;
      unsigned int r=r0+d;
      for(unsigned int t=0; t < p; ++t) {
        Complex *Zetar=ZetaShift+n*t+r;
        Complex *Wt=W+Cm*t;
        for(unsigned int s=0; s < m; ++s) {
          Complex Zeta=Zetar[q*s];
          for(unsigned int c=0; c < C; ++c)
            Wt[C*s+c] *= Zeta;
        }
      }
    }
  }

  // p=2 && q odd
  void forward2(Complex *f, Complex *F0, unsigned int r0, Complex *W) {
    if(W == NULL) W=F0;

    unsigned int D0=Q-r0;
    if(D0 > D) D0=D;

    unsigned int H=L/2;
    unsigned int mH=m-H;
    unsigned int LH=L-H;
    unsigned int first=r0 == 0;
    Complex *fmH=f-mH;
    Complex *fH=f+H;
    if(first) {
      for(unsigned int s=0; s < mH; ++s)
        W[s]=fH[s];
      for(unsigned int s=mH; s < LH; ++s)
        W[s]=fmH[s]+fH[s];
      for(unsigned int s=LH; s < m; ++s)
        W[s]=fmH[s];
    }
    for(unsigned int d=first; d < D0; ++d) {
      Complex *F=W+m*d;
      unsigned int r=r0+d;
      Complex Zetaqr=conj(Zetaq[r]);
      Complex *Zetar=Zetaqm+m*r;
      for(unsigned int s=0; s < mH; ++s)
        F[s]=Zetar[s]*fH[s];
      for(unsigned int s=mH; s < LH; ++s)
        F[s]=Zetar[s]*(Zetaqr*fmH[s]+fH[s]);
      for(unsigned int s=LH; s < m; ++s)
        F[s]=Zetar[s]*Zetaqr*fmH[s];
    }
    (D0 == D ? fftm : fftm2)->fft(W,F0);
  }

  void forward2Many(Complex *f, Complex *F, unsigned int r, Complex *W)
  {
    if(W == NULL) W=F;

    unsigned int H=L/2;
    unsigned int mH=m-H;
    unsigned int LH=L-H;
    Complex *fH=f+C*H;
    Complex *fmH=f-C*mH;
    if(r == 0) {
      for(unsigned int s=0; s < mH; ++s) {
        unsigned int Cs=C*s;
        Complex *Fs=W+Cs;
        Complex *fHs=fH+Cs;
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=fHs[c];
      }
      for(unsigned int s=mH; s < LH; ++s) {
        unsigned int Cs=C*s;
        Complex *Fs=W+Cs;
        Complex *fmHs=fmH+Cs;
        Complex *fHs=fH+Cs;
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=fmHs[c]+fHs[c];
      }
      for(unsigned int s=LH; s < m; ++s) {
        unsigned int Cs=C*s;
        Complex *Fs=W+Cs;
        Complex *fmHs=fmH+Cs;
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=fmHs[c];
      }
    } else {
      Complex Zetaqr=conj(Zetaq[r]);
      Complex *Zetar=Zetaqm+m*r;
      for(unsigned int s=0; s < mH; ++s) {
        unsigned int Cs=C*s;
        Complex *Fs=W+Cs;
        Complex *fHs=fH+Cs;
        Complex Zetars=Zetar[s];
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=Zetars*fHs[c];
      }
      for(unsigned int s=mH; s < LH; ++s) {
        unsigned int Cs=C*s;
        Complex *Fs=W+Cs;
        Complex *fHs=fH+Cs;
        Complex *fmHs=fmH+Cs;
        Complex Zetars=Zetar[s];
        Complex Zetarsq=Zetars*Zetaqr;
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=Zetarsq*fmHs[c]+Zetars*fHs[c];
      }
      for(unsigned int s=LH; s < m; ++s) {
        unsigned int Cs=C*s;
        Complex *Fs=W+Cs;
        Complex *fmHs=fmH+Cs;
        Complex Zetars=Zetar[s]*Zetaqr;
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=Zetars*fmHs[c];
      }
    }
    fftm->fft(W,F);
  }

  void backward2(Complex *F0, Complex *f, unsigned int r0, Complex *W) {
    if(W == NULL) W=F0;

    unsigned int D0=Q-r0;
    if(D0 > D) D0=D;

    (D0 == D ? ifftm : ifftm2)->fft(F0,W);

    unsigned int H=L/2;
    unsigned int mH=m-H;
    unsigned int LH=L-H;
    unsigned int first=r0 == 0;
    Complex *fmH=f-mH;
    Complex *fH=f+H;
    if(first) {
      for(unsigned int s=mH; s < m; ++s)
        fmH[s]=W[s];
      for(unsigned int s=0; s < LH; ++s)
        fH[s]=W[s];
    }
    for(unsigned int d=first; d < D0; ++d) {
      Complex *F=W+m*d;
      unsigned int r=r0+d;
      Complex Zetaqr=Zetaq[r];
      Complex *Zetamr=Zetaqm+m*r;
      for(unsigned int s=mH; s < m; ++s)
        fmH[s] += conj(Zetamr[s])*Zetaqr*F[s];
      for(unsigned int s=0; s < LH; ++s)
        fH[s] += conj(Zetamr[s])*F[s];
    }
  }

  void backward2Many(Complex *F, Complex *f, unsigned int r, Complex *W) {
    if(W == NULL) W=F;

    ifftm->fft(F,W);

    unsigned int H=L/2;
    unsigned int mH=m-H;
    unsigned int LH=L-H;
    Complex *fmH=f-C*mH;
    Complex *fH=f+C*H;
    if(r == 0) {
      for(unsigned int s=mH; s < m; ++s) {
        unsigned int Cs=C*s;
        Complex *fmHs=fmH+Cs;
        Complex *Fs=W+Cs;
        for(unsigned int c=0; c < C; ++c)
          fmHs[c]=Fs[c];
      }
      for(unsigned int s=0; s < LH; ++s) {
        unsigned int Cs=C*s;
        Complex *fHs=fH+Cs;
        Complex *Fs=W+Cs;
        for(unsigned int c=0; c < C; ++c)
          fHs[c]=Fs[c];
      }
    } else {
      Complex Zetaqr=Zetaq[r];
      Complex *Zetamr=Zetaqm+m*r;
      for(unsigned int s=mH; s < m; ++s) {
        unsigned int Cs=C*s;
        Complex *fmHs=fmH+Cs;
        Complex *Fs=W+Cs;
        Complex Zetamrs=conj(Zetamr[s])*Zetaqr;
        for(unsigned int c=0; c < C; ++c)
          fmHs[c] += Zetamrs*Fs[c];
      }
      for(unsigned int s=0; s < LH; ++s) {
        unsigned int Cs=C*s;
        Complex *fHs=fH+Cs;
        Complex *Fs=W+Cs;
        Complex Zetamrs=conj(Zetamr[s]);
        for(unsigned int c=0; c < C; ++c)
          fHs[c] += Zetamrs*Fs[c];
      }
    }
  }
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

  void init() {
    common();
    e=m/2;
    if(q == 1) {
      if(C == 1) {
        Forward=&fftBase::forwardExplicit;
        Backward=&fftBase::backwardExplicit;
      } else {
        Forward=&fftBase::forwardExplicitMany;
        Backward=&fftBase::backwardExplicitMany;
      }

      Complex *G=ComplexAlign(C*(e+1));

      crfftm=new mcrfft1d(m,C, C,C,1,1, G);
      rcfftm=new mrcfft1d(m,C, C,C,1,1, (double *) G);
      deleteAlign(G);
      Q=1;
    } else {
      bool twop=p == 2;
      if(twop)
        initZetaq();
      else {
        cerr << "Unimplemented!" << endl;
        exit(-1);
      }

      unsigned int size=(e+1)*D*C;
      Complex *G=ComplexAlign(size);
      Complex *H=inplace ? G : ComplexAlign(size);

      if(C == 1) {
        crfftm=new mcrfft1d(m,D, 1,1, e+1,m, G,(double *) H);
        rcfftm=new mrcfft1d(m,D, 1,1, m,e+1, (double *) G,H);
        Forward=&fftBase::forward2;
        Backward=&fftBase::backward2;
      } else {
        crfftm=new mcrfft1d(m,C, C,C, 1,1, G,(double *) H);
        rcfftm=new mrcfft1d(m,C, C,C, 1,1, (double *) G,H);
        Forward=&fftBase::forward2Many;
        Backward=&fftBase::backward2Many;
      }

      unsigned int x=Q % D;
      if(x > 0) {
        crfftm2=new mcrfft1d(m,x, 1,1, e+1,m, G,(double *) H);
        rcfftm2=new mrcfft1d(m,x, 1,1, m,e+1, (double *) G,H);
      }

      if(!inplace)
        deleteAlign(H);
      deleteAlign(G);

      initZetaqm();
    }

    fftBase::Forward=Forward;
    fftBase::Backward=Backward;
  }

  ~fftPadHermitian() {
    delete crfftm;
    delete rcfftm;
  }

  void forward(Complex *f, Complex *F) {
    unsigned int b=C*e;
    for(unsigned int r=0; r < Q; r += D)
      (this->*Forward)(f,F+b*r,r,W0);
  }

  void backward(Complex *F, Complex *f) {
    unsigned int b=C*e;
    for(unsigned int r=0; r < Q; r += D)
      (this->*Backward)(F+b*r,f,r,W0);
  }

  void forwardExplicit(Complex *f, Complex *F, unsigned int, Complex *W) {
    unsigned int H=L/2;
    for(unsigned int s=0; s <= H; ++s)
      F[s]=f[s];
    for(unsigned int s=H+1; s <= e; ++s)
      F[s]=0.0;

    crfftm->fft(F);
  }

  void forwardExplicitMany(Complex *f, Complex *F, unsigned int, Complex *W) {
    unsigned int H=L/2;
    for(unsigned int s=0; s <= H; ++s) {
      Complex *Fs=F+C*s;
      Complex *fs=f+C*s;
      for(unsigned int c=0; c < C; ++c)
        Fs[c]=fs[c];
    }
    for(unsigned int s=H+1; s <= e; ++s) {
      Complex *Fs=F+C*s;
      for(unsigned int c=0; c < C; ++c)
        Fs[c]=0.0;
    }

    crfftm->fft(F);
  }

  void backwardExplicit(Complex *F, Complex *f, unsigned int, Complex *W) {
    unsigned int H=L/2;
    rcfftm->fft(F);
    for(unsigned int s=0; s <= H; ++s)
      f[s]=F[s];
  }

  void backwardExplicitMany(Complex *F, Complex *f, unsigned int, Complex *W) {
    unsigned int H=L/2;
    rcfftm->fft(F);
    for(unsigned int s=0; s <= H; ++s) {
      Complex *fs=f+C*s;
      Complex *Fs=F+C*s;
      for(unsigned int c=0; c < C; ++c)
        fs[c]=Fs[c];
    }
  }

  void forward2(Complex *f, Complex *F0, unsigned int r0, Complex *W) {
    if(W == NULL) W=F0;

    unsigned int D0=Q-r0;
    if(D0 > D) D0=D;

    unsigned int first=r0 == 0;
    Complex *fm=f+m;
    if(first) {
      W[0]=f[0];
      for(unsigned int s=1; s <= e; ++s)
        W[s]=f[s]+conj(*(fm-s));
    }
    unsigned int e1=e+1;
    for(unsigned int d=first; d < D0; ++d) {
      Complex *F=W+e1*d;
      F[0]=f[0];
      unsigned int r=r0+d;
      Complex Zetaqr=Zetaq[r];
      Complex *Zetar=Zetaqm+m*r;
      for(unsigned int s=1; s <= e; ++s)
        F[s]=Zetar[s]*(f[s]+conj(*(fm-s)*Zetaqr));
    }
    (D0 == D ? crfftm : crfftm2)->fft(W,F0);
  }

  void forward2Many(Complex *f, Complex *F, unsigned int r, Complex *W) {
    if(W == NULL) W=F;

    Complex *fm=f+Cm;
    if(r == 0) {
      for(unsigned int c=0; c < C; ++c)
        W[c]=f[c];
      for(unsigned int s=1; s <= e; ++s) {
        unsigned int Cs=C*s;
        Complex *Ws=W+Cs;
        Complex *fs=f+Cs;
        Complex *fms=fm-Cs;
        for(unsigned int c=0; c < C; ++c)
          Ws[c]=fs[c]+conj(fms[c]);
      }
    } else {
      for(unsigned int c=0; c < C; ++c)
        W[c]=f[c];
      Complex Zetaqr=Zetaq[r];
      Complex *Zetar=Zetaqm+m*r;
      for(unsigned int s=1; s <= e; ++s) {
        unsigned int Cs=C*s;
        Complex *Ws=W+Cs;
        Complex *fs=f+Cs;
        Complex *fms=fm-Cs;
        Complex Zetars=Zetar[s];
        for(unsigned int c=0; c < C; ++c)
          Ws[c]=Zetars*(fs[c]+conj(fms[c]*Zetaqr));
      }
    }
    crfftm->fft(W,F);
  }

  void backward2(Complex *F0, Complex *f, unsigned int r0, Complex *W) {
    if(W == NULL) W=F0;

    unsigned int D0=Q-r0;
    if(D0 > D) D0=D;

    Complex Nyquist[D0];
    if(W == F0) {
      for(unsigned int d=0; d < D0; ++d)
        Nyquist[d]=F0[D0*e+d]; // Save before being overwritten
    }

    (D0 == D ? rcfftm : rcfftm2)->fft(F0,W);

    unsigned int first=r0 == 0;
    if(first) {
      for(unsigned int s=0; s <= e; ++s)
        f[s]=W[s];
      for(unsigned int s=1; s < m-e; ++s)
        f[m-s]=conj(W[s]);
    }
    unsigned int e1=e+1;
    for(unsigned int d=first; d < D0; ++d) {
      Complex *F=W+e1*d;
      unsigned int r=r0+d;
      Complex Zetaqr=Zetaq[r];
      Complex *Zetamr=Zetaqm+m*r;
      for(unsigned int s=0; s <= e; ++s)
        f[s] += conj(Zetamr[s])*F[s];
      for(unsigned int s=1; s < m-e; ++s)
        f[m-s] += Zetamr[s]*conj(Zetaqr*F[s]);
    }

    if(W == F0) {
      for(unsigned int d=0; d < D0; ++d)
        F0[D0*e+d]=Nyquist[d]; // Restore initial input of next residue
    }
  }

  void backward2Many(Complex *F, Complex *f, unsigned int r, Complex *W) {
    if(W == NULL) W=F;

    Complex Nyquist[C];
    if(W == F) {
      for(unsigned int c=0; c < C; ++c)
        Nyquist[c]=F[C*e+c]; // Save before being overwritten
    }

    rcfftm->fft(F,W);

    Complex *fm=f+Cm;

    if(r == 0) {
      for(unsigned int s=0; s <= e; ++s) {
        unsigned int Cs=C*s;
        Complex *fs=f+Cs;
        Complex *Ws=W+Cs;
        for(unsigned int c=0; c < C; ++c)
          fs[c]=Ws[c];
      }
      for(unsigned int s=1; s < m-e; ++s) {
        unsigned int Cs=C*s;
        Complex *fms=fm-Cs;
        Complex *Ws=W+Cs;
        for(unsigned int c=0; c < C; ++c)
          fms[c]=conj(Ws[c]);
      }
    } else {
      Complex Zetaqr=conj(Zetaq[r]);
      Complex *Zetamr=Zetaqm+m*r;
      for(unsigned int s=0; s <= e; ++s) {
        unsigned int Cs=C*s;
        Complex *fs=f+Cs;
        Complex *Ws=W+Cs;
        Complex Zetamrs=conj(Zetamr[s]);
        for(unsigned int c=0; c < C; ++c)
          fs[c] += Zetamrs*Ws[c];
      }
      for(unsigned int s=1; s < m-e; ++s) {
        unsigned int Cs=C*s;
        Complex *fms=fm-Cs;
        Complex *Ws=W+Cs;
        Complex Zeta=Zetamr[s]*Zetaqr;
        for(unsigned int c=0; c < C; ++c)
          fms[c] += Zeta*conj(Ws[c]);
      }
    }

    if(W == F) {
      for(unsigned int c=0; c < C; ++c)
        F[C*e+c]=Nyquist[c]; // Restore initial input of next residue
    }
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

  void init(fftBase &fft) {
    Forward=fft.Forward;
    Backward=fft.Backward;
    C=fft.C;
    D=fft.D;
    Q=fft.Q;

    unsigned int L=fft.L;
    unsigned int Lf=C*fft.length();
    unsigned int LF=fft.worksizeF();
    unsigned int E=max(A,B);

    f=new Complex*[E];
    F=new Complex*[E];
    h=new Complex*[B];

    for(unsigned int a=0; a < A; ++a)
      f[a]=ComplexAlign(Lf);

    for(unsigned int e=0; e < E; ++e)
      F[e]=ComplexAlign(LF);

    for(unsigned int b=0; b < B; ++b)
      h[b]=ComplexAlign(Lf);

    W=ComplexAlign(fft.worksizeW());
    (fft.*fft.Pad)(W);

    for(unsigned int a=0; a < E; ++a) {
      Complex *fa=f[a];
      for(unsigned int j=0; j < L; ++j) {
        unsigned int Cj=C*j;
        Complex *faj=fa+Cj;
        for(unsigned int c=0; c < C; ++c) {
          faj[c]=0.0;
        }
      }
    }
  }

  double time(fftBase &fft, unsigned int K) {
    double t0=totalseconds();
    for(unsigned int k=0; k < K; ++k) {
      for(unsigned int r=0; r < Q; r += D) {
        for(unsigned int a=0; a < A; ++a)
          (fft.*Forward)(f[a],F[a],r,W);
        for(unsigned int b=0; b < B; ++b)
          (fft.*Backward)(F[b],h[b],r,W);
      }
    }
    double t=totalseconds();
    return t-t0;
  }

  void clear() {
    if(W) {
      deleteAlign(W);
      W=NULL;
    }

    unsigned int E=max(A,B);

    if(h) {
      for(unsigned int b=0; b < B; ++b)
        deleteAlign(h[b]);
      delete[] h;
      h=NULL;
    }

    if(F) {
      for(unsigned int e=0; e < E; ++e)
        deleteAlign(F[e]);
      delete[] F;
      F=NULL;
    }

    if(f) {
      for(unsigned int a=0; a < A; ++a)
        deleteAlign(f[a]);
      delete[] f;
      f=NULL;
    }
  }
};

typedef void multiplier(Complex **, unsigned int e, unsigned int threads);

// This multiplication routine is for binary convolutions and takes two inputs
// of size e.
// F0[j] *= F1[j];
void multbinary(Complex **F, unsigned int e, unsigned int threads)
{
  Complex *F0=F[0];
  Complex *F1=F[1];

  PARALLEL(
    for(unsigned int j=0; j < e; ++j)
      F0[j] *= F1[j];
    );
}

// L is the number of unpadded Complex data values
// M is the minimum number of padded Complex data values

unsigned int threads=1;

class HybridConvolution {
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
  // F is an optional work array of size max(A,B)*fft->worksizeF(),
  // V is an optional work array of size B*fft->worksizeV() (for inplace usage)
  // W is an optional work array of size fft->worksizeW();
  //   if changed between calls to convolve(), be sure to call pad()
  // TODO: add inplace flag to avoid allocating W.
  HybridConvolution(fftPad &fft, unsigned int A=2, unsigned int B=1,
                    Complex *F=NULL, Complex *V=NULL, Complex *W=NULL) :
    fft(&fft), A(A), B(B), W(W), allocateU(false) {
    Forward=fft.Forward;
    Backward=fft.Backward;

    L=fft.L;
    unsigned int N=fft.size();
    scale=1.0/N;
    c=fft.worksizeF();

    unsigned int K=max(A,B);
    this->F=new Complex*[K];
    if(F) {
      for(unsigned int i=0; i < K; ++i)
        this->F[i]=F+i*c;
    } else {
      allocateU=true;
      for(unsigned int i=0; i < K; ++i)
        this->F[i]=ComplexAlign(c);
    }

    if(fft.q > 1) {
      allocateV=false;
      if(V) {
        this->V=new Complex*[B];
        unsigned int size=fft.worksizeV();
        for(unsigned int i=0; i < B; ++i)
          this->V[i]=V+i*size;
      } else
        this->V=NULL;

      if(!this->W) {
        allocateW=true;
        this->W=ComplexAlign(c);
      }

      Pad=fft.Pad;
      (fft.*Pad)(this->W);

      loop2=fft.loop2(); // Two loops and A > B
      int extra;
      if(loop2) {
        Fp=new Complex*[A];
        Fp[0]=this->F[A-1];
        for(unsigned int a=1; a < A; ++a)
          Fp[a]=this->F[a-1];
        extra=1;
      } else
        extra=0;

      if(A > B+extra) {
        W0=this->F[B];
        Pad=&fftBase::padNone;
      } else {
        W0=this->W;
      }
    }

    Q=fft.Q;
    D=fft.D;
  }

  void initV() {
    allocateV=true;
    V=new Complex*[B];
    unsigned int size=fft->worksizeV();
    for(unsigned int i=0; i < B; ++i)
      V[i]=ComplexAlign(size);
  }

  ~HybridConvolution() {
    if(fft->q > 1) {
      if(allocateW)
        deleteAlign(W);

      if(loop2)
        delete[] Fp;

      if(allocateV) {
        for(unsigned int i=0; i < B; ++i)
          deleteAlign(V[i]);
      }
      if(V)
        delete [] V;
    }

    if(allocateU) {
      unsigned int K=max(A,B);
      for(unsigned int i=0; i < K; ++i)
        deleteAlign(F[i]);
    }
    delete [] F;
  }

  // f is an input array of A pointers to distinct data blocks each of size
  // fft->length()
  // h is an output array of B pointers to distinct data blocks each of size
  // fft->length(), which may coincide with f.
  // offset is applied to each input and output component

  void convolve0(Complex **f, Complex **h, multiplier *mult,
                 unsigned int offset=0) {
    if(fft->q == 1) {
      for(unsigned int a=0; a < A; ++a)
        (fft->*Forward)(f[a]+offset,F[a],0,NULL);
      (*mult)(F,fft->M,threads);
      for(unsigned int b=0; b < B; ++b)
        (fft->*Backward)(F[b],h[b]+offset,0,NULL);
    } else {
      if(loop2) {
        for(unsigned int a=0; a < A; ++a)
          (fft->*Forward)(f[a]+offset,F[a],0,W);
        (*mult)(F,c,threads);

        for(unsigned int b=0; b < B; ++b) {
          (fft->*Forward)(f[b]+offset,Fp[b],D,W);
          (fft->*Backward)(F[b],h[b]+offset,0,W0);
          (fft->*Pad)(W);
        }
        for(unsigned int a=B; a < A; ++a)
          (fft->*Forward)(f[a]+offset,Fp[a],D,W);
        (*mult)(Fp,c,threads);
        Complex *UpB=Fp[B];
        for(unsigned int b=0; b < B; ++b)
          (fft->*Backward)(Fp[b],h[b]+offset,D,UpB);
      } else {
        unsigned int Offset;
        bool useV=h == f && D < Q; // Inplace and more than one loop
        Complex **h0;
        if(useV) {
          if(!V) initV();
          h0=V;
          Offset=0;
        } else {
          Offset=offset;
          h0=h;
        }
        for(unsigned int r=0; r < Q; r += D) {
          for(unsigned int a=0; a < A; ++a)
            (fft->*Forward)(f[a]+offset,F[a],r,W);
          (*mult)(F,c,threads);
          for(unsigned int b=0; b < B; ++b)
            (fft->*Backward)(F[b],h0[b]+Offset,r,W0);
          (fft->*Pad)(W);
        }

        if(useV) {
          for(unsigned int b=0; b < B; ++b) {
            Complex *fb=f[b]+offset;
            Complex *hb=h0[b];
            for(unsigned int i=0; i < L; ++i)
              fb[i]=hb[i];
          }
        }
      }
    }
  }

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

class HybridConvolution2 {
  fftPad *fftx;
  HybridConvolution *convolvey;
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
  HybridConvolution2(fftPad &fftx,
                     HybridConvolution &convolvey, Complex *Fx=NULL) :
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
    unsigned int K=max(A,B);
    this->Fx=new Complex*[K];
    if(Fx) {
      for(unsigned int i=0; i < K; ++i)
        this->Fx[i]=Fx+i*c;
    } else {
      allocateUx=true;
      for(unsigned int i=0; i < K; ++i)
        this->Fx[i]=ComplexAlign(c);
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
    HybridConvolution2(unsigned int Lx, unsigned int Ly,
    unsigned int Mx, unsigned int My,
    unsigned int A, unsigned int B, Complex *Fx=NULL,
    Complex *Fy=NULL, Complex *Vy=NULL, Complex *Wy=NULL): Fy(Fy), Vy(Vy), Wy(Wy) {}
  */

  ~HybridConvolution2() {
    unsigned int K=max(A,B);
    if(allocateUx) {
      for(unsigned int i=0; i < K; ++i)
        deleteAlign(Fx[i]);
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

unsigned int L=512;
unsigned int M=1024;

void usage()
{
  std::cerr << "Options: " << std::endl;
  std::cerr << "-c\t\t test centered FFTs" << std::endl;
  std::cerr << "-h\t\t help" << std::endl;
  std::cerr << "-m\t\t subtransform size" << std::endl;
  std::cerr << "-C\t\t number of padded FFTs to compute" << std::endl;
  std::cerr << "-D\t\t number of blocks to process at a time" << std::endl;
  std::cerr << "-I\t\t use in-place FFTs [by default only for C > 1]" << std::endl;
  std::cerr << "-L\t\t number of physical data values" << std::endl;
  std::cerr << "-M\t\t minimal number of padded data values" << std::endl;
  std::cerr << "-S\t\t number of surplus FFT sizes" << std::endl;
  std::cerr << "-T\t\t number of threads" << std::endl;
}

}

int main(int argc, char* argv[])
{
  fftw::maxthreads=1;//get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

#ifdef __GNUC__
  optind=0;
#endif
  for (;;) {
    int c = getopt(argc,argv,"c:hC:D:I:L:M:O:S:T:m:");
    if (c == -1) break;

    switch (c) {
      case 0:
        break;
      case 'c':
        centered=atoi(optarg) > 0;
        break;
      case 'C':
        C=max(atoi(optarg),1);
        break;
      case 'D':
        DOption=max(atoi(optarg),0);
        break;
      case 'L':
        L=atoi(optarg);
        break;
      case 'I':
        IOption=atoi(optarg) > 0;
        break;
      case 'M':
        M=atoi(optarg);
        break;
      case 'S':
        surplusFFTsizes=atoi(optarg);
        break;
      case 'T':
        fftw::maxthreads=max(atoi(optarg),1);
        break;
      case 'm':
        mOption=max(atoi(optarg),0);
        break;
      case 'h':
      default:
        usage();
        exit(1);
    }
  }

  cout << "L=" << L << endl;
  cout << "M=" << M << endl;
  cout << "C=" << C << endl;

  if(centered)
    cout << "Centered" << endl;

  cout << endl;

  ForwardBackward FB;
  Application *app=&FB;


#if 1
  cout << "Explicit:" << endl;
  // Minimal explicit padding
  fftPad fft0(L,M,*app,C,true,true);

  double mean0=fft0.report(*app);

  // Optimal explicit padding
  fftPad fft1(L,M,*app,C,true,false);
  double mean1=min(mean0,fft1.report(*app));

  // Hybrid padding
  fftPad fft(L,M,*app,C);
//  fftPadCentered fft(L,M,*app,C);

  double mean=fft.report(*app);

  if(mean0 > 0)
    cout << "minimal ratio=" << mean/mean0 << endl;
  cout << endl;

  if(mean1 > 0)
    cout << "optimal ratio=" << mean/mean1 << endl;
  cout << endl;

  unsigned int N=fft.size();

  Complex *f=ComplexAlign(C*fft.length());
  // C*qm
  Complex *F=ComplexAlign(fft.q*fft.worksizeF()/fft.D);// Improve
  fft.W0=ComplexAlign(fft.worksizeW());

//  unsigned int Length=L;
  unsigned int Length=L/2+1; // For Hermitian case

  for(unsigned int c=0; c < C; ++c)
    f[c]=1;
  for(unsigned int j=1; j < Length; ++j)
    for(unsigned int c=0; c < C; ++c)
      f[C*j+c]=Complex(j+1,j+1);

  fft.forward(f,F);

  fftPadHermitian fftH(L,M,*app,C);
  F=ComplexAlign(fftH.q*fftH.worksizeF()/fftH.D);// Improve
//  (fftH.*fftH.Forward)(f,F,0,NULL);
  fftH.W0=ComplexAlign(fftH.worksizeW());
  fftH.forward(f,F);

#if 0 // Hermitian case
  double *Fr=(double *) F;
  for(unsigned int j=0; j < C*fftH.size(); ++j)
    cout << Fr[j] << endl;

  fftH.backward(F,f);

  cout << endl;
  if(L < 30) {
    cout << endl;
    cout << "Inverse:" << endl;
    unsigned int N=fftH.size();
    for(unsigned int j=0; j < (L+1)/2*C; ++j)
      cout << f[j]/N << endl;
    cout << endl;
  }
#endif

#if 0
  for(unsigned int j=0; j < fft.size(); ++j)
    for(unsigned int c=0; c < C; ++c)
      cout << F[C*j+c] << endl;
#endif

  exit(0);

  Complex *f0=ComplexAlign(C*fft.length());
  Complex *F0=ComplexAlign(C*N);

  for(unsigned int j=0; j < fft.size(); ++j)
    for(unsigned int c=0; c < C; ++c)
      F0[C*j+c]=F[C*j+c];

  fft.backward(F0,f0);

  if(L < 30) {
    cout << endl;
    cout << "Inverse:" << endl;
    unsigned int N=fft.size();
    for(unsigned int j=0; j < C*L; ++j)
      cout << f0[j]/N << endl;
    cout << endl;
  }

  Complex *F2=ComplexAlign(N*C);
  fftPad fft2(L,N,C,N,1,1);

  for(unsigned int j=0; j < L; ++j)
    for(unsigned int c=0; c < C; ++c)
      f[C*j+c]=Complex(j+1,0);
  fft2.forward(f,F2);

#if 0
  cout << endl;
  for(unsigned int j=0; j < fft.size(); ++j)
    for(unsigned int c=0; c < C; ++c)
      cout << F2[C*j+c] << endl;
  cout << endl;
#endif

  double error=0.0, norm=0.0;
  double error2=0.0, norm2=0.0;

  unsigned int m=fft.m;
  unsigned int p=fft.p;
  unsigned int n=fft.n;

  for(unsigned int s=0; s < m; ++s) {
    for(unsigned int t=0; t < p; ++t) {
      for(unsigned int r=0; r < n; ++r) {
        for(unsigned int c=0; c < C; ++c) {
          unsigned int i=C*(n*(p*s+t)+r)+c;
          error += abs2(F[C*(m*(p*r+t)+s)+c]-F2[i]);
          norm += abs2(F2[i]);
        }
      }
    }
  }

  for(unsigned int j=0; j < C*L; ++j) {
    error2 += abs2(f0[j]/N-f[j]);
    norm2 += abs2(f[j]);
  }

  if(norm > 0) error=sqrt(error/norm);
  double eps=1e-12;
  if(error > eps || error2 > eps)
    cerr << endl << "WARNING: " << endl;
  cout << "forward error=" << error << endl;
  cout << "backward error=" << error2 << endl;

  exit(-1);
#endif
  {

#if 0
    {
      unsigned int Lx=L;
      unsigned int Ly=Lx;
      unsigned int Mx=M;
      unsigned int My=Mx;

      cout << "Lx=" << Lx << endl;
      cout << "Mx=" << Mx << endl;
      cout << endl;

//      fftPad fftx(Lx,Mx,Ly,Lx,2,1);
      fftPad fftx(Lx,Mx,*app,Ly);

//      fftPad ffty(Ly,My,1,Ly,2,1);
      fftPad ffty(Ly,My,FB,1);

      HybridConvolution convolvey(ffty);

      Complex **f=new Complex *[A];
      Complex **h=new Complex *[B];
      for(unsigned int a=0; a < A; ++a)
        f[a]=ComplexAlign(Lx*Ly);
      for(unsigned int b=0; b < B; ++b)
        h[b]=ComplexAlign(Lx*Ly);

      array2<Complex> f0(Lx,Ly,f[0]);
      array2<Complex> f1(Lx,Ly,f[1]);

      for(unsigned int i=0; i < Lx; ++i) {
        for(unsigned int j=0; j < Ly; ++j) {
          f0[i][j]=Complex(i,j);
          f1[i][j]=Complex(2*i,j+1);
        }
      }

      if(Lx*Ly < 200) {
        for(unsigned int i=0; i < Lx; ++i) {
          for(unsigned int j=0; j < Ly; ++j) {
            cout << f0[i][j] << " ";
          }
          cout << endl;
        }
      }
      HybridConvolution2 Convolve2(fftx,convolvey);

      unsigned int K=1000;
      double t0=totalseconds();

      for(unsigned int k=0; k < K; ++k)
        Convolve2.convolve(f,h,multbinary);

      double t=totalseconds();
      cout << (t-t0)/K << endl;
      cout << endl;

      array2<Complex> h0(Lx,Ly,h[0]);

      Complex sum=0.0;
      for(unsigned int i=0; i < Lx; ++i) {
        for(unsigned int j=0; j < Ly; ++j) {
          sum += h0[i][j];
        }
      }

      cout << "sum=" << sum << endl;
      cout << endl;

      if(Lx*Ly < 200) {
        for(unsigned int i=0; i < Lx; ++i) {
          for(unsigned int j=0; j < Ly; ++j) {
            cout << h0[i][j] << " ";
          }
          cout << endl;
        }
      }
    }
#endif

#if 0
    fftPad fft(L,M);

    unsigned int L0=fft.length();
    Complex *f=ComplexAlign(L0);
    Complex *g=ComplexAlign(L0);

    for(unsigned int j=0; j < L0; ++j) {
#if OUTPUT
      f[j]=Complex(j,j+1);
      g[j]=Complex(j,2*j+1);
#else
      f[j]=0.0;
      g[j]=0.0;
#endif
    }

    HybridConvolution Convolve(fft);

    Complex *F[]={f,g};
//  Complex *h=ComplexAlign(L0);
//  Complex *H[]={h};
#if OUTPUT
    unsigned int K=1;
#else
    unsigned int K=10000;
#endif
    double t0=totalseconds();

    for(unsigned int k=0; k < K; ++k)
      Convolve.convolve(F,F,multbinary);

    double t=totalseconds();
    cout << (t-t0)/K << endl;
    cout << endl;
#if OUTPUT
    for(unsigned int j=0; j < L; ++j)
      cout << F[0][j] << endl;
#endif

#endif

  }

  return 0;
}
