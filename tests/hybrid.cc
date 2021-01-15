// TODO:
// Use power of two m values in certain cases
// Generalize meantime to A and B; use A=B=1 and null mult when C > 1.
//
// Can user request allowing overlap of input and output arrays,
// for possibly reduced performance?

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

class FFTpad {
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

  typedef void (FFTpad::*FFTcall)(Complex *f, Complex *F, unsigned int r, Complex *W);

  typedef void (FFTpad::*FFTPad)(Complex *W);

  FFTcall Forward,Backward;
  FFTPad Pad;

protected:
  mfft1d *fftM,*ifftM;
  mfft1d *fftm,*ifftm;
  mfft1d *fftm2,*ifftm2;
  mfft1d *fftp,*ifftp;
  Complex *Zetaqp;
  Complex *Zetaqm;
  bool inplace;
public:

  void init() {
    if(C > 1) D=1;
    inplace=IOption == -1 ? C > 1 : IOption;

    Cm=C*m;
    p=ceilquotient(L,m);
    n=q/p;
    if(m > M) M=m;
    Pad=&FFTpad::padNone;

    if(q == 1) {
      if(C == 1) {
        Forward=&FFTpad::forwardExplicit1;
        Backward=&FFTpad::backwardExplicit1;
      } else {
        Forward=&FFTpad::forwardExplicit;
        Backward=&FFTpad::backwardExplicit;
      }
      Complex *G=ComplexAlign(C*M);
      fftM=new mfft1d(M,1,C, C,1, G);
      ifftM=new mfft1d(M,-1,C, C,1, G);
      deleteAlign(G);
      Q=1;
    } else {
      unsigned int N=m*q;
      double twopibyN=twopi/N;

      if(p > 1) { // Implies L > m
        if(C == 1) {
          Forward=&FFTpad::forwardInner1;
          Backward=&FFTpad::backwardInner1;
         } else {
          Forward=&FFTpad::forwardInner;
          Backward=&FFTpad::backwardInner;
        }
        Q=n;
        Zetaqp=ComplexAlign((n-1)*(p-1))-p;
        double twopibyq=twopi/q;
        for(unsigned int r=1; r < n; ++r)
          for(unsigned int t=1; t < p; ++t)
            Zetaqp[p*r-r+t]=expi(r*t*twopibyq);
      } else {
        if(C == 1) {
          Forward=&FFTpad::forward1;
          Backward=&FFTpad::backward1;
          if(padding())
            Pad=&FFTpad::pad1;
        } else {
          Forward=&FFTpad::forward;
          Backward=&FFTpad::backward;
          if(padding())
            Pad=&FFTpad::padC;
        }
        Q=q;
        Zetaqp=ComplexAlign((q-1)*(p-1))-p;
        for(unsigned int r=1; r < q; ++r)
          for(unsigned int t=1; t < p; ++t)
            Zetaqp[p*r-r+t]=expi((m*r*t % N)*twopibyN);
      }

      unsigned int d=p*D*C;
      Complex *G=ComplexAlign(m*d);
      Complex *H=inplace ? G : ComplexAlign(m*d);

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

      if(p > 1) {// L'=p, M'=q, m'=p, p'=1, q'=n
        fftp=new mfft1d(p,1,Cm, Cm,1, G);
        ifftp=new mfft1d(p,-1,Cm, Cm,1, G);
      }

      if(!inplace)
        deleteAlign(H);
      deleteAlign(G);

      Zetaqm=ComplexAlign((q-1)*(m-1))-m;
      for(unsigned int r=1; r < q; ++r)
        for(unsigned int s=1; s < m; ++s)
          Zetaqm[m*r-r+s]=expi(r*s*twopibyN);
    }
  }

  // Compute an fft padded to N=m*q >= M >= L
  FFTpad(unsigned int L, unsigned int M, unsigned int C,
         unsigned int m, unsigned int q, unsigned int D) :
    L(L), M(M), C(C), m(m), p(ceilquotient(L,m)), q(q), D(D) {
    init();
  }

  ~FFTpad() {
    if(q == 1) {
      delete fftM;
      delete ifftM;
    } else {
      if(p > 1) {
        delete fftp;
        delete ifftp;
      }
      deleteAlign(Zetaqp+p);
      deleteAlign(Zetaqm+m);
      delete fftm;
      delete ifftm;
      if(Q % D > 0) {
        delete fftm2;
        delete ifftm2;
      }
    }
  }

  class Opt {
  public:
    unsigned int m,q,D;
    double T;

    void check(unsigned int L, unsigned int M, unsigned int C,
               unsigned int m, bool fixed=false, bool mForced=false) {
//    cout << "m=" << m << endl;
      unsigned int p=ceilquotient(L,m);
      unsigned int q=ceilquotient(M,m);

//      if(C > 1 && p > 1) return;
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
            FFTpad fft(L,M,C,m,q2,D);
            double t=fft.meantime();
            if(t < T) {
              this->m=m;
              this->q=q2;
              this->D=D;
              T=t;
            }
          }
        }
      }

      if(q % p != 0) return;

      unsigned int start=DOption > 0 ? min(DOption,q) : 1;
      unsigned int stop=DOption > 0 ? min(DOption,q) : q;
      if(fixed || C > 1) start=stop=1;
      for(unsigned int D=start; D <= stop; D *= 2) {
        if(2*D > stop) D=stop;
//        cout << "q=" << q << endl;
//        cout << "D=" << D << endl;
        FFTpad fft(L,M,C,m,q,D);
        double t=fft.meantime();

        if(t < T) {
          this->m=m;
          this->q=q;
          this->D=D;
          T=t;
        }
      }
    }

    // Determine optimal m,q values for padding L data values to
    // size >= M
    // If fixed=true then an FFT of size M is enforced.
    Opt(unsigned int L, unsigned int M, unsigned int C,
        bool Explicit=false, bool fixed=false)
    {
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
        check(L,M,C,mOption,fixed,true);
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
            check(L,M,C,m0,fixed || Explicit);
          ++i;
        }

      unsigned int p=ceilquotient(L,m);
      cout << endl;
      cout << "Optimal values:" << endl;
      cout << "m=" << m << endl;
      cout << "p=" << p << endl;
      cout << "q=" << q << endl;
      cout << "D=" << D << endl;
      cout << "Padding:" << m*p-L << endl;
    }
  };

  // Normal entry point.
  // Compute C ffts of length L and distance 1 padded to at least M
  // (or exactly M if fixed=true)
  FFTpad(unsigned int L, unsigned int M, unsigned int C=1,
         bool Explicit=false, bool fixed=false) :
    L(L), M(M), C(C) {
    Opt opt=Opt(L,M,C,Explicit,fixed);
    m=opt.m;
    if(Explicit)
      this->M=M=m;
    p=ceilquotient(L,M);
    q=opt.q;
    D=opt.D;
    init();
  }

  void padNone(Complex *W) {}

  // Explicitly pad to m.
  void pad1(Complex *W) {
    for(unsigned int d=0; d < D; ++d) {
      Complex *F=W+m*d;
      for(unsigned int s=L; s < m; ++s)
        F[s]=0.0;
    }
  }

  // Explicitly pad C FFTs to m.
  void padC(Complex *W) {
    for(unsigned int s=L; s < m; ++s) {
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

  void forwardExplicit1(Complex *f, Complex *F, unsigned int, Complex *W=NULL) {
    for(unsigned int s=0; s < L; ++s)
      F[s]=f[s];
    for(unsigned int s=L; s < M; ++s)
      F[s]=0.0;
    fftM->fft(F);
  }

  void forwardExplicit(Complex *f, Complex *F, unsigned int, Complex *W=NULL) {
    for(unsigned int s=0; s < L; ++s) {
      Complex *Fs=F+C*s;
      Complex *fs=f+C*s;
      for(unsigned int c=0; c < C; ++c)
        Fs[c]=fs[c];
    }
    padC(F);
    fftM->fft(F);
  }

  void backwardExplicit1(Complex *F, Complex *f, unsigned int, Complex *W=NULL)
  {
    ifftM->fft(F);
    for(unsigned int s=0; s < L; ++s)
      f[s]=F[s];
  }

  void backwardExplicit(Complex *F, Complex *f, unsigned int,
                            Complex *W=NULL) {
    ifftM->fft(F);
    for(unsigned int s=0; s < L; ++s) {
      Complex *fs=f+C*s;
      Complex *Fs=F+C*s;
      for(unsigned int c=0; c < C; ++c)
        fs[c]=Fs[c];
    }
  }

  void forward1(Complex *f, Complex *F0, unsigned int r0, Complex *W=NULL) {
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
      Complex *Zetar=Zetaqm+m*r-r;
      for(unsigned int s=1; s < L; ++s)
        F[s]=Zetar[s]*f[s];
    }
    (D0 == D ? fftm : fftm2)->fft(W,F0);
  }

  void forward(Complex *f, Complex *F, unsigned int r, Complex *W=NULL) {
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
        Complex *Fs=W+C*s;
        Complex *fs=f+C*s;
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=fs[c];
      }
    } else {
      for(unsigned int c=0; c < C; ++c)
        W[c]=f[c];
      Complex *Zetar=Zetaqm+m*r-r;
      for(unsigned int s=1; s < L; ++s) {
        Complex *Fs=W+C*s;
        Complex *fs=f+C*s;
        Complex Zetars=Zetar[s];
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=Zetars*fs[c];
      }
    }
    fftm->fft(W,F);
  }

  void forwardInner1(Complex *f, Complex *F0, unsigned int r0, Complex *W) {
    if(W == NULL) W=F0;

    unsigned int D0=Q-r0;
    if(D0 > D) D0=D;

    unsigned int first=r0 == 0;
    unsigned int pm1=p-1;
    unsigned stop=L-m*pm1;

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
        Complex *Zetar=Zetaqm+m*R-R;
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
        Complex *Zetar=Zetaqm+m*R-R;
        for(unsigned int s=1; s < m; ++s)
          Ft[s] *= Zetar[s];
      }
    }
    (D0 == D ? fftm : fftm2)->fft(W,F0);
  }

  void forwardInner(Complex *f, Complex *F, unsigned int r, Complex *W) {
    if(W == NULL) W=F;

    unsigned int pm1=p-1;
    unsigned stop=L-m*pm1;

    if(r == 0) {
      for(unsigned int t=0; t < pm1; ++t) {
        unsigned int Cmt=Cm*t;
        Complex *Ft=W+Cmt;
        Complex *ft=f+Cmt;
        for(unsigned int s=0; s < m; ++s) {
          Complex *Fts=Ft+C*s;
          Complex *fts=ft+C*s;
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
        Complex *Zetar=Zetaqm+m*R-R;
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
        Complex *Zetar=Zetaqm+m*R-R;
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
  void backward1(Complex *F0, Complex *f, unsigned int r0, Complex *W=NULL) {
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
      Complex *Zetamr=Zetaqm+m*r-r;
      for(unsigned int s=1; s < L; ++s)
        f[s] += conj(Zetamr[s])*F[s];
    }
  }

  void backward(Complex *F, Complex *f, unsigned int r, Complex *W=NULL) {
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
      Complex *Zetamr=Zetaqm+m*r-r;
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

  void backwardInner1(Complex *F0, Complex *f, unsigned int r0, Complex *W) {
    if(W == NULL) W=F0;

    unsigned int D0=Q-r0;
    if(D0 > D) D0=D;

    (D0 == D ? ifftm : ifftm2)->fft(F0,W);

    unsigned int first=r0 == 0;
    unsigned int pm1=p-1;
    unsigned stop=L-m*pm1;

    if(first) {
      for(unsigned int t=1; t < p; ++t) {
        unsigned int R=n*t;
        Complex *Ft=W+m*t;
        Complex *Zetar=Zetaqm+m*R-R;
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
        Complex *Zetar=Zetaqm+m*R-R;
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

  void backwardInner(Complex *F, Complex *f, unsigned int r, Complex *W) {
    if(W == NULL) W=F;

    for(unsigned int t=0; t < p; ++t) {
      unsigned int Cmt=Cm*t;
      ifftm->fft(F+Cmt,W+Cmt);
    }
    unsigned int pm1=p-1;
    unsigned stop=L-m*pm1;

    if(r == 0) {
      for(unsigned int t=1; t < p; ++t) {
        unsigned int R=n*t;
        Complex *Ft=W+Cm*t;
        Complex *Zetar=Zetaqm+m*R-R;
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
        Complex *Zetar=Zetaqm+m*R-R;
        for(unsigned int s=1; s < m; ++s) {
          Complex Zetars=Zetar[s];
          Complex *Fts=Ft+C*s;
          for(unsigned int c=0; c < C; ++c)
            Fts[c] *= conj(Zetars);
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

  // FFT input length
  unsigned int length() {
    return q == 1 ? L : m*p;
  }

  // FFT output length
  unsigned int Length() {
    return q == 1 ? M : m*p;
  }

  unsigned int size() {
    return q == 1 ? M : m*q;
  }

  unsigned int worksizeU() {
    return C*(q == 1 ? M : m*p*D);
  }

  bool loop2() {
    return D < Q && 2*D >= Q && A > B;
  }

  unsigned int worksizeV() {
    return q == 1 || D >= Q || loop2() ? 0 : length();
  }

  unsigned int worksizeW() {
    return q == 1 || inplace ? 0 : worksizeU();
  }

  unsigned int padding() {
    return !inplace && L < m;
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

  double meantime(double *Stdev=NULL) {
    utils::statistics Stats;
    unsigned int L0=C*length();

    Complex *f=ComplexAlign(L0);
    Complex *g=ComplexAlign(L0);
    Complex *h;
    bool loop2=this->loop2(); // Two loops?
    if(C > 1) loop2=false;
    if(loop2)
      h=f;
    else
      h=D >= Q ? f : ComplexAlign(L0);

    unsigned int e=worksizeU();

    Complex *F=ComplexAlign(e);
    Complex *G=ComplexAlign(e);
    Complex *W=ComplexAlign(worksizeW());

// Assume f != F (out-of-place)
    for(unsigned int j=0; j < L; ++j) {
      unsigned int Cj=C*j;
      Complex *fj=f+Cj;
      Complex *gj=g+Cj;
      for(unsigned int c=0; c < C; ++c) {
        fj[c]=0.0;
        gj[c]=0.0;
      }
    }

    unsigned int K=1;
    double eps=0.1;
    unsigned int N=size();
    double scale=1.0/N;

#define OUTPUT 0
    for(;;) {
      double t0,t;
      if(q == 1) {
        t0=totalseconds();
        if(C == 1) {
          for(unsigned int k=0; k < K; ++k) {
#if OUTPUT
            initialize(f,g);
#endif
            (this->*Forward)(f,F,0,NULL);
            (this->*Forward)(g,G,0,NULL);
            for(unsigned int i=0; i < N; ++i)
              F[i] *= G[i];
            (this->*Backward)(F,f,0,NULL);
          }
          t=totalseconds();
        } else {
          for(unsigned int k=0; k < K; ++k) {
#if OUTPUT
            initialize(f,g);
#endif
            (this->*Forward)(f,F,0,NULL);
            (this->*Forward)(g,G,0,NULL);
            (this->*Backward)(F,f,0,NULL);
          }
          t=totalseconds();
        }
      } else {
        (this->*Pad)(W);

        if(C == 1) {
          Complex *G0=inplace ? NULL : G;
          if(loop2) {
            t0=totalseconds();
            for(unsigned int k=0; k < K; ++k) {
#if OUTPUT
              initialize(f,g);
#endif
              (this->*Forward)(f,F,0,W);
              (this->*Forward)(g,G,0,W);
              for(unsigned int i=0; i < e; ++i)
                F[i] *= G[i];
              (this->*Forward)(f,G,D,W);
              (this->*Backward)(F,f,0,W);
              (this->*Pad)(W);
              (this->*Forward)(g,F,D,W);
              for(unsigned int i=0; i < e; ++i)
                F[i] *= G[i];
              (this->*Backward)(F,f,D,G0);
            }
            t=totalseconds();
          } else {
            t0=totalseconds();
            for(unsigned int k=0; k < K; ++k) {
#if OUTPUT
              initialize(f,g);
#endif
              for(unsigned int r=0; r < Q; r += D) {
                (this->*Forward)(f,F,r,W);
                (this->*Forward)(g,G,r,W);
                for(unsigned int i=0; i < e; ++i)
                  F[i] *= G[i];
                (this->*Backward)(F,h,r,G0);
              }
            }
            t=totalseconds();
          }
        } else {
          t0=totalseconds();
          for(unsigned int k=0; k < K; ++k) {
#if OUTPUT
            initialize(f,g);
#endif
            for(unsigned int r=0; r < Q; ++r) {
              (this->*Forward)(f,F,r,W);
              (this->*Forward)(g,G,r,W);
              (this->*Backward)(F,h,r,W);
            }
          }
          t=totalseconds();
        }
      }

      for(unsigned int j=0; j < L; ++j) {
        Complex *fj=f+C*j;
        Complex *hj=h+C*j;
        for(unsigned int c=0; c < C; ++c)
          fj[c]=hj[c]*scale;
      }
      Stats.add(t-t0);
      double mean=Stats.mean();
      double stdev=Stats.stdev();
      if(Stats.count() < 7) continue;
      int threshold=5000;
      if(mean*CLOCKS_PER_SEC < threshold || eps*mean < stdev) {
        K *= 2;
        Stats.clear();
      } else {
        if(Stdev) *Stdev=stdev/K;
#if OUTPUT
        for(unsigned int j=0; j < L; ++j) {
          Complex *fj=f+C*j;
          for(unsigned int c=0; c < C; ++c)
            cout << fj[c] << endl;
        }
#endif

#undef OUTPUT
#define OUTPUT 1

        deleteAlign(F);
        deleteAlign(f);
        deleteAlign(G);
        deleteAlign(g);
        if(W)
          deleteAlign(W);
        if(D < Q && !loop2)
          deleteAlign(h);
        return mean/K;
      }
    }
    return 0.0;
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
  FFTpad *fft;
  unsigned int A;
  unsigned int B;
  unsigned int L;
private:
  unsigned int Q;
  unsigned int D;
  unsigned int c;
  Complex **U,**Up;
  Complex **V;
  Complex *W;
  Complex *H;
  Complex *W0;
  double scale;
  bool allocateU;
  bool allocateV;
  bool allocateW;
  bool loop2;

  typedef void (FFTpad::*FFTPad)(Complex *W);
  FFTPad Pad;

public:
  // A is the number of inputs.
  // B is the number of outputs.
  // U is an optional work array of size max(A,B)*fft->worksizeU(),
  // V is an optional work array of size B*fft->worksizeV() (for inplace usage)
  // W is an optional work array of size fft->worksizeW();
  //   if changed between calls to convolve(), be sure to call pad()
  // TODO: add inplace flag to avoid allocating W.
  HybridConvolution(FFTpad &fft, unsigned int A=2, unsigned int B=1,
                    Complex *U=NULL, Complex *V=NULL, Complex *W=NULL) :
    fft(&fft), A(A), B(B), W(W), allocateU(false) {
    L=fft.L;
    unsigned int N=fft.size();
    scale=1.0/N;
    c=fft.worksizeU();

    unsigned int K=max(A,B);
    this->U=new Complex*[K];
    if(U) {
      for(unsigned int i=0; i < K; ++i)
        this->U[i]=U+i*c;
    } else {
      allocateU=true;
      for(unsigned int i=0; i < K; ++i)
        this->U[i]=ComplexAlign(c);
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
        Up=new Complex*[A];
        Up[0]=this->U[A-1];
        for(unsigned int a=1; a < A; ++a)
          Up[a]=this->U[a-1];
        extra=1;
      } else
        extra=0;

      if(A > B+extra) {
        W0=this->U[B];
        Pad=&FFTpad::padNone;
      } else {
        W0=this->W;
      }
    }

    Q=fft.Q;
    D=fft.D;
  }

  void initV() {
    allocateV=true;
    this->V=new Complex*[B];
    unsigned int size=fft->worksizeV();
    for(unsigned int i=0; i < B; ++i)
      this->V[i]=ComplexAlign(size);
  }

  ~HybridConvolution() {
    if(fft->q > 1) {
      if(allocateW)
        deleteAlign(W);

      if(allocateV) {
        for(unsigned int i=0; i < B; ++i)
          deleteAlign(V[i]);
      }
      delete [] V;
    }

    if(allocateU) {
      unsigned int K=max(A,B);
      for(unsigned int i=0; i < K; ++i)
        deleteAlign(U[i]);
    }
    delete [] U;
  }

  // F is an input array of A pointers to distinct data blocks each of size
  // fft->length()
  // H is an output array of B pointers to distinct data blocks each of size
  // fft->length(), which may coincide with F.
  // offset is applied to each input and output component

  void convolve0(Complex **F, Complex **H, multiplier *mult,
                 unsigned int offset=0) {
    if(fft->q == 1) {
      for(unsigned int a=0; a < A; ++a)
        (fft->*fft->Forward)(F[a]+offset,U[a],0,NULL);
      (*mult)(U,fft->M,threads);
      for(unsigned int b=0; b < B; ++b)
        (fft->*fft->Backward)(U[b],H[b]+offset,0,NULL);
    } else {
      if(loop2) {
        for(unsigned int a=0; a < A; ++a)
          (fft->*fft->Forward)(F[a]+offset,U[a],0,W);
        (*mult)(U,c,threads);

        for(unsigned int b=0; b < B; ++b) {
          (fft->*fft->Forward)(F[b]+offset,Up[b],D,W);
          (fft->*fft->Backward)(U[b],H[b]+offset,0,W0);
          (fft->*Pad)(W);
        }
        for(unsigned int a=B; a < A; ++a)
          (fft->*fft->Forward)(F[a]+offset,Up[a],D,W);
        (*mult)(Up,c,threads);
        Complex *UpB=Up[B];
        for(unsigned int b=0; b < B; ++b)
          (fft->*fft->Backward)(Up[b],H[b]+offset,D,UpB);
      } else {
        unsigned int Offset;
        bool useV=H == F && D < Q; // Inplace and more than one loop
        Complex **H0;
        if(useV) {
          if(!V) initV();
          H0=V;
          Offset=0;
        } else {
          Offset=offset;
          H0=H;
        }
        for(unsigned int r=0; r < Q; r += D) {
          for(unsigned int a=0; a < A; ++a)
            (fft->*fft->Forward)(F[a]+offset,U[a],r,W);
          (*mult)(U,c,threads);
          for(unsigned int b=0; b < B; ++b)
            (fft->*fft->Backward)(U[b],H0[b]+Offset,r,W0);
          (fft->*Pad)(W);
        }

        if(useV) {
          for(unsigned int b=0; b < B; ++b) {
            Complex *f=F[b]+offset;
            Complex *h0=H0[b];
            for(unsigned int i=0; i < L; ++i)
              f[i]=h0[i];
          }
        }
      }
    }
  }

  void convolve(Complex **F, Complex **H, multiplier *mult,
                unsigned int offset=0) {
    convolve0(F,H,mult,offset);

    for(unsigned int b=0; b < B; ++b) {
      Complex *h=H[b]+offset;
      for(unsigned int i=0; i < L; ++i)
        h[i] *= scale;
    }
  }
};

class HybridConvolution2 {
  FFTpad *fftx;
  HybridConvolution *convolvey;
  unsigned int Sx; // x dimension of Ux buffer
  unsigned int Lx,Ly; // x,y dimensions of input arrays
  unsigned int A;
  unsigned int B;
  unsigned int Q;
  unsigned int D;
  unsigned int qx,Qx;
  Complex **Ux;
  Complex *Uy;
  Complex *Vy;
  Complex *Wy;
  bool allocateUx;
  double scale;

public:
  // Ux is an optional work array of size max(A,B)*fftx->worksizeU(),
  HybridConvolution2(FFTpad &fftx,
                     HybridConvolution &convolvey, Complex *Ux=NULL) :
    fftx(&fftx), convolvey(&convolvey), allocateUx(false) {

    A=convolvey.A;
    B=convolvey.B;

    qx=fftx.q;
    Qx=fftx.Q;
    Sx=fftx.Length();
    scale=1.0/(fftx.size()*convolvey.fft->size());

    unsigned int c=fftx.worksizeU();
    unsigned int K=max(A,B);
    this->Ux=new Complex*[K];
    if(Ux) {
      for(unsigned int i=0; i < K; ++i)
        this->Ux[i]=Ux+i*c;
    } else {
      allocateUx=true;
      for(unsigned int i=0; i < K; ++i)
        this->Ux[i]=ComplexAlign(c);
    }

    Lx=fftx.L;
    Ly=convolvey.L;
  }

  // A is the number of inputs.
  // B is the number of outputs.
  // Uy is an optional work array of size max(A,B)*ffty->worksizeU(),
  // Vy is an optional work array of size B*ffty->worksizeV() (inplace usage)
  // Wy is an optional work array of size ffty->worksizeW();
  //   if changed between calls to convolve(), be sure to call pad()
  /*
  HybridConvolution2(unsigned int Lx, unsigned int Ly,
                     unsigned int Mx, unsigned int My,
                     unsigned int A, unsigned int B, Complex *Ux=NULL,
                     Complex *Uy=NULL, Complex *Vy=NULL, Complex *Wy=NULL): Uy(Uy), Vy(Vy), Wy(Wy) {}
  */

  ~HybridConvolution2() {
    unsigned int K=max(A,B);
    if(allocateUx) {
      for(unsigned int i=0; i < K; ++i)
        deleteAlign(Ux[i]);
    }
    delete [] Ux;
  }

  void forward(Complex **F, Complex **U, unsigned int rx) {
    for(unsigned int a=0; a < A; ++a)
      (fftx->*fftx->Forward)(F[a],U[a],rx,NULL); // C=Ly <= my py, Dx=1
  }

  void subconvolution(Complex **F, multiplier *mult,
                      unsigned int C, unsigned int stride,
                      unsigned int offset=0) {
    for(unsigned int i=0; i < C; ++i)
      convolvey->convolve0(F,F,mult,offset+i*stride);
  }

  void backward(Complex **U, Complex **F, unsigned int rx) {
    // TODO: Support out-of-place
    for(unsigned int b=0; b < B; ++b)
      (fftx->*fftx->Backward)(U[b],F[b],rx,NULL);
  }

// F is a pointer to A distinct data blocks each of size Lx*Ly,
// shifted by offset (contents not preserved).
  virtual void convolve(Complex **F, Complex **H, multiplier *mult,
                        unsigned int offset=0) {
    for(unsigned int rx=0; rx < Qx; ++rx) {
      forward(F,Ux,rx);
      subconvolution(Ux,mult,Sx,Ly,offset);
      backward(Ux,H,rx);
    }
    for(unsigned int b=0; b < B; ++b) {
      Complex *Hb=H[b];
      for(unsigned int i=0; i < Lx; ++i)
        for(unsigned int j=0; j < Ly; ++j)
          Hb[Ly*i+j] *= scale;
    }
  }
};

unsigned int L=683;
unsigned int M=1025;

double report(FFTpad &fft)
{
  double stdev;
  cout << endl;

  double mean=fft.meantime(&stdev);

  cout << "mean=" << mean << " +/- " << stdev << endl;

  return mean;
}

void usage()
{
  std::cerr << "Options: " << std::endl;
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
    int c = getopt(argc,argv,"hC:D:I:L:M:O:S:T:m:");
    if (c == -1) break;

    switch (c) {
      case 0:
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

#if 1
  cout << "Explicit:" << endl;
  // Minimal explicit padding
  FFTpad fft0(L,M,C,true,true);
  double mean0=report(fft0);

  // Optimal explicit padding
  FFTpad fft1(L,M,C,true,false);
  double mean1=min(mean0,report(fft1));

  // Hybrid padding
  FFTpad fft(L,M,C);

  double mean=report(fft);

  if(mean0 > 0)
    cout << "minimal ratio=" << mean/mean0 << endl;
  cout << endl;

  if(mean1 > 0)
    cout << "optimal ratio=" << mean/mean1 << endl;
  cout << endl;

  unsigned int N=fft.size();

  Complex *f=ComplexAlign(C*fft.length());
  Complex *F=ComplexAlign(C*N);
  fft.W0=ComplexAlign(fft.worksizeW());

  for(unsigned int j=0; j < L; ++j)
    for(unsigned int c=0; c < C; ++c)
      f[C*j+c]=j+1;
  fft.forward(f,F);

  Complex *f0=ComplexAlign(C*fft.length());
  Complex *F0=ComplexAlign(C*N);

  for(unsigned int j=0; j < fft.size(); ++j)
    for(unsigned int c=0; c < C; ++c)
      F0[C*j+c]=F[C*j+c];

  fft.backward(F0,f0);

  if(L < 30) {
#if 0
    for(unsigned int j=0; j < fft.size(); ++j)
      for(unsigned int c=0; c < C; ++c)
        cout << F[C*j+c] << endl;
#endif
    cout << endl;
    cout << "Inverse:" << endl;
    unsigned int N=fft.size();
    for(unsigned int j=0; j < C*L; ++j)
      cout << f0[j]/N << endl;
    cout << endl;
  }

  Complex *F2=ComplexAlign(N*C);
  FFTpad fft2(L,N,C,N,1,1);
  for(unsigned int j=0; j < L; ++j)
    for(unsigned int c=0; c < C; ++c)
      f[C*j+c]=j+1;
  fft2.forward(f,F2);

  double error=0.0, norm=0.0;
  double error2=0.0, norm2=0.0;

  unsigned int i=0;
  unsigned int m=fft.m;
  unsigned int p=fft.p;
  unsigned int q=fft.q;
  unsigned int n=fft.n;

  if(q == 1) {
    for(unsigned int i=0; i < C*N; ++i) {
      error += abs2(F[i]-F2[i]);
      norm += abs2(F2[i]);
      ++i;
    }
  } else {
    for(unsigned int s=0; s < m; ++s) {
      for(unsigned int t=0; t < p; ++t) {
        for(unsigned int r=0; r < n; ++r) {
          for(unsigned int c=0; c < C; ++c) {
            error += abs2(F[C*(m*(p*r+t)+s)+c]-F2[i]);
            norm += abs2(F2[i]);
            ++i;
          }
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
#endif
  {

#if 1
    {
      unsigned int Lx=L;
      unsigned int Ly=Lx;
      unsigned int Mx=M;
      unsigned int My=Mx;

      cout << "Lx=" << Lx << endl;
      cout << "Mx=" << Mx << endl;
      cout << endl;

//      FFTpad fftx(Lx,Mx,Ly,Lx,2,1);
      FFTpad fftx(Lx,Mx,Ly);

//      FFTpad ffty(Ly,My,1,Ly,2,1);
      FFTpad ffty(Ly,My,1);

      HybridConvolution convolvey(ffty);

      Complex **F=new Complex *[A];
      Complex **H=new Complex *[B];
      for(unsigned int a=0; a < A; ++a)
        F[a]=ComplexAlign(Lx*Ly);
      for(unsigned int b=0; b < B; ++b)
        H[b]=ComplexAlign(Lx*Ly);

      array2<Complex> f(Lx,Ly,F[0]);
      array2<Complex> g(Lx,Ly,F[1]);

      for(unsigned int i=0; i < Lx; ++i) {
        for(unsigned int j=0; j < Ly; ++j) {
          f[i][j]=Complex(i,j);
          g[i][j]=Complex(2*i,j+1);
        }
      }

      if(Lx*Ly < 200) {
        for(unsigned int i=0; i < Lx; ++i) {
          for(unsigned int j=0; j < Ly; ++j) {
            cout << f[i][j] << " ";
          }
          cout << endl;
        }
      }
      HybridConvolution2 Convolve2(fftx,convolvey);

      unsigned int K=1000;
      double t0=totalseconds();

      for(unsigned int k=0; k < K; ++k)
        Convolve2.convolve(F,H,multbinary);

      double t=totalseconds();
      cout << (t-t0)/K << endl;
      cout << endl;

      array2<Complex> h(Lx,Ly,H[0]);

      Complex sum=0.0;
      for(unsigned int i=0; i < Lx; ++i) {
        for(unsigned int j=0; j < Ly; ++j) {
          sum += h[i][j];
        }
      }

      cout << "sum=" << sum << endl;
      cout << endl;

      if(Lx*Ly < 200) {
        for(unsigned int i=0; i < Lx; ++i) {
          for(unsigned int j=0; j < Ly; ++j) {
            cout << h[i][j] << " ";
          }
          cout << endl;
        }
      }
    }
#endif

#if 0
    FFTpad fft(L,M);

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
