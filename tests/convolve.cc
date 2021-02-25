#include "convolve.h"
#include "cmult-sse2.h"

using namespace std;
using namespace utils;
using namespace Array;

namespace fftwpp {

unsigned int threads=1;

unsigned int mOption=0;
unsigned int DOption=0;

int IOption=-1;

unsigned int A=2; // number of inputs
unsigned int B=1; // number of outputs
unsigned int C=1; // number of copies
unsigned int L;
unsigned int M;

unsigned int surplusFFTsizes=25;

// This multiplication routine is for binary convolutions and takes
// two Complex inputs of size e.
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

// This multiplication routine is for binary convolutions and takes
// two real inputs of size e.
// F0[j] *= F1[j];
void realmultbinary(Complex **F, unsigned int e, unsigned int threads)
{
  double *F0=(double *) F[0];
  double *F1=(double *) F[1];

  PARALLEL(
    for(unsigned int j=0; j < e; ++j)
      F0[j] *= F1[j];
    );
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

void fftBase::OptBase::check(unsigned int L, unsigned int M,
                             Application& app, unsigned int C, unsigned int m,
                             bool fixed, bool mForced)
{
//    cout << "m=" << m << endl;
  unsigned int q=ceilquotient(M,m);
  unsigned int p=ceilquotient(L,m);

//  if(p != 2) return; // Temporary ***********************************

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

void fftBase::OptBase::scan(unsigned int L, unsigned int M, Application& app,
                            unsigned int C, bool Explicit, bool fixed)
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

fftBase::~fftBase()
{
  if(q > 1) {
    if(Zetaq)
      deleteAlign(Zetaq);
    deleteAlign(Zetaqm+m);
  }
}

double fftBase::meantime(Application& app, double *Stdev)
{
  unsigned int K=1;
  double eps=0.1;

  statistics Stats;
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

void fftBase::initialize(Complex *f, Complex *g)
{
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

double fftBase::report(Application& app)
{
  double stdev;
  cout << endl;

  double mean=meantime(app,&stdev);

  cout << "mean=" << mean << " +/- " << stdev << endl;

  return mean;
}

void fftBase::common()
{
  if(C > 1) D=1;
  inplace=IOption == -1 ? C > 1 : IOption;

  Cm=C*m;
  p=ceilquotient(L,m);
  n=q/p;
  M=m*q;
  Pad=&fftBase::padNone;
  Zetaqp=Zetaq=NULL;
}

void fftPad::init()
{
  common();
//    if(m >  M) M=m;

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
    b=C*M;
  } else {
    double twopibyN=twopi/M;
    double twopibyq=twopi/q;

    bool twop=p == 2 && 2*n != q;

    unsigned int d;

    if(twop) {
      initZetaq();
      b=Cm*p/2;
      d=C*D;
    } else {
      b=Cm*p;
      d=C*D*p;
    }

    Complex *G,*H;
    unsigned int size=m*d;

    G=ComplexAlign(size);
    H=inplace ? G : ComplexAlign(size);

    if(twop) {
      unsigned int Lm=L-m;
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
        if(repad())
          Pad=&fftBase::padSingle;
      } else {
        Forward=&fftBase::forwardMany;
        Backward=&fftBase::backwardMany;
        if(repad())
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

    unsigned int x=Q % D;
    if(x > 0) {
      x *= p;
      fftm2=new mfft1d(m,1,x, 1,m, G,H);
      ifftm2=new mfft1d(m,-1,x, 1,m, G,H);
    }

    if(!inplace)
      deleteAlign(H);
    deleteAlign(G);

    initZetaqm(q);
  }

  fftBase::Forward=Forward;
  fftBase::Backward=Backward;
}

fftPad:: ~fftPad() {
  if(q == 1) {
    delete fftm;
    delete ifftm;
  } else {
    if(Zetaqp) {
      deleteAlign(Zetaqp+p);
      delete fftp;
      delete ifftp;
    }
    delete fftm;
    delete ifftm;
    if(Q % D > 0) {
      delete fftm2;
      delete ifftm2;
    }
  }
}

void fftPad::padSingle(Complex *W)
{
  unsigned int mp=p*m;
  for(unsigned int d=0; d < D; ++d) {
    Complex *F=W+m*d;
    for(unsigned int s=L; s < mp; ++s)
      F[s]=0.0;
  }
}

void fftPad::padMany(Complex *W)
{
  unsigned int mp=p*m;
  for(unsigned int s=L; s < mp; ++s) {
    Complex *F=W+C*s;
    for(unsigned int c=0; c < C; ++c)
      F[c]=0.0;
  }
}

void fftPad::forward(Complex *f, Complex *F)
{
  (this->*Pad)(W0);
  for(unsigned int r=0; r < Q; r += D)
    (this->*Forward)(f,F+b*r,r,W0);
}

void fftPad::backward(Complex *F, Complex *f)
{
  for(unsigned int r=0; r < Q; r += D)
    (this->*Backward)(F+b*r,f,r,W0);
}

void fftPad::forwardExplicit(Complex *f, Complex *F, unsigned int, Complex *W)
{
  for(unsigned int s=0; s < L; ++s)
    F[s]=f[s];
  for(unsigned int s=L; s < M; ++s)
    F[s]=0.0;
  fftm->fft(F);
}

void fftPad::backwardExplicit(Complex *F, Complex *f, unsigned int, Complex *W) {
  ifftm->fft(F);
  for(unsigned int s=0; s < L; ++s)
    f[s]=F[s];
}

void fftPad::forwardExplicitMany(Complex *f, Complex *F, unsigned int, Complex *W) {
  for(unsigned int s=0; s < L; ++s) {
    Complex *Fs=F+C*s;
    Complex *fs=f+C*s;
    for(unsigned int c=0; c < C; ++c)
      Fs[c]=fs[c];
  }
  padMany(F);
  fftm->fft(F);
}

void fftPad::backwardExplicitMany(Complex *F, Complex *f, unsigned int, Complex *W) {
  ifftm->fft(F);
  for(unsigned int s=0; s < L; ++s) {
    Complex *fs=f+C*s;
    Complex *Fs=F+C*s;
    for(unsigned int c=0; c < C; ++c)
      fs[c]=Fs[c];
  }
}
void fftPad::forward(Complex *f, Complex *F0, unsigned int r0, Complex *W)
{
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

void fftPad::forwardMany(Complex *f, Complex *F, unsigned int r, Complex *W) {
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

void fftPad::forward2(Complex *f, Complex *F0, unsigned int r0, Complex *W)
{
  if(W == NULL) W=F0;

  unsigned int D0=Q-r0;
  if(D0 > D) D0=D;

  unsigned int Lm=L-m;
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

void fftPad::forward2Many(Complex *f, Complex *F, unsigned int r, Complex *W)
{
  if(W == NULL) W=F;

  unsigned int Lm=L-m;
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

void fftPad::forwardInner(Complex *f, Complex *F0, unsigned int r0, Complex *W)
{
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

void fftPad::forwardInnerMany(Complex *f, Complex *F, unsigned int r, Complex *W)
{
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

void fftPad::backward(Complex *F0, Complex *f, unsigned int r0, Complex *W)
{
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
    Complex *Zetar=Zetaqm+m*r;
    for(unsigned int s=1; s < L; ++s)
      f[s] += conj(Zetar[s])*F[s];
  }
}

void fftPad::backwardMany(Complex *F, Complex *f, unsigned int r, Complex *W)
{
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
    Complex *Zetar=Zetaqm+m*r;
    for(unsigned int s=1; s < L; ++s) {
      unsigned int Cs=C*s;
      Complex *fs=f+Cs;
      Complex *Fs=W+Cs;
      Complex Zetars=Zetar[s];
      for(unsigned int c=0; c < C; ++c)
        fs[c] += conj(Zetars)*Fs[c];
    }
  }
}

void fftPad::backward2(Complex *F0, Complex *f, unsigned int r0, Complex *W)
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
    Complex *Zetar=Zetaqm+m*r;
    for(unsigned int s=1; s < m; ++s)
      f[s] += conj(Zetar[s])*F[s];
    Complex *Zetar2=Zetaqm2+Lm*r;
    Complex *Fm=F-m;
    for(unsigned int s=m; s < L; ++s)
      f[s] += conj(Zetar2[s])*Fm[s];
  }
}

void fftPad::backward2Many(Complex *F, Complex *f, unsigned int r, Complex *W)
{
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
    Complex *Zetar=Zetaqm+m*r;
    for(unsigned int s=1; s < m; ++s) {
      unsigned int Cs=C*s;
      Complex *fs=f+Cs;
      Complex *Fs=W+Cs;
      Complex Zetars=conj(Zetar[s]);
      for(unsigned int c=0; c < C; ++c)
        fs[c] += Zetars*Fs[c];
    }
    Complex *Zetar2=Zetaqm2+Lm*r;
    Complex *WCm=W-Cm;
    for(unsigned int s=m; s < L; ++s) {
      unsigned int Cs=C*s;
      Complex *fs=f+Cs;
      Complex *Fs=WCm+Cs;
      Complex Zetars2=conj(Zetar2[s]);
      for(unsigned int c=0; c < C; ++c)
        fs[c] += Zetars2*Fs[c];
    }
  }
}

void fftPad::backwardInner(Complex *F0, Complex *f, unsigned int r0, Complex *W)
{
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

void fftPad::backwardInnerMany(Complex *F, Complex *f, unsigned int r, Complex *W)
{
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

void fftPadCentered::init()
{
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
    ZetaShift=NULL;
  } else {
    initShift();
    Forward=fftBase::Forward;
    Backward=fftBase::Backward;

    fftBase::Forward=&fftBase::forwardShifted;
    fftBase::Backward=&fftBase::backwardShifted;
  }
}

void fftPadCentered::initShift()
{
  ZetaShift=ComplexAlign(M);
  double factor=L/2*twopi/M;
  for(unsigned int r=0; r < q; ++r) {
    Complex *Zetar=ZetaShift+r;
    for(unsigned int s=0; s < m; ++s) {
      Zetar[q*s]=expi(factor*(q*s+r));
    }
  }
}

void fftPadCentered::forwardShifted(Complex *f, Complex *F, unsigned int r, Complex *W)
{
  (this->*Forward)(f,F,r,W);
  forwardShift(F,r);
}

void fftPadCentered::backwardShifted(Complex *F, Complex *f, unsigned int r, Complex *W)
{
  backwardShift(F,r);
  (this->*Backward)(F,f,r,W);
}

void fftPadCentered::forwardShift(Complex *F, unsigned int r0) {
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

void fftPadCentered::backwardShift(Complex *F, unsigned int r0)
{
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

void fftPadCentered::forward2(Complex *f, Complex *F0, unsigned int r0, Complex *W)
{
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
      F[s]=Zetar[s]*Zetaqr*fmH[s]; // TODO: Can we use Zetaqm2 here?
  }
  (D0 == D ? fftm : fftm2)->fft(W,F0);
}

void fftPadCentered::forward2Many(Complex *f, Complex *F, unsigned int r, Complex *W)
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

void fftPadCentered::backward2(Complex *F0, Complex *f, unsigned int r0, Complex *W)
{
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
    Complex *Zetar=Zetaqm+m*r;
    for(unsigned int s=mH; s < m; ++s)
      fmH[s] += conj(Zetar[s])*Zetaqr*F[s];
    for(unsigned int s=0; s < LH; ++s)
      fH[s] += conj(Zetar[s])*F[s];
  }
}

void fftPadCentered::backward2Many(Complex *F, Complex *f, unsigned int r, Complex *W)
{
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
    Complex *Zetar=Zetaqm+m*r;
    for(unsigned int s=mH; s < m; ++s) {
      unsigned int Cs=C*s;
      Complex *fmHs=fmH+Cs;
      Complex *Fs=W+Cs;
      Complex Zetars=conj(Zetar[s])*Zetaqr;
      for(unsigned int c=0; c < C; ++c)
        fmHs[c] += Zetars*Fs[c];
    }
    for(unsigned int s=0; s < LH; ++s) {
      unsigned int Cs=C*s;
      Complex *fHs=fH+Cs;
      Complex *Fs=W+Cs;
      Complex Zetars=conj(Zetar[s]);
      for(unsigned int c=0; c < C; ++c)
        fHs[c] += Zetars*Fs[c];
    }
  }
}

void fftPadHermitian::init()
{
  common();
  e=m/2;

  if(q == 1) {
    b=C*e;
    if(C == 1) {
      Forward=&fftBase::forwardExplicit;
      Backward=&fftBase::backwardExplicit;
    } else {
      Forward=&fftBase::forwardExplicitMany;
      Backward=&fftBase::backwardExplicitMany;
    }

    Complex *G=ComplexAlign(C*(e+1));
    double *H=(double *) G;

    crfftm=new mcrfft1d(m,C, C,C,1,1, G,H);
    rcfftm=new mrcfft1d(m,C, C,C,1,1, H,G);
    deleteAlign(G);
    Q=1;
  } else {
    b=C*(m-e);
    D=1; // Temporary
    bool twop=p == 2;
    if(!twop) {
      cout << "p=" << p << endl;
      cerr << "Unimplemented!" << endl;
      exit(-1);
    }

    Q=ceilquotient(q,2);

    Complex *G=ComplexAlign(C*(e+1)*D);
    double *H=inplace ? (double *) G : doubleAlign(Cm*D);

    unsigned int m0=m+(m % 2);
    if(C == 1) {
      crfftm=new mcrfft1d(m,D, 1,1, e+1,m0, G,H);
      rcfftm=new mrcfft1d(m,D, 1,1, m0,e+1, H,G);
      Forward=&fftBase::forward2;
      Backward=&fftBase::backward2;
    } else {
      crfftm=new mcrfft1d(m,C, C,C, 1,1, G,H);
      rcfftm=new mrcfft1d(m,C, C,C, 1,1, H,G);
      Forward=&fftBase::forward2Many;
      Backward=&fftBase::backward2Many;
    }

    unsigned int x=Q % D;
    if(x > 0) {
      crfftm2=new mcrfft1d(m,x, 1,1, e+1,m0, G,H);
      rcfftm2=new mrcfft1d(m,x, 1,1, m0,e+1, H,G);
    }

    if(!inplace)
      deleteAlign(H);
    deleteAlign(G);

    initZetaqm(q/2+1);
  }

  fftBase::Forward=Forward;
  fftBase::Backward=Backward;
}

fftPadHermitian::~fftPadHermitian()
{
  delete crfftm;
  delete rcfftm;
}

void fftPadHermitian::forward(Complex *f, Complex *F)
{
  for(unsigned int r=0; r < Q; r += D)
    (this->*Forward)(f,F+blockOffset(r),r,W0);
}

void fftPadHermitian::backward(Complex *F, Complex *f)
{
  for(unsigned int r=0; r < Q; r += D)
    (this->*Backward)(F+blockOffset(r),f,r,W0);
}

void fftPadHermitian::forwardExplicit(Complex *f, Complex *F, unsigned int, Complex *W)
{
  unsigned int H=ceilquotient(L,2);
  for(unsigned int s=0; s < H; ++s)
    F[s]=f[s];
  for(unsigned int s=H; s <= e; ++s)
    F[s]=0.0;

  crfftm->fft(F);
}

void fftPadHermitian::forwardExplicitMany(Complex *f, Complex *F, unsigned int, Complex *W)
{
  unsigned int H=ceilquotient(L,2);
  for(unsigned int s=0; s < H; ++s) {
    Complex *Fs=F+C*s;
    Complex *fs=f+C*s;
    for(unsigned int c=0; c < C; ++c)
      Fs[c]=fs[c];
  }
  for(unsigned int s=H; s <= e; ++s) {
    Complex *Fs=F+C*s;
    for(unsigned int c=0; c < C; ++c)
      Fs[c]=0.0;
  }

  crfftm->fft(F);
}

void fftPadHermitian::backwardExplicit(Complex *F, Complex *f, unsigned int, Complex *W)
{
  rcfftm->fft(F);
  unsigned int H=ceilquotient(L,2);
  for(unsigned int s=0; s < H; ++s)
    f[s]=F[s];
}

void fftPadHermitian::backwardExplicitMany(Complex *F, Complex *f,
                                           unsigned int, Complex *W)
{
  rcfftm->fft(F);
  unsigned int H=ceilquotient(L,2);
  for(unsigned int s=0; s < H; ++s) {
    Complex *fs=f+C*s;
    Complex *Fs=F+C*s;
    for(unsigned int c=0; c < C; ++c)
      fs[c]=Fs[c];
  }
}

void fftPadHermitian::forward2(Complex *f, Complex *F, unsigned int r, Complex *W)
{
  if(W == NULL) W=F;

  unsigned int e1=e+1;
  unsigned int first=r == 0;
  Complex *fm=f+m;
  unsigned int H=ceilquotient(L,2);
  unsigned int mH1=m-H+1;
  if(first) {
    unsigned int q2=q/2;
    if(2*q2 < q) { // q odd
      for(unsigned int s=0; s < mH1; ++s)
        W[s]=f[s];
      for(unsigned int s=mH1; s <= e; ++s)
        W[s]=f[s]+conj(*(fm-s));

      crfftm->fft(W,F);

      if(m > 2*e) {
        unsigned int offset=e1+e;
        double *Fr=(double *) F+offset;
        *Fr=0.0;
      }
    } else { // q even
      Complex *G=F+b;
      Complex *V=W == F ? G : W+e1;

      Complex *Zetar=Zetaqm+m*q2;

      for(unsigned int s=1; s < mH1; ++s) {
        Complex fs=f[s];
        W[s]=fs;
        V[s]=Zetar[s]*fs;
      }

      for(unsigned int s=mH1; s <= e; ++s) {
        Complex fs=f[s];
        Complex fms=conj(*(fm-s));
        W[s]=fs+fms;
        V[s]=Zetar[s]*(fs-fms);
      }

      W[0]=f[0];
      crfftm->fft(W,F);

      V[0]=f[0];
      crfftm->fft(V,G);

      if(m > 2*e) {
        unsigned int offset=e1+e;
        double *Fr=(double *) F+offset;
        double *Gr=(double *) G+offset;
        *Gr=*Fr=0.0;
      }
    }
  } else {
    Complex *G=F+b;
    Complex *V=W == F ? G : W+e1;
    Complex *Zetar=Zetaqm+m*r;
    for(unsigned int s=1; s < mH1; ++s) {
//      W[s]=Zetar[s]*f[s];
//      V[s]=conj(Zetar[s])*f[s];
      Vec Zeta=LOAD(Zetar+s);
      Vec fs=LOAD(f+s);
      Vec A=Zeta*UNPACKL(fs,fs);
      Vec B=ZMULTI(Zeta*UNPACKH(fs,fs));
      STORE(W+s,A+B);
      STORE(V+s,CONJ(A-B));
    }
    Complex *Zetarm=Zetar+m;
    for(unsigned int s=mH1; s <= e; ++s) {
//      W[s]=Zetar[s]*(f[s]+conj(*(fm-s)*Zetaqr));
//      V[s]=conj(Zetar[s])*(f[s]+conj(*(fm-s))*Zetaqr);
//      Complex A=Zeta*fs.re+Zetam*fms.re;
//      Complex B=Zeta*fs.im-Zetam*fms.im;
//      W[s]=Complex(A.re-B.im,A.im+B.re);
//      V[s]=Complex(A.re+B.im,B.re-A.im);
      Vec Zeta=LOAD(Zetar+s);
      Vec Zetam=CONJ(LOAD(Zetarm-s));
      Vec fs=LOAD(f+s);
      Vec fms=LOAD(fm-s);
      Vec A=Zeta*UNPACKL(fs,fs)+Zetam*UNPACKL(fms,fms);
      Vec B=ZMULTI(Zeta*UNPACKH(fs,fs)-Zetam*UNPACKH(fms,fms));
      STORE(W+s,A+B);
      STORE(V+s,CONJ(A-B));
    }
    W[0]=f[0];
    crfftm->fft(W,F);

    V[0]=f[0];
    crfftm->fft(V,G);

    if(m > 2*e) {
      unsigned int offset=e1+e;
      double *Fr=(double *) F+offset;
      double *Gr=(double *) G+offset;
      *Gr=*Fr=0.0;
    }
  }
}

void fftPadHermitian::forward2Many(Complex *f, Complex *F, unsigned int r, Complex *W)
{
  if(W == NULL) W=F;

  Complex *fm=f+Cm;
  unsigned int H=ceilquotient(L,2);
  unsigned int mH1=m-H+1;
  if(r == 0) {
    for(unsigned int s=0; s < mH1; ++s) {
      unsigned int Cs=C*s;
      Complex *Ws=W+Cs;
      Complex *fs=f+Cs;
      for(unsigned int c=0; c < C; ++c)
        Ws[c]=fs[c];
    }
    for(unsigned int s=mH1; s <= e; ++s) {
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
    Complex *Zetar=Zetaqm+m*r;
    for(unsigned int s=1; s < mH1; ++s) {
      unsigned int Cs=C*s;
      Complex *Ws=W+Cs;
      Complex *fs=f+Cs;
      Complex Zetars=Zetar[s];
      for(unsigned int c=0; c < C; ++c)
        Ws[c]=Zetars*fs[c];
    }
    Complex Zetaqr=Zetaq[r];
    for(unsigned int s=mH1; s <= e; ++s) {
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

void fftPadHermitian::backward2(Complex *F, Complex *f, unsigned int r, Complex *W)
{
  if(W == NULL) W=F;

  Complex Nyquist(0,0);
  bool inplace=W == F;
  if(inplace)
    Nyquist=F[e]; // Save before being overwritten

  rcfftm->fft(F,W);

  Complex *fm=f+m;
  bool even=m == 2*e;
  bool overlap=even && inplace;
  unsigned int me=m-e;
  unsigned int H=ceilquotient(L,2);
  unsigned int mH1=m-H+1;
  unsigned int first=r == 0;
  if(first) {
    unsigned int q2=q/2;
    if(2*q2 < q) { // q odd
      for(unsigned int s=0; s < mH1; ++s)
        f[s]=W[s];
      for(unsigned int s=mH1; s < me; ++s) {
        Complex A=W[s];
        f[s]=A;
        *(fm-s)=conj(A);
      }
      if(even) {
        f[e]=W[e];
        if(inplace)
          F[e]=Nyquist; // Restore initial input of next residue
      }
    } else { // q even
      unsigned int e1=e+1;
      Complex *G=F+b;
      Complex *V=inplace ? G : W+e1;

      Complex We(0,0);
      if(overlap) {
        We=W[e];
        W[e]=Nyquist; // Restore initial input of next residue
        Nyquist=G[e];
      }

      rcfftm->fft(G,V);

      f[0]=W[0]+V[0];

      if(overlap)
        W[e]=We;

      Complex *Zetar=Zetaqm+m*q2;

      for(unsigned int s=1; s < mH1; ++s)
        f[s]=W[s]+conj(Zetar[s])*V[s];

      for(unsigned int s=mH1; s < me; ++s) {
        Complex A=W[s];
        Complex B=conj(Zetar[s])*V[s];
        f[s]=A+B;
        *(fm-s)=conj(A-B);
      }

      if(even) {
        f[e]=W[e]-I*V[e];
        if(inplace)
          G[e]=Nyquist; // Restore initial input of next residue
      }
    }
  } else {
    unsigned int e1=e+1;
    Complex *G=F+b;
    Complex *V=inplace ? G : W+e1;

    Complex We(0,0);
    if(overlap) {
      We=W[e];
      W[e]=Nyquist; // Restore initial input of next residue
      Nyquist=G[e];
    }

    rcfftm->fft(G,V);

    f[0] += W[0]+V[0];

    if(overlap)
      W[e]=We;

    Complex *Zetar=Zetaqm+m*r;
    for(unsigned int s=1; s < mH1; ++s) {
//      f[s] += conj(Zeta)*W[s]+Zeta*V[s];
      Vec Zeta=LOAD(Zetar+s);
      Vec Ws=LOAD(W+s);
      Vec Vs=LOAD(V+s);
      STORE(f+s,LOAD(f+s)+ZMULTC(Zeta,Ws)+ZMULT(Zeta,Vs));
    }
    Complex *Zetarm=Zetar+m;
    for(unsigned int s=mH1; s < me; ++s) {
//    f[s] += conj(Zeta)*W[s]+Zeta*V[s];
//    *(fm-s) += conj(Zetam*W[s])+Zetam*conj(V[s]);
      Vec Zeta=LOAD(Zetar+s);
      Vec Zetam=LOAD(Zetarm-s);
      Vec Ws=LOAD(W+s);
      Vec Vs=LOAD(V+s);
      STORE(f+s,LOAD(f+s)+ZMULTC(Zeta,Ws)+ZMULT(Zeta,Vs));
      STORE(fm-s,LOAD(fm-s)+CONJ(ZMULT(Zetam,Ws))+ZMULTC(Vs,Zetam));
    }
    if(even) {
//      f[e] += conj(Zetar[e])*W[e]+Zetar[e]*V[e];
      Vec Zeta=LOAD(Zetar+e);
      Vec Ws=LOAD(W+e);
      Vec Vs=LOAD(V+e);
      STORE(f+e,LOAD(f+e)+ZMULTC(Zeta,Ws)+ZMULT(Zeta,Vs));
      if(inplace)
        G[e]=Nyquist; // Restore initial input of next residue
    }
  }
}

void fftPadHermitian::backward2Many(Complex *F, Complex *f, unsigned int r, Complex *W)
{
  if(W == NULL) W=F;

  Complex Nyquist[C];
  if(W == F) {
    Complex *FCe=F+C*e;
    for(unsigned int c=0; c < C; ++c)
      Nyquist[c]=FCe[c]; // Save before being overwritten
  }

  rcfftm->fft(F,W);

  Complex *fm=f+Cm;
  unsigned int me=m-e;
  unsigned int H=ceilquotient(L,2);
  unsigned int mH1=m-H+1;
  if(r == 0) {
    for(unsigned int s=0; s < mH1; ++s) {
      unsigned int Cs=C*s;
      Complex *fs=f+Cs;
      Complex *Ws=W+Cs;
      for(unsigned int c=0; c < C; ++c)
        fs[c]=Ws[c];
    }
    for(unsigned int s=mH1; s < me; ++s) {
      unsigned int Cs=C*s;
      Complex *fs=f+Cs;
      Complex *fms=fm-Cs;
      Complex *Ws=W+Cs;
      for(unsigned int c=0; c < C; ++c) {
        Complex A=Ws[c];
        fs[c]=A;
        fms[c]=conj(A);
      }
    }
    if(m == 2*e) {
      unsigned int Ce=C*e;
      Complex *fe=f+Ce;
      Complex *We=W+Ce;
      for(unsigned int c=0; c < C; ++c)
        fe[c]=We[c];
    }
  } else {
    for(unsigned int c=0; c < C; ++c)
      f[c] += W[c];
    Complex *Zetar=Zetaqm+m*r;
    for(unsigned int s=1; s < mH1; ++s) {
      unsigned int Cs=C*s;
      Complex *fs=f+Cs;
      Complex *Ws=W+Cs;
      Complex Zetars=conj(Zetar[s]);
      for(unsigned int c=0; c < C; ++c)
        fs[c] += Zetars*Ws[c];
    }
    Complex Zetaqr=Zetaq[r];
    for(unsigned int s=mH1; s < me; ++s) {
      unsigned int Cs=C*s;
      Complex *fs=f+Cs;
      Complex *fms=fm-Cs;
      Complex *Ws=W+Cs;
      Complex Zetars=conj(Zetar[s]);
      for(unsigned int c=0; c < C; ++c) {
        Complex A=Zetars*Ws[c];
        fs[c] += A;
        fms[c] += conj(A*Zetaqr);
      }
    }
    if(m == 2*e) {
      Complex Zetare=conj(Zetar[e]);
      unsigned int Ce=C*e;
      Complex *fe=f+Ce;
      Complex *We=W+Ce;
      for(unsigned int c=0; c < C; ++c)
        fe[c] += Zetare*We[c];
    }
  }

  if(W == F) {
    Complex *FCe=F+C*e;
    for(unsigned int c=0; c < C; ++c)
      FCe[c]=Nyquist[c]; // Restore initial input of next residue
  }
}

void Convolution::init(Complex *F, Complex *V)
{
  Forward=fft->Forward;
  Backward=fft->Backward;

  L=fft->L;
  q=fft->q;
  Q=fft->Q;
  D=fft->D;
  b=fft->b;

  unsigned int M=fft->normalization();
  scale=1.0/M;
  unsigned int c=fft->outputSize();

  unsigned int N=max(A,B);
  this->F=new Complex*[N];
  if(F) {
    for(unsigned int i=0; i < N; ++i)
      this->F[i]=F+i*c;
  } else {
    allocate=true;
    for(unsigned int i=0; i < N; ++i)
      this->F[i]=ComplexAlign(c);
  }

  if(q > 1) {
    allocateV=false;
    if(V) {
      this->V=new Complex*[B];
      unsigned int size=fft->workSizeV();
      for(unsigned int i=0; i < B; ++i)
        this->V[i]=V+i*size;
    } else
      this->V=NULL;

    allocateW=!this->W && !fft->inplace;
    this->W=allocateW ? ComplexAlign(c) : NULL;

    Pad=fft->Pad;
    (fft->*Pad)(this->W);

    loop2=fft->loop2(); // Two loops and A > B
    int extra;
    if(loop2) {
      Fp=new Complex*[A];
      Fp[0]=this->F[A-1];
      for(unsigned int a=1; a < A; ++a)
        Fp[a]=this->F[a-1];
      FpB=fft->inplace ? NULL : Fp[B];
      extra=1;
    } else
      extra=0;

    if(A > B+extra && !fft->inplace) {
      W0=this->F[B];
      Pad=&fftBase::padNone;
    } else
      W0=this->W;
  }
}

Convolution::~Convolution()
{
  if(q > 1) {
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

  if(allocate) {
    unsigned int N=max(A,B);
    for(unsigned int i=0; i < N; ++i)
      deleteAlign(F[i]);
  }
  delete [] F;
}

// f is an input array of A pointers to distinct data blocks each of size
// fft->length()
// h is an output array of B pointers to distinct data blocks each of size
// fft->length(), which may coincide with f.
// offset is applied to each input and output component
void Convolution::convolve0(Complex **f, Complex **h, multiplier *mult,
                            unsigned int offset)
{
  if(q == 1) {
    for(unsigned int a=0; a < A; ++a)
      (fft->*Forward)(f[a]+offset,F[a],0,NULL);
    (*mult)(F,b,threads);
    for(unsigned int b=0; b < B; ++b)
      (fft->*Backward)(F[b],h[b]+offset,0,NULL);
  } else {
    if(loop2) {
      for(unsigned int a=0; a < A; ++a)
        (fft->*Forward)(f[a]+offset,F[a],0,W);
      (*mult)(F,fft->conjugates(0)*b*D,threads);

      for(unsigned int b=0; b < B; ++b) {
        (fft->*Forward)(f[b]+offset,Fp[b],D,W);
        (fft->*Backward)(F[b],h[b]+offset,0,W0);
        (fft->*Pad)(W);
      }
      for(unsigned int a=B; a < A; ++a)
        (fft->*Forward)(f[a]+offset,Fp[a],D,W);
      (*mult)(Fp,fft->conjugates(1)*b*D,threads);
      for(unsigned int b=0; b < B; ++b)
        (fft->*Backward)(Fp[b],h[b]+offset,D,FpB);
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
        unsigned int D0=Q-r;
        if(D0 > D) D0=D;
        for(unsigned int a=0; a < A; ++a)
          (fft->*Forward)(f[a]+offset,F[a],r,W);
        (*mult)(F,fft->conjugates(r)*b*D0,threads);
        for(unsigned int b=0; b < B; ++b)
          (fft->*Backward)(F[b],h0[b]+Offset,r,W0);
        (fft->*Pad)(W);
      }

      if(useV) {
        for(unsigned int b=0; b < B; ++b) {
          Complex *fb=f[b]+offset;
          Complex *hb=h0[b];
          for(unsigned int i=0; i < noutputs; ++i)
            fb[i]=hb[i];
        }
      }
    }
  }
}

void ForwardBackward::init(fftBase &fft)
{
  Forward=fft.Forward;
  Backward=fft.Backward;
  C=fft.C;
  D=fft.D;
  Q=fft.Q;

  unsigned int L0=fft.outputSize();
  unsigned int N=max(A,B);

  f=new Complex*[N];
  F=new Complex*[N];
  h=new Complex*[B];

  unsigned CL=C*L;

  for(unsigned int a=0; a < A; ++a)
    f[a]=ComplexAlign(CL);

  for(unsigned int a=0; a < N; ++a)
    F[a]=ComplexAlign(L0);

  for(unsigned int b=0; b < B; ++b)
    h[b]=ComplexAlign(CL);

  W=ComplexAlign(fft.workSizeW());
  (fft.*fft.Pad)(W);

  // Initialize entire array to 0 to avoid overflow when timing.
  for(unsigned int a=0; a < A; ++a) {
    Complex *fa=f[a];
    for(unsigned int j=0; j < CL; ++j)
      fa[j]=0.0;
  }
}

double ForwardBackward::time(fftBase &fft, unsigned int K)
{
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

void ForwardBackward::clear()
{
  if(W) {
    deleteAlign(W);
    W=NULL;
  }

  unsigned int N=max(A,B);

  if(h) {
    for(unsigned int b=0; b < B; ++b)
      deleteAlign(h[b]);
    delete[] h;
    h=NULL;
  }

  if(F) {
    for(unsigned int a=0; a < N; ++a)
      deleteAlign(F[a]);
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

void optionsHybrid(int argc, char* argv[])
{
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
        usageHybrid();
        exit(1);
    }
  }
  cout << "L=" << L << endl;
  cout << "M=" << M << endl;
  cout << "C=" << C << endl;

  cout << endl;
}

}
