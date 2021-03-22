#include "convolve.h"
#include "cmult-sse2.h"

// TODO:
// Implement built-in shift for p > 2 centered case
// Optimize shift when M=2L for p=1
// Implement 3D convolutions
// Abort timing when best time exceeded
// Exploit zeta -> -conj(zeta) symmetry for even q
// Precompute best D and inline options for each m value
// Only check m <= M/2 and m=M; how many surplus sizes to check?
// Use experience or heuristics (sparse distribution?) to determine best m value
// Use power of P values for m when L,M,M-L are powers of P?
// Multithread
// Port to MPI

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

//unsigned int surplusFFTsizes=25;
unsigned int surplusFFTsizes=0; // TEMPORARY

#ifdef __SSE2__
const union uvec sse2_pm = {
  { 0x00000000,0x00000000,0x00000000,0x80000000 }
};
const union uvec sse2_mp = {
  { 0x00000000,0x80000000,0x00000000,0x00000000 }
};
const union uvec sse2_mm = {
  { 0x00000000,0x80000000,0x00000000,0x80000000 }
};
#endif

const double twopi=2.0*M_PI;

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

void fftBase::initZetaqm(unsigned int q, unsigned int m)
{
  double twopibyM=twopi/M;
  Zetaqm0=utils::ComplexAlign((q-1)*m);
  Zetaqm=Zetaqm0-m;
  for(unsigned int r=1; r < q; ++r) {
    Zetaqm[m*r]=1.0;
    for(unsigned int s=1; s < m; ++s)
      Zetaqm[m*r+s]=expi(r*s*twopibyM);
  }
}

void fftBase::OptBase::check(unsigned int L, unsigned int M,
                             Application& app, unsigned int C, unsigned int m,
                             bool fixed, bool mForced)
{
//    cout << "m=" << m << endl;
  unsigned int q=ceilquotient(M,m);
  unsigned int p=ceilquotient(L,m);

  if(p == q && p > 1 && !mForced) return;

  if(p > 2 && !fixed) {
    unsigned int n=ceilquotient(M,m*p);
    unsigned int q2=p*n;
    if(q2 != q) {
      unsigned int start=DOption > 0 ? min(DOption,n) : 1;
      unsigned int stop=DOption > 0 ? min(DOption,n) : n;
      if(fixed || C > 1) start=stop=1;
      for(unsigned int D=start; D <= stop; D *= 2) {
        if(2*D > stop) D=stop;
        if(!validD(D)) continue;
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

  if(p > 2 && q % p != 0) return;

  unsigned int start=DOption > 0 ? min(DOption,q) : 1;
  unsigned int stop=DOption > 0 ? min(DOption,q) : q;
  if(fixed || C > 1) start=stop=1;
  for(unsigned int D=start; D <= stop; D *= 2) {
    if(2*D > stop) D=stop;
    if(!validD(D)) continue;
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
  if(q > 1)
    deleteAlign(Zetaqm0);
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
  p=ceilquotient(L,m);
  dr=conjugates() ? D/2 : D;

  inplace=IOption == -1 ? C > 1 : IOption;

  Cm=C*m;
  n=q/p;
  M=m*q;
  Pad=&fftBase::padNone;
  Zetaqp=NULL;
}

unsigned int fftBase::residue(unsigned int r, unsigned int q)
{
  if(r == 0) return 0;
  unsigned int q2=q/2;
  if(2*q2 == q) {
    if(r == 1) return q2;
    --r;
  }
  unsigned int h=ceilquotient(r,2);
  return r % 2 == 0 ? q-h : h;
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
    D0=R=Q=1;
    b=C*M;
  } else {
    double twopibyN=twopi/M;
    double twopibyq=twopi/q;

    unsigned int d;

    if(p == 2) {
      Q=n=q;
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

    if(p == 2) {
      if(!centered) {
        unsigned int Lm=L-m;
        Zetaqm2=ComplexAlign((q-1)*Lm)-L;
        for(unsigned int r=1; r < q; ++r) {
          for(unsigned int s=m; s < L; ++s)
            Zetaqm2[Lm*r+s]=expi(r*s*twopibyN);
        }
      }

      if(C == 1) {
        Forward=&fftBase::forward2;
        Backward=&fftBase::backward2;
      } else {
        Forward=&fftBase::forward2Many;
        Backward=&fftBase::backward2Many;
      }
    } else if(p > 2) { // Implies L > 2m
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

    R=residueBlocks();
    D0=Q % D;
    if(D0 == 0) D0=D;

    if(C == 1) {
      fftm=new mfft1d(m,1,d, 1,m, G,H);
      ifftm=new mfft1d(m,-1,d, 1,m, G,H);
    } else {
      fftm=new mfft1d(m,1,C, C,1, G,H);
      ifftm=new mfft1d(m,-1,C, C,1, G,H);
    }

    if(D0 != D) {
      unsigned int x=D0*p;
      fftm2=new mfft1d(m,1,x, 1,m, G,H);
      ifftm2=new mfft1d(m,-1,x, 1,m, G,H);
    } else
      fftm2=NULL;

    if(!inplace)
      deleteAlign(H);
    deleteAlign(G);

    initZetaqm(q,centered ? m+1 : m);
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
    if(fftm2) {
      delete fftm2;
      delete ifftm2;
    }
  }
}

void fftPad::padSingle(Complex *W)
{
  unsigned int mp=m*p;
  for(unsigned int d=0; d < D; ++d) {
    Complex *F=W+m*d;
    for(unsigned int s=L; s < mp; ++s)
      F[s]=0.0;
  }
}

void fftPad::padMany(Complex *W)
{
  unsigned int mp=m*p;
  for(unsigned int s=L; s < mp; ++s) {
    Complex *F=W+C*s;
    for(unsigned int c=0; c < C; ++c)
      F[c]=0.0;
  }
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

  bool inplace=W == F0;

  Complex *W0=W;
  unsigned int dr0=dr;
  mfft1d *fftm0;

  if(r0 == 0) {
    unsigned int residues;
    unsigned int q2=q/2;
    if(D == 1 || 2*q2 < q) { // q odd, r=0
      residues=1;
      for(unsigned int s=0; s < L; ++s)
        W[s]=f[s];
    } else { // q even, r=q/2
      residues=2;
      Complex *V=W+m;
      V[0]=W[0]=f[0];
      Complex *Zetar=Zetaqm+m*q2;
      for(unsigned int s=1; s < L; ++s) {
        Complex fs=f[s];
        W[s]=fs;
        V[s]=Zetar[s]*fs;
      }
      if(inplace) {
        for(unsigned int s=L; s < m; ++s)
          V[s]=0.0;
      }
    }
    if(inplace) {
      for(unsigned int s=L; s < m; ++s)
        W[s]=0.0;
    }
    W += residues*m;
    r0=1;
    dr0=(D0-residues)/2;
    fftm0=D0 == D ? fftm : fftm2;
  } else
    fftm0=fftm;

  if(D == 1) {
    if(dr0 > 0) {
      W[0]=f[0];
      Complex *Zetar=Zetaqm+m*r0;
      for(unsigned int s=1; s < L; ++s)
        W[s]=Zetar[s]*f[s];
      if(inplace) {
        for(unsigned int s=L; s < m; ++s)
          W[s]=0.0;
      }
    }
  } else {
    for(unsigned int d=0; d < dr0; ++d) {
      Complex *F=W+2*m*d;
      Complex *G=F+m;
      unsigned int r=r0+d;
      F[0]=G[0]=f[0];
      Complex *Zetar=Zetaqm+m*r;
      for(unsigned int s=1; s < L; ++s) {
//        F[s]=Zeta*fs;
//        G[s]=conj(Zeta)*fs;
        Vec Zeta=LOAD(Zetar+s);
        Vec fs=LOAD(f+s);
        Vec A=Zeta*UNPACKL(fs,fs);
        Vec B=ZMULTI(Zeta*UNPACKH(fs,fs));
        STORE(F+s,A+B);
        STORE(G+s,CONJ(A-B));
      }
    }
    if(inplace) {
      for(unsigned int d=0; d < dr0; ++d) {
        Complex *F=W+2*m*d;
        Complex *G=F+m;
        for(unsigned int s=L; s < m; ++s)
          F[s]=0.0;
        for(unsigned int s=L; s < m; ++s)
          G[s]=0.0;
      }
    }
  }

  fftm0->fft(W0,F0);
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
      Vec Zeta=LOAD(Zetar+s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=CONJ(UNPACKH(Zeta,Zeta));
      for(unsigned int c=0; c < C; ++c)
        STORE(Fs+c,ZMULT(X,Y,LOAD(fs+c)));
    }
  }
  fftm->fft(W,F);
}

void fftPad::forward2(Complex *f, Complex *F0, unsigned int r0, Complex *W)
{
  if(W == NULL) W=F0;

  Complex *W0=W;
  unsigned int dr0=dr;
  mfft1d *fftm0;

  unsigned int Lm=L-m;
  if(r0 == 0) {
    unsigned int residues;
    unsigned int q2=q/2;
    if(D == 1 || 2*q2 < q) { // q odd, r=0
      residues=1;
      for(unsigned int s=0; s < Lm; ++s)
        W[s]=f[s]+f[m+s];
      for(unsigned int s=Lm; s < m; ++s)
        W[s]=f[s];
    } else {
      residues=2;
      Complex *V=W+m;
      Complex *fm=f+m;
      Complex f0=f[0];
      Complex fm0=fm[0];
      W[0]=f0+fm0;
      V[0]=f0-fm0;
      Complex *Zetar=Zetaqm+m*q2;
      for(unsigned int s=1; s < Lm; ++s) {
        Complex fs=f[s];
        Complex fms=fm[s];
        W[s]=fs+fms;
        V[s]=conj(Zetar[s])*(fs-fms);
      }
      for(unsigned int s=Lm; s < m; ++s) {
        Complex fs=f[s];
        W[s]=fs;
        V[s]=conj(Zetar[s])*fs;
      }
    }
    W += residues*m;
    r0=1;
    dr0=(D0-residues)/2;
    fftm0=D0 == D ? fftm : fftm2;
  } else
    fftm0=fftm;

  if(D == 1) {
    if(dr0 > 0) {
      Complex *Zetar2=Zetaqm2+Lm*r0+m;
      Complex *fm=f+m;
      W[0]=f[0]+Zetar2[0]*fm[0];
      Complex *Zetar=Zetaqm+m*r0;
      for(unsigned int s=1; s < Lm; ++s) {
//        W[s]=Zetar[s]*f[s]+Zetar2[s]*fm[s];
        Vec Zeta=LOAD(Zetar+s);
        Vec Zeta2=LOAD(Zetar2+s);
        Vec fs=LOAD(f+s);
        Vec fms=LOAD(fm+s);
        STORE(W+s,ZMULT(Zeta,fs)+ZMULT(Zeta2,fms));
      }
      for(unsigned int s=Lm; s < m; ++s)
        W[s]=Zetar[s]*f[s];
    }
  } else {
    for(unsigned int d=0; d < dr0; ++d) {
      Complex *F=W+2*m*d;
      Complex *G=F+m;
      unsigned int r=r0+d;
      Complex *Zetar=Zetaqm+m*r;
      Complex *Zetar2=Zetaqm2+Lm*r+m;
      Complex *fm=f+m;
      Vec fs=LOAD(f);
      Vec fms=LOAD(fm);
      Vec Zetam=LOAD(Zetar2);
      Vec A=Zetam*UNPACKL(fms,fms);
      Vec B=ZMULTI(Zetam*UNPACKH(fms,fms));
      STORE(F,fs+A+B);
      STORE(G,fs+CONJ(A-B));
      for(unsigned int s=1; s < Lm; ++s) {
//        F[s]=Zetar[s]*fs+Zetar2[s]*fms;
//        G[s]=conj(Zetar[s])*fs+conj(Zetar2[s])*fms;
        /*
          Complex fs=f[s];
          Complex fms=fm[s];
          Complex Zeta=Zetar[s];
          Complex Zetam=Zetar2[s];
          Complex A=Zeta*fs.re+Zetam*fms.re;
          Complex B=Zeta*fs.im+Zetam*fms.im;
          F[s]=Complex(A.re-B.im,A.im+B.re);
          G[s]=Complex(A.re+B.im,B.re-A.im);
        */
        Vec fs=LOAD(f+s);
        Vec fms=LOAD(fm+s);
        Vec Zeta=LOAD(Zetar+s);
        Vec Zetam=LOAD(Zetar2+s);
        Vec A=Zeta*UNPACKL(fs,fs)+Zetam*UNPACKL(fms,fms);
        Vec B=ZMULTI(Zeta*UNPACKH(fs,fs)+Zetam*UNPACKH(fms,fms));
        STORE(F+s,A+B);
        STORE(G+s,CONJ(A-B));
      }
      for(unsigned int s=Lm; s < m; ++s) {
//        F[s]=Zetar[s]*f[s];
//        G[s]=conj(Zetar[s])*f[s];
        Vec fs=LOAD(f+s);
        Vec Zeta=LOAD(Zetar+s);
        Vec A=Zeta*UNPACKL(fs,fs);
        Vec B=ZMULTI(Zeta*UNPACKH(fs,fs));
        STORE(F+s,A+B);
        STORE(G+s,CONJ(A-B));
      }
    }
  }

  fftm0->fft(W0,F0);
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
    Complex *Zetar2=Zetaqm2+Lm*r+m;
    Complex Zeta2=Zetar2[0];
    Complex *fm=f+Cm;
    for(unsigned int c=0; c < C; ++c)
      W[c]=f[c]+Zeta2*fm[c];
    Complex *Zetar=Zetaqm+m*r;
    for(unsigned int s=1; s < Lm; ++s) {
      unsigned int Cs=C*s;
      Complex *Fs=W+Cs;
      Complex *fs=f+Cs;
      Complex *fms=f+Cm+Cs;
      Complex Zetars=Zetar[s];
      Complex Zetarms=Zetar2[s];
      for(unsigned int c=0; c < C; ++c)
        Fs[c]=Zetars*fs[c]+Zetarms*fms[c];
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

  Complex *W0=W;
  unsigned int dr0=dr;
  mfft1d *fftm0;

  unsigned int pm1=p-1;
  unsigned int stop=L-m*pm1;

  if(r0 == 0) {
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

    W += b;
    r0=1;
    dr0=D0-1;
    fftm0=D0 == D ? fftm : fftm2;
  } else
    fftm0=fftm;

  for(unsigned int d=0; d < dr0; ++d) {
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

  fftm0->fft(W0,F0);
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

  (r0 > 0 || D0 == D ? ifftm : ifftm2)->fft(F0,W);

  unsigned int dr0=dr;

  if(r0 == 0) {
    unsigned int residues;
    unsigned int q2=q/2;
    if(D == 1 || 2*q2 < q) { // q odd, r=0
      residues=1;
      for(unsigned int s=0; s < L; ++s)
        f[s]=W[s];
    } else { // q even, r=q/2
      residues=2;
      Complex *V=W+m;
      f[0]=W[0]+V[0];
      Complex *Zetar=Zetaqm+m*q2;
      for(unsigned int s=1; s < L; ++s)
        f[s]=W[s]+conj(Zetar[s])*V[s];
    }
    W += residues*m;
    r0=1;
    dr0=(D0-residues)/2;
  }

  if(D == 1) {
    if(dr0 > 0) {
      f[0] += W[0];
      Complex *Zetar=Zetaqm+m*r0;
      for(unsigned int s=1; s < L; ++s)
        f[s] += conj(Zetar[s])*W[s];
    }
  } else {
    for(unsigned int d=0; d < dr0; ++d) {
      Complex *U=W+2*m*d;
      Complex *V=U+m;
      f[0] += U[0]+V[0];
      unsigned int r=r0+d;
      Complex *Zetar=Zetaqm+m*r;
      for(unsigned int s=1; s < L; ++s) {
        Vec Zeta=LOAD(Zetar+s);
        Vec Us=LOAD(U+s);
        Vec Vs=LOAD(V+s);
        STORE(f+s,LOAD(f+s)+ZCMULT(Zeta,Us)+ZMULT(Zeta,Vs));
      }
    }
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

  (r0 > 0 || D0 == D ? ifftm : ifftm2)->fft(F0,W);

  unsigned int dr0=dr;
  unsigned int Lm=L-m;

  if(r0 == 0) {
    unsigned int residues;
    unsigned int q2=q/2;
    if(D == 1 || 2*q2 < q) { // q odd, r=0
      residues=1;
      for(unsigned int s=0; s < m; ++s)
        f[s]=W[s];
      Complex *Wm=W-m;
      for(unsigned int s=m; s < L; ++s)
        f[s]=Wm[s];
    } else { // q even, r=q/2
      residues=2;
      Complex *V=W+m;
      f[0]=W[0]+V[0];
      Complex *Zetar=Zetaqm+m*q2;
      for(unsigned int s=1; s < m; ++s)
        f[s]=W[s]+Zetar[s]*V[s];
      Complex *Zetar2=Zetaqm2+Lm*q2;
      Complex *Wm=W-m;
      for(unsigned int s=m; s < L; ++s)
        f[s]=Wm[s]+Zetar2[s]*W[s];
    }
    W += residues*m;
    r0=1;
    dr0=(D0-residues)/2;
  }

  if(D == 1) {
    if(dr0 > 0) {
      f[0] += W[0];
      Complex *Zetar=Zetaqm+m*r0;
      for(unsigned int s=1; s < m; ++s)
        f[s] += conj(Zetar[s])*W[s];
      Complex *Zetar2=Zetaqm2+Lm*r0;
      Complex *Wm=W-m;
      for(unsigned int s=m; s < L; ++s)
        f[s] += conj(Zetar2[s])*Wm[s];
    }
  } else {
    for(unsigned int d=0; d < dr0; ++d) {
      Complex *U=W+2*m*d;
      Complex *V=U+m;
      unsigned int r=r0+d;
      f[0] += U[0]+V[0];
      Complex *Zetar=Zetaqm+m*r;
      for(unsigned int s=1; s < m; ++s) {
//      f[s] += conj(Zetar[s])*U[s]+Zetar[s]*V[s];
        Vec Zeta=LOAD(Zetar+s);
        Vec Us=LOAD(U+s);
        Vec Vs=LOAD(V+s);
        STORE(f+s,LOAD(f+s)+ZCMULT(Zeta,Us)+ZMULT(Zeta,Vs));
      }
      Complex *Zetar2=Zetaqm2+Lm*r;
      Complex *Um=U-m;
      Complex *Vm=V-m;
      for(unsigned int s=m; s < L; ++s) {
//      f[s] += conj(Zetar2[s])*Um[s]+Zetar2[s]*Vm[s];
        Vec Zeta=LOAD(Zetar2+s);
        Vec Ums=LOAD(Um+s);
        Vec Vms=LOAD(Vm+s);
        STORE(f+s,LOAD(f+s)+ZCMULT(Zeta,Ums)+ZMULT(Zeta,Vms));
      }
    }
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

  (r0 > 0 || D0 == D ? ifftm : ifftm2)->fft(F0,W);

  unsigned int dr0=dr;

  unsigned int pm1=p-1;
  unsigned int stop=L-m*pm1;

  if(r0 == 0) {
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

    W += b;
    r0=1;
    dr0=D0-1;
  }

  for(unsigned int d=0; d < dr0; ++d) {
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
  if(p == 2 && q > 1) {
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
    Forward=&fftBase::forwardShifted;
    Backward=&fftBase::backwardShifted;
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
  (this->*fftBase::Forward)(f,F,r,W);
  forwardShift(F,r);
}

void fftPadCentered::backwardShifted(Complex *F, Complex *f, unsigned int r, Complex *W)
{
  backwardShift(F,r);
  (this->*fftBase::Backward)(F,f,r,W);
}

void fftPadCentered::forwardShift(Complex *F, unsigned int r0) {
  unsigned int dr0=r0 == 0 ? D0 : D;
  for(unsigned int d=0; d < dr0; ++d) {
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
  unsigned int dr0=r0 == 0 ? D0 : D;
  for(unsigned int d=0; d < dr0; ++d) {
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

  Complex *W0=W;
  unsigned int dr0=dr;
  mfft1d *fftm0;

  unsigned int H=L/2;
  unsigned int mH=m-H;
  unsigned int LH=L-H;
  Complex *fmH=f-mH;
  Complex *fH=f+H;
  if(r0 == 0) {
    unsigned int residues;
    unsigned int q2=q/2;
    if(D == 1 || 2*q2 < q) { // q odd, r=0
      residues=1;
      for(unsigned int s=0; s < mH; ++s)
        W[s]=fH[s];
      for(unsigned int s=mH; s < LH; ++s)
        W[s]=fmH[s]+fH[s];
      for(unsigned int s=LH; s < m; ++s)
        W[s]=fmH[s];
    } else {
      residues=2;
      Complex *V=W+m;
      Complex *Zetar=Zetaqm+(m+1)*q2;
      Complex *Zetarm=Zetar+m;
      for(unsigned int s=0; s < mH; ++s) {
        W[s]=fH[s];
        V[s]=Zetar[s]*fH[s];
      }
      for(unsigned int s=mH; s < LH; ++s) {
//        W[s]=fmH[s]+fH[s];
//        V[s]=conj(*(Zetarm-s))*fmH[s]+Zetar[s]*fH[s];
        Vec Zeta=LOAD(Zetar+s);
        Vec Zetam=LOAD(Zetarm-s);
        Vec fs=LOAD(fH+s);
        Vec fms=LOAD(fmH+s);
        STORE(W+s,fms+fs);
        STORE(V+s,ZCMULT(Zetam,fms)+ZMULT(Zeta,fs));
      }
      for(unsigned int s=LH; s < m; ++s) {
        W[s]=fmH[s];
        V[s]=conj(*(Zetarm-s))*fmH[s];
      }
    }
    W += residues*m;
    r0=1;
    dr0=(D0-residues)/2;
    fftm0=D0 == D ? fftm : fftm2;
  } else
    fftm0=fftm;

  if(D == 1) {
    if(dr0 > 0) {
      Complex *Zetar=Zetaqm+(m+1)*r0;
      for(unsigned int s=0; s < mH; ++s)
        W[s]=Zetar[s]*fH[s];
      Complex *Zetarm=Zetar+m;
      for(unsigned int s=mH; s < LH; ++s) {
//        W[s]=conj(*(Zetarm-s))*fmH[s]+Zetar[s]*fH[s];
        Vec Zeta=LOAD(Zetar+s);
        Vec Zetam=LOAD(Zetarm-s);
        Vec fs=LOAD(fH+s);
        Vec fms=LOAD(fmH+s);
        STORE(W+s,ZCMULT(Zetam,fms)+ZMULT(Zeta,fs));
      }
      for(unsigned int s=LH; s < m; ++s)
        W[s]=conj(*(Zetarm-s))*fmH[s];
    }
  } else {
    for(unsigned int d=0; d < dr0; ++d) {
      Complex *F=W+2*m*d;
      Complex *G=F+m;
      unsigned int r=r0+d;
      Complex *Zetar=Zetaqm+(m+1)*r;
      for(unsigned int s=0; s < mH; ++s) {
//        F[s]=Zetar[s]*fH[s];
//        G[s]=conj(Zetar[s])*fH[s];
        Vec Zeta=LOAD(Zetar+s);
        Vec fs=LOAD(fH+s);
        Vec A=Zeta*UNPACKL(fs,fs);
        Vec B=ZMULTI(Zeta*UNPACKH(fs,fs));
        STORE(F+s,A+B);
        STORE(G+s,CONJ(A-B));
      }
      Complex *Zetarm=Zetar+m;
      for(unsigned int s=mH; s < LH; ++s) {
//      F[s]=conj(*(Zetarm-s))*fmH[s]+Zetar[s]*fH[s];
//      G[s]=*(Zetarm-s)*fmH[s]+conj(Zetar[s])*fH[s];
        Vec Zeta=LOAD(Zetar+s);
        Vec Zetam=CONJ(LOAD(Zetarm-s));
        Vec fms=LOAD(fmH+s);
        Vec fs=LOAD(fH+s);
        Vec A=Zeta*UNPACKL(fs,fs)+Zetam*UNPACKL(fms,fms);
        Vec B=ZMULTI(Zeta*UNPACKH(fs,fs)+Zetam*UNPACKH(fms,fms));
        STORE(F+s,A+B);
        STORE(G+s,CONJ(A-B));
      }
      for(unsigned int s=LH; s < m; ++s) {
//        F[s]=conj(*(Zetarm-s))*fmH[s];
//        G[s]=*(Zetarm-s)*fmH[s];
        Vec Zetam=CONJ(LOAD(Zetarm-s));
        Vec fms=LOAD(fmH+s);
        Vec A=Zetam*UNPACKL(fms,fms);
        Vec B=ZMULTI(Zetam*UNPACKH(fms,fms));
        STORE(F+s,A+B);
        STORE(G+s,CONJ(A-B));
      }
    }
  }

  fftm0->fft(W0,F0);
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
    Complex *Zetar=Zetaqm+(m+1)*r;
    for(unsigned int s=0; s < mH; ++s) {
      unsigned int Cs=C*s;
      Complex *Fs=W+Cs;
      Complex *fHs=fH+Cs;
//      Complex Zeta=Zetar[s];
      Vec Zeta=LOAD(Zetar+s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=CONJ(UNPACKH(Zeta,Zeta));
      for(unsigned int c=0; c < C; ++c)
//        Fs[c]=Zeta*fHs[c];
        STORE(Fs+c,ZMULT(X,Y,LOAD(fHs+c)));
    }
    Complex *Zetarm=Zetar+m;
    for(unsigned int s=mH; s < LH; ++s) {
      unsigned int Cs=C*s;
      Complex *Fs=W+Cs;
      Complex *fHs=fH+Cs;
      Complex *fmHs=fmH+Cs;
//      Complex Zetars=Zetar[s];
//      Complex Zetarms=conj(*(Zetarm-s));
      Vec Zeta=LOAD(Zetar+s);
      Vec Zetam=LOAD(Zetarm-s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=CONJ(UNPACKH(Zeta,Zeta));
      Vec Xm=UNPACKL(Zetam,Zetam);
      Vec Ym=REFL(UNPACKH(Zetam,Zetam));
      for(unsigned int c=0; c < C; ++c)
//        Fs[c]=Zetarms*fmHs[c]+Zetars*fHs[c];
//        STORE(Fs+c,ZMULT(Xm,Ym,LOAD(fmHs+c))+ZMULT(X,Y,LOAD(fHs+c)));
        STORE(Fs+c,ZMULT(Xm,Ym,LOAD(fmHs+c))+ZMULT(X,Y,LOAD(fHs+c)));
    }
    for(unsigned int s=LH; s < m; ++s) {
      unsigned int Cs=C*s;
      Complex *Fs=W+Cs;
      Complex *fmHs=fmH+Cs;
//      Complex Zetam=conj(*(Zetarm-s));
      Vec Zetam=LOAD(Zetarm-s);
      Vec Xm=UNPACKL(Zetam,Zetam);
      Vec Ym=REFL(UNPACKH(Zetam,Zetam));
      for(unsigned int c=0; c < C; ++c)
//        Fs[c]=Zetam*fmHs[c];
        STORE(Fs+c,ZMULT(Xm,Ym,LOAD(fmHs+c)));
    }
  }
  fftm->fft(W,F);
}

void fftPadCentered::backward2(Complex *F0, Complex *f, unsigned int r0, Complex *W)
{
  if(W == NULL) W=F0;

  (r0 > 0 || D0 == D ? ifftm : ifftm2)->fft(F0,W);

  unsigned int dr0=dr;

  unsigned int H=L/2;
  unsigned int mH=m-H;
  unsigned int LH=L-H;
  Complex *fmH=f-mH;
  Complex *fH=f+H;
  if(r0 == 0) {
    unsigned int residues;
    unsigned int q2=q/2;
    if(D == 1 || 2*q2 < q) { // q odd, r=0
      residues=1;
      for(unsigned int s=mH; s < m; ++s)
        fmH[s]=W[s];
      for(unsigned int s=0; s < LH; ++s)
        fH[s]=W[s];
    } else { // q even, r=q/2
      residues=2;
      Complex *V=W+m;
      Complex *Zetar=Zetaqm+(m+1)*q2;
      Complex *Zetarm=Zetar+m;
      for(unsigned int s=mH; s < m; ++s)
        fmH[s]=W[s]+*(Zetarm-s)*V[s];
      for(unsigned int s=0; s < LH; ++s) {
        fH[s]=W[s]+conj(Zetar[s])*V[s];
      }
    }
    W += residues*m;
    r0=1;
    dr0=(D0-residues)/2;
  }

  if(D == 1) {
    if(dr0 > 0) {
      Complex *Zetar=Zetaqm+(m+1)*r0;
      Complex *Zetarm=Zetar+m;
      for(unsigned int s=mH; s < m; ++s)
        fmH[s] += *(Zetarm-s)*W[s];
      for(unsigned int s=0; s < LH; ++s)
        fH[s] += conj(Zetar[s])*W[s];
    }
  } else {
    for(unsigned int d=0; d < dr0; ++d) {
      Complex *U=W+2*m*d;
      Complex *V=U+m;
      unsigned int r=r0+d;
      Complex *Zetar=Zetaqm+(m+1)*r;
      Complex *Zetarm=Zetar+m;
      for(unsigned int s=mH; s < m; ++s)
        fmH[s] += *(Zetarm-s)*U[s]+conj(*(Zetarm-s))*V[s];
      for(unsigned int s=0; s < LH; ++s)
        fH[s] += conj(Zetar[s])*U[s]+Zetar[s]*V[s];
    }
  }
}

void fftPadCentered::backward2Many(Complex *F, Complex *f, unsigned int r, Complex *W)
{
  if(W == NULL) W=F;

  ifftm->fft(F,W);

  unsigned int H=L/2;
  unsigned int odd=L-2*H;
  unsigned int mH=m-H;
  Complex *fmH=f-C*mH;
  Complex *fH=f+C*H;
  if(r == 0) {
    for(unsigned int s=H; s < m; ++s) {
      unsigned int Cs=C*s;
      Complex *fmHs=fmH+Cs;
      Complex *Fs=W+Cs;
      for(unsigned int c=0; c < C; ++c)
        fmHs[c]=Fs[c];
    }
    for(unsigned int s=mH; s < H; ++s) {
      unsigned int Cs=C*s;
      Complex *fmHs=fmH+Cs;
      Complex *fHs=fH+Cs;
      Complex *Fs=W+Cs;
      for(unsigned int c=0; c < C; ++c)
        fHs[c]=fmHs[c]=Fs[c];
    }
    for(unsigned int s=0; s < mH; ++s) {
      unsigned int Cs=C*s;
      Complex *fHs=fH+Cs;
      Complex *Fs=W+Cs;
      for(unsigned int c=0; c < C; ++c)
        fHs[c]=Fs[c];
    }
    if(odd) {
      unsigned int CH=C*H;
      Complex *fHs=fH+CH;
      Complex *Fs=W+CH;
      for(unsigned int c=0; c < C; ++c)
        fHs[c]=Fs[c];
    }
  } else {
    Complex *Zetar=Zetaqm+(m+1)*r;
    Complex *Zetarm=Zetar+m;
    for(unsigned int s=H; s < m; ++s) {
      unsigned int Cs=C*s;
      Complex *fmHs=fmH+Cs;
      Complex *Fs=W+Cs;
      Vec Zetam=LOAD(Zetarm-s);
      Vec Xm=UNPACKL(Zetam,Zetam);
      Vec Ym=CONJ(UNPACKH(Zetam,Zetam));
      for(unsigned int c=0; c < C; ++c)
        STORE(fmHs+c,LOAD(fmHs+c)+ZMULT(Xm,Ym,LOAD(Fs+c)));
    }
    for(unsigned int s=mH; s < H; ++s) {
      unsigned int Cs=C*s;
      Complex *fmHs=fmH+Cs;
      Complex *fHs=fH+Cs;
      Complex *Fs=W+Cs;
//      Complex Zetam=*(Zetarm-s);
//      Complex Zeta=conj(Zetar[s]);
      Vec Zetam=LOAD(Zetarm-s);
      Vec Zeta=LOAD(Zetar+s);
      Vec Xm=UNPACKL(Zetam,Zetam);
      Vec Ym=CONJ(UNPACKH(Zetam,Zetam));
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=REFL(UNPACKH(Zeta,Zeta));
      for(unsigned int c=0; c < C; ++c) {
//        fmHs[c] += Zetam*Fsc;
//        fHs[c] += Zeta*Fsc;
        Vec Fsc=LOAD(Fs+c);
        STORE(fmHs+c,LOAD(fmHs+c)+ZMULT(Xm,Ym,Fsc));
        STORE(fHs+c,LOAD(fHs+c)+ZMULT(X,Y,Fsc));
      }
    }
    for(unsigned int s=0; s < mH; ++s) {
      unsigned int Cs=C*s;
      Complex *fHs=fH+Cs;
      Complex *Fs=W+Cs;
      Vec Zeta=LOAD(Zetar+s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=REFL(UNPACKH(Zeta,Zeta));
      for(unsigned int c=0; c < C; ++c)
        STORE(fHs+c,LOAD(fHs+c)+ZMULT(X,Y,LOAD(Fs+c)));
    }
    if(odd) {
      unsigned int CH=C*H;
      Complex *fHs=fH+CH;
      Complex *Fs=W+CH;
      Vec Zeta=LOAD(Zetar+H);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=REFL(UNPACKH(Zeta,Zeta));
      for(unsigned int c=0; c < C; ++c)
        STORE(fHs+c,LOAD(fHs+c)+ZMULT(X,Y,LOAD(Fs+c)));
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
    D0=R=Q=1;
  } else {
    D=2;
    dr=1;

    b=C*(m-e);
    bool twop=p == 2;
    if(!twop) {
      cout << "p=" << p << endl;
      cerr << "Unimplemented!" << endl;
      exit(-1);
    }

    Q=q;
    R=residueBlocks();
    D0=Q % D;
    if(D0 == 0) D0=D;

    Complex *G=ComplexAlign(C*(e+1));
    double *H=inplace ? (double *) G : doubleAlign(Cm);

    unsigned int m0=m+(m % 2);
    if(C == 1) {
      crfftm=new mcrfft1d(m,1, 1,1, e+1,m0, G,H);
      rcfftm=new mrcfft1d(m,1, 1,1, m0,e+1, H,G);
      Forward=&fftBase::forward2;
      Backward=&fftBase::backward2;
    } else {
      crfftm=new mcrfft1d(m,C, C,C, 1,1, G,H);
      rcfftm=new mrcfft1d(m,C, C,C, 1,1, H,G);
      Forward=&fftBase::forward2Many;
      Backward=&fftBase::backward2Many;
    }

    if(!inplace)
      deleteAlign(H);
    deleteAlign(G);

    initZetaqm(q/2+1,m);
  }

  fftBase::Forward=Forward;
  fftBase::Backward=Backward;
}

fftPadHermitian::~fftPadHermitian()
{
  delete crfftm;
  delete rcfftm;
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
  Complex *fm=f+m;
  unsigned int H=ceilquotient(L,2);
  unsigned int mH1=m-H+1;

  if(r == 0) {
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
//      W[s]=Zetar[s]*f[s]+conj(*(fm-s)*Zetar[m-s]);
//      V[s]=conj(Zetar[s])*f[s]+conj(*(fm-s))*Zetar[m-s];
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

  unsigned int e1=e+1;
  Complex *fm=f+Cm;
  unsigned int H=ceilquotient(L,2);
  unsigned int mH1=m-H+1;
  if(r == 0) {
    unsigned int q2=q/2;
    if(2*q2 < q) { // q odd
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

      crfftm->fft(W,F);

      if(m > 2*e) {
        unsigned int offset=C*(e1+e);
        double *Fr=(double *) F+offset;
        for(unsigned int c=0; c < C; ++c)
          Fr[c]=0.0;
      }
    } else { // q even
      Complex *G=F+b;
      Complex *V=W == F ? G : W+C*e1;

      Complex *Zetar=Zetaqm+m*q2;

      for(unsigned int s=1; s < mH1; ++s) {
        unsigned int Cs=C*s;
        Complex *fs=f+Cs;
        Complex *Ws=W+Cs;
        Complex *Vs=V+Cs;
        Complex Zetars=Zetar[s];
        for(unsigned int c=0; c < C; ++c) {
          Complex fsc=fs[c];
          Ws[c]=fsc;
          Vs[c]=Zetars*fsc;
        }
      }

      for(unsigned int s=mH1; s <= e; ++s) {
        unsigned int Cs=C*s;
        Complex *fs=f+Cs;
        Complex *fms=fm-Cs;
        Complex *Ws=W+Cs;
        Complex *Vs=V+Cs;
        Complex Zetars=Zetar[s];
        for(unsigned int c=0; c < C; ++c) {
          Complex fsc=fs[c];
          Complex fmsc=conj(fms[c]);
          Ws[c]=fsc+fmsc;
          Vs[c]=Zetars*(fsc-fmsc);
        }
      }

      for(unsigned int c=0; c < C; ++c)
        W[c]=f[c];
      crfftm->fft(W,F);

      for(unsigned int c=0; c < C; ++c)
        V[c]=f[c];
      crfftm->fft(V,G);

      if(m > 2*e) {
        unsigned int offset=C*(e1+e);
        double *Fr=(double *) F+offset;
        double *Gr=(double *) G+offset;
        for(unsigned int c=0; c < C; ++c)
          Fr[c]=0.0;
        for(unsigned int c=0; c < C; ++c)
          Gr[c]=0.0;
      }
    }
  } else {
    Complex *G=F+b;
    Complex *V=W == F ? G : W+C*e1;
    Complex *Zetar=Zetaqm+m*r;
//    cout << r << " " << Zetar[1] << endl;

    for(unsigned int s=1; s < mH1; ++s) {
      unsigned int Cs=C*s;
      Complex *Ws=W+Cs;
      Complex *Vs=V+Cs;
      Complex *fs=f+Cs;
      Vec Zeta=LOAD(Zetar+s);
      for(unsigned int c=0; c < C; ++c) {
//        Ws[c]=Zetars*fs[c];
//        Vs[c]=conj(Zetars)*fs[c];
        Vec fsc=LOAD(fs+c);
        Vec A=Zeta*UNPACKL(fsc,fsc);
        Vec B=ZMULTI(Zeta*UNPACKH(fsc,fsc));
        STORE(Ws+c,A+B);
        STORE(Vs+c,CONJ(A-B));
      }
    }
    Complex *Zetarm=Zetar+m;
    for(unsigned int s=mH1; s <= e; ++s) {
      unsigned int Cs=C*s;
      Complex *Ws=W+Cs;
      Complex *Vs=V+Cs;
      Complex *fs=f+Cs;
      Complex *fms=fm-Cs;
      Vec Zeta=LOAD(Zetar+s);
      Vec Zetam=CONJ(LOAD(Zetarm-s));
      for(unsigned int c=0; c < C; ++c) {
//        Complex A=Zeta*fs[c].re+Zetam*fms[c].re;
//        Complex B=Zeta*fs[c].im-Zetam*fms[c].im;
//        Ws[c]=Complex(A.re-B.im,A.im+B.re);
//        Vs[c]=Complex(A.re+B.im,B.re-A.im);
        Vec fsc=LOAD(fs+c);
        Vec fmsc=LOAD(fms+c);
        Vec A=Zeta*UNPACKL(fsc,fsc)+Zetam*UNPACKL(fmsc,fmsc);
        Vec B=ZMULTI(Zeta*UNPACKH(fsc,fsc)-Zetam*UNPACKH(fmsc,fmsc));
        STORE(Ws+c,A+B);
        STORE(Vs+c,CONJ(A-B));
      }
    }

    for(unsigned int c=0; c < C; ++c)
      W[c]=f[c];

    crfftm->fft(W,F);

    for(unsigned int c=0; c < C; ++c)
      V[c]=f[c];

    crfftm->fft(V,G);

    if(m > 2*e) {
      unsigned int offset=C*(e1+e);
      double *Fr=(double *) F+offset;
      double *Gr=(double *) G+offset;
      for(unsigned int c=0; c < C; ++c)
        Fr[c]=0.0;
      for(unsigned int c=0; c < C; ++c)
        Gr[c]=0.0;
    }
  }
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
  if(r == 0) {
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
      STORE(f+s,LOAD(f+s)+ZCMULT(Zeta,LOAD(W+s))+ZMULT(Zeta,LOAD(V+s)));
    }
    Complex *Zetarm=Zetar+m;
    for(unsigned int s=mH1; s < me; ++s) {
//    f[s] += conj(Zeta)*W[s]+Zeta*V[s];
//    *(fm-s) += conj(Zetam*W[s])+Zetam*conj(V[s]);
      Vec Zeta=LOAD(Zetar+s);
      Vec Zetam=LOAD(Zetarm-s);
      Vec Ws=LOAD(W+s);
      Vec Vs=LOAD(V+s);
      STORE(f+s,LOAD(f+s)+ZMULT2(Zeta,Ws,Vs));
      STORE(fm-s,LOAD(fm-s)+CONJ(ZMULT2(Zetam,Vs,Ws)));
    }
    if(even) {
//      f[e] += conj(Zetar[e])*W[e]+Zetar[e]*V[e];
      Vec Zeta=LOAD(Zetar+e);
      STORE(f+e,LOAD(f+e)+ZCMULT(Zeta,LOAD(W+e))+ZMULT(Zeta,LOAD(V+e)));
      if(inplace)
        G[e]=Nyquist; // Restore initial input of next residue
    }
  }
}

void fftPadHermitian::backward2Many(Complex *F, Complex *f, unsigned int r, Complex *W)
{
  if(W == NULL) W=F;

  Complex Nyquist[C];
  bool inplace=W == F;
  if(inplace) {
    Complex *FCe=F+C*e;
    for(unsigned int c=0; c < C; ++c)
      Nyquist[c]=FCe[c]; // Save before being overwritten
  }

  rcfftm->fft(F,W);

  Complex *fm=f+Cm;
  bool even=m == 2*e;
  bool overlap=even && inplace;
  unsigned int me=m-e;
  unsigned int H=ceilquotient(L,2);
  unsigned int mH1=m-H+1;
  unsigned int Ce=C*e;
  if(r == 0) {
    unsigned int q2=q/2;
    if(2*q2 < q) { // q odd
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
      if(even) {
        Complex *fe=f+Ce;
        Complex *WCe=W+Ce;
        for(unsigned int c=0; c < C; ++c)
          fe[c]=WCe[c];
        if(inplace) {
          Complex *Fe=F+Ce;
          for(unsigned int c=0; c < C; ++c)
            Fe[c]=Nyquist[c]; // Restore initial input of next residue
        }
      }
    } else { // q even
      unsigned int e1=e+1;
      Complex *G=F+b;
      Complex *V=inplace ? G : W+C*e1;

      Complex We[C];
      if(overlap) {
        Complex *WCe=W+Ce;
        Complex *Ge=G+Ce;
        for(unsigned int c=0; c < C; ++c) {
          We[c]=WCe[c];
          WCe[c]=Nyquist[c]; // Restore initial input of next residue
          Nyquist[c]=Ge[c];
        }
      }

      rcfftm->fft(G,V);

      for(unsigned int c=0; c < C; ++c)
        f[c]=W[c]+V[c];

      if(overlap) {
        Complex *WCe=W+Ce;
        for(unsigned int c=0; c < C; ++c)
          WCe[c]=We[c];
      }

      Complex *Zetar=Zetaqm+m*q2;

      for(unsigned int s=1; s < mH1; ++s) {
        unsigned int Cs=C*s;
        Complex *fs=f+Cs;
        Complex *Ws=W+Cs;
        Complex *Vs=V+Cs;
        Complex Zetars=conj(Zetar[s]);
        for(unsigned int c=0; c < C; ++c)
          fs[c]=Ws[c]+Zetars*Vs[c];
      }

      for(unsigned int s=mH1; s < me; ++s) {
        unsigned int Cs=C*s;
        Complex *fs=f+Cs;
        Complex *fms=fm-Cs;
        Complex *Ws=W+Cs;
        Complex *Vs=V+Cs;
        Complex Zetars=conj(Zetar[s]);
        for(unsigned int c=0; c < C; ++c) {
          Complex A=Ws[c];
          Complex B=Zetars*Vs[c];
          fs[c]=A+B;
          fms[c]=conj(A-B);
        }
      }

      if(even) {
        Complex *fe=f+Ce;
        Complex *We=W+Ce;
        Complex *Ve=V+Ce;
        for(unsigned int c=0; c < C; ++c)
          fe[c]=We[c]-I*Ve[c];
        if(inplace) {
          Complex *Ge=G+Ce;
          for(unsigned int c=0; c < C; ++c)
            Ge[c]=Nyquist[c]; // Restore initial input of next residue
        }
      }
    }
  } else {
    unsigned int e1=e+1;
    Complex *G=F+b;
    Complex *V=inplace ? G : W+C*e1;

    Complex We[C];
    if(overlap) {
      Complex *WCe=W+Ce;
      for(unsigned int c=0; c < C; ++c)
        We[c]=WCe[c];
      for(unsigned int c=0; c < C; ++c)
        WCe[c]=Nyquist[c]; // Restore initial input of next residue
      Complex *Ge=G+Ce;
      for(unsigned int c=0; c < C; ++c)
        Nyquist[c]=Ge[c];
    }

    rcfftm->fft(G,V);

    for(unsigned int c=0; c < C; ++c)
      f[c] += W[c]+V[c];

    if(overlap) {
      Complex *WCe=W+Ce;
      for(unsigned int c=0; c < C; ++c)
        WCe[c]=We[c];
    }

    Complex *Zetar=Zetaqm+m*r;
    for(unsigned int s=1; s < mH1; ++s) {
      unsigned int Cs=C*s;
      Complex *fs=f+Cs;
      Complex *Ws=W+Cs;
      Complex *Vs=V+Cs;
      Vec Zeta=LOAD(Zetar+s);
      for(unsigned int c=0; c < C; ++c)
//        fs[c] += conj(Zetars)*Ws[c]+Zetars*Vs[c];
        STORE(fs+c,LOAD(fs+c)+ZCMULT(Zeta,LOAD(Ws+c))+ZMULT(Zeta,LOAD(Vs+c)));
    }

    Complex *Zetarm=Zetar+m;
    for(unsigned int s=mH1; s < me; ++s) {
      unsigned int Cs=C*s;
      Complex *fs=f+Cs;
      Complex *fms=fm-Cs;
      Complex *Ws=W+Cs;
      Complex *Vs=V+Cs;
      Vec Zeta=LOAD(Zetar+s);
      Vec Zetam=LOAD(Zetarm-s);
      for(unsigned int c=0; c < C; ++c) {
//        fs[c] += conj(Zeta)*Ws[c]+Zeta*Vs[c];
//        fms[c] += conj(Zetam*Ws[c])+Zetam*conj(Vs[c]);
        Vec Wsc=LOAD(Ws+c);
        Vec Vsc=LOAD(Vs+c);
        STORE(fs+c,LOAD(fs+c)+ZCMULT(Zeta,Wsc)+ZMULT(Zeta,Vsc));
        STORE(fms+c,LOAD(fms+c)+CONJ(ZMULT(Zetam,Wsc))+ZCMULT(Vsc,Zetam));
      }
    }
    if(even) {
      Complex *fe=f+Ce;
      Complex *We=W+Ce;
      Complex *Ve=V+Ce;
      Vec Zeta=LOAD(Zetar+e);
      for(unsigned int c=0; c < C; ++c)
//        fe[c] += conj(Zetare)*We[c]+Zetare*Ve[c];
        STORE(fe+c,LOAD(fe+c)+ZCMULT(Zeta,LOAD(We+c))+ZMULT(Zeta,LOAD(Ve+c)));

    }
    if(inplace) {
      Complex *GCe=G+Ce;
      for(unsigned int c=0; c < C; ++c)
        GCe[c]=Nyquist[c]; // Restore initial input of next residue
    }
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
  D0=fft->D0;
  dr=fft->dr;
  b=fft->b;
  R=fft->residueBlocks();

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

    nloops=fft->nloops();
    loop2=fft->loop2();
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
      unsigned int r=ceilquotient(D0,2);
      for(unsigned int a=0; a < A; ++a)
        (fft->*Forward)(f[a]+offset,F[a],0,W);
      (*mult)(F,D0*b,threads);

      for(unsigned int b=0; b < B; ++b) {
        (fft->*Forward)(f[b]+offset,Fp[b],r,W);
        (fft->*Backward)(F[b],h[b]+offset,0,W0);
        (fft->*Pad)(W);
      }
      for(unsigned int a=B; a < A; ++a)
        (fft->*Forward)(f[a]+offset,Fp[a],r,W);
      (*mult)(Fp,D*b,threads);
      for(unsigned int b=0; b < B; ++b)
        (fft->*Backward)(Fp[b],h[b]+offset,r,FpB);
    } else {
      unsigned int Offset;
      bool useV=h == f && nloops > 1;
      Complex **h0;
      if(useV) {
        if(!V) initV();
        h0=V;
        Offset=0;
      } else {
        Offset=offset;
        h0=h;
      }

      for(unsigned int r=0; r < R; r += fft->increment(r)) {
        for(unsigned int a=0; a < A; ++a)
          (fft->*Forward)(f[a]+offset,F[a],r,W);
        (*mult)(F,(r == 0 ? D0 : D)*b,threads);
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
  D0=fft.D0;
  dr=fft.dr;

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
  unsigned int R=fft.R;
  double t0=totalseconds();
  for(unsigned int k=0; k < K; ++k) {
    for(unsigned int r=0; r < R; r += fft.increment(r)) {
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
        if(DOption > 1 && DOption % 2) ++DOption;
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
