#include "convolve.h"
#include "cmult-sse2.h"

// TODO:
// Implement overwrite optimization in fftpad::forwardMany
// Implement loop2 optimization in Convolution2
// Optimize over -I0 and -I1
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

//unsigned int surplusFFTsizes=25;
double epsilon=0.1; // TEMPORARY

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
// two Complex inputs of size n.
// F0[j] *= F1[j];
void multbinary(Complex **F, unsigned int n, unsigned int threads)
{
  Complex *F0=F[0];
  Complex *F1=F[1];

  PARALLEL(
    for(unsigned int j=0; j < n; ++j)
      F0[j] *= F1[j];
    );
}

// This multiplication routine is for binary convolutions and takes
// two real inputs of size n.
// F0[j] *= F1[j];
void realmultbinary(Complex **F, unsigned int n, unsigned int threads)
{
  double *F0=(double *) F[0];
  double *F1=(double *) F[1];

  PARALLEL(
    for(unsigned int j=0; j < n; ++j)
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

unsigned int prevfftsize(unsigned int M, bool mixed)
{
  int P[]={2,3,5,7};
  int i;
  for(i=M; i > 10; --i) {
    int m=i;
    for(int j=0; j < 4; j++) {
      int p=P[j];
      while(m % p == 0)
        m /= p;
      if(!mixed) {
        if(m != 1)
          m=i;
        else
          return i;
      }
    }
    if(m == 1) return i;
  }
  return i;
}

void fftBase::initZetaqm(unsigned int q, unsigned int m)
{
  double twopibyM=twopi/M;
  Zetaqm0=ComplexAlign((q-1)*m);
  Zetaqm=Zetaqm0-m;
  for(unsigned int r=1; r < q; ++r) {
    unsigned int mr=m*r;
    Zetaqm[mr]=1.0;
    for(unsigned int s=1; s < m; ++s)
      Zetaqm[mr+s]=expi(r*s*twopibyM);
  }
}

void fftBase::OptBase::check(unsigned int L, unsigned int M,
                             Application& app, unsigned int C, unsigned int m,
                             bool fixed, bool mForced, bool centered)
{
//    cout << "m=" << m << endl;
  unsigned int q=ceilquotient(M,m);
  unsigned int p=ceilquotient(L,m);
  unsigned int p2=p/2;
  unsigned int P=(centered && p == 2*p2) || p == 2 ? p2 : p;

  if(p == q && p > 1 && !mForced) return;

  unsigned int n=ceilquotient(M,m*P);

  if(p > 2 && !fixed) {
    unsigned int q2=P*n;
    if(q2 != q) {
      unsigned int start=DOption > 0 ? min(DOption,n) : 1;
      unsigned int stop=DOption > 0 ? min(DOption,n) : n;
      if(fixed || C > 1) start=stop=1;
      unsigned int stop2=2*stop;
      for(unsigned int D=start; D < stop2; D *= 2) {
        if(D > stop) D=stop;
        if(!valid(D,p)) continue;
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

  if(p > 2 && q % P != 0) return;

  unsigned int start=DOption > 0 ? min(DOption,n) : 1;
  unsigned int stop=DOption > 0 ? min(DOption,n) : n;
  if(fixed || C > 1) start=stop=1;
  unsigned int stop2=2*stop;
  for(unsigned int D=start; D < stop2; D *= 2) {
    if(D > stop) D=stop;
    if(!valid(D,p)) continue;
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
                            unsigned int C, bool Explicit, bool fixed,
                            bool centered)
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

  double sqrtM=sqrt(M);
  unsigned int Mmore=M+max(M*epsilon,1);
  unsigned int ub=Mmore;
  unsigned int lb=M;
  unsigned int stop=M-1;
  unsigned int m0=ub;
  unsigned int denom=2;
  bool mixed=true;

  if(mOption >= 1 && !Explicit)
    check(L,M,app,C,mOption,fixed,true,centered);
  else
    while(true){
      m0=prevfftsize(m0-1,mixed);
      if(mixed == true && m0 < L/2) mixed=false;
      if(m0 < lb){
        double factor=1.0/denom;
        lb=M*factor;
        ub=Mmore*factor;
        denom++;
        while(m0 > ub)
          m0=prevfftsize(m0-1,mixed);
      }
      //cout << m0 << endl;
      if(Explicit) {
        if(m0 > stop) break;
        if(m0 < M) {++i; continue;}
        M=m0;
      } else if(m0 <= sqrtM) break;
//        } else if(m0 > L) break;
      if(!fixed || Explicit || M % m0 == 0)
        check(L,M,app,C,m0,fixed || Explicit,centered);
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
  if(ZetaqmS)
    deleteAlign(ZetaqmS0);
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
  if(q*m < M) {
    cerr << "Invalid parameters." << endl;
    exit(-1);
  }

  if(C > 1) D=1;
  p=ceilquotient(L,m);

  inplace=IOption == -1 ? C > 1 : IOption;

  Cm=C*m;
  n=q/p;
  M=m*q;
  Pad=&fftBase::padNone;
  Zetaqp=NULL;
  ZetaqmS=NULL;
  overwrite=false;
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
    Complex *G=ComplexAlign(Cm);
    fftm=new mfft1d(m,1,C, C,1, G);
    ifftm=new mfft1d(m,-1,C, C,1, G);
    deleteAlign(G);
    dr=D0=R=Q=1;
    b=C*M;
  } else {
    double twopibyN=twopi/M;
    double twopibyq=twopi/q;

    unsigned int d;

    unsigned int P,P1;
    unsigned int p2=p/2;

    if(p == 2) {
      Q=n=q;
      P1=P=1;
    } else if (centered && p == 2*p2) {
      Q=n=q/p2;
      P=p2;
      P1=P+1;
    } else
      P1=P=p;

    b=Cm*P;
    d=C*D*P;

    Complex *G,*H;
    unsigned int size=m*d;

    G=ComplexAlign(size);
    H=inplace ? G : ComplexAlign(size);

    if(p > 2) { // Implies L > 2m
      if(C == 1) {
        Forward=&fftBase::forwardInner;
        Backward=&fftBase::backwardInner;
      } else {
        Forward=&fftBase::forwardInnerMany;
        Backward=&fftBase::backwardInnerMany;
      }
      Q=n;

      Zetaqp0=ComplexAlign((n-1)*(P1-1));
      Zetaqp=Zetaqp0-P1;
      for(unsigned int r=1; r < n; ++r)
        for(unsigned int t=1; t < P1; ++t)
          Zetaqp[(P1-1)*r+t]=expi(r*t*twopibyq);

      // L'=p, M'=q, m'=p, p'=1, q'=n
      fftp=new mfft1d(P,1,Cm, Cm,1, G);
      ifftp=new mfft1d(P,-1,Cm, Cm,1, G);
    } else {
      if(p == 2) {
        if(!centered) {
          unsigned int Lm=L-m;
          ZetaqmS0=ComplexAlign((q-1)*Lm);
          ZetaqmS=ZetaqmS0-L;
          for(unsigned int r=1; r < q; ++r) {
            for(unsigned int s=m; s < L; ++s)
              ZetaqmS[Lm*r+s]=expi(r*s*twopibyN);
          }
        }

        if(C == 1) {
          Forward=&fftBase::forward2;
          Backward=&fftBase::backward2;
        } else {
          Forward=&fftBase::forward2Many;
          Backward=&fftBase::backward2Many;
        }
      } else { // p == 1
        if(C == 1) {
          Forward=&fftBase::forward1;
          Backward=&fftBase::backward1;
          if(repad())
            Pad=&fftBase::padSingle;
        } else {
          Forward=&fftBase::forward1Many;
          Backward=&fftBase::backward1Many;
          if(repad())
            Pad=&fftBase::padMany;
        }
        Q=q;
      }
    }

    dr=Dr();
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
      unsigned int x=D0*P;
      fftm2=new mfft1d(m,1,x, 1,m, G,H);
      ifftm2=new mfft1d(m,-1,x, 1,m, G,H);
    } else
      fftm2=NULL;

    if(!inplace)
      deleteAlign(H);
    deleteAlign(G);

    initZetaqm(q,centered && p == 2 ? m+1 : m);
  }
}

fftPad:: ~fftPad() {
  if(q == 1) {
    delete fftm;
    delete ifftm;
  } else {
    if(Zetaqp) {
      deleteAlign(Zetaqp0);
      delete fftp;
      delete ifftp;
    }
    if(fftm2) {
      delete fftm2;
      delete ifftm2;
    }
    delete fftm;
    delete ifftm;
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

void fftPad::backwardExplicit(Complex *F, Complex *f, unsigned int, Complex *W, double) {
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

void fftPad::backwardExplicitMany(Complex *F, Complex *f, unsigned int, Complex *W, double) {
  ifftm->fft(F);
  for(unsigned int s=0; s < L; ++s) {
    Complex *fs=f+C*s;
    Complex *Fs=F+C*s;
    for(unsigned int c=0; c < C; ++c)
      fs[c]=Fs[c];
  }
}

void fftPad::forward1(Complex *f, Complex *F0, unsigned int r0, Complex *W)
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
      if(!inplace && D == 1 && L >= m)
        return fftm->fft(f,F0);
      residues=1;
      for(unsigned int s=0; s < L; ++s)
        W[s]=f[s];
    } else { // q even, r=0,q/2
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

void fftPad::forward1Many(Complex *f, Complex *F, unsigned int r, Complex *W) {
  if(W == NULL) W=F;

  if(W == F) {
    for(unsigned int s=L; s < m; ++s) {
      Complex *Fs=W+C*s;
      for(unsigned int c=0; c < C; ++c)
        Fs[c]=0.0;
    }
  }

  if(r == 0) {
    if(!inplace && L >= m)
      return fftm->fft(f,F);
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
      Complex *Zetar2=ZetaqmS+Lm*r0+m;
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
      Complex *Zetar2=ZetaqmS+Lm*r+m;
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
    Complex *Zetar2=ZetaqmS+Lm*r+m;
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

void fftPad::backward1(Complex *F0, Complex *f, unsigned int r0, Complex *W, double)
{
  if(W == NULL) W=F0;

  if(r0 == 0 && !inplace && D == 1 && L >= m)
    return ifftm->fft(F0,f);

  (r0 > 0 || D0 == D ? ifftm : ifftm2)->fft(F0,W);

  unsigned int dr0=dr;

  if(r0 == 0) {
    unsigned int residues;
    unsigned int q2=q/2;
    if(D == 1 || 2*q2 < q) { // q odd, r=0
      residues=1;
      for(unsigned int s=0; s < L; ++s)
        f[s]=W[s];
    } else { // q even, r=0,q/2
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

void fftPad::backward1Many(Complex *F, Complex *f, unsigned int r, Complex *W, double)
{
  if(W == NULL) W=F;

  if(r == 0 && !inplace && L >= m)
    return ifftm->fft(F,f);

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

void fftPad::backward2(Complex *F0, Complex *f, unsigned int r0, Complex *W, double)
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
    } else { // q even, r=0,q/2
      residues=2;
      Complex *V=W+m;
      f[0]=W[0]+V[0];
      Complex *Zetar=Zetaqm+m*q2;
      for(unsigned int s=1; s < m; ++s)
        f[s]=W[s]+Zetar[s]*V[s];
      Complex *Zetar2=ZetaqmS+Lm*q2;
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
      Complex *Zetar2=ZetaqmS+Lm*r0;
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
//        STORE(f+s,LOAD(f+s)+ZCMULT(Zeta,Us)+ZMULT(Zeta,Vs));
        STORE(f+s,LOAD(f+s)+ZMULT2(Zeta,Vs,Us)); // CHECK
      }
      Complex *Zetar2=ZetaqmS+Lm*r;
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

void fftPad::backward2Many(Complex *F, Complex *f, unsigned int r, Complex *W, double)
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
    Complex *Zetar2=ZetaqmS+Lm*r;
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

void fftPad::backwardInner(Complex *F0, Complex *f, unsigned int r0, Complex *W, double)
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

void fftPad::backwardInnerMany(Complex *F, Complex *f, unsigned int r, Complex *W, double)
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

void fftPadCentered::init(bool fast)
{
  if(q == 1 && fast) {
    ZetaShift=NULL;
    fftBaseForward=Forward;
    fftBaseBackward=Backward;
    Forward=&fftBase::forwardShiftedExplicit;
    Backward=&fftBase::backwardShiftedExplicit;
  } else {
    ZetaShift=NULL;
//  if(q > 1 && p % 2 == 0) { // Eventually
    if(q > 1 && p % 2 == 0 && (p == 2 || C == 1)) { // Temporary
      if(p == 2) {
        if(C == 1) {
          Forward=&fftBase::forward2C;
          Backward=&fftBase::backward2C;
        } else {
          Forward=&fftBase::forward2CMany;
          Backward=&fftBase::backward2CMany;
          overwrite=inplace && L == p*m && q == p+1;
        }
      } else { // p even > 2
        if(C == 1) {
          Forward=&fftBase::forwardInnerC;
          Backward=&fftBase::backwardInnerC;
        } else {
//        Forward=&fftBase::forwardInnerCMany;
//        Backward=&fftBase::backwardInnerCMany;
        }
      }
    } else {
      initShift();
      fftBaseForward=Forward;
      fftBaseBackward=Backward;
      Forward=&fftBase::forwardShifted;
      Backward=&fftBase::backwardShifted;
    }
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

void fftPadCentered::forwardShiftedExplicit(Complex *f, Complex *F, unsigned int, Complex *)
{
  (this->*fftBaseForward)(f,F,0,NULL);

  for(unsigned int s=1; s < m; s += 2) {
    for(unsigned int c=0; c < C; ++c) {
      F[C*s+c] *= -1;
    }
  }
}

void fftPadCentered::backwardShiftedExplicit(Complex *F, Complex *f, unsigned int, Complex *, double)
{
  for(unsigned int s=1; s < m; s += 2) {
    for(unsigned int c=0; c < C; ++c) {
      F[C*s+c] *= -1;
    }
  }
  (this->*fftBaseBackward)(F,f,0,NULL,1.0);
}

void fftPadCentered::forwardShifted(Complex *f, Complex *F, unsigned int r, Complex *W)
{
  (this->*fftBaseForward)(f,F,r,W);
  forwardShift(F,r);
}

void fftPadCentered::backwardShifted(Complex *F, Complex *f, unsigned int r, Complex *W, double)
{
  backwardShift(F,r);
  (this->*fftBaseBackward)(F,f,r,W,1.0);
}

void fftPadCentered::forwardShift(Complex *F, unsigned int r0)
{
  unsigned int dr0=r0 == 0 ? D0 : D;
  for(unsigned int d=0; d < dr0; ++d) {
    Complex *W=F+b*d;
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
    Complex *W=F+b*d;
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

void fftPadCentered::forward2C(Complex *f, Complex *F0, unsigned int r0, Complex *W)
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
        Vec A=Zetam*UNPACKL(fms,fms)+Zeta*UNPACKL(fs,fs);
        Vec B=ZMULTI(Zetam*UNPACKH(fms,fms)+Zeta*UNPACKH(fs,fs));
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

void fftPadCentered::forward2CMany(Complex *f, Complex *F, unsigned int r, Complex *W)
{
  if(r == R) {
    Complex *g=f+Cm;
    Complex *Zetar=Zetaqm+(m+1);
    Complex *Zetarm=Zetar+m;

    for(unsigned int s=0; s < m; ++s) {
      unsigned int Cs=C*s;
      Complex *F0=f+Cs;
      Complex *F1=g+Cs;
      Complex *F2=F+Cs;
      Vec Zeta=LOAD(Zetar+s);
      Vec Zetam=CONJ(LOAD(Zetarm-s));
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=CONJ(UNPACKH(Zeta,Zeta));
      Vec Xm=UNPACKL(Zetam,Zetam);
      Vec Ym=CONJ(UNPACKH(Zetam,Zetam));
      for(unsigned int c=0; c < C; ++c) {
//    Fs[c]=Zetarms*fmHs[c]+Zetars*fHs[c];
//    Gs[c]=conj(Zetarms)*fmHs[c]+conj(Zetars)*fHs[c];
        Vec F0s=LOAD(F0+c);
        Vec F1s=LOAD(F1+c);
        Vec A=Xm*F0s+X*F1s;
        Vec B=FLIP(Ym*F0s+Y*F1s);
        STORE(F0+c,F0s+F1s);
        STORE(F1+c,A+B);
        STORE(F2+c,A-B);
      }
    }
    fftm->fft(f);
    fftm->fft(g);
    fftm->fft(F);
    return;
  }

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

void fftPadCentered::backward2C(Complex *F0, Complex *f, unsigned int r0, Complex *W, double)
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
    } else { // q even, r=0,q/2
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

void fftPadCentered::backward2CMany(Complex *F, Complex *f, unsigned int r, Complex *W, double scale)
{
  if(r == R) {
    Complex *g=f+Cm;

    ifftm->fft(f);
    ifftm->fft(g);
    ifftm->fft(F);

    Complex *Zetar=Zetaqm+(m+1);
    Complex *Zetarm=Zetar+m;
    Vec Scale=LOAD2(scale);
    for(unsigned int s=0; s < m; ++s) {
      unsigned int Cs=C*s;
      Complex *F0=f+Cs;
      Complex *F1=g+Cs;
      Complex *F2=F+Cs;
      Vec Zetam=Scale*LOAD(Zetarm-s);
      Vec Zeta=Scale*LOAD(Zetar+s);
      Vec Xm=UNPACKL(Zetam,Zetam);
      Vec Ym=CONJ(UNPACKH(Zetam,Zetam));
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=REFL(UNPACKH(Zeta,Zeta));
      for(unsigned int c=0; c < C; ++c) {
//    fmHs[c] += Zetarms*Fs[c]+conj(Zetarms)*Gs[c];
//    fHs[c] += Zetars*Fs[c]+conj(Zetars)*Gs[c];
        Vec F0c=Scale*LOAD(F0+c);
        Vec F1c=LOAD(F1+c);
        Vec F2c=LOAD(F2+c);
        STORE(F0+c,F0c+ZMULT2(Xm,Ym,F1c,F2c));
        STORE(F1+c,F0c+ZMULT2(X,Y,F1c,F2c));
      }
    }
    return;
  }

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

void fftPadCentered::forwardInnerC(Complex *f, Complex *F0, unsigned int r0, Complex *W)
{
  if(W == NULL) W=F0;

  Complex *W0=W;
  unsigned int dr0=dr;
  mfft1d *fftm0;

  unsigned int p2=p/2;
  unsigned int H=L/2;
  unsigned int p2s1=p2-1;
  unsigned int p2s1m=p2s1*m;
  unsigned int p2m=p2*m;
  unsigned int m0=p2m-H;
  unsigned int m1=L-H-p2s1m;
  Complex *fm0=f-m0;
  Complex *fH=f+H;
  if(r0 == 0) {
    unsigned int residues;
    unsigned int n2=n/2;
    if(D == 1 || 2*n2 < n) { // n odd, r=0
      residues=1;
      for(unsigned int s=0; s < m0; ++s)
        W[s]=fH[s];
      for(unsigned int s=m0; s < m; ++s)
        W[s]=fm0[s]+fH[s];
      for(unsigned int t=1; t < p2s1; ++t) {
        unsigned int tm=t*m;
        Complex *Wt=W+tm;
        Complex *fm0t=fm0+tm;
        Complex *fHt=fH+tm;
        for(unsigned int s=0; s < m; ++s) {
          Wt[s]=fm0t[s]+fHt[s];
        }
      }
      Complex *Wt=W+p2s1m;
      Complex *fm0t=fm0+p2s1m;
      Complex *fHt=fH+p2s1m;
      for(unsigned int s=0; s < m1; ++s)
        Wt[s]=fm0t[s]+fHt[s];
      for(unsigned int s=m1; s < m; ++s)
        Wt[s]=fm0t[s];

      fftp->fft(W);

      unsigned int mn=m*n;
      for(unsigned int u=1; u < p2; ++u) {
        Complex *Wu=W+u*m;
        Complex *Zeta0=Zetaqm+mn*u;
        for(unsigned int s=1; s < m; ++s)
          Wu[s] *= Zeta0[s];
      }
    } else { // n even, r=0,n/2
      residues=2;
      Complex *V=W+b;
      for(unsigned int s=0; s < m0; ++s)
        V[s]=W[s]=fH[s];
      for(unsigned int s=m0; s < m; ++s) {
        Complex fm0s=fm0[s];
        Complex fHs=fH[s];
        W[s]=fm0s+fHs;
        V[s]=-fm0s+fHs;
      }
      Complex *Zetaqn2=Zetaqp+p2*n2;
      for(unsigned int t=1; t < p2s1; ++t) {
        unsigned int tm=t*m;
        Complex *Wt=W+tm;
        Complex *Vt=V+tm;
        Complex *fm0t=fm0+tm;
        Complex *fHt=fH+tm;
        Complex Zeta=Zetaqn2[t]; //*zeta_q^{tn/2}
        for(unsigned int s=0; s < m; ++s) {
          Complex fm0ts=fm0t[s];
          Complex fHts=fHt[s];
          Wt[s]=fm0ts+fHts;
          Vt[s]=Zeta*(fHts-fm0ts);
        }
      }
      Complex *Wt=W+p2s1m;
      Complex *Vt=V+p2s1m;
      Complex *fm0t=fm0+p2s1m;
      Complex *fHt=fH+p2s1m;
      Complex Zeta=Zetaqn2[p2s1];
      for(unsigned int s=0; s < m1; ++s) {
        Complex fm0ts=fm0t[s];
        Complex fHts=fHt[s];
        Wt[s]=fm0ts+fHts;
        Vt[s]=Zeta*(fHts-fm0ts);
      }
      Complex mZeta=-Zeta;
      for(unsigned int s=m1; s < m; ++s) {
        Complex fm0ts=fm0t[s];
        Wt[s]=fm0ts;
        Vt[s]=mZeta*fm0ts;
      }
      fftp->fft(W);
      fftp->fft(V);

      unsigned int mn=m*n;
      unsigned int mn2=m*n2;
      Complex *Zetan2=Zetaqm+mn2;
      for(unsigned int s=0; s < m; ++s)
        V[s] *= Zetan2[s];
      for(unsigned int u=1; u < p2; ++u) {
        unsigned int um=u*m;
        Complex *Wu=W+um;
        Complex *Zeta0=Zetaqm+mn*u;
        for(unsigned int s=1; s < m; ++s)
          Wu[s] *= Zeta0[s];
        Complex *Vu=V+um;
        Complex *Zetan2=Zeta0+mn2;
        for(unsigned int s=1; s < m; ++s)
          Vu[s] *= Zetan2[s];
      }
    }
    W += residues*b;
    r0=1;
    dr0=(D0-residues)/2;
    fftm0=D0 == D ? fftm : fftm2;
  } else
    fftm0=fftm;

  if(D == 1) {
    if(dr0 > 0) {
      Vec Zetanr=CONJ(LOAD(Zetaqp+p2*r0+p2)); // zeta_n^-r

      for(unsigned int s=0; s < m0; ++s)
        W[s]=fH[s];

      Vec X=UNPACKL(Zetanr,Zetanr);
      Vec Y=CONJ(UNPACKH(Zetanr,Zetanr));
      for(unsigned int s=m0; s < m; ++s)
        STORE(W+s,ZMULT(X,Y,LOAD(fm0+s))+LOAD(fH+s));
      Complex *Zetaqr=Zetaqp+p2*r0;
      for(unsigned int t=1; t < p2s1; ++t) {
        unsigned int tm=t*m;
        Complex *Ft=W+tm;
        Complex *fm0t=fm0+tm;
        Complex *fHt=fH+tm;
        Vec Zeta=LOAD(Zetaqr+t); // zeta_q^{tr}
        Vec Zetam=ZMULT(Zeta,Zetanr);

        Vec Xm=UNPACKL(Zetam,Zetam);
        Vec Ym=CONJ(UNPACKH(Zetam,Zetam));
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=CONJ(UNPACKH(Zeta,Zeta));
        for(unsigned int s=0; s < m; ++s)
          STORE(Ft+s,ZMULT(Xm,Ym,LOAD(fm0t+s))+ZMULT(X,Y,LOAD(fHt+s)));
      }
      Complex *Ft=W+p2s1m;
      Complex *fm0t=fm0+p2s1m;
      Complex *fHt=fH+p2s1m;
      Vec Zeta=LOAD(Zetaqr+p2s1);
      Vec Zetam=ZMULT(Zeta,Zetanr);
      Vec Xm=UNPACKL(Zetam,Zetam);
      Vec Ym=CONJ(UNPACKH(Zetam,Zetam));
      X=UNPACKL(Zeta,Zeta);
      Y=CONJ(UNPACKH(Zeta,Zeta));
      for(unsigned int s=0; s < m1; ++s)
        STORE(Ft+s,ZMULT(Xm,Ym,LOAD(fm0t+s))+ZMULT(X,Y,LOAD(fHt+s)));
      for(unsigned int s=m1; s < m; ++s)
        STORE(Ft+s,ZMULT(Xm,Ym,LOAD(fm0t+s)));

      fftp->fft(W);

      unsigned int mr=m*r0;
      Complex *Zetar=Zetaqm+mr;
      for(unsigned int s=1; s < m; ++s) {
        Complex Zeta=Zetar[s];
        W[s] *= Zeta;
      }
      for(unsigned int u=1; u < p2; ++u) {
        unsigned int mu=m*u;
        Complex *Zetar0=Zetaqm+n*mu;
        Complex *Zetar=Zetar0+mr;
        Complex *Wu=W+mu;
        for(unsigned int s=1; s < m; ++s)
          Wu[s] *= Zetar[s];
      }
    }
  } else {
    for(unsigned int d=0; d < dr0; ++d) {
      Complex *F=W+2*b*d;
      unsigned int r=r0+d;
      Vec Zetanr=CONJ(LOAD(Zetaqp+p2*r+p2)); // zeta_n^-r
      Complex *G=F+b;

      for(unsigned int s=0; s < m0; ++s)
        G[s]=F[s]=fH[s];
      for(unsigned int s=m0; s < m; ++s) {
        Vec fm0ts=LOAD(fm0+s);
        Vec fHts=LOAD(fH+s);
        Vec A=Zetanr*UNPACKL(fm0ts,fm0ts);
        Vec B=ZMULTI(Zetanr*UNPACKH(fm0ts,fm0ts));
        STORE(F+s,A+B+fHts);
        STORE(G+s,CONJ(A-B)+fHts);
      }
      Complex *Zetaqr=Zetaqp+p2*r;
      for(unsigned int t=1; t < p2s1; ++t) {
        unsigned int tm=t*m;
        Complex *Ft=F+tm;
        Complex *Gt=G+tm;
        Complex *fm0t=fm0+tm;
        Complex *fHt=fH+tm;
        Vec Zeta=LOAD(Zetaqr+t); // zeta_q^{tr}
        Vec Zetam=ZMULT(Zeta,Zetanr);
        for(unsigned int s=0; s < m; ++s) {
          Vec fm0ts=LOAD(fm0t+s);
          Vec fHts=LOAD(fHt+s);
          Vec A=Zetam*UNPACKL(fm0ts,fm0ts)+Zeta*UNPACKL(fHts,fHts);
          Vec B=ZMULTI(Zetam*UNPACKH(fm0ts,fm0ts)+Zeta*UNPACKH(fHts,fHts));
          STORE(Ft+s,A+B);
          STORE(Gt+s,CONJ(A-B));
        }
      }
      Complex *Ft=F+p2s1m;
      Complex *Gt=G+p2s1m;
      Complex *fm0t=fm0+p2s1m;
      Complex *fHt=fH+p2s1m;
      Vec Zeta=LOAD(Zetaqr+p2s1);
      Vec Zetam=ZMULT(Zeta,Zetanr);
      for(unsigned int s=0; s < m1; ++s) {
        Vec fm0ts=LOAD(fm0t+s);
        Vec fHts=LOAD(fHt+s);
        Vec A=Zetam*UNPACKL(fm0ts,fm0ts)+Zeta*UNPACKL(fHts,fHts);
        Vec B=ZMULTI(Zetam*UNPACKH(fm0ts,fm0ts)+Zeta*UNPACKH(fHts,fHts));
        STORE(Ft+s,A+B);
        STORE(Gt+s,CONJ(A-B));
      }
      for(unsigned int s=m1; s < m; ++s) {
        Vec fm0ts=LOAD(fm0t+s);
        Vec A=Zetam*UNPACKL(fm0ts,fm0ts);
        Vec B=ZMULTI(Zetam*UNPACKH(fm0ts,fm0ts));
        STORE(Ft+s,A+B);
        STORE(Gt+s,CONJ(A-B));
      }

      fftp->fft(F);
      fftp->fft(G);

      unsigned int mr=m*r;
      Complex *Zetar=Zetaqm+mr;
      for(unsigned int s=1; s < m; ++s) {
        Complex Zeta=Zetar[s];
        F[s] *= Zeta;
        G[s] *= conj(Zeta);
      }
      for(unsigned int u=1; u < p2; ++u) {
        unsigned int mu=m*u;
        Complex *Zetar0=Zetaqm+n*mu;
        Complex *Zetar=Zetar0+mr;
        Complex *Wu=F+mu;
        for(unsigned int s=1; s < m; ++s)
          Wu[s] *= Zetar[s];
        Complex *Zetar2=Zetar0-mr;
        Complex *Vu=G+mu;
        for(unsigned int s=1; s < m; ++s)
          Vu[s] *= Zetar2[s];
      }
    }
  }

  fftm0->fft(W0,F0);
}

void fftPadCentered::backwardInnerC(Complex *F0, Complex *f, unsigned int r0, Complex *W, double)
{
  if(W == NULL) W=F0;

  (r0 > 0 || D0 == D ? ifftm : ifftm2)->fft(F0,W);

  unsigned int dr0=dr;

  unsigned int p2=p/2;
  unsigned int H=L/2;
  unsigned int p2s1=p2-1;
  unsigned int p2s1m=p2s1*m;
  unsigned int p2m=p2*m;
  unsigned int m0=p2m-H;
  unsigned int m1=L-H-p2s1m;
  Complex *fm0=f-m0;
  Complex *fH=f+H;

  if(r0 == 0) {
    unsigned int residues;
    unsigned int n2=n/2;
    if(D == 1 || 2*n2 < n) { // n odd, r=0
      residues=1;
      for(unsigned int u=1; u < p2; ++u) {
        unsigned int mu=m*u;
        Complex *Wu=W+mu;
        Complex *Zeta0=Zetaqm+n*mu;
        for(unsigned int s=1; s < m; ++s)
          Wu[s] *= conj(Zeta0[s]);
      }

      ifftp->fft(W);

      for(unsigned int s=0; s < m0; ++s)
        fH[s]=W[s];

      for(unsigned int s=m0; s < m; ++s)
        fH[s]=fm0[s]=W[s];

      for(unsigned int t=1; t < p2s1; ++t) {
        unsigned int tm=t*m;
        Complex *Wt=W+tm;
        Complex *fm0t=fm0+tm;
        Complex *fHt=fH+tm;
        for(unsigned int s=0; s < m; ++s)
          fHt[s]=fm0t[s]=Wt[s];
      }
      Complex *Wt=W+p2s1m;
      Complex *fm0t=fm0+p2s1m;
      Complex *fHt=fH+p2s1m;
      for(unsigned int s=0; s < m1; ++s)
        fHt[s]=fm0t[s]=Wt[s];

      for(unsigned int s=m1; s < m; ++s)
        fm0t[s]=Wt[s];

    } else { // n even, r=0,n/2
      residues=2;
      Complex *V=W+b;
      unsigned int mn=m*n;
      unsigned int mn2=m*n2;

      Complex *Zetan2=Zetaqm+mn2;
      for(unsigned int s=1; s < m; ++s)
        V[s] *= conj(Zetan2[s]);

      for(unsigned int u=1; u < p2; ++u) {
        unsigned int mu=m*u;
        Complex *Wu=W+mu;
        Complex *Zeta0=Zetaqm+mn*u;
        for(unsigned int s=1; s < m; ++s)
          Wu[s] *= conj(Zeta0[s]);
        Complex *Vu=V+mu;
        Complex *Zetan2=Zeta0+mn2;
        for(unsigned int s=1; s < m; ++s)
          Vu[s] *= conj(Zetan2[s]);
      }
      ifftp->fft(W);
      ifftp->fft(V);
      for(unsigned int s=0; s < m0; ++s)
        fH[s]=W[s]+V[s];

      for(unsigned int s=m0; s < m; ++s) {
        Complex Wts=W[s];
        Complex Vts=V[s];
        fm0[s]=Wts-Vts;
        fH[s]=Wts+Vts;
      }
      Complex *Zetaqn2=Zetaqp+p2*n2;
      for(unsigned int t=1; t < p2s1; ++t) {
        unsigned int tm=t*m;
        Complex *Wt=W+tm;
        Complex *Vt=V+tm;
        Complex *fm0t=fm0+tm;
        Complex *fHt=fH+tm;
        Complex Zeta=conj(Zetaqn2[t]);
        for(unsigned int s=0; s < m; ++s) {
          Complex Wts=Wt[s];
          Complex Vts=Zeta*Vt[s];
          fm0t[s]=Wts-Vts;
          fHt[s]=Wts+Vts;
        }
      }
      Complex *Wt=W+p2s1m;
      Complex *Vt=V+p2s1m;
      Complex *fm0t=fm0+p2s1m;
      Complex *fHt=fH+p2s1m;
      Complex Zeta=conj(Zetaqn2[p2s1]);
      for(unsigned int s=0; s < m1; ++s) {
        Complex Wts=Wt[s];
        Complex Vts=Zeta*Vt[s];
        fm0t[s]=Wts-Vts;
        fHt[s]=Wts+Vts;
      }
      for(unsigned int s=m1; s < m; ++s)
        fm0t[s]=Wt[s]-Zeta*Vt[s];
    }
    W += residues*b;
    r0=1;
    dr0=(D0-residues)/2;
  }

  if(D == 1) {
    if(dr0 > 0) {
      Vec Zetanr=LOAD(Zetaqp+p2*r0+p2); // zeta_n^r
      unsigned int mr=m*r0;
      Complex *Zetar=Zetaqm+mr;
      for(unsigned int s=1; s < m; ++s)
        W[s] *= conj(Zetar[s]);
      for(unsigned int u=1; u < p2; ++u) {
        unsigned int mu=m*u;
        Complex *Zetar0=Zetaqm+n*mu;
        Complex *Zetar=Zetar0+mr;
        Complex *Wu=W+mu;
        for(unsigned int s=1; s < m; ++s)
          Wu[s] *= conj(Zetar[s]);
      }

      ifftp->fft(W);

      for(unsigned int s=0; s < m0; ++s)
        fH[s] += W[s];

      Vec Xm=UNPACKL(Zetanr,Zetanr);
      Vec Ym=CONJ(UNPACKH(Zetanr,Zetanr));
      for(unsigned int s=m0; s < m; ++s) {
        Vec Fts=LOAD(W+s);
        STORE(fm0+s,LOAD(fm0+s)+ZMULT(Xm,Ym,Fts));
        STORE(fH+s,LOAD(fH+s)+Fts);
      }
      Complex *Zetaqr=Zetaqp+p2*r0;
      for(unsigned int t=1; t < p2s1; ++t) {
        unsigned int tm=t*m;
        Complex *Ft=W+tm;
        Complex *fm0t=fm0+tm;
        Complex *fHt=fH+tm;
        Vec Zeta=LOAD(Zetaqr+t);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=REFL(UNPACKH(Zeta,Zeta));
        Vec Zeta2=ZMULT(X,Y,Zetanr);
        Vec Xm=UNPACKL(Zeta2,Zeta2);
        Vec Ym=CONJ(UNPACKH(Zeta2,Zeta2));
        for(unsigned int s=0; s < m; ++s) {
          Vec Fts=LOAD(Ft+s);
          STORE(fm0t+s,LOAD(fm0t+s)+ZMULT(Xm,Ym,Fts));
          STORE(fHt+s,LOAD(fHt+s)+ZMULT(X,Y,Fts));
        }
      }
      Complex *Ft=W+p2s1m;
      Complex *fm0t=fm0+p2s1m;
      Complex *fHt=fH+p2s1m;
      Vec Zeta=LOAD(Zetaqr+p2s1);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=REFL(UNPACKH(Zeta,Zeta));
      Vec Zeta2=ZMULT(X,Y,Zetanr);
      Xm=UNPACKL(Zeta2,Zeta2);
      Ym=CONJ(UNPACKH(Zeta2,Zeta2));
      for(unsigned int s=0; s < m1; ++s) {
        Vec Fts=LOAD(Ft+s);
        STORE(fm0t+s,LOAD(fm0t+s)+ZMULT(Xm,Ym,Fts));
        STORE(fHt+s,LOAD(fHt+s)+ZMULT(X,Y,Fts));
      }
      for(unsigned int s=m1; s < m; ++s)
        STORE(fm0t+s,LOAD(fm0t+s)+ZMULT(Xm,Ym,LOAD(Ft+s)));
    }
  } else {
    for(unsigned int d=0; d < dr0; ++d) {
      Complex *F=W+2*b*d;
      unsigned int r=r0+d;
      Vec Zetanr=LOAD(Zetaqp+p2*r+p2); // zeta_n^r
      Complex *G=F+b;

      unsigned int mr=m*r;
      Complex *Zetar=Zetaqm+mr;
      for(unsigned int s=1; s < m; ++s) {
        Complex Zeta=Zetar[s];
        F[s] *= conj(Zeta);
        G[s] *= Zeta;
      }
      for(unsigned int u=1; u < p2; ++u) {
        unsigned int mu=m*u;
        Complex *Zetar0=Zetaqm+n*mu;
        Complex *Zetar=Zetar0+mr;
        Complex *Wu=F+mu;
        for(unsigned int s=1; s < m; ++s)
          Wu[s] *= conj(Zetar[s]);
        Complex *Zetar2=Zetar0-mr;
        Complex *Vu=G+mu;
        for(unsigned int s=1; s < m; ++s)
          Vu[s] *= conj(Zetar2[s]);
      }

      ifftp->fft(F);
      ifftp->fft(G);

      for(unsigned int s=0; s < m0; ++s)
        fH[s] += F[s]+G[s];

      Vec Xm=UNPACKL(Zetanr,Zetanr);
      Vec Ym=CONJ(UNPACKH(Zetanr,Zetanr));
      for(unsigned int s=m0; s < m; ++s) {
        Vec Fts=LOAD(F+s);
        Vec Gts=LOAD(G+s);
        STORE(fm0+s,LOAD(fm0+s)+ZMULT2(Xm,Ym,Fts,Gts));
        STORE(fH+s,LOAD(fH+s)+Fts+Gts);
      }
      Complex *Zetaqr=Zetaqp+p2*r;
      for(unsigned int t=1; t < p2s1; ++t) {
        unsigned int tm=t*m;
        Complex *Ft=F+tm;
        Complex *Gt=G+tm;
        Complex *fm0t=fm0+tm;
        Complex *fHt=fH+tm;
        Vec Zeta=LOAD(Zetaqr+t);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=CONJ(UNPACKH(Zeta,Zeta));
        Vec Zeta2=ZMULT(X,-Y,Zetanr);
        Vec Xm=UNPACKL(Zeta2,Zeta2);
        Vec Ym=CONJ(UNPACKH(Zeta2,Zeta2));
        for(unsigned int s=0; s < m; ++s) {
          Vec Fts=LOAD(Ft+s);
          Vec Gts=LOAD(Gt+s);
          STORE(fm0t+s,LOAD(fm0t+s)+ZMULT2(Xm,Ym,Fts,Gts));
          STORE(fHt+s,LOAD(fHt+s)+ZMULT2(X,Y,Gts,Fts));
        }
      }
      Complex *Ft=F+p2s1m;
      Complex *Gt=G+p2s1m;
      Complex *fm0t=fm0+p2s1m;
      Complex *fHt=fH+p2s1m;
      Vec Zeta=LOAD(Zetaqr+p2s1);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=CONJ(UNPACKH(Zeta,Zeta));
      Vec Zeta2=ZMULT(X,-Y,Zetanr);
      Xm=UNPACKL(Zeta2,Zeta2);
      Ym=CONJ(UNPACKH(Zeta2,Zeta2));
      for(unsigned int s=0; s < m1; ++s) {
        Vec Fts=LOAD(Ft+s);
        Vec Gts=LOAD(Gt+s);
        STORE(fm0t+s,LOAD(fm0t+s)+ZMULT2(Xm,Ym,Fts,Gts));
        STORE(fHt+s,LOAD(fHt+s)+ZMULT2(X,Y,Gts,Fts));
      }
      for(unsigned int s=m1; s < m; ++s)
        STORE(fm0t+s,LOAD(fm0t+s)+ZMULT2(Xm,Ym,LOAD(Ft+s),LOAD(Gt+s)));
    }
  }
}

void fftPadHermitian::init()
{
  common();
  e=m/2;
  unsigned int e1=e+1;
  unsigned int Ce1=C*e1;

  if(q == 1) {
    B=b=Ce1;
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
    dr=D0=R=Q=1;
  } else {
    D=2;
    dr=Dr();

    double twopibyq=twopi/q;
    unsigned int p2=p/2;

    Complex *G=ComplexAlign(Ce1*p2);
    double *H=inplace ? (double *) G : doubleAlign(Cm*p2);

    if(p > 2) { // p must be even
      if(C == 1) {
        Forward=&fftBase::forwardInnerC;
        Backward=&fftBase::backwardInnerC;
      } else {
//        Forward=&fftBase::forwardInnerCMany;
//        Backward=&fftBase::backwardInnerCMany;
      }
      Q=n=q/p2;

      Zetaqp0=ComplexAlign((n-1)*p2); // CHECK
      Zetaqp=Zetaqp0-p2-1;
      for(unsigned int r=1; r < n; ++r)
        for(unsigned int t=1; t <= p2; ++t)
          Zetaqp[p2*r+t]=expi(r*t*twopibyq);

      fftp=new mfft1d(p2,1,Ce1, Ce1,1, G);
      ifftp=new mfft1d(p2,-1,Ce1, Ce1,1, G);
    } else { // p=2
      Q=n=q;
      if(C == 1) {
        Forward=&fftBase::forward2;
        Backward=&fftBase::backward2;
      } else {
        Forward=&fftBase::forward2Many;
        Backward=&fftBase::backward2Many;
      }
    }

//    b=p2*C*(m-e); // Output block size
    b=p2*C*e+m%2; // Output block size
    B=p2*Ce1; // Work block size
//    if(inplace) b=B; // TEMP

    R=residueBlocks();
    D0=Q % D;
    if(D0 == 0) D0=D;

    unsigned int m0=m;//+(m % 2);
    if(C == 1) {
      crfftm=new mcrfft1d(m,p2, 1,1, e1,m0, G,H);
      rcfftm=new mrcfft1d(m,p2, 1,1, m0,e1, H,G);
    } else {
      unsigned int d=C*p2;
      crfftm=new mcrfft1d(m,d, C,C, 1,1, G,H);
      rcfftm=new mrcfft1d(m,d, C,C, 1,1, H,G);
    }

    if(!inplace)
      deleteAlign(H);
    deleteAlign(G);

    initZetaqm(p == 2 ? q/2+1 : q,m);
  }
}

fftPadHermitian::~fftPadHermitian()
{
  delete crfftm;
  delete rcfftm;
  // delete fftp;
  // delete ifftp;
}

void fftPadHermitian::forwardExplicit(Complex *f, Complex *F, unsigned int, Complex *)
{
  unsigned int H=ceilquotient(L,2);
  for(unsigned int s=0; s < H; ++s)
    F[s]=f[s];
  for(unsigned int s=H; s <= e; ++s)
    F[s]=0.0;

  crfftm->fft(F);
}

void fftPadHermitian::forwardExplicitMany(Complex *f, Complex *F, unsigned int, Complex *)
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

void fftPadHermitian::backwardExplicit(Complex *F, Complex *f, unsigned int, Complex *W, double)
{
  rcfftm->fft(F);
  unsigned int H=ceilquotient(L,2);
  for(unsigned int s=0; s < H; ++s)
    f[s]=F[s];
}

void fftPadHermitian::backwardExplicitMany(Complex *F, Complex *f,
                                           unsigned int, Complex *, double)
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
    if(2*q2 < q) { // q odd, r=0
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
    } else { // q even, r=0,q/2
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
    } else { // q even, r=0,q/2
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

void fftPadHermitian::backward2(Complex *F, Complex *f, unsigned int r, Complex *W, double)
{
  if(W == NULL) W=F;

  Complex Nyquist(0,0);
  bool inplace=W == F;
  bool even=m == 2*e;
  bool overlap=even && inplace;
  if(overlap)
    Nyquist=F[e]; // Save before being overwritten

  rcfftm->fft(F,W);

  Complex *fm=f+m;
  unsigned int me=m-e;
  unsigned int H=ceilquotient(L,2);
  unsigned int mH1=m-H+1;
  if(r == 0) {
    unsigned int q2=q/2;
    if(2*q2 < q) { // q odd, r=0
      for(unsigned int s=0; s < mH1; ++s)
        f[s]=W[s];
      for(unsigned int s=mH1; s < me; ++s) {
        Complex A=W[s];
        f[s]=A;
        *(fm-s)=conj(A);
      }
      if(even) {
        f[e]=W[e];
        if(overlap)
          F[e]=Nyquist; // Restore initial input of next residue
      }

    } else { // q even, r=0,q/2
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
        if(overlap)
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
      STORE(f+s,LOAD(f+s)+ZMULT2(Zeta,Vs,Ws));
      STORE(fm-s,LOAD(fm-s)+CONJ(ZMULT2(Zetam,Ws,Vs)));
    }
    if(even) {
//      f[e] += conj(Zetar[e])*W[e]+Zetar[e]*V[e];
      Vec Zeta=LOAD(Zetar+e);
      STORE(f+e,LOAD(f+e)+ZCMULT(Zeta,LOAD(W+e))+ZMULT(Zeta,LOAD(V+e)));
      if(overlap)
        G[e]=Nyquist; // Restore initial input of next residue
    }
  }
}

void fftPadHermitian::backward2Many(Complex *F, Complex *f, unsigned int r, Complex *W, double)
{
  if(W == NULL) W=F;

  Complex Nyquist[C];
  bool inplace=W == F;
  bool even=m == 2*e;
  bool overlap=even && inplace;
  if(overlap) {
    Complex *FCe=F+C*e;
    for(unsigned int c=0; c < C; ++c)
      Nyquist[c]=FCe[c]; // Save before being overwritten
  }

  rcfftm->fft(F,W);

  Complex *fm=f+Cm;
  unsigned int me=m-e;
  unsigned int H=ceilquotient(L,2);
  unsigned int mH1=m-H+1;
  unsigned int Ce=C*e;
  if(r == 0) {
    unsigned int q2=q/2;
    if(2*q2 < q) { // q odd, r=0
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
        if(overlap) {
          Complex *Fe=F+Ce;
          for(unsigned int c=0; c < C; ++c)
            Fe[c]=Nyquist[c]; // Restore initial input of next residue
        }
      }
    } else { // q even, r=0,q/2
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
        if(overlap) {
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
    if(overlap) {
      Complex *GCe=G+Ce;
      for(unsigned int c=0; c < C; ++c)
        GCe[c]=Nyquist[c]; // Restore initial input of next residue
    }
  }
}

void fftPadHermitian::forwardInnerC(Complex *f, Complex *F, unsigned int r, Complex *W)
{
  if(W == NULL) W=F;

  unsigned int p2=p/2;
  unsigned int H=L/2;
  unsigned int e1=e+1;
  unsigned int p2s1=p2-1;
  unsigned int p2s1m=p2s1*m;
  unsigned int p2s1e1=p2s1*e1;
  unsigned int p2m=p2*m;
  unsigned int p2mH=p2m+H-L;
  unsigned int m0=min(p2mH+1,e1);
  unsigned int m1=min(m-p2mH,e1);
  Complex *fm=f+p2m;
  if(r == 0) {
    unsigned int n2=n/2;
    if(2*n2 < n) { // n odd, r=0
      W[0]=f[0];
      for(unsigned int s=1; s < m0; ++s)
        W[s]=f[s];
      for(unsigned int s=m0; s < e1; ++s)
        W[s]=conj(*(fm-s))+f[s];
      for(unsigned int t=1; t < p2s1; ++t) {
        unsigned int tm=t*m;
        unsigned int te1=t*e1;
        Complex *fmt=fm-tm;
        Complex *ft=f+tm;
        Complex *Wt=W+te1;
        for(unsigned int s=0; s < e1; ++s)
          Wt[s]=conj(*(fmt-s))+ft[s];
      }
      Complex *Wt=W+p2s1e1;
      Complex *ft=f+p2s1m;
      Complex *fmt=f+m;
      for(unsigned int s=0; s < m1; ++s)
        Wt[s]=conj(*(fmt-s))+ft[s];
      for(unsigned int s=m1; s < e1; ++s)
        Wt[s]=conj(*(fmt-s));

      fftp->fft(W);

      unsigned int mn=m*n;
      for(unsigned int u=1; u < p2; ++u) {
        Complex *Wu=W+u*e1;
        Complex *Zeta0=Zetaqm+mn*u;
        for(unsigned int s=1; s < e1; ++s)
          Wu[s] *= Zeta0[s];
      }
      crfftm->fft(W,F);
    } else { // n even, r=0,n/2
      Complex *V=W+B;
      V[0]=W[0]=f[0];
      for(unsigned int s=1; s < m0; ++s)
        V[s]=W[s]=f[s];
      for(unsigned int s=m0; s < e1; ++s) {
        Complex fms=conj(*(fm-s));
        Complex fs=f[s];
        W[s]=fms+fs;
        V[s]=-fms+fs;
      }
      Complex *Zetaqn2=Zetaqp+p2*n2;
      for(unsigned int t=1; t < p2s1; ++t) {
        unsigned int tm=t*m;
        unsigned int te1=t*e1;
        Complex *Wt=W+te1;
        Complex *Vt=V+te1;
        Complex *fmt=fm-tm;
        Complex *ft=f+tm;
        Vec Zeta=LOAD(Zetaqn2+t); //*zeta_q^{tn/2}
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=CONJ(UNPACKH(Zeta,Zeta));
        for(unsigned int s=0; s < e1; ++s) {
          Vec fmts=CONJ(LOAD(fmt-s));
          Vec fts=LOAD(ft+s);
          STORE(Wt+s,fmts+fts);
          STORE(Vt+s,ZMULT(X,Y,fts-fmts));
        }
      }
      Complex *Wt=W+p2s1e1;
      Complex *Vt=V+p2s1e1;
      Complex *ft=f+p2s1m;
      Vec Zeta=LOAD(Zetaqn2+p2s1);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=CONJ(UNPACKH(Zeta,Zeta));
      Complex *fmt=f+m;
      for(unsigned int s=0; s < m1; ++s) {
        Vec fmts=CONJ(LOAD(fmt-s));
        Vec fts=LOAD(ft+s);
        STORE(Wt+s,fmts+fts);
        STORE(Vt+s,ZMULT(X,Y,fts-fmts));
      }
      X=-X;
      Y=-Y;
      // TODO: Does this every happen?
      for(unsigned int s=m1; s < e1; ++s) {
        Vec fmts=CONJ(LOAD(fmt-s));
        STORE(Wt+s,fmts);
        STORE(Vt+s,ZMULT(X,Y,fmts));
      }
      fftp->fft(W);
      fftp->fft(V);

      unsigned int mn=m*n;
      unsigned int mn2=m*n2;
      Complex *Zetan2=Zetaqm+mn2;
      for(unsigned int s=0; s < e1; ++s)
        V[s] *= Zetan2[s];
      for(unsigned int u=1; u < p2; ++u) {
        unsigned int ue1=u*e1;
        Complex *Wu=W+ue1;
        Complex *Zeta0=Zetaqm+mn*u;
        for(unsigned int s=1; s < e1; ++s)
          Wu[s] *= Zeta0[s];
        Complex *Vu=V+ue1;
        Complex *Zetan2=Zeta0+mn2;
        for(unsigned int s=1; s < e1; ++s)
          Vu[s] *= Zetan2[s];
      }

      crfftm->fft(W,F);
      crfftm->fft(V,F+b);
    }
  } else {
    Vec Zetanr=CONJ(LOAD(Zetaqp+p2*r+p2)); // zeta_n^-r
    Complex *G=F+b;
    Complex *V=inplace ? G : W+B;
    V[0]=W[0]=f[0];
    for(unsigned int s=1; s < m0; ++s)
      V[s]=W[s]=f[s];
    for(unsigned int s=m0; s < e1; ++s) {
      Vec fms=CONJ(LOAD(fm-s));
      Vec fs=LOAD(f+s);
      Vec A=Zetanr*UNPACKL(fms,fms);
      Vec B=ZMULTI(Zetanr*UNPACKH(fms,fms));
      STORE(W+s,A+B+fs);
      STORE(V+s,CONJ(A-B)+fs);
    }
    Complex *Zetaqr=Zetaqp+p2*r;
    for(unsigned int t=1; t < p2s1; ++t) {
      unsigned int tm=t*m;
      unsigned int te1=t*e1;
      Complex *Wt=W+te1;
      Complex *Vt=V+te1;
      Complex *fmt=fm-tm;
      Complex *ft=f+tm;
      Vec Zeta=LOAD(Zetaqr+t); // zeta_q^{tr}
      Vec Zetam=ZMULT(Zeta,Zetanr);
      for(unsigned int s=0; s < e1; ++s) {
        Vec fmts=CONJ(LOAD(fmt-s));
        Vec fts=LOAD(ft+s);
        Vec A=Zetam*UNPACKL(fmts,fmts)+Zeta*UNPACKL(fts,fts);
        Vec B=ZMULTI(Zetam*UNPACKH(fmts,fmts)+Zeta*UNPACKH(fts,fts));
        STORE(Wt+s,A+B);
        STORE(Vt+s,CONJ(A-B));
      }
    }
    Complex *Wt=W+p2s1e1;
    Complex *Vt=V+p2s1e1;
    Complex *fmt=fm-p2s1m;
    Complex *ft=f+p2s1m;
    Vec Zeta=LOAD(Zetaqr+p2s1);
    Vec Zetam=ZMULT(Zeta,Zetanr);
    for(unsigned int s=0; s < m1; ++s) {
      Vec fmts=CONJ(LOAD(fmt-s));
      Vec fts=LOAD(ft+s);
      Vec A=Zetam*UNPACKL(fmts,fmts)+Zeta*UNPACKL(fts,fts);
      Vec B=ZMULTI(Zetam*UNPACKH(fmts,fmts)+Zeta*UNPACKH(fts,fts));
      STORE(Wt+s,A+B);
      STORE(Vt+s,CONJ(A-B));
    }
    for(unsigned int s=m1; s < e1; ++s) {
      Vec fmts=CONJ(LOAD(fmt-s));
      Vec A=Zetam*UNPACKL(fmts,fmts);
      Vec B=ZMULTI(Zetam*UNPACKH(fmts,fmts));
      STORE(Wt+s,A+B);
      STORE(Vt+s,CONJ(A-B));
    }

    fftp->fft(W);
    fftp->fft(V);

    unsigned int mr=m*r;
    Complex *Zetar=Zetaqm+mr;
    for(unsigned int s=1; s < e1; ++s) {
      Complex Zeta=Zetar[s];
      W[s] *= Zeta;
      V[s] *= conj(Zeta);
    }

    for(unsigned int u=1; u < p2; ++u) {
      unsigned int mu=m*u;
      unsigned int e1u=e1*u;
      Complex *Zetar0=Zetaqm+n*mu;
      Complex *Zetar=Zetar0+mr;
      Complex *Wu=W+e1u;
      for(unsigned int s=1; s < e1; ++s)
        Wu[s] *= Zetar[s];
      Complex *Zetar2=Zetar0-mr;
      Complex *Vu=V+e1u;
      for(unsigned int s=1; s < e1; ++s)
        Vu[s] *= Zetar2[s];
    }
    crfftm->fft(W,F);
    crfftm->fft(V,G);
  }
}

void fftPadHermitian::backwardInnerC(Complex *F, Complex *f, unsigned int r, Complex *W, double)
{
  if(W == NULL) W=F;

  unsigned int p2=p/2;
  unsigned int H=L/2;
  unsigned int e1=e+1;
  unsigned int p2s1=p2-1;
  unsigned int p2s1m=p2s1*m;
  unsigned int p2s1e1=p2s1*e1;
  unsigned int p2m=p2*m;
  unsigned int p2mH=p2m+H-L;
  unsigned int S=m-e;
  unsigned int m0=min(p2mH+1,e1);
  unsigned int m1=min(m-p2mH,S);
  unsigned int T= m1 < e ? p2s1 : p2;
  Complex *fm=f+p2m;

  if(r == 0) {
    unsigned int n2=n/2;
    if(2*n2 < n) { // n odd, r=0
      rcfftm->fft(F,W);
      for(unsigned int u=1; u < p2; ++u) {
        unsigned int mu=m*u;
        unsigned int e1u=u*e1;
        Complex *Wu=W+e1u;
        Complex *Zeta0=Zetaqm+n*mu;
        for(unsigned int s=1; s < e1; ++s)
          Wu[s] *= conj(Zeta0[s]);
      }

      ifftp->fft(W);

      for(unsigned int s=0; s < m0; ++s)
        f[s]=W[s];

      for(unsigned int s=m0; s < S; ++s) {
        f[s]=W[s];
        *(fm-s)=conj(W[s]);
      }

      for(unsigned int t=1; t < p2s1; ++t) {
        unsigned int tm=t*m;
        unsigned int te1=t*e1;
        Complex *Wt=W+te1;
        Complex *ft=f+tm;
        // s=0 case (this is important to avoid overlap)
        ft[0]=Wt[0];
        Complex *fmt=fm-tm;
        for(unsigned int s=1; s < S; ++s) {
          ft[s]=Wt[s];
          *(fmt-s)=conj(Wt[s]);
        }
      }

      Complex *Wt=W+p2s1e1;
      Complex *ft=f+p2s1m;

      // s=0 case (this is important to avoid overlap)
      ft[0]=Wt[0];

      Complex *fmt=fm-p2s1m;
      for(unsigned int s=1; s < m1; ++s) {
        ft[s]=Wt[s];
        *(fmt-s)=conj(Wt[s]);
      }
      for(unsigned int s=m1; s < S; ++s)
        *(fmt-s)=conj(Wt[s]);

      if(S < e1) {
        f[e]=W[e];
        for(unsigned int t=1; t < T; ++t)
          f[t*m+e]=W[t*e1+e];
      }

    } else { // n even, r=0,n/2
      Complex *V=W+B;
      Complex *G=F+b;
      rcfftm->fft(F,W);
      rcfftm->fft(G,V);
      unsigned int mn=m*n;
      unsigned int mn2=m*n2;

      Complex *Zetan2=Zetaqm+mn2;
      for(unsigned int s=1; s < e1; ++s)
        V[s] *= conj(Zetan2[s]);

      for(unsigned int u=1; u < p2; ++u) {
        unsigned int e1u=u*e1;
        Complex *Wu=W+e1u;
        Complex *Zeta0=Zetaqm+mn*u;
        for(unsigned int s=1; s < e1; ++s)
          Wu[s] *= conj(Zeta0[s]);
        Complex *Vu=V+e1u;
        Complex *Zetan2=Zeta0+mn2;
        for(unsigned int s=1; s < e1; ++s)
          Vu[s] *= conj(Zetan2[s]);
      }
      ifftp->fft(W);
      ifftp->fft(V);

      f[0]=W[0]+V[0];
      for(unsigned int s=1; s < m0; ++s)
        f[s]=W[s]+V[s];

      for(unsigned int s=m0; s < S; ++s) {
        Complex Wts=W[s];
        Complex Vts=V[s];
        *(fm-s)=conj(Wts-Vts);
        f[s]=Wts+Vts;
      }

      Complex *Zetaqn2=Zetaqp+p2*n2;
      for(unsigned int t=1; t < p2s1; ++t) {
        unsigned int tm=t*m;
        unsigned int te1=t*e1;
        Complex *Wt=W+te1;
        Complex *Vt=V+te1;
        Complex *ft=f+tm;
        Vec Zeta=LOAD(Zetaqn2+t);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=REFL(UNPACKH(Zeta,Zeta));
        STORE(ft,LOAD(Wt)+ZMULT(X,Y,LOAD(Vt)));
        Complex *fmt=fm-tm;
        for(unsigned int s=1; s < S; ++s) {
          Vec Wts=LOAD(Wt+s);
          Vec Vts=ZMULT(X,Y,LOAD(Vt+s));
          STORE(fmt-s,CONJ(Wts-Vts));
          STORE(ft+s,Wts+Vts);
        }
      }
      Complex *Wt=W+p2s1e1;
      Complex *Vt=V+p2s1e1;
      Complex *ft=f+p2s1m;
      Vec Zeta=LOAD(Zetaqn2+p2s1);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=REFL(UNPACKH(Zeta,Zeta));
      STORE(ft,LOAD(Wt)+ZMULT(X,Y,LOAD(Vt)));
      Complex *fmt=fm-p2s1m;
      for(unsigned int s=1; s < m1; ++s) {
        Vec Wts=LOAD(Wt+s);
        Vec Vts=ZMULT(X,Y,LOAD(Vt+s));
        STORE(fmt-s,CONJ(Wts-Vts));
        STORE(ft+s,Wts+Vts);
      }
      for(unsigned int s=m1; s < S; ++s)
        *(fmt-s)=conj(Wt[s]-Vt[s]);

      if(S < e1) {
        f[e]=W[e]+V[e];
        for(unsigned int t=1; t < T; ++t) {
          unsigned int te1e=t*e1+e;
          f[t*m+e]=W[te1e]+conj(Zetaqn2[t])*V[te1e];
        }
      }
    }
  } else { // r > 0
    Complex *V=W+B;
    Complex *G=F+b;
    rcfftm->fft(F,W);
    rcfftm->fft(G,V);
    Vec Zetanr=LOAD(Zetaqp+p2*r+p2); // zeta_n^r

    unsigned int mr=m*r;
    Complex *Zetar=Zetaqm+mr;
    for(unsigned int s=1; s < e1; ++s) {
      Complex Zeta=Zetar[s];
      W[s] *= conj(Zeta);
      V[s] *= Zeta;
    }
    for(unsigned int u=1; u < p2; ++u) {
      unsigned int mu=m*u;
      unsigned int e1u=u*e1;
      Complex *Zetar0=Zetaqm+n*mu;
      Complex *Zetar=Zetar0+mr;
      Complex *Wu=W+e1u;
      for(unsigned int s=1; s < e1; ++s)
        Wu[s] *= conj(Zetar[s]);
      Complex *Zetar2=Zetar0-mr;
      Complex *Vu=V+e1u;
      for(unsigned int s=1; s < e1; ++s)
        Vu[s] *= conj(Zetar2[s]);
    }

    ifftp->fft(W);
    ifftp->fft(V);

    f[0] += W[0]+V[0];
    for(unsigned int s=1; s < m0; ++s){
      f[s] += W[s]+V[s];
    }

    Vec Xm=UNPACKL(Zetanr,Zetanr);
    Vec Ym=CONJ(UNPACKH(Zetanr,Zetanr));
    for(unsigned int s=m0; s < S; ++s) {
      Vec Wts=LOAD(W+s);
      Vec Vts=LOAD(V+s);
      STORE(fm-s,LOAD(fm-s)+CONJ(ZMULT2(Xm,Ym,Wts,Vts)));
      STORE(f+s,LOAD(f+s)+Wts+Vts);
    }

    Complex *Zetaqr=Zetaqp+p2*r;
    for(unsigned int t=1; t < p2s1; ++t) {
      unsigned int tm=t*m;
      unsigned int te1=t*e1;
      Complex *Wt=W+te1;
      Complex *Vt=V+te1;
      Complex *ft=f+tm;
      Vec Zeta=LOAD(Zetaqr+t);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=CONJ(UNPACKH(Zeta,Zeta));
      STORE(ft,LOAD(ft)+ZMULT2(X,Y,LOAD(Vt),LOAD(Wt)));
      Complex *fmt=fm-tm;
      Vec Zeta2=ZMULT(X,-Y,Zetanr); //=zeta_q^(-rt)z_n^r
      Vec Xm=UNPACKL(Zeta2,Zeta2);
      Vec Ym=CONJ(UNPACKH(Zeta2,Zeta2));
      for(unsigned int s=1; s < S; ++s) {
        Vec Wts=LOAD(Wt+s);
        Vec Vts=LOAD(Vt+s);
        STORE(fmt-s,LOAD(fmt-s)+CONJ(ZMULT2(Xm,Ym,Wts,Vts)));
        STORE(ft+s,LOAD(ft+s)+ZMULT2(X,Y,Vts,Wts));
      }
    }
    Complex *Wt=W+p2s1e1;
    Complex *Vt=V+p2s1e1;
    Complex *ft=f+p2s1m;
    Vec Zeta=LOAD(Zetaqr+p2s1);
    Vec X=UNPACKL(Zeta,Zeta);
    Vec Y=CONJ(UNPACKH(Zeta,Zeta));

    // s=0 case (this is important to avoid overlap)
    STORE(ft,LOAD(ft)+ZMULT2(X,Y,LOAD(Vt),LOAD(Wt)));

    Complex *fmt=fm-p2s1m;
    Vec Zeta2=ZMULT(X,-Y,Zetanr);
    Xm=UNPACKL(Zeta2,Zeta2);
    Ym=CONJ(UNPACKH(Zeta2,Zeta2));
    for(unsigned int s=1; s < m1; ++s) {
      Vec Wts=LOAD(Wt+s);
      Vec Vts=LOAD(Vt+s);
      STORE(fmt-s,LOAD(fmt-s)+CONJ(ZMULT2(Xm,Ym,Wts,Vts)));
      STORE(ft+s,LOAD(ft+s)+ZMULT2(X,Y,Vts,Wts));
    }
    for(unsigned int s=m1; s < S; ++s)
      STORE(fmt-s,LOAD(fmt-s)+CONJ(ZMULT2(Xm,Ym,LOAD(Wt+s),LOAD(Vt+s))));

    if(S < e1) {
      f[e]+=W[e]+V[e];
      for(unsigned int t=1; t < T; ++t) {
        unsigned int te1e=t*e1+e;
        unsigned int tme=t*m+e;
        Vec Zeta=LOAD(Zetaqr+t);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=CONJ(UNPACKH(Zeta,Zeta));
        STORE(f+tme,LOAD(f+tme)+ZMULT2(X,Y,LOAD(V+te1e),LOAD(W+te1e)));
      }
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
  b=fft->b;
  R=fft->residueBlocks();

  unsigned int M=fft->normalization();
  scale=1.0/M;
  unsigned int c=fft->outputSize();
  noutputs=fft->ninputs();

  N=max(A,B);
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
      unsigned int size=fft->workSizeV(A,B);
      for(unsigned int i=0; i < B; ++i)
        this->V[i]=V+i*size;
    } else
      this->V=NULL;

    allocateW=!this->W && !fft->inplace;
    this->W=allocateW ? ComplexAlign(c) : NULL;

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
      FpB=fft->inplace ? NULL : Fp[B];
      extra=1;
    } else {
      extra=0;
      G=new Complex*[N];
      for(unsigned int b=0; b < B; ++b)
        G[b]=this->F[b];
    }

    if(A > B+extra && !fft->inplace) {
      W0=this->F[B];
      Pad=&fftBase::padNone;
    } else
      W0=this->W;
  }

  if(fft->outputSize() <= fft->C*L && false) // Doesn't work for Hermitian case
    for(unsigned int r=0; r < R; r += fft->increment(r))
      overwrite=r; // Store F in h on the last loop
  else
    overwrite=R;
}

Convolution::~Convolution()
{
  if(q > 1) {
    if(allocateW)
      deleteAlign(W);

    if(loop2)
      delete[] Fp;
    else
      delete [] G;

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

// f is an array of max(A,B) pointers to distinct data blocks
// each of size fft->length()
// offset is applied to each input and output component
void Convolution::convolve0(Complex **f, multiplier *mult, unsigned int offset)
{
  if(q == 1) {
    for(unsigned int a=0; a < A; ++a)
      (fft->*Forward)(f[a]+offset,F[a],0,NULL);
    (*mult)(F,b,threads);
    for(unsigned int b=0; b < B; ++b)
      (fft->*Backward)(F[b],f[b]+offset,0,NULL,1.0);
  } else {
    if(fft->overwrite) {
      for(unsigned int a=0; a < N; ++a)
        G[a]=f[a]+offset;
      for(unsigned int a=0; a < A; ++a)
        (fft->*Forward)(G[a],F[a],R,NULL);
      (*mult)(G,fft->p*b,threads);
      (*mult)(F,b,threads);
      for(unsigned int b=0; b < B; ++b)
        (fft->*Backward)(F[b],G[b],R,NULL,1.0);
    } else {
      if(loop2) {
        for(unsigned int a=0; a < A; ++a)
          (fft->*Forward)(f[a]+offset,F[a],0,W);
        (*mult)(F,D0*b,threads);

        for(unsigned int b=0; b < B; ++b) {
          (fft->*Forward)(f[b]+offset,Fp[b],r,W);
          (fft->*Backward)(F[b],f[b]+offset,0,W0,1.0);
          (fft->*Pad)(W);
        }
        for(unsigned int a=B; a < A; ++a)
          (fft->*Forward)(f[a]+offset,Fp[a],r,W);
        (*mult)(Fp,D*b,threads);
        for(unsigned int b=0; b < B; ++b)
          (fft->*Backward)(Fp[b],f[b]+offset,r,FpB,1.0);
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

        for(unsigned int r=0; r < R; r += fft->increment(r)) {
          for(unsigned int a=0; a < A; ++a) {
            Complex *output=r == overwrite ? f[a]+offset : F[a];
            G[a]=output;
            (fft->*Forward)(f[a]+offset,output,r,W);
          }
          (*mult)(G,(r == 0 ? D0 : D)*b,threads);
          for(unsigned int b=0; b < B; ++b)
            (fft->*Backward)(G[b],h0[b]+Offset,r,W0,1.0);
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

  unsigned CL=C*fft.L;

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
  unsigned int start,stop;
  if(fft.overwrite) {
    start=R;
    stop=R+1;
  } else {
    start=0;
    stop=R;
  }
  double t0=totalseconds();
  for(unsigned int k=0; k < K; ++k) {
    for(unsigned int r=start; r < stop; r += fft.increment(r)) {
      for(unsigned int a=0; a < A; ++a)
        (fft.*Forward)(f[a],F[a],r,W);
      for(unsigned int b=0; b < B; ++b)
        (fft.*Backward)(F[b],h[b],r,W,1.0);
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

}
