#include "convolve.h"
#include "cmult-sse2.h"

// TODO:
// Replace optimization tests with actual problem
// Optimize on the fly
// Apply const to arguments that should be read-only
//
// Exploit zeta -> -conj(zeta) symmetry for even q
// Precompute best D and inline options for each m value
// Use power of P values for m when L,M,M-L are powers of P?
// Port to MPI

using namespace std;
using namespace utils;
using namespace Array;

namespace fftwpp {

bool Output=false;
bool testError=false;
bool showOptTimes=false;
bool Centered=false;
bool normalized=true;
bool Tforced=false;

#ifdef __SSE2__
const union uvec sse2_pm={
  {0x00000000,0x00000000,0x00000000,0x80000000}
};

const union uvec sse2_mm={
  {0x00000000,0x80000000,0x00000000,0x80000000}
};
#endif

const double twopi=2.0*M_PI;

void multNone(Complex **F, unsigned int n, Indices *indices,
              unsigned int threads)
{
}

// This multiplication routine is for binary convolutions and takes
// two Complex inputs of size n and outputs one Complex value.
// F0[j] *= F1[j];
void multbinary(Complex **F, unsigned int n, Indices *indices,
                unsigned int threads)
{
  Complex *F0=F[0];
  Complex *F1=F[1];

#if 0 // Transformed indices are available, if needed.
  size_t N=indices->size;
  fftBase *fft=indices->fft;
  unsigned int r=indices->r;
  for(unsigned int j=0; j < n; ++j) {
    for(unsigned int d=0; d < N; ++d)
      cout << indices->index[N-1-d] << ",";
    cout << fft->index(r,j) << endl;
  }
#endif

  PARALLEL(
    for(unsigned int j=0; j < n; ++j)
      F0[j] *= F1[j];
    );
}

// This multiplication routine is for binary convolutions and takes
// two real inputs of size n.
// F0[j] *= F1[j];
void realmultbinary(Complex **F, unsigned int n, Indices *indices,
                    unsigned int threads)
{
  double *F0=(double *) F[0];
  double *F1=(double *) F[1];

#if 0 // Transformed indices are available, if needed.
  size_t N=indices->size;
  fftBase *fft=indices->fft;
  unsigned int r=indices->r;
  for(unsigned int j=0; j < n; ++j) {
    for(unsigned int d=0; d < N; ++d)
      cout << indices->index[N-1-d] << ",";
    cout << fft->index(r,j) << endl;
  }
#endif

  PARALLEL(
    for(unsigned int j=0; j < n; ++j)
      F0[j] *= F1[j];
    );
}

bool notPow2(unsigned int m)
{
  return m != ceilpow2(m);
}

// Returns the smallest natural number greater than m of the form
// 2^a 3^b 5^c 7^d for some nonnegative integers a, b, c, and d.
unsigned int nextfftsize(unsigned int m)
{
  if(m == ceilpow2(m))
    return m;
  unsigned int N=-1;
  unsigned int ni=1;
  for(unsigned int i=0; ni < 7*m; ni=pow(7,i),++i) {
    unsigned int nj=ni;
    for(unsigned int j=0; nj < 5*m; nj=ni*pow(5,j),++j) {
      unsigned int nk=nj;
      for(unsigned int k=0; nk < 3*m; nk=nj*pow(3,k),++k) {
        N=min(N,nk*ceilpow2(ceilquotient(m,nk)));
      }
      if(N == m)
        return N;
    }
  }
  return N;
}

void fftBase::initZetaqm(unsigned int q, unsigned int m)
{
  double twopibyM=twopi/M;
  Zetaqm0=ComplexAlign((q-1)*m);
  Zetaqm=Zetaqm0-m;
  for(unsigned int r=1; r < q; ++r) {
    unsigned int mr=m*r;
    Zetaqm[mr]=1.0;
    PARALLEL(
      for(unsigned int s=1; s < m; ++s)
        Zetaqm[mr+s]=expi(r*s*twopibyM);
      );
  }
}

// Returns the smallest natural number greater than m that is a
// power of 2, 3, 5, or 7.
unsigned int nextpuresize(unsigned int m)
{
  unsigned int M=ceilpow2(m);
  if(m == M)
    return m;
  M=min(M,ceilpow(3,m));
  if(m == M)
    return m;
  M=min(M,ceilpow(5,m));
  if(m == M)
    return m;
  return min(M,ceilpow(7,m));
}

// Returns true iff m is a power of 2, 3, 5, or 7.
bool ispure(unsigned int m)
{
  if(m == ceilpow2(m))
    return true;
  if(m == ceilpow(3,m))
    return true;
  if(m == ceilpow(5,m))
    return true;
  if(m == ceilpow(7,m))
    return true;
  return false;
}


template<class Convolution>
double time(fftBase *fft, Application &app, double &threshold)
{
  unsigned int N=max(app.A,app.B);
  unsigned int inputSize=fft->inputSize();

  unsigned int size=fft->embed() ? fft->outputSize() : inputSize;
  Complex **f=ComplexAlign(N,size);

  // Initialize entire array to 0 to avoid overflow when timing.
  for(unsigned int a=0; a < app.A; ++a) {
    Complex *fa=f[a];
    for(unsigned int j=0; j < inputSize; ++j)
      fa[j]=0.0;
  }

  Convolution Convolve(fft,app.A,app.B,fft->embed() ? f : NULL);

  statistics Stats(true);
  statistics medianStats(false);
  double eps=0.02;

  do {
    double t0=nanoseconds();
    Convolve.convolveRaw(f);
    Stats.add(nanoseconds()-t0);
    if(Stats.count() > 1 && Stats.min() >= threshold) break;
    medianStats.add(Stats.median());
  } while(medianStats.stderror() > eps*medianStats.mean());

  threshold=min(threshold,Stats.max());
  deleteAlign(f[0]);
  delete [] f;
  return Stats.median();
}

double timePad(fftBase *fft, Application &app, double& threshold)
{
  return time<Convolution>(fft,app,threshold);
}

double timePadHermitian(fftBase *fft, Application &app, double& threshold)
{
  return time<ConvolutionHermitian>(fft,app,threshold);
}

void fftBase::OptBase::optloop(unsigned int& m, unsigned int L,
                               unsigned int M, Application& app,
                               unsigned int C, unsigned int S,
                               bool centered, unsigned int itmax,
                               bool useTimer, bool Explicit, bool inner)
{
  unsigned int i=(inner ? m : 0);
  // If inner == true, i is an m value and itmax is the largest m value that
  // we consider. If inner == false, i is a counter starting at zero, and
  // itmax is maximum number of m values we consider before exiting optloop.
  while(i < itmax) {
    unsigned int p=ceilquotient(L,m);
    // P is the effective p value
    unsigned int P=(centered && p == 2*(p/2)) || p == 2 ? (p/2) : p;
    unsigned int n=ceilquotient(M,m*P);
    //cout<<"inner="<<inner<<", p="<<p<<", P="<<P<<", n="<<n<<", centered="<<centered<<endl;

    if(!Explicit && app.m >= 1 && app.m < M && centered && p%2 != 0) {
      cerr << "Odd values of p are incompatible with the centered and Hermitian routines." << endl;
      cerr << "Using explicit routines with m=" << M << " instead." << endl;
    }

    // In the inner loop we must have the following:
    // p must be a power of 2, 3, 5, or 7.
    // p must be even in the centered case.
    // p != q.
    if(inner && (((!ispure(p) || p == P*n) && !mForced) || (centered && p%2 != 0)))
      i=m=nextpuresize(m+1);
    else {
      bool forceD=app.D > 0 && valid(app.D, p, S);
      unsigned int q=(inner ? P*n : ceilquotient(M,m));
      unsigned int Dstart=forceD ? app.D : 1;
      unsigned int Dstop=forceD ? app.D : n;
      unsigned int Dstop2=2*Dstop;

      // Check inplace and out-of-place unless C > 1.
      unsigned int Istart=app.I == -1 ? C > 1 : app.I;

      unsigned int Istop=app.I == -1 ? 2 : app.I+1;

        for(unsigned int D=Dstart; D < Dstop2; D *= 2) {
          if(D > Dstop) D=Dstop;
          for(unsigned int inplace=Istart; inplace < Istop; ++inplace)
            if((q == 1 || valid(D,p,S)) && D <= n) {
              for(unsigned int pass=app.threads == 1 || Tforced; pass < 2;
                  ++pass) {
                unsigned int threads=pass ? app.threads : 1;
                check(L,M,C,S,m,p,q,D,inplace,threads,app,useTimer);
              }
        }
      }
      if(mForced) break;
      if(inner) {
        m=nextpuresize(m+1);
        i=m;
      } else {
        if(ispure(m)) {
          m=nextfftsize(m+1);
          break;
        }
        m=nextfftsize(m+1);
        i++;
      }
    }
  }
}

void fftBase::OptBase::opt(unsigned int L, unsigned int M, Application& app,
                           unsigned int C, unsigned int S,
                           unsigned int minsize, unsigned int itmax,
                           bool Explicit, bool centered, bool useTimer)
{
  if(!Explicit) {
    if(mForced) {
      if(app.m >= ceilquotient(L,2))
        optloop(app.m,L,M,app,C,S,centered,1,useTimer,false);
      else
        optloop(app.m,L,M,app,C,S,centered,app.m+1,useTimer,false,true);
    } else {
      unsigned int m=nextfftsize(minsize);

      optloop(m,L,M,app,C,S,centered,max(L/2,32),useTimer,false,true);

      m=nextfftsize(L/2);
      optloop(m,L,M,app,C,S,centered,itmax,useTimer,false);

      if(L > M/2) {
        m=nextfftsize(max(M/2,m));
        optloop(m,L,M,app,C,S,centered,itmax,useTimer,false);

        m=nextfftsize(max(L,m));
        optloop(m,L,M,app,C,S,centered,itmax,useTimer,false);
      } else {
        m=nextfftsize(max(L <= M/2 ? L : M/2,m));
        optloop(m,L,M,app,C,S,centered,itmax,useTimer,false);
      }
      m=nextfftsize(max(M,m));
      optloop(m,L,M,app,C,S,centered,itmax,useTimer,false);
    }
  } else {
    unsigned int m=nextfftsize(M);
    optloop(m,L,m,app,C,S,centered,itmax,useTimer,true);
  }
}

void fftBase::OptBase::check(unsigned int L, unsigned int M,
                             unsigned int C, unsigned int S, unsigned int m,
                             unsigned int p, unsigned int q, unsigned int D,
                             bool inplace, unsigned int threads,
                             Application& app, bool useTimer)
{
  //cout << "m=" << m << ", p=" << p << ", q=" << q << ", D=" << D << " I=" << inplace << endl;
  if(useTimer) {
    double t=time(L,M,C,S,m,q,D,inplace,threads,app);
    if(showOptTimes)
      cout << "m=" << m << ", p=" << p << ", q=" << q << ", C=" << C << ", S=" << S << ", D=" << D << ", I=" << inplace << ", threads=" << threads << ": t=" << t*1.0e-9 << endl;
    if(t < T) {
      this->m=m;
      this->q=q;
      this->D=D;
      this->inplace=inplace;
      this->threads=threads;
      T=t;
    }
  } else {
    counter += 1;
    if(m != mlist.back())
      mlist.push_back(m);
    if(counter == 1) {
      this->m=m;
      this->q=q;
      this->D=D;
      this->inplace=inplace;
      this->threads=threads;
    }
  }
}

void fftBase::OptBase::scan(unsigned int L, unsigned int M, Application& app,
                            unsigned int C, unsigned int S, bool Explicit,
                            bool centered)
{
  m=M;
  q=1;
  D=1;
  threads=app.threads;
  inplace=false;
  T=DBL_MAX;
  threshold=T;

  mForced=(app.m >= 1);

  unsigned int mStart=2;
  unsigned int itmax=3;
  opt(L,M,app,C,S,mStart,itmax,Explicit,centered,false);

  if(counter == 0) {
    cerr << "Optimizer found no valid cases with specified parameters." << endl;
    cerr << "Using explicit routines with m=" << M << " instead." << endl << endl;
  } else if(counter > 1) {
    if(showOptTimes) cout << "Optimizer Timings:" << endl;
    mForced=true;
    for(mList::reverse_iterator r=mlist.rbegin(); r != mlist.rend(); ++r) {
      app.m=*r;
      opt(L,M,app,C,S,mStart,itmax,Explicit,centered);
    }
    if(showOptTimes)
      cout << endl << "Optimal time: t=" << T*1.0e-9 << endl << endl;
  }

  unsigned int p=ceilquotient(L,m);
  unsigned int mpL=m*p-L;
  cout << "Optimal padding: ";
  if(p == q)
    cout << "Explicit" << endl;
  else if(mpL > 0)
    cout << "Hybrid" << endl;
  else
    cout << "Implicit" << endl;
  cout << "m=" << m << endl;
  cout << "p=" << p << endl;
  cout << "q=" << q << endl;
  cout << "C=" << C << endl;
  cout << "S=" << S << endl;
  cout << "D=" << D << endl;
  cout << "I=" << inplace << endl;
  cout << "threads=" << threads << endl;
  cout << endl;
  cout << "Padding: " << mpL << endl;
}

fftBase::~fftBase()
{
  if(q > 1)
    deleteAlign(Zetaqm0);
  if(ZetaqmS)
    deleteAlign(ZetaqmS0);
}

void fftBase::checkParameters()
{
  if(L > M) {
    cerr << "L=" << L << " is greater than M=" << M << "." << endl;
    exit(-1);
  }

  if(S < C) {
    cerr << "stride S cannot be less than count C" << endl;
    exit(-1);
  }
}

void fftBase::common()
{
  if(q*m < M) {
    cerr << "Invalid parameters: " << endl
         << " q=" << q << " m=" << m << " M=" << M << endl;
    exit(-1);
  }
  p=ceilquotient(L,m);
  Cm=C*(size_t) m;
  Sm=S*(size_t) m;
  n=q/p;
  M=m*q;
  Pad=&fftBase::padNone;
  Zetaqp=NULL;
  ZetaqmS=NULL;
  overwrite=false;
}

void fftPad::init()
{
  common();
  if(q == 1) {
    if(C == 1 && S == 1) {
      Forward=&fftBase::forwardExplicit;
      Backward=&fftBase::backwardExplicit;
    } else {
      Forward=&fftBase::forwardExplicitMany;
      Backward=&fftBase::backwardExplicitMany;
    }

    Complex *G=ComplexAlign(Sm);
    Complex *H=inplace ? G : ComplexAlign(Sm);

    fftm=new mfft1d(m,1,C, S,1, G,H);
    ifftm=new mfft1d(m,-1,C, S,1, G,H);
    deleteAlign(G);
    if(!inplace)
      deleteAlign(H);
    dr=D=D0=R=Q=1;
    l=M;
    b=S*l;
  } else {
    double twopibyN=twopi/M;
    double twopibyq=twopi/q;

    unsigned int d;

    unsigned int P,P1;
    unsigned int p2=p/2;

    if(p == 2) {
      Q=n=q;
      P1=P=1;
    } else if(centered && p == 2*p2) {
      Q=n=q/p2;
      P=p2;
      P1=P+1;
    } else
      P1=P=p;

    l=m*P;
    b=S*l;
    d=C*D*P;

    Complex *G,*H;
    unsigned int size=b*D;

    G=ComplexAlign(size);
    H=inplace ? G : ComplexAlign(size);

    overwrite=inplace && L == p*m && n == (centered ? 3 : p+1) && D == 1;

    if(p > 2) { // Implies L > 2m
      if(!centered) overwrite=false;
      if(C == 1 && S == 1) {
        Forward=&fftBase::forwardInner;
        Backward=&fftBase::backwardInner;
        ForwardAll=&fftBase::forwardInnerAll;
        BackwardAll=&fftBase::backwardInnerAll;
      } else {
        Forward=&fftBase::forwardInnerMany;
        Backward=&fftBase::backwardInnerMany;
        ForwardAll=&fftBase::forwardInnerManyAll;
        BackwardAll=&fftBase::backwardInnerManyAll;
      }
      Q=n;

      Zetaqp0=ComplexAlign((n-1)*(P1-1));
      Zetaqp=Zetaqp0-P1;
      for(unsigned int r=1; r < n; ++r)
        for(unsigned int t=1; t < P1; ++t)
          Zetaqp[(P1-1)*r+t]=expi(r*t*twopibyq);

      // L'=p, M'=q, m'=p, p'=1, q'=n
      if(S == C) {
        fftp=new mfft1d(P,1,Cm, Cm,1, G);
        ifftp=new mfft1d(P,-1,Cm, Cm,1, G);
      } else {
        fftp=new mfft1d(P,1,C, Sm,1, G);
        ifftp=new mfft1d(P,-1,C, Sm,1, G);
      }
    } else {
      if(p == 2) {
        if(!centered) {
          unsigned int Lm=L-m;
          ZetaqmS0=ComplexAlign((q-1)*Lm);
          ZetaqmS=ZetaqmS0-L;
          for(unsigned int r=1; r < q; ++r)
            for(unsigned int s=m; s < L; ++s)
              ZetaqmS[Lm*r+s]=expi(r*s*twopibyN);
        }

        if(C == 1 && S == 1) {
          Forward=&fftBase::forward2;
          Backward=&fftBase::backward2;
          ForwardAll=&fftBase::forward2All;
          BackwardAll=&fftBase::backward2All;
        } else {
          Forward=&fftBase::forward2Many;
          Backward=&fftBase::backward2Many;
          ForwardAll=&fftBase::forward2ManyAll;
          BackwardAll=&fftBase::backward2ManyAll;
        }
      } else { // p == 1
        if(C == 1 && S == 1) {
          Forward=&fftBase::forward1;
          Backward=&fftBase::backward1;
          ForwardAll=&fftBase::forward1All;
          BackwardAll=&fftBase::backward1All;
          if(repad())
            Pad=&fftBase::padSingle;
        } else {
          Forward=&fftBase::forward1Many;
          Backward=&fftBase::backward1Many;
          ForwardAll=&fftBase::forward1ManyAll;
          BackwardAll=&fftBase::backward1ManyAll;
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

    if(C == 1 && S == 1) {
      fftm=new mfft1d(m,1,d, 1,m, G,H);
      ifftm=new mfft1d(m,-1,d, 1,m, G,H);
    } else {
      fftm=new mfft1d(m,1,C, S,1, G,H);
      ifftm=new mfft1d(m,-1,C, S,1, G,H);
    }

    if(D0 != D) {
      unsigned int x=D0*P;
      fftm0=new mfft1d(m,1,x, 1,Sm, G,H);
      ifftm0=new mfft1d(m,-1,x, 1,Sm, G,H);
    } else
      fftm0=NULL;

    if(!inplace)
      deleteAlign(H);
    deleteAlign(G);

    initZetaqm(q,centered && p == 2 ? m+1 : m);
  }
}

fftPad::~fftPad() {
  if(q == 1) {
    delete fftm;
    delete ifftm;
  } else {
    if(Zetaqp) {
      deleteAlign(Zetaqp0);
      delete fftp;
      delete ifftp;
    }
    if(fftm0) {
      delete fftm0;
      delete ifftm0;
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
    Complex *F=W+S*s;
    for(unsigned int c=0; c < C; ++c)
      F[c]=0.0;
  };
}

void fftPad::forwardExplicit(Complex *f, Complex *F, unsigned int, Complex *W)
{
  if(W == NULL) W=F;

  if(W != f) {
    PARALLEL(
      for(unsigned int s=0; s < L; ++s)
        W[s]=f[s];
      );
  }
  PARALLEL(
    for(unsigned int s=L; s < M; ++s)
      W[s]=0.0;
    );
  fftm->fft(W,F);
}

void fftPad::backwardExplicit(Complex *F, Complex *f, unsigned int, Complex *W)
{
  if(W == NULL) W=F;

  ifftm->fft(F,W);
  if(W != f) {
    PARALLEL(
      for(unsigned int s=0; s < L; ++s)
        f[s]=W[s];
      );
  }
}

void fftPad::forwardExplicitMany(Complex *f, Complex *F, unsigned int,
                                 Complex *W)
{
  if(W == NULL) W=F;

  if(W != f) {
    PARALLEL(
      for(unsigned int s=0; s < L; ++s) {
        unsigned int Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fs=f+Ss;
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=fs[c];
      });
  }
  padMany(W);
  fftm->fft(W,F);
}

void fftPad::backwardExplicitMany(Complex *F, Complex *f, unsigned int,
                                  Complex *W)
{
  if(W == NULL) W=F;

  ifftm->fft(F,W);
  if(W != f) {
    PARALLEL(
      for(unsigned int s=0; s < L; ++s) {
        unsigned int Ss=S*s;
        Complex *fs=f+Ss;
        Complex *Fs=W+Ss;
        for(unsigned int c=0; c < C; ++c)
          fs[c]=Fs[c];
      });
  }
}

void fftPad::forward1All(Complex *f, Complex *F, unsigned int, Complex *)
{
  F[0]=f[0];
  Complex *Zetar=Zetaqm+m;
  PARALLEL(
    for(unsigned int s=1; s < m; ++s)
      F[s]=Zetar[s]*f[s];
    );
  fftm->fft(f);
  fftm->fft(F);
}

void fftPad::forward1(Complex *f, Complex *F0, unsigned int r0, Complex *W)
{
  if(W == NULL) W=F0;

  Complex *W0=W;
  unsigned int dr0=dr;
  mfft1d *fftm1;

  if(r0 == 0) {
    unsigned int residues;
    unsigned int q2=q/2;
    if(D == 1 || 2*q2 < q) { // q odd, r=0
      if(!inplace && D == 1 && L >= m)
        return fftm->fft(f,F0);
      residues=1;
      if(W != f) {
        PARALLEL(
          for(unsigned int s=0; s < L; ++s)
            W[s]=f[s];
          );
      }
    } else { // q even, r=0,q/2
      residues=2;
      Complex *V=W+m;
      V[0]=W[0]=f[0];
      Complex *Zetar=Zetaqm+m*q2;
      if(W != f) {
        PARALLEL(
          for(unsigned int s=1; s < L; ++s) {
            Complex fs=f[s];
            W[s]=fs;
            V[s]=Zetar[s]*fs;
          });
      } else {
        PARALLEL(
          for(unsigned int s=1; s < L; ++s)
            V[s]=Zetar[s]*f[s];
          );
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
    fftm1=D0 == D ? fftm : fftm0;
  } else
    fftm1=fftm;

  if(D == 1) {
    if(dr0 > 0) {
      W[0]=f[0];
      Complex *Zetar=Zetaqm+m*r0;
      PARALLEL(
        for(unsigned int s=1; s < L; ++s)
          W[s]=Zetar[s]*f[s];
        );
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
      PARALLEL(
        for(unsigned int s=1; s < L; ++s) {
//        F[s]=Zeta*fs;
//        G[s]=conj(Zeta)*fs;
          Vec Zeta=LOAD(Zetar+s);
          Vec fs=LOAD(f+s);
          Vec A=Zeta*UNPACKL(fs,fs);
          Vec B=ZMULTI(Zeta*UNPACKH(fs,fs));
          STORE(F+s,A+B);
          STORE(G+s,CONJ(A-B));
        });
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

  fftm1->fft(W0,F0);
}

void fftPad::forward1ManyAll(Complex *f, Complex *F, unsigned int, Complex *)
{
  Complex *Zetar=Zetaqm+m;
  for(unsigned int c=0; c < C; ++c)
    F[c]=f[c];
  PARALLEL(
    for(unsigned int s=1; s < m; ++s) {
      unsigned int Ss=S*s;
      Complex *fs=f+Ss;
      Complex *Fs=F+Ss;
      Vec Zeta=LOAD(Zetar+s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(Zeta,-Zeta);
      for(unsigned int c=0; c < C; ++c)
        STORE(Fs+c,ZMULT(X,Y,LOAD(fs+c)));
    });
  fftm->fft(f);
  fftm->fft(F);
}

void fftPad::forward1Many(Complex *f, Complex *F, unsigned int r, Complex *W)
{
  if(W == NULL) W=F;

  if(inplace) {
    for(unsigned int s=L; s < m; ++s) {
      Complex *Fs=W+S*s;
      for(unsigned int c=0; c < C; ++c)
        Fs[c]=0.0;
    }
  }

  if(r == 0) {
    if(!inplace && L >= m && S == C)
      return fftm->fft(f,F);
    PARALLEL(
      for(unsigned int s=0; s < L; ++s) {
        unsigned int Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fs=f+Ss;
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=fs[c];
      });
  } else {
    for(unsigned int c=0; c < C; ++c)
      W[c]=f[c];
    Complex *Zetar=Zetaqm+m*r;
    PARALLEL(
      for(unsigned int s=1; s < L; ++s) {
        unsigned int Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fs=f+Ss;
        Vec Zeta=LOAD(Zetar+s);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(Zeta,-Zeta);
        for(unsigned int c=0; c < C; ++c)
          STORE(Fs+c,ZMULT(X,Y,LOAD(fs+c)));
      });
  }
  fftm->fft(W,F);
}

void fftPad::forward2All(Complex *f, Complex *F, unsigned int, Complex *)
{
  Complex *g=f+m;
  Complex *Zetar2=ZetaqmS+L;
  Vec V1=LOAD(g);
  Vec Zetam=LOAD(Zetar2);
  Vec A=Zetam*UNPACKL(V1,V1);
  Vec B=ZMULTI(Zetam*UNPACKH(V1,V1));
  Vec V0=LOAD(f);
  STORE(f,V0+V1);
  STORE(g,V0+A+B);
  STORE(F,V0+CONJ(A-B));
  Complex *Zetar=Zetaqm+m;
  PARALLEL(
    for(unsigned int s=1; s < m; ++s) {
      Complex *v0=f+s;
      Complex *v1=g+s;
      Vec V0=LOAD(v0);
      Vec V1=LOAD(v1);
      Vec Zeta=LOAD(Zetar+s);
      Vec Zetam=LOAD(Zetar2+s);
      Vec A=Zeta*UNPACKL(V0,V0)+Zetam*UNPACKL(V1,V1);
      Vec B=ZMULTI(Zeta*UNPACKH(V0,V0)+Zetam*UNPACKH(V1,V1));
      STORE(v0,V0+V1);
      STORE(v1,A+B);
      STORE(F+s,CONJ(A-B));
    });
  fftm->fft(f);
  fftm->fft(g);
  fftm->fft(F);
}

void fftPad::forward2(Complex *f, Complex *F0, unsigned int r0, Complex *W)
{
  if(W == NULL) W=F0;

  Complex *W0=W;
  unsigned int dr0=dr;
  mfft1d *fftm1;

  unsigned int Lm=L-m;
  if(r0 == 0) {
    unsigned int residues;
    unsigned int q2=q/2;
    if(D == 1 || 2*q2 < q) { // q odd, r=0
      residues=1;
      PARALLEL(
        for(unsigned int s=0; s < Lm; ++s)
          W[s]=f[s]+f[m+s];
        );
      PARALLEL(
        for(unsigned int s=Lm; s < m; ++s)
          W[s]=f[s];
        );
    } else {
      residues=2;
      Complex *V=W+m;
      Complex *fm=f+m;
      Complex f0=f[0];
      Complex fm0=fm[0];
      W[0]=f0+fm0;
      V[0]=f0-fm0;
      Complex *Zetar=Zetaqm+m*q2;
      PARALLEL(
        for(unsigned int s=1; s < Lm; ++s) {
          Complex fs=f[s];
          Complex fms=fm[s];
          W[s]=fs+fms;
          V[s]=conj(Zetar[s])*(fs-fms);
        });
      PARALLEL(
        for(unsigned int s=Lm; s < m; ++s) {
          Complex fs=f[s];
          W[s]=fs;
          V[s]=conj(Zetar[s])*fs;
        });
    }
    W += residues*m;
    r0=1;
    dr0=(D0-residues)/2;
    fftm1=D0 == D ? fftm : fftm0;
  } else
    fftm1=fftm;

  if(D == 1) {
    if(dr0 > 0) {
      Complex *Zetar2=ZetaqmS+Lm*r0+m;
      Complex *fm=f+m;
      W[0]=f[0]+Zetar2[0]*fm[0];
      Complex *Zetar=Zetaqm+m*r0;
      PARALLEL(
        for(unsigned int s=1; s < Lm; ++s) {
//        W[s]=Zetar[s]*f[s]+Zetar2[s]*fm[s];
          Vec Zeta=LOAD(Zetar+s);
          Vec Zeta2=LOAD(Zetar2+s);
          Vec fs=LOAD(f+s);
          Vec fms=LOAD(fm+s);
          STORE(W+s,ZMULT(Zeta,fs)+ZMULT(Zeta2,fms));
        });
      PARALLEL(
        for(unsigned int s=Lm; s < m; ++s)
          W[s]=Zetar[s]*f[s];
        );
    }
  } else {
    for(unsigned int d=0; d < dr0; ++d) {
      Complex *F=W+2*m*d;
      Complex *G=F+m;
      unsigned int r=r0+d;
      Complex *Zetar2=ZetaqmS+Lm*r+m;
      Complex *fm=f+m;
      Vec fs=LOAD(f);
      Vec fms=LOAD(fm);
      Vec Zetam=LOAD(Zetar2);
      Vec A=Zetam*UNPACKL(fms,fms);
      Vec B=ZMULTI(Zetam*UNPACKH(fms,fms));
      STORE(F,fs+A+B);
      STORE(G,fs+CONJ(A-B));
      Complex *Zetar=Zetaqm+m*r;
      PARALLEL(
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
        });
      PARALLEL(
        for(unsigned int s=Lm; s < m; ++s) {
//        F[s]=Zetar[s]*f[s];
//        G[s]=conj(Zetar[s])*f[s];
          Vec fs=LOAD(f+s);
          Vec Zeta=LOAD(Zetar+s);
          Vec A=Zeta*UNPACKL(fs,fs);
          Vec B=ZMULTI(Zeta*UNPACKH(fs,fs));
          STORE(F+s,A+B);
          STORE(G+s,CONJ(A-B));
        });
    }
  }

  fftm1->fft(W0,F0);
}

void fftPad::forward2ManyAll(Complex *f, Complex *F, unsigned int, Complex *)
{
  Complex *g=f+Sm;
  Complex *Zetar2=ZetaqmS+L;
  Vec Zetam=LOAD(Zetar2);
  for(unsigned int c=0; c < C; ++c) {
    Complex *v0=f+c;
    Complex *v1=g+c;
    Vec V1=LOAD(v1);
    Vec A=Zetam*UNPACKL(V1,V1);
    Vec B=ZMULTI(Zetam*UNPACKH(V1,V1));
    Vec V0=LOAD(v0);
    STORE(v0,V0+V1);
    STORE(v1,V0+A+B);
    STORE(F+c,V0+CONJ(A-B));
  }
  Complex *Zetar=Zetaqm+m;
  PARALLEL(
    for(unsigned int s=1; s < m; ++s) {
      unsigned int Ss=S*s;
      Complex *v0=f+Ss;
      Complex *v1=g+Ss;
      Complex *v2=F+Ss;
      Vec Zeta=LOAD(Zetar+s);
      Vec Zetam=LOAD(Zetar2+s);
      for(unsigned int c=0; c < C; ++c) {
        Complex *v0c=v0+c;
        Complex *v1c=v1+c;
        Vec V0=LOAD(v0c);
        Vec V1=LOAD(v1c);
        Vec A=Zeta*UNPACKL(V0,V0)+Zetam*UNPACKL(V1,V1);
        Vec B=ZMULTI(Zeta*UNPACKH(V0,V0)+Zetam*UNPACKH(V1,V1));
        STORE(v0c,V0+V1);
        STORE(v1c,A+B);
        STORE(v2+c,CONJ(A-B));
      }
    });
  fftm->fft(f);
  fftm->fft(g);
  fftm->fft(F);
}

void fftPad::forward2Many(Complex *f, Complex *F, unsigned int r, Complex *W)
{
  if(W == NULL) W=F;

  unsigned int Lm=L-m;
  if(r == 0) {
    PARALLEL(
      for(unsigned int s=0; s < Lm; ++s) {
        unsigned int Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fs=f+Ss;
        Complex *fms=f+Sm+Ss;
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=fs[c]+fms[c];
      });
    PARALLEL(
      for(unsigned int s=Lm; s < m; ++s) {
        unsigned int Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fs=f+Ss;
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=fs[c];
      });
  } else {
    Complex *Zetar2=ZetaqmS+Lm*r+m;
    Complex Zeta2=Zetar2[0];
    Complex *fm=f+Sm;
    for(unsigned int c=0; c < C; ++c)
      W[c]=f[c]+Zeta2*fm[c];
    Complex *Zetar=Zetaqm+m*r;
    PARALLEL(
      for(unsigned int s=1; s < Lm; ++s) {
        unsigned int Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fs=f+Ss;
        Complex *fms=f+Sm+Ss;
        Complex Zetars=Zetar[s];
        Complex Zetarms=Zetar2[s];
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=Zetars*fs[c]+Zetarms*fms[c];
      });
    PARALLEL(
      for(unsigned int s=Lm; s < m; ++s) {
        unsigned int Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fs=f+Ss;
        Complex Zetars=Zetar[s];
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=Zetars*fs[c];
      });
  }
  fftm->fft(W,F);
}

void fftPad::forwardInner(Complex *f, Complex *F0, unsigned int r0, Complex *W)
{
  if(W == NULL) W=F0;

  Complex *W0=W;
  unsigned int dr0=dr;
  mfft1d *fftm1;

  unsigned int pm1=p-1;
  unsigned int stop=L-m*pm1;

  if(r0 == 0) {
    for(unsigned int t=0; t < pm1; ++t) {
      unsigned int mt=m*t;
      Complex *Ft=W+mt;
      Complex *ft=f+mt;
      PARALLEL(
        for(unsigned int s=0; s < m; ++s)
          Ft[s]=ft[s];
        );
    }

    unsigned int mt=m*pm1;
    Complex *Ft=W+mt;
    Complex *ft=f+mt;
    PARALLEL(
      for(unsigned int s=0; s < stop; ++s)
        Ft[s]=ft[s];
      );
    for(unsigned int s=stop; s < m; ++s)
      Ft[s]=0.0;

    fftp->fft(W);
    for(unsigned int t=1; t < p; ++t) {
      unsigned int R=n*t;
      Complex *Ft=W+m*t;
      Complex *Zetar=Zetaqm+m*R;
      PARALLEL(
        for(unsigned int s=1; s < m; ++s)
          Ft[s] *= Zetar[s];
        );
    }

    W += b;
    r0=1;
    dr0=D0-1;
    fftm1=D0 == D ? fftm : fftm0;
  } else
    fftm1=fftm;

  for(unsigned int d=0; d < dr0; ++d) {
    Complex *F=W+b*d;
    unsigned int r=r0+d;
    PARALLEL(
      for(unsigned int s=0; s < m; ++s)
        F[s]=f[s];
      );
    Complex *Zetaqr=Zetaqp+pm1*r;
    for(unsigned int t=1; t < pm1; ++t) {
      unsigned int mt=m*t;
      Complex *Ft=F+mt;
      Complex *ft=f+mt;
      Complex Zeta=Zetaqr[t];
      PARALLEL(
        for(unsigned int s=0; s < m; ++s)
          Ft[s]=Zeta*ft[s];
        );
    }
    unsigned int mt=m*pm1;
    Complex *Ft=F+mt;
    Complex *ft=f+mt;
    Complex Zeta=Zetaqr[pm1];
    PARALLEL(
      for(unsigned int s=0; s < stop; ++s)
        Ft[s]=Zeta*ft[s];
      );
    for(unsigned int s=stop; s < m; ++s)
      Ft[s]=0.0;

    fftp->fft(F);
    for(unsigned int t=0; t < p; ++t) {
      unsigned int R=n*t+r;
      Complex *Ft=F+m*t;
      Complex *Zetar=Zetaqm+m*R;
      PARALLEL(
        for(unsigned int s=1; s < m; ++s)
          Ft[s] *= Zetar[s];
        );
    }
  }

  fftm1->fft(W0,F0);
}

void fftPad::forwardInnerMany(Complex *f, Complex *F, unsigned int r,
                              Complex *W)
{
  if(W == NULL) W=F;

  unsigned int pm1=p-1;
  unsigned int stop=L-m*pm1;

  if(r == 0) {
    for(unsigned int t=0; t < pm1; ++t) {
      unsigned int Smt=Sm*t;
      Complex *Ft=W+Smt;
      Complex *ft=f+Smt;
      PARALLEL(
        for(unsigned int s=0; s < m; ++s) {
          unsigned int Ss=S*s;
          Complex *Fts=Ft+Ss;
          Complex *fts=ft+Ss;
          for(unsigned int c=0; c < C; ++c)
            Fts[c]=fts[c];
        });
    }
    unsigned int Smt=Sm*pm1;
    Complex *Ft=W+Smt;
    Complex *ft=f+Smt;
    PARALLEL(
      for(unsigned int s=0; s < stop; ++s) {
        unsigned int Ss=S*s;
        Complex *Fts=Ft+Ss;
        Complex *fts=ft+Ss;
        for(unsigned int c=0; c < C; ++c)
          Fts[c]=fts[c];
      });
    for(unsigned int s=stop; s < m; ++s) {
      Complex *Fts=Ft+S*s;
      for(unsigned int c=0; c < C; ++c)
        Fts[c]=0.0;
    };

    if(S == C)
      fftp->fft(W);
    else
      for(unsigned int s=0; s < m; ++s)
        fftp->fft(W+S*s);

    for(unsigned int t=1; t < p; ++t) {
      unsigned int R=n*t;
      Complex *Ft=W+Sm*t;
      Complex *Zetar=Zetaqm+m*R;
      PARALLEL(
        for(unsigned int s=1; s < m; ++s) {
          Complex *Fts=Ft+S*s;
          Complex Zetars=Zetar[s];
          for(unsigned int c=0; c < C; ++c)
            Fts[c] *= Zetars;
        });
    }
  } else {
    PARALLEL(
      for(unsigned int s=0; s < m; ++s) {
        unsigned int Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fs=f+Ss;
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=fs[c];
      });
    Complex *Zetaqr=Zetaqp+pm1*r;
    for(unsigned int t=1; t < pm1; ++t) {
      unsigned int Smt=Sm*t;
      Complex *Ft=W+Smt;
      Complex *ft=f+Smt;
      Complex Zeta=Zetaqr[t];
      PARALLEL(
        for(unsigned int s=0; s < m; ++s) {
          unsigned int Ss=S*s;
          Complex *Fts=Ft+Ss;
          Complex *fts=ft+Ss;
          for(unsigned int c=0; c < C; ++c)
            Fts[c]=Zeta*fts[c];
        });
    }
    Complex *Ft=W+Sm*pm1;
    Complex *ft=f+Sm*pm1;
    Vec Zeta=LOAD(Zetaqr+pm1);
    PARALLEL(
      for(unsigned int s=0; s < stop; ++s) {
        unsigned int Ss=S*s;
        Complex *Fts=Ft+Ss;
        Complex *fts=ft+Ss;
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(Zeta,-Zeta);
        for(unsigned int c=0; c < C; ++c)
          STORE(Fts+c,ZMULT(X,Y,LOAD(fts+c)));
      });
    for(unsigned int s=stop; s < m; ++s) {
      Complex *Fts=Ft+S*s;
      for(unsigned int c=0; c < C; ++c)
        Fts[c]=0.0;
    }

    if(S == C)
      fftp->fft(W);
    else
      for(unsigned int s=0; s < m; ++s)
        fftp->fft(W+S*s);

    for(unsigned int t=0; t < p; ++t) {
      unsigned int R=n*t+r;
      Complex *Ft=W+Sm*t;
      Complex *Zetar=Zetaqm+m*R;
      PARALLEL(
        for(unsigned int s=1; s < m; ++s) {
          Complex *Fts=Ft+S*s;
          Vec Zetars=LOAD(Zetar+s);
          Vec X=UNPACKL(Zetars,Zetars);
          Vec Y=UNPACKH(Zetars,-Zetars);
          for(unsigned int c=0; c < C; ++c)
            STORE(Fts+c,ZMULT(X,Y,LOAD(Fts+c)));
        });
    }
  }
  for(unsigned int t=0; t < p; ++t) {
    unsigned int Smt=Sm*t;
    fftm->fft(W+Smt,F+Smt);
  }
}

void fftPad::backward1All(Complex *F, Complex *f, unsigned int, Complex *)
{
  ifftm->fft(f);
  ifftm->fft(F);

  f[0] += F[0];
  Complex *Zetar=Zetaqm+m;
  PARALLEL(
    for(unsigned int s=1; s < m; ++s)
      f[s] += conj(Zetar[s])*F[s];
    );
}

void fftPad::backward1(Complex *F0, Complex *f, unsigned int r0, Complex *W)
{
  if(W == NULL) W=F0;

  if(r0 == 0 && !inplace && D == 1 && L >= m)
    return ifftm->fft(F0,f);

  (r0 > 0 || D0 == D ? ifftm : ifftm0)->fft(F0,W);

  unsigned int dr0=dr;

  if(r0 == 0) {
    unsigned int residues;
    unsigned int q2=q/2;
    if(D == 1 || 2*q2 < q) { // q odd, r=0
      residues=1;
      PARALLEL(
        for(unsigned int s=0; s < L; ++s)
          f[s]=W[s];
        );
    } else { // q even, r=0,q/2
      residues=2;
      Complex *V=W+m;
      f[0]=W[0]+V[0];
      Complex *Zetar=Zetaqm+m*q2;
      PARALLEL(
        for(unsigned int s=1; s < L; ++s)
          f[s]=W[s]+conj(Zetar[s])*V[s];
        );
    }
    W += residues*m;
    r0=1;
    dr0=(D0-residues)/2;
  }

  if(D == 1) {
    if(dr0 > 0) {
      f[0] += W[0];
      Complex *Zetar=Zetaqm+m*r0;
      PARALLEL(
        for(unsigned int s=1; s < L; ++s)
          f[s] += conj(Zetar[s])*W[s];
        );
    }
  } else {
    for(unsigned int d=0; d < dr0; ++d) {
      Complex *U=W+2*m*d;
      Complex *V=U+m;
      f[0] += U[0]+V[0];
      unsigned int r=r0+d;
      Complex *Zetar=Zetaqm+m*r;
      PARALLEL(
        for(unsigned int s=1; s < L; ++s) {
          Vec Zeta=LOAD(Zetar+s);
          Vec Us=LOAD(U+s);
          Vec Vs=LOAD(V+s);
          STORE(f+s,LOAD(f+s)+ZMULT2(Zeta,Vs,Us));
        });
    }
  }
}

void fftPad::backward1ManyAll(Complex *F, Complex *f, unsigned int, Complex *)
{
  ifftm->fft(f);
  ifftm->fft(F);

  Complex *Zetar=Zetaqm+m;
  for(unsigned int c=0; c < C; ++c)
    f[c] += F[c];
  PARALLEL(
    for(unsigned int s=1; s < m; ++s) {
      unsigned int Ss=S*s;
      Complex *fs=f+Ss;
      Complex *Fs=F+Ss;
      Vec Zeta=LOAD(Zetar+s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(-Zeta,Zeta);
      for(unsigned int c=0; c < C; ++c)
        STORE(fs+c,LOAD(fs+c)+ZMULT(X,Y,LOAD(Fs+c)));
    });
}

void fftPad::backward1Many(Complex *F, Complex *f, unsigned int r, Complex *W)
{
  if(W == NULL) W=F;

  if(r == 0 && !inplace && L >= m)
    return ifftm->fft(F,f);

  ifftm->fft(F,W);

  if(r == 0) {
    PARALLEL(
      for(unsigned int s=0; s < L; ++s) {
        unsigned int Ss=S*s;
        Complex *fs=f+Ss;
        Complex *Fs=W+Ss;
        for(unsigned int c=0; c < C; ++c)
          fs[c]=Fs[c];
      });
  } else {
    for(unsigned int c=0; c < C; ++c)
      f[c] += W[c];
    Complex *Zetar=Zetaqm+m*r;
    PARALLEL(
      for(unsigned int s=1; s < L; ++s) {
        unsigned int Ss=S*s;
        Complex *fs=f+Ss;
        Complex *Fs=W+Ss;
        Vec Zeta=LOAD(Zetar+s);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(-Zeta,Zeta);
        for(unsigned int c=0; c < C; ++c)
          STORE(fs+c,LOAD(fs+c)+ZMULT(X,Y,LOAD(Fs+c)));
      }
      );
  }
}

void fftPad::backward2All(Complex *F, Complex *f, unsigned int, Complex *)
{
  Complex *g=f+m;

  ifftm->fft(f);
  ifftm->fft(g);
  ifftm->fft(F);

  Complex *Zetar2=ZetaqmS+L;
  Vec V0=LOAD(f);
  Vec V1=LOAD(g);
  Vec V2=LOAD(F);
  STORE(f,V0+V1+V2);
  STORE(g,V0+ZMULT2(LOAD(Zetar2),V2,V1));
  Complex *Zetar=Zetaqm+m;
  PARALLEL(
    for(unsigned int s=1; s < m; ++s) {
      Complex *v0=f+s;
      Complex *v1=g+s;
      Complex *v2=F+s;
      Vec V0=LOAD(v0);
      Vec V1=LOAD(v1);
      Vec V2=LOAD(v2);
      STORE(v0,V0+ZMULT2(LOAD(Zetar+s),V2,V1));
      STORE(v1,V0+ZMULT2(LOAD(Zetar2+s),V2,V1));
    });
}

void fftPad::backward2(Complex *F0, Complex *f, unsigned int r0, Complex *W)
{
  if(W == NULL) W=F0;

  (r0 > 0 || D0 == D ? ifftm : ifftm0)->fft(F0,W);

  unsigned int dr0=dr;

  unsigned int Lm=L-m;
  if(r0 == 0) {
    unsigned int residues;
    unsigned int q2=q/2;
    if(D == 1 || 2*q2 < q) { // q odd, r=0
      residues=1;
      PARALLEL(
        for(unsigned int s=0; s < m; ++s)
          f[s]=W[s];
        );
      Complex *Wm=W-m;
      PARALLEL(
        for(unsigned int s=m; s < L; ++s)
          f[s]=Wm[s];
        );
    } else { // q even, r=0,q/2
      residues=2;
      Complex *V=W+m;
      f[0]=W[0]+V[0];
      Complex *Zetar=Zetaqm+m*q2;
      PARALLEL(
        for(unsigned int s=1; s < m; ++s)
          f[s]=W[s]+Zetar[s]*V[s];
        );
      Complex *Zetar2=ZetaqmS+Lm*q2;
      Complex *Wm=W-m;
      PARALLEL(
        for(unsigned int s=m; s < L; ++s)
          f[s]=Wm[s]+Zetar2[s]*W[s];
        );
    }
    W += residues*m;
    r0=1;
    dr0=(D0-residues)/2;
  }

  if(D == 1) {
    if(dr0 > 0) {
      f[0] += W[0];
      Complex *Zetar1=Zetaqm+m*r0;
      PARALLEL(
        for(unsigned int s=1; s < m; ++s)
          f[s] += conj(Zetar1[s])*W[s];
        );
      Complex *Zetar2=ZetaqmS+Lm*r0;
      Complex *Wm=W-m;
      PARALLEL(
        for(unsigned int s=m; s < L; ++s)
          f[s] += conj(Zetar2[s])*Wm[s];
        );
    }
  } else {
    for(unsigned int d=0; d < dr0; ++d) {
      Complex *U=W+2*m*d;
      Complex *V=U+m;
      unsigned int r=r0+d;
      f[0] += U[0]+V[0];
      Complex *Zetar=Zetaqm+m*r;
      PARALLEL(
        for(unsigned int s=1; s < m; ++s)
//      f[s] += conj(Zetar[s])*U[s]+Zetar[s]*V[s];
          STORE(f+s,LOAD(f+s)+ZMULT2(LOAD(Zetar+s),LOAD(V+s),LOAD(U+s)));
        );
      Complex *Zetar2=ZetaqmS+Lm*r;
      Complex *Um=U-m;
      Complex *Vm=V-m;
      PARALLEL(
        for(unsigned int s=m; s < L; ++s)
//      f[s] += conj(Zetar2[s])*Um[s]+Zetar2[s]*Vm[s];
          STORE(f+s,LOAD(f+s)+ZMULT2(LOAD(Zetar2+s),LOAD(Vm+s),LOAD(Um+s)));
        );
    }
  }
}

void fftPad::backward2ManyAll(Complex *F, Complex *f, unsigned int, Complex *)
{
  Complex *g=f+Sm;

  ifftm->fft(f);
  ifftm->fft(g);
  ifftm->fft(F);

  Complex *Zetar2=ZetaqmS+L;
  Vec Zetam=LOAD(Zetar2);
  Vec Xm=UNPACKL(Zetam,Zetam);
  Vec Ym=UNPACKH(Zetam,-Zetam);
  for(unsigned int c=0; c < C; ++c) {
    Complex *v0=f+c;
    Complex *v1=g+c;
    Vec V0=LOAD(v0);
    Vec V1=LOAD(v1);
    Vec V2=LOAD(F+c);
    STORE(v0,V0+V1+V2);
    STORE(v1,V0+ZMULT2(Xm,Ym,V2,V1));
  };
  Complex *Zetar=Zetaqm+m;
  PARALLEL(
    for(unsigned int s=1; s < m; ++s) {
      unsigned int Ss=S*s;
      Complex *v0=f+Ss;
      Complex *v1=g+Ss;
      Complex *v2=F+Ss;
      Vec Zeta=LOAD(Zetar+s);
      Vec Zetam=LOAD(Zetar2+s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(Zeta,-Zeta);
      Vec Xm=UNPACKL(Zetam,Zetam);
      Vec Ym=UNPACKH(Zetam,-Zetam);
      for(unsigned int c=0; c < C; ++c) {
        Complex *v0c=v0+c;
        Complex *v1c=v1+c;
        Vec V0=LOAD(v0c);
        Vec V1=LOAD(v1c);
        Vec V2=LOAD(v2+c);
        STORE(v0c,V0+ZMULT2(X,Y,V2,V1));
        STORE(v1c,V0+ZMULT2(Xm,Ym,V2,V1));
      }
    });
}

void fftPad::backward2Many(Complex *F, Complex *f, unsigned int r, Complex *W)
{
  if(W == NULL) W=F;

  ifftm->fft(F,W);

  if(r == 0) {
    PARALLEL(
      for(unsigned int s=0; s < m; ++s) {
        unsigned int Ss=S*s;
        Complex *fs=f+Ss;
        Complex *Fs=W+Ss;
        for(unsigned int c=0; c < C; ++c)
          fs[c]=Fs[c];
      });
    Complex *WSm=W-Sm;
    PARALLEL(
      for(unsigned int s=m; s < L; ++s) {
        unsigned int Ss=S*s;
        Complex *fs=f+Ss;
        Complex *Fs=WSm+Ss;
        for(unsigned int c=0; c < C; ++c)
          fs[c]=Fs[c];
      });
  } else {
    unsigned int Lm=L-m;
    for(unsigned int c=0; c < C; ++c)
      f[c] += W[c];
    Complex *Zetar=Zetaqm+m*r;
    PARALLEL(
      for(unsigned int s=1; s < m; ++s) {
        unsigned int Ss=S*s;
        Complex *fs=f+Ss;
        Complex *Fs=W+Ss;
        Complex Zetars=conj(Zetar[s]);
        for(unsigned int c=0; c < C; ++c)
          fs[c] += Zetars*Fs[c];
      });
    Complex *Zetar2=ZetaqmS+Lm*r;
    Complex *WSm=W-Sm;
    PARALLEL(
      for(unsigned int s=m; s < L; ++s) {
        unsigned int Ss=S*s;
        Complex *fs=f+Ss;
        Complex *Fs=WSm+Ss;
        Complex Zetars2=conj(Zetar2[s]);
        for(unsigned int c=0; c < C; ++c)
          fs[c] += Zetars2*Fs[c];
      });
  }
}

void fftPad::backwardInner(Complex *F0, Complex *f, unsigned int r0,
                           Complex *W)
{
  if(W == NULL) W=F0;

  (r0 > 0 || D0 == D ? ifftm : ifftm0)->fft(F0,W);

  unsigned int dr0=dr;

  unsigned int pm1=p-1;
  unsigned int stop=L-m*pm1;

  if(r0 == 0) {
    for(unsigned int t=1; t < p; ++t) {
      unsigned int R=n*t;
      Complex *Ft=W+m*t;
      Complex *Zetar=Zetaqm+m*R;
      PARALLEL(
        for(unsigned int s=1; s < m; ++s)
          Ft[s] *= conj(Zetar[s]);
        );
    }
    ifftp->fft(W);
    for(unsigned int t=0; t < pm1; ++t) {
      unsigned int mt=m*t;
      Complex *ft=f+mt;
      Complex *Ft=W+mt;
      PARALLEL(
        for(unsigned int s=0; s < m; ++s)
          ft[s]=Ft[s];
        );
    }
    unsigned int mt=m*pm1;
    Complex *ft=f+mt;
    Complex *Ft=W+mt;
    PARALLEL(
      for(unsigned int s=0; s < stop; ++s)
        ft[s]=Ft[s];
      );

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
      PARALLEL(
        for(unsigned int s=1; s < m; ++s)
          Ft[s] *= conj(Zetar[s]);
        );
    }
    ifftp->fft(F);
    PARALLEL(
      for(unsigned int s=0; s < m; ++s)
        f[s] += F[s];
      );
    Complex *Zetaqr=Zetaqp+pm1*r;
    for(unsigned int t=1; t < pm1; ++t) {
      unsigned int mt=m*t;
      Complex *ft=f+mt;
      Complex *Ft=F+mt;
      Complex Zeta=conj(Zetaqr[t]);
      PARALLEL(
        for(unsigned int s=0; s < m; ++s)
          ft[s] += Zeta*Ft[s];
        );
    }
    unsigned int mt=m*pm1;
    Complex *Ft=F+mt;
    Complex *ft=f+mt;
    Complex Zeta=conj(Zetaqr[pm1]);
    PARALLEL(
      for(unsigned int s=0; s < stop; ++s)
        ft[s] += Zeta*Ft[s];
      );
  }
}

void fftPad::backwardInnerMany(Complex *F, Complex *f, unsigned int r,
                               Complex *W)
{
  if(W == NULL) W=F;

  for(unsigned int t=0; t < p; ++t) {
    unsigned int Smt=Sm*t;
    ifftm->fft(F+Smt,W+Smt);
  }
  unsigned int pm1=p-1;
  unsigned int stop=L-m*pm1;

  if(r == 0) {
    for(unsigned int t=1; t < p; ++t) {
      unsigned int R=n*t;
      Complex *Ft=W+Sm*t;
      Complex *Zetar=Zetaqm+m*R;
      PARALLEL(
        for(unsigned int s=1; s < m; ++s) {
          Complex *Fts=Ft+S*s;
          Complex Zetars=Zetar[s];
          for(unsigned int c=0; c < C; ++c)
            Fts[c] *= conj(Zetar[s]);
        });
    }

    if(S == C)
      ifftp->fft(W);
    else
      for(unsigned int s=0; s < m; ++s)
        ifftp->fft(W+S*s);

    for(unsigned int t=0; t < pm1; ++t) {
      unsigned int Smt=Sm*t;
      Complex *ft=f+Smt;
      Complex *Ft=W+Smt;
      PARALLEL(
        for(unsigned int s=0; s < m; ++s) {
          unsigned int Ss=S*s;
          Complex *fts=ft+Ss;
          Complex *Fts=Ft+Ss;
          for(unsigned int c=0; c < C; ++c)
            fts[c]=Fts[c];
        });
    }
    unsigned int Smt=Sm*pm1;
    Complex *ft=f+Smt;
    Complex *Ft=W+Smt;
    PARALLEL(
      for(unsigned int s=0; s < stop; ++s) {
        unsigned int Ss=S*s;
        Complex *fts=ft+Ss;
        Complex *Fts=Ft+Ss;
        for(unsigned int c=0; c < C; ++c)
          fts[c]=Fts[c];
      });
  } else {
    for(unsigned int t=0; t < p; ++t) {
      unsigned int R=n*t+r;
      Complex *Ft=W+Sm*t;
      Complex *Zetar=Zetaqm+m*R;
      PARALLEL(
        for(unsigned int s=1; s < m; ++s) {
          Complex *Fts=Ft+S*s;
          Complex Zetars=conj(Zetar[s]);
          for(unsigned int c=0; c < C; ++c)
            Fts[c] *= Zetars;
        });
    }

    if(S == C)
      ifftp->fft(W);
    else
      for(unsigned int s=0; s < m; ++s)
        ifftp->fft(W+S*s);

    PARALLEL(
      for(unsigned int s=0; s < m; ++s) {
        unsigned int Ss=S*s;
        Complex *fs=f+Ss;
        Complex *Fs=W+Ss;
        for(unsigned int c=0; c < C; ++c)
          fs[c] += Fs[c];
      });
    Complex *Zetaqr=Zetaqp+pm1*r;
    for(unsigned int t=1; t < pm1; ++t) {
      unsigned int Smt=Sm*t;
      Complex *ft=f+Smt;
      Complex *Ft=W+Smt;
      Complex Zeta=conj(Zetaqr[t]);
      PARALLEL(
        for(unsigned int s=0; s < m; ++s) {
          unsigned int Ss=S*s;
          Complex *fts=ft+Ss;
          Complex *Fts=Ft+Ss;
          for(unsigned int c=0; c < C; ++c)
            fts[c] += Zeta*Fts[c];
        });
    }
    Complex *ft=f+Sm*pm1;
    Complex *Ft=W+Sm*pm1;
    Complex Zeta=conj(Zetaqr[pm1]);
    PARALLEL(
      for(unsigned int s=0; s < stop; ++s) {
        unsigned int Ss=S*s;
        Complex *fts=ft+Ss;
        Complex *Fts=Ft+Ss;
        for(unsigned int c=0; c < C; ++c)
          fts[c] += Zeta*Fts[c];
      });
  }
}

void fftPadCentered::init(bool fast)
{
  ZetaShift=NULL;
  if(q == 1) {
    if(M % 2 == 0 && fast) {
      if(C == 1 && S == 1) {
        Forward=&fftBase::forwardExplicitFast;
        Backward=&fftBase::backwardExplicitFast;
      } else {
        Forward=&fftBase::forwardExplicitManyFast;
        Backward=&fftBase::backwardExplicitManyFast;
      }
    } else {
      initShift();
      if(C == 1 && S == 1) {
        Forward=&fftBase::forwardExplicitSlow;
        Backward=&fftBase::backwardExplicitSlow;
      } else {
        Forward=&fftBase::forwardExplicitManySlow;
        Backward=&fftBase::backwardExplicitManySlow;
      }
    }
  }
}

void fftPadCentered::forwardExplicitFast(Complex *f, Complex *F,
                                         unsigned int, Complex *W)
{
  if(W == NULL) W=F;

  unsigned int H=ceilquotient(M-L,2);
  PARALLEL(
    for(unsigned int s=0; s < H; ++s)
      W[s]=0.0;
    );
  Complex *FH=W+S*H;
  PARALLEL(
    for(unsigned int s=0; s < L; ++s)
      FH[s]=f[s];
    );
  PARALLEL(
    for(unsigned int s=H+L; s < M; ++s)
      W[s]=0.0;
    );

  fftm->fft(W,F);

  PARALLEL(
    for(unsigned int s=1; s < m; s += 2)
      F[s] *= -1;
    );
}

void fftPadCentered::backwardExplicitFast(Complex *F, Complex *f,
                                          unsigned int, Complex *W)
{
  if(W == NULL) W=F;

  PARALLEL(
    for(unsigned int s=1; s < m; s += 2)
      F[s] *= -1;
    );

  ifftm->fft(F,W);

  unsigned int H=ceilquotient(M-L,2);
  Complex *FH=W+S*H;
  PARALLEL(
    for(unsigned int s=0; s < L; ++s)
      f[s]=FH[s];
    );
}

void fftPadCentered::forwardExplicitManyFast(Complex *f, Complex *F,
                                             unsigned int, Complex *W)
{
  if(W == NULL) W=F;

  unsigned int H=ceilquotient(M-L,2);
  PARALLEL(
    for(unsigned int s=0; s < H; ++s) {
      unsigned int Ss=S*s;
      Complex *Fs=W+Ss;
      for(unsigned int c=0; c < C; ++c)
        Fs[c]=0.0;
    });
  Complex *FH=W+S*H;
  PARALLEL(
    for(unsigned int s=0; s < L; ++s) {
      unsigned int Ss=S*s;
      Complex *fs=f+Ss;
      Complex *FHs=FH+Ss;
      for(unsigned int c=0; c < C; ++c)
        FHs[c]=fs[c];
    });
  PARALLEL(
    for(unsigned int s=H+L; s < M; ++s) {
      unsigned int Ss=S*s;
      Complex *Fs=W+Ss;
      for(unsigned int c=0; c < C; ++c)
        Fs[c]=0.0;
    });

  fftm->fft(W,F);

  PARALLEL(
    for(unsigned int s=1; s < m; s += 2) {
      unsigned int Ss=S*s;
      Complex *Fs=F+Ss;
      for(unsigned int c=0; c < C; ++c)
        Fs[c] *= -1;
    });
}

void fftPadCentered::backwardExplicitManyFast(Complex *F, Complex *f,
                                              unsigned int, Complex *W)
{
  if(W == NULL) W=F;

  PARALLEL(
    for(unsigned int s=1; s < m; s += 2) {
      unsigned int Ss=S*s;
      Complex *Fs=F+Ss;
      for(unsigned int c=0; c < C; ++c)
        Fs[c] *= -1;
    });

  ifftm->fft(F,W);

  unsigned int H=ceilquotient(M-L,2);
  Complex *FH=W+S*H;
  PARALLEL(
    for(unsigned int s=0; s < L; ++s) {
      unsigned int Ss=S*s;
      Complex *fs=f+Ss;
      Complex *FHs=FH+Ss;
      for(unsigned int c=0; c < C; ++c)
        fs[c]=FHs[c];
    });
}

void fftPadCentered::initShift()
{
  ZetaShift=ComplexAlign(M);
  double factor=L/2*twopi/M;
  for(unsigned int r=0; r < q; ++r) {
    Complex *Zetar=ZetaShift+r;
    PARALLEL(
      for(unsigned int s=0; s < m; ++s)
        Zetar[q*s]=expi(factor*(q*s+r));
      );
  }
}

void fftPadCentered::forwardExplicitSlow(Complex *f, Complex *F,
                                         unsigned int, Complex *W)
{
  fftPad::forwardExplicit(f,F,0,W);
  PARALLEL(
    for(unsigned int s=0; s < m; ++s)
      F[s] *= conj(ZetaShift[s]);
    );
}

void fftPadCentered::backwardExplicitSlow(Complex *F, Complex *f,
                                          unsigned int, Complex *W)
{
  PARALLEL(
    for(unsigned int s=0; s < m; ++s)
      F[s] *= ZetaShift[s];
    );
  fftPad::backwardExplicit(F,f,0,W);
}

void fftPadCentered::forwardExplicitManySlow(Complex *f, Complex *F,
                                             unsigned int, Complex *W)
{
  fftPad::forwardExplicitMany(f,F,0,W);
  PARALLEL(
    for(unsigned int s=0; s < m; ++s) {
      Complex *Fs=F+S*s;
      Complex Zeta=conj(ZetaShift[s]);
      for(unsigned int c=0; c < C; ++c)
        Fs[c] *= Zeta;
    });
}

void fftPadCentered::backwardExplicitManySlow(Complex *F, Complex *f,
                                              unsigned int, Complex *W)
{
  PARALLEL(
    for(unsigned int s=0; s < m; ++s) {
      Complex *Fs=F+S*s;
      Complex Zeta=ZetaShift[s];
      for(unsigned int c=0; c < C; ++c)
        Fs[c] *= Zeta;
    });
  fftPad::backwardExplicitMany(F,f,0,W);
}

void fftPadCentered::forward2All(Complex *f, Complex *F, unsigned int,
                                 Complex *)
{
  Complex *g=f+m;
  Complex *Zetar=Zetaqm+(m+1);
  Complex *Zetarm=Zetar+m;

  Vec V0=LOAD(f);
  Vec Zetam=CONJ(LOAD(Zetarm));
  Vec A=Zetam*UNPACKL(V0,V0);
  Vec B=ZMULTI(Zetam*UNPACKH(V0,V0));
  Vec V1=LOAD(g);
  STORE(f,V0+V1);
  STORE(g,V1+A+B);
  STORE(F,V1+CONJ(A-B));
  PARALLEL(
    for(unsigned int s=1; s < m; ++s) {
      Complex *v0=f+s;
      Complex *v1=g+s;
      Vec Zeta=LOAD(Zetar+s);
      Vec Zetam=CONJ(LOAD(Zetarm-s));
      Vec V0=LOAD(v0);
      Vec V1=LOAD(v1);
      Vec A=Zetam*UNPACKL(V0,V0)+Zeta*UNPACKL(V1,V1);
      Vec B=ZMULTI(Zetam*UNPACKH(V0,V0)+Zeta*UNPACKH(V1,V1));
      STORE(v0,V0+V1);
      STORE(v1,A+B);
      STORE(F+s,CONJ(A-B));
    });
  fftm->fft(f);
  fftm->fft(g);
  fftm->fft(F);
}

void fftPadCentered::forward2(Complex *f, Complex *F0, unsigned int r0,
                              Complex *W)
{
  if(W == NULL) W=F0;

  Complex *W0=W;
  unsigned int dr0=dr;
  mfft1d *fftm1;

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
      PARALLEL(
        for(unsigned int s=0; s < mH; ++s)
          W[s]=fH[s];
        );
      PARALLEL(
        for(unsigned int s=mH; s < LH; ++s)
          W[s]=fmH[s]+fH[s];
        );
      PARALLEL(
        for(unsigned int s=LH; s < m; ++s)
          W[s]=fmH[s];
        );
    } else {
      residues=2;
      Complex *V=W+m;
      Complex *Zetar=Zetaqm+(m+1)*q2;
      Complex *Zetarm=Zetar+m;
      PARALLEL(
        for(unsigned int s=0; s < mH; ++s) {
          W[s]=fH[s];
          V[s]=Zetar[s]*fH[s];
        });
      PARALLEL(
        for(unsigned int s=mH; s < LH; ++s) {
//        W[s]=fmH[s]+fH[s];
//        V[s]=conj(*(Zetarm-s))*fmH[s]+Zetar[s]*fH[s];
          Vec Zeta=LOAD(Zetar+s);
          Vec Zetam=LOAD(Zetarm-s);
          Vec fs=LOAD(fH+s);
          Vec fms=LOAD(fmH+s);
          STORE(W+s,fms+fs);
          STORE(V+s,ZCMULT(Zetam,fms)+ZMULT(Zeta,fs));
        });
      PARALLEL(
        for(unsigned int s=LH; s < m; ++s) {
          W[s]=fmH[s];
          V[s]=conj(*(Zetarm-s))*fmH[s];
        });
    }
    W += residues*m;
    r0=1;
    dr0=(D0-residues)/2;
    fftm1=D0 == D ? fftm : fftm0;
  } else
    fftm1=fftm;

  if(D == 1) {
    if(dr0 > 0) {
      Complex *Zetar=Zetaqm+(m+1)*r0;
      PARALLEL(
        for(unsigned int s=0; s < mH; ++s)
          W[s]=Zetar[s]*fH[s];
        );
      Complex *Zetarm=Zetar+m;
      PARALLEL(
        for(unsigned int s=mH; s < LH; ++s) {
//        W[s]=conj(*(Zetarm-s))*fmH[s]+Zetar[s]*fH[s];
          Vec Zeta=LOAD(Zetar+s);
          Vec Zetam=LOAD(Zetarm-s);
          Vec fs=LOAD(fH+s);
          Vec fms=LOAD(fmH+s);
          STORE(W+s,ZCMULT(Zetam,fms)+ZMULT(Zeta,fs));
        });
      PARALLEL(
        for(unsigned int s=LH; s < m; ++s)
          W[s]=conj(*(Zetarm-s))*fmH[s];
        );
    }
  } else {
    for(unsigned int d=0; d < dr0; ++d) {
      Complex *F=W+2*m*d;
      Complex *G=F+m;
      unsigned int r=r0+d;
      Complex *Zetar=Zetaqm+(m+1)*r;
      PARALLEL(
        for(unsigned int s=0; s < mH; ++s) {
//        F[s]=Zetar[s]*fH[s];
//        G[s]=conj(Zetar[s])*fH[s];
          Vec Zeta=LOAD(Zetar+s);
          Vec fs=LOAD(fH+s);
          Vec A=Zeta*UNPACKL(fs,fs);
          Vec B=ZMULTI(Zeta*UNPACKH(fs,fs));
          STORE(F+s,A+B);
          STORE(G+s,CONJ(A-B));
        });
      Complex *Zetarm=Zetar+m;
      PARALLEL(
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
        });
      PARALLEL(
        for(unsigned int s=LH; s < m; ++s) {
//        F[s]=conj(*(Zetarm-s))*fmH[s];
//        G[s]=*(Zetarm-s)*fmH[s];
          Vec Zetam=CONJ(LOAD(Zetarm-s));
          Vec fms=LOAD(fmH+s);
          Vec A=Zetam*UNPACKL(fms,fms);
          Vec B=ZMULTI(Zetam*UNPACKH(fms,fms));
          STORE(F+s,A+B);
          STORE(G+s,CONJ(A-B));
        });
    }
  }

  fftm1->fft(W0,F0);
}

void fftPadCentered::forward2ManyAll(Complex *f, Complex *F, unsigned int,
                                     Complex *)
{
  Complex *g=f+Sm;
  Complex *Zetar=Zetaqm+(m+1);
  Complex *Zetarm=Zetar+m;

  Vec Zetam=CONJ(LOAD(Zetarm));
  for(unsigned int c=0; c < C; ++c) {
    Complex *v0=f+c;
    Complex *v1=g+c;
    Vec V0=LOAD(v0);
    Vec A=Zetam*UNPACKL(V0,V0);
    Vec B=ZMULTI(Zetam*UNPACKH(V0,V0));
    Vec V1=LOAD(v1);
    STORE(v0,V0+V1);
    STORE(v1,V1+A+B);
    STORE(F+c,V1+CONJ(A-B));
  };

  PARALLEL(
    for(unsigned int s=1; s < m; ++s) {
      unsigned int Ss=S*s;
      Complex *v0=f+Ss;
      Complex *v1=g+Ss;
      Complex *v2=F+Ss;
      Vec Zeta=LOAD(Zetar+s);
      Vec Zetam=LOAD(Zetarm-s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(Zeta,-Zeta);
      Vec Xm=UNPACKL(Zetam,Zetam);
      Vec Ym=UNPACKH(-Zetam,Zetam);
      for(unsigned int c=0; c < C; ++c) {
        Complex *v0c=v0+c;
        Complex *v1c=v1+c;
        Vec V0=LOAD(v0c);
        Vec V1=LOAD(v1c);
        Vec A=Xm*V0+X*V1;
        Vec B=FLIP(Ym*V0+Y*V1);
        STORE(v0c,V0+V1);
        STORE(v1c,A+B);
        STORE(v2+c,A-B);
      }
    });
  fftm->fft(f);
  fftm->fft(g);
  fftm->fft(F);
}

void fftPadCentered::forward2Many(Complex *f, Complex *F, unsigned int r,
                                  Complex *W)
{
  if(W == NULL) W=F;

  unsigned int H=L/2;
  unsigned int mH=m-H;
  unsigned int LH=L-H;
  Complex *fH=f+S*H;
  Complex *fmH=f-S*mH;
  if(r == 0) {
    PARALLEL(
      for(unsigned int s=0; s < mH; ++s) {
        unsigned int Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fHs=fH+Ss;
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=fHs[c];
      });
    PARALLEL(
      for(unsigned int s=mH; s < LH; ++s) {
        unsigned int Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fmHs=fmH+Ss;
        Complex *fHs=fH+Ss;
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=fmHs[c]+fHs[c];
      });
    PARALLEL(
      for(unsigned int s=LH; s < m; ++s) {
        unsigned int Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fmHs=fmH+Ss;
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=fmHs[c];
      });
  } else {
    Complex *Zetar=Zetaqm+(m+1)*r;
    PARALLEL(
      for(unsigned int s=0; s < mH; ++s) {
        unsigned int Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fHs=fH+Ss;
//      Complex Zeta=Zetar[s];
        Vec Zeta=LOAD(Zetar+s);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(Zeta,-Zeta);
        for(unsigned int c=0; c < C; ++c)
//        Fs[c]=Zeta*fHs[c];
          STORE(Fs+c,ZMULT(X,Y,LOAD(fHs+c)));
      });
    Complex *Zetarm=Zetar+m;
    PARALLEL(
      for(unsigned int s=mH; s < LH; ++s) {
        unsigned int Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fHs=fH+Ss;
        Complex *fmHs=fmH+Ss;
//      Complex Zetars=Zetar[s];
//      Complex Zetarms=conj(*(Zetarm-s));
        Vec Zeta=LOAD(Zetar+s);
        Vec Zetam=LOAD(Zetarm-s);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(Zeta,-Zeta);
        Vec Xm=UNPACKL(Zetam,Zetam);
        Vec Ym=UNPACKH(-Zetam,Zetam);
        for(unsigned int c=0; c < C; ++c)
//        Fs[c]=Zetarms*fmHs[c]+Zetars*fHs[c];
          STORE(Fs+c,ZMULT(Xm,Ym,LOAD(fmHs+c))+ZMULT(X,Y,LOAD(fHs+c)));
      });
    PARALLEL(
      for(unsigned int s=LH; s < m; ++s) {
        unsigned int Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fmHs=fmH+Ss;
//      Complex Zetam=conj(*(Zetarm-s));
        Vec Zetam=LOAD(Zetarm-s);
        Vec Xm=UNPACKL(Zetam,Zetam);
        Vec Ym=UNPACKH(-Zetam,Zetam);
        for(unsigned int c=0; c < C; ++c)
//        Fs[c]=Zetam*fmHs[c];
          STORE(Fs+c,ZMULT(Xm,Ym,LOAD(fmHs+c)));
      });
  }
  fftm->fft(W,F);
}

void fftPadCentered::backward2All(Complex *F, Complex *f, unsigned int,
                                  Complex *)
{
  Complex *g=f+m;

  ifftm->fft(f);
  ifftm->fft(g);
  ifftm->fft(F);

  Complex *Zetar=Zetaqm+(m+1);
  Complex *Zetarm=Zetar+m;
  Vec V0=LOAD(f);
  Vec V1=LOAD(g);
  Vec V2=LOAD(F);
  STORE(f,V0+ZMULT2(LOAD(Zetarm),V1,V2));
  STORE(g,V0+V1+V2);

  PARALLEL(
    for(unsigned int s=1; s < m; ++s) {
      Complex *v0=f+s;
      Complex *v1=g+s;
//    f[s]=F0[s]+Zetarm[s]*F1[s]+conj(Zetarm[s])*F2[s];
//    g[s]=F0[s]+Zetar[s]*F1[s]+conj(Zetar[s])*F2[s];
      Vec V0=LOAD(v0);
      Vec V1=LOAD(v1);
      Vec V2=LOAD(F+s);
      STORE(v0,V0+ZMULT2(LOAD(Zetarm-s),V1,V2));
      STORE(v1,V0+ZMULT2(LOAD(Zetar+s),V2,V1));
    });
}

void fftPadCentered::backward2(Complex *F0, Complex *f, unsigned int r0,
                               Complex *W)
{
  if(W == NULL) W=F0;

  (r0 > 0 || D0 == D ? ifftm : ifftm0)->fft(F0,W);

  unsigned int dr0=dr;

  unsigned int H=L/2;
  unsigned int odd=L-2*H;
  unsigned int mH=m-H;
  unsigned int LH=L-H;
  Complex *fmH=f-mH;
  Complex *fH=f+H;
  if(r0 == 0) {
    unsigned int residues;
    unsigned int q2=q/2;
    if(D == 1 || 2*q2 < q) { // q odd, r=0
      residues=1;
      PARALLEL(
        for(unsigned int s=mH; s < m; ++s)
          fmH[s]=W[s];
        );
      PARALLEL(
        for(unsigned int s=0; s < LH; ++s)
          fH[s]=W[s];
        );
    } else { // q even, r=0,q/2
      residues=2;
      Complex *V=W+m;
      Complex *Zetar=Zetaqm+(m+1)*q2;
      Complex *Zetarm=Zetar+m;
      PARALLEL(
        for(unsigned int s=mH; s < m; ++s)
          fmH[s]=W[s]+*(Zetarm-s)*V[s];
        );
      PARALLEL(
        for(unsigned int s=0; s < LH; ++s)
          fH[s]=W[s]+conj(Zetar[s])*V[s];
        );
    }
    W += residues*m;
    r0=1;
    dr0=(D0-residues)/2;
  }

  if(D == 1) {
    if(dr0 > 0) {
      Complex *Zetar=Zetaqm+(m+1)*r0;
      Complex *Zetarm=Zetar+m;
      PARALLEL(
        for(unsigned int s=mH; s < m; ++s)
          fmH[s] += *(Zetarm-s)*W[s];
        );
      PARALLEL(
        for(unsigned int s=0; s < LH; ++s)
          fH[s] += conj(Zetar[s])*W[s];
        );
    }
  } else {
    for(unsigned int d=0; d < dr0; ++d) {
      Complex *U=W+2*m*d;
      Complex *V=U+m;
      unsigned int r=r0+d;
      Complex *Zetar=Zetaqm+(m+1)*r;
      Complex *Zetarm=Zetar+m;
      PARALLEL(
        for(unsigned int s=H; s < m; ++s)
          STORE(fmH+s,LOAD(fmH+s)+ZMULT2(LOAD(Zetarm-s),LOAD(U+s),LOAD(V+s)));
        );
      PARALLEL(
        for(unsigned int s=mH; s < H; ++s) {
//        fmH[s] += *(Zetarm-s)*Us+conj(*(Zetarm-s))*Vs;
//        fH[s] += conj(Zetar[s])*Us+Zetar[s]*Vs;
          Vec Us=LOAD(U+s);
          Vec Vs=LOAD(V+s);
          Vec Zeta=LOAD(Zetar+s);
          Vec Zetam=LOAD(Zetarm-s);
          STORE(fmH+s,LOAD(fmH+s)+ZMULT2(Zetam,Us,Vs));
          STORE(fH+s,LOAD(fH+s)+ZMULT2(Zeta,Vs,Us));
        });
      PARALLEL(
        for(unsigned int s=0; s < mH; ++s)
          STORE(fH+s,LOAD(fH+s)+ZMULT2(LOAD(Zetar+s),LOAD(V+s),LOAD(U+s)));
        );
      if(odd)
        STORE(fH+H,LOAD(fH+H)+ZMULT2(LOAD(Zetar+H),LOAD(V+H),LOAD(U+H)));
    }
  }
}

void fftPadCentered::backward2ManyAll(Complex *F, Complex *f, unsigned int,
                                      Complex *)
{
  Complex *g=f+Sm;

  ifftm->fft(f);
  ifftm->fft(g);
  ifftm->fft(F);

  Complex *Zetar=Zetaqm+(m+1);
  Complex *Zetarm=Zetar+m;
  Vec Zetam=LOAD(Zetarm);
  Vec Xm=UNPACKL(Zetam,Zetam);
  Vec Ym=UNPACKH(Zetam,-Zetam);
  for(unsigned int c=0; c < C; ++c) {
    Complex *v0=f+c;
    Complex *v1=g+c;
    Vec V0=LOAD(v0);
    Vec V1=LOAD(v1);
    Vec V2=LOAD(F+c);
    STORE(v0,V0+ZMULT2(Xm,Ym,V1,V2));
    STORE(v1,V0+V1+V2);
  };

  PARALLEL(
    for(unsigned int s=1; s < m; ++s) {
      unsigned int Ss=S*s;
      Complex *v0=f+Ss;
      Complex *v1=g+Ss;
      Complex *v2=F+Ss;
      Vec Zetam=LOAD(Zetarm-s);
      Vec Zeta=LOAD(Zetar+s);
      Vec Xm=UNPACKL(Zetam,Zetam);
      Vec Ym=UNPACKH(Zetam,-Zetam);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(-Zeta,Zeta);
      for(unsigned int c=0; c < C; ++c) {
        Complex *v0c=v0+c;
        Complex *v1c=v1+c;
        Vec V0=LOAD(v0c);
        Vec V1=LOAD(v1c);
        Vec V2=LOAD(v2+c);
        STORE(v0c,V0+ZMULT2(Xm,Ym,V1,V2));
        STORE(v1c,V0+ZMULT2(X,Y,V1,V2));
      }
    });
}

void fftPadCentered::backward2Many(Complex *F, Complex *f, unsigned int r,
                                   Complex *W)
{
  if(W == NULL) W=F;

  ifftm->fft(F,W);

  unsigned int H=L/2;
  unsigned int odd=L-2*H;
  unsigned int mH=m-H;
  Complex *fmH=f-S*mH;
  Complex *fH=f+S*H;
  if(r == 0) {
    PARALLEL(
      for(unsigned int s=H; s < m; ++s) {
        unsigned int Ss=S*s;
        Complex *fmHs=fmH+Ss;
        Complex *Fs=W+Ss;
        for(unsigned int c=0; c < C; ++c)
          fmHs[c]=Fs[c];
      });
    PARALLEL(
      for(unsigned int s=mH; s < H; ++s) {
        unsigned int Ss=S*s;
        Complex *fmHs=fmH+Ss;
        Complex *fHs=fH+Ss;
        Complex *Fs=W+Ss;
        for(unsigned int c=0; c < C; ++c)
          fHs[c]=fmHs[c]=Fs[c];
      });
    PARALLEL(
      for(unsigned int s=0; s < mH; ++s) {
        unsigned int Ss=S*s;
        Complex *fHs=fH+Ss;
        Complex *Fs=W+Ss;
        for(unsigned int c=0; c < C; ++c)
          fHs[c]=Fs[c];
      });
    if(odd) {
      unsigned int SH=S*H;
      Complex *fHs=fH+SH;
      Complex *Fs=W+SH;
      for(unsigned int c=0; c < C; ++c)
        fHs[c]=Fs[c];
    }
  } else {
    Complex *Zetar=Zetaqm+(m+1)*r;
    Complex *Zetarm=Zetar+m;
    PARALLEL(
      for(unsigned int s=H; s < m; ++s) {
        unsigned int Ss=S*s;
        Complex *fmHs=fmH+Ss;
        Complex *Fs=W+Ss;
        Vec Zetam=LOAD(Zetarm-s);
        Vec Xm=UNPACKL(Zetam,Zetam);
        Vec Ym=UNPACKH(Zetam,-Zetam);
        for(unsigned int c=0; c < C; ++c)
          STORE(fmHs+c,LOAD(fmHs+c)+ZMULT(Xm,Ym,LOAD(Fs+c)));
      });
    PARALLEL(
      for(unsigned int s=mH; s < H; ++s) {
        unsigned int Ss=S*s;
        Complex *fmHs=fmH+Ss;
        Complex *fHs=fH+Ss;
        Complex *Fs=W+Ss;
//      Complex Zetam=*(Zetarm-s);
//      Complex Zeta=conj(Zetar[s]);
        Vec Zetam=LOAD(Zetarm-s);
        Vec Zeta=LOAD(Zetar+s);
        Vec Xm=UNPACKL(Zetam,Zetam);
        Vec Ym=UNPACKH(Zetam,-Zetam);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(-Zeta,Zeta);
        for(unsigned int c=0; c < C; ++c) {
//        fmHs[c] += Zetam*Fsc;
//        fHs[c] += Zeta*Fsc;
          Vec Fsc=LOAD(Fs+c);
          STORE(fmHs+c,LOAD(fmHs+c)+ZMULT(Xm,Ym,Fsc));
          STORE(fHs+c,LOAD(fHs+c)+ZMULT(X,Y,Fsc));
        }
      });
    PARALLEL(
      for(unsigned int s=0; s < mH; ++s) {
        unsigned int Ss=S*s;
        Complex *fHs=fH+Ss;
        Complex *Fs=W+Ss;
        Vec Zeta=LOAD(Zetar+s);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(-Zeta,Zeta);
        for(unsigned int c=0; c < C; ++c)
          STORE(fHs+c,LOAD(fHs+c)+ZMULT(X,Y,LOAD(Fs+c)));
      });
    if(odd) {
      unsigned int SH=S*H;
      Complex *fHs=fH+SH;
      Complex *Fs=W+SH;
      Vec Zeta=LOAD(Zetar+H);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(-Zeta,Zeta);
      for(unsigned int c=0; c < C; ++c)
        STORE(fHs+c,LOAD(fHs+c)+ZMULT(X,Y,LOAD(Fs+c)));
    }
  }
}

void fftPadCentered::forwardInnerAll(Complex *f, Complex *F, unsigned int,
                                     Complex *)
{
  unsigned int p2=p/2;
  Vec Zetanr=CONJ(LOAD(Zetaqp+p)); // zeta_n^-r
  Complex *g=f+b;
  PARALLEL(
    for(unsigned int s=0; s < m; ++s) {
      Complex *v0=f+s;
      Complex *v1=g+s;
      Complex *v2=F+s;
      Vec V0=LOAD(v0);
      Vec V1=LOAD(v1);
      Vec A=Zetanr*UNPACKL(V0,V0);
      Vec B=ZMULTI(Zetanr*UNPACKH(V0,V0));
      STORE(v0,V0+V1);
      STORE(v1,A+B+V1);
      STORE(v2,CONJ(A-B)+V1);
    });
  Complex *Zetaqr=Zetaqp+p2;
  for(unsigned int t=1; t < p2; ++t) {
    unsigned int tm=t*m;
    Complex *v0=f+tm;
    Complex *v1=g+tm;
    Complex *v2=F+tm;
    Vec Zeta=LOAD(Zetaqr+t); // zeta_q^{tr}
    Vec Zetam=ZMULT(Zeta,Zetanr);
    PARALLEL(
      for(unsigned int s=0; s < m; ++s) {
        Complex *v0s=v0+s;
        Complex *v1s=v1+s;
        Complex *v2s=v2+s;
        Vec V0=LOAD(v0s);
        Vec V1=LOAD(v1s);
        Vec A=Zetam*UNPACKL(V0,V0)+Zeta*UNPACKL(V1,V1);
        Vec B=ZMULTI(Zetam*UNPACKH(V0,V0)+Zeta*UNPACKH(V1,V1));
        STORE(v0s,V0+V1);
        STORE(v1s,A+B);
        STORE(v2s,CONJ(A-B));
      });
  }

  fftp->fft(f);
  fftp->fft(g);
  fftp->fft(F);

  Complex *Zetar=Zetaqm+m;
  PARALLEL(
    for(unsigned int s=1; s < m; ++s) {
      Complex Zeta=Zetar[s];
      g[s] *= Zeta;
      F[s] *= conj(Zeta);
    });
  for(unsigned int u=1; u < p2; ++u) {
    unsigned int mu=m*u;
    Complex *Zetar0=Zetaqm+n*mu;
    Complex *fu=f+mu;
    PARALLEL(
      for(unsigned int s=1; s < m; ++s)
        fu[s] *= Zetar0[s];
      );
    Complex *Wu=g+mu;
    Complex *Zeta=Zetar0+m;
    PARALLEL(
      for(unsigned int s=1; s < m; ++s)
        Wu[s] *= Zeta[s];
      );
    Complex *Zeta2=Zetar0-m;
    Complex *Vu=F+mu;
    PARALLEL(
      for(unsigned int s=1; s < m; ++s)
        Vu[s] *= Zeta2[s];
      );
  }
  fftm->fft(f);
  fftm->fft(g);
  fftm->fft(F);
}

void fftPadCentered::forwardInner(Complex *f, Complex *F0, unsigned int r0,
                                  Complex *W)
{
  if(W == NULL) W=F0;

  Complex *W0=W;
  unsigned int dr0=dr;
  mfft1d *fftm1;

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
      PARALLEL(
        for(unsigned int s=0; s < m0; ++s)
          W[s]=fH[s];
        );
      PARALLEL(
        for(unsigned int s=m0; s < m; ++s)
          W[s]=fm0[s]+fH[s];
        );
      for(unsigned int t=1; t < p2s1; ++t) {
        unsigned int tm=t*m;
        Complex *Wt=W+tm;
        Complex *fm0t=fm0+tm;
        Complex *fHt=fH+tm;
        PARALLEL(
          for(unsigned int s=0; s < m; ++s)
            Wt[s]=fm0t[s]+fHt[s];
          );
      }
      Complex *Wt=W+p2s1m;
      Complex *fm0t=fm0+p2s1m;
      Complex *fHt=fH+p2s1m;
      PARALLEL(
        for(unsigned int s=0; s < m1; ++s)
          Wt[s]=fm0t[s]+fHt[s];
        );
      PARALLEL(
        for(unsigned int s=m1; s < m; ++s)
          Wt[s]=fm0t[s];
        );

      fftp->fft(W);

      unsigned int mn=m*n;
      for(unsigned int u=1; u < p2; ++u) {
        Complex *Wu=W+u*m;
        Complex *Zeta0=Zetaqm+mn*u;
        PARALLEL(
          for(unsigned int s=1; s < m; ++s)
            Wu[s] *= Zeta0[s];
          );
      }
    } else { // n even, r=0,n/2
      residues=2;
      Complex *V=W+b;
      PARALLEL(
        for(unsigned int s=0; s < m0; ++s)
          V[s]=W[s]=fH[s];
        );
      PARALLEL(
        for(unsigned int s=m0; s < m; ++s) {
          Complex fm0s=fm0[s];
          Complex fHs=fH[s];
          W[s]=fm0s+fHs;
          V[s]=-fm0s+fHs;
        });
      Complex *Zetaqn2=Zetaqp+p2*n2;
      for(unsigned int t=1; t < p2s1; ++t) {
        unsigned int tm=t*m;
        Complex *Wt=W+tm;
        Complex *Vt=V+tm;
        Complex *fm0t=fm0+tm;
        Complex *fHt=fH+tm;
        Complex Zeta=Zetaqn2[t]; //*zeta_q^{tn/2}
        PARALLEL(
          for(unsigned int s=0; s < m; ++s) {
            Complex fm0ts=fm0t[s];
            Complex fHts=fHt[s];
            Wt[s]=fm0ts+fHts;
            Vt[s]=Zeta*(fHts-fm0ts);
          });
      }
      Complex *Wt=W+p2s1m;
      Complex *Vt=V+p2s1m;
      Complex *fm0t=fm0+p2s1m;
      Complex *fHt=fH+p2s1m;
      Complex Zeta=Zetaqn2[p2s1];
      PARALLEL(
        for(unsigned int s=0; s < m1; ++s) {
          Complex fm0ts=fm0t[s];
          Complex fHts=fHt[s];
          Wt[s]=fm0ts+fHts;
          Vt[s]=Zeta*(fHts-fm0ts);
        });
      Complex mZeta=-Zeta;
      PARALLEL(
        for(unsigned int s=m1; s < m; ++s) {
          Complex fm0ts=fm0t[s];
          Wt[s]=fm0ts;
          Vt[s]=mZeta*fm0ts;
        });
      fftp->fft(W);
      fftp->fft(V);

      unsigned int mn=m*n;
      unsigned int mn2=m*n2;
      Complex *Zetan2=Zetaqm+mn2;
      PARALLEL(
        for(unsigned int s=0; s < m; ++s)
          V[s] *= Zetan2[s];
        );
      for(unsigned int u=1; u < p2; ++u) {
        unsigned int um=u*m;
        Complex *Wu=W+um;
        Complex *Zeta0=Zetaqm+mn*u;
        PARALLEL(
          for(unsigned int s=1; s < m; ++s)
            Wu[s] *= Zeta0[s];
          );
        Complex *Vu=V+um;
        Complex *Zetan2=Zeta0+mn2;
        PARALLEL(
          for(unsigned int s=1; s < m; ++s)
            Vu[s] *= Zetan2[s];
          );
      }
    }
    W += residues*b;
    r0=1;
    dr0=(D0-residues)/2;
    fftm1=D0 == D ? fftm : fftm0;
  } else
    fftm1=fftm;

  if(D == 1) {
    if(dr0 > 0) {
      Complex *Zetaqr=Zetaqp+p2*r0;
      Vec Zetanr=CONJ(LOAD(Zetaqr+p2)); // zeta_n^-r

      PARALLEL(
        for(unsigned int s=0; s < m0; ++s)
          W[s]=fH[s];
        );

      Vec X=UNPACKL(Zetanr,Zetanr);
      Vec Y=UNPACKH(Zetanr,-Zetanr);
      PARALLEL(
        for(unsigned int s=m0; s < m; ++s)
          STORE(W+s,ZMULT(X,Y,LOAD(fm0+s))+LOAD(fH+s));
        );
      for(unsigned int t=1; t < p2s1; ++t) {
        unsigned int tm=t*m;
        Complex *Ft=W+tm;
        Complex *fm0t=fm0+tm;
        Complex *fHt=fH+tm;
        Vec Zeta=LOAD(Zetaqr+t); // zeta_q^{tr}
        Vec Zetam=ZMULT(Zeta,Zetanr);

        Vec Xm=UNPACKL(Zetam,Zetam);
        Vec Ym=UNPACKH(Zetam,-Zetam);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(Zeta,-Zeta);
        PARALLEL(
          for(unsigned int s=0; s < m; ++s)
            STORE(Ft+s,ZMULT(Xm,Ym,LOAD(fm0t+s))+ZMULT(X,Y,LOAD(fHt+s)));
          );
      }
      Complex *Ft=W+p2s1m;
      Complex *fm0t=fm0+p2s1m;
      Complex *fHt=fH+p2s1m;
      Vec Zeta=LOAD(Zetaqr+p2s1);
      Vec Zetam=ZMULT(Zeta,Zetanr);
      Vec Xm=UNPACKL(Zetam,Zetam);
      Vec Ym=UNPACKH(Zetam,-Zetam);
      X=UNPACKL(Zeta,Zeta);
      Y=UNPACKH(Zeta,-Zeta);
      PARALLEL(
        for(unsigned int s=0; s < m1; ++s)
          STORE(Ft+s,ZMULT(Xm,Ym,LOAD(fm0t+s))+ZMULT(X,Y,LOAD(fHt+s)));
        );
      PARALLEL(
        for(unsigned int s=m1; s < m; ++s)
          STORE(Ft+s,ZMULT(Xm,Ym,LOAD(fm0t+s)));
        );

      fftp->fft(W);

      unsigned int mr=m*r0;
      Complex *Zetar=Zetaqm+mr;
      PARALLEL(
        for(unsigned int s=1; s < m; ++s)
          W[s] *= Zetar[s];
        );
      for(unsigned int u=1; u < p2; ++u) {
        unsigned int mu=m*u;
        Complex *Zetar0=Zetar+n*mu;
        Complex *Wu=W+mu;
        PARALLEL(
          for(unsigned int s=1; s < m; ++s)
            Wu[s] *= Zetar0[s];
          );
      }
    }
  } else {
    for(unsigned int d=0; d < dr0; ++d) {
      Complex *F=W+2*b*d;
      unsigned int r=r0+d;
      Vec Zetanr=CONJ(LOAD(Zetaqp+p2*r+p2)); // zeta_n^-r
      Complex *G=F+b;

      PARALLEL(
        for(unsigned int s=0; s < m0; ++s)
          G[s]=F[s]=fH[s];
        );
      PARALLEL(
        for(unsigned int s=m0; s < m; ++s) {
          Vec fm0ts=LOAD(fm0+s);
          Vec fHts=LOAD(fH+s);
          Vec A=Zetanr*UNPACKL(fm0ts,fm0ts);
          Vec B=ZMULTI(Zetanr*UNPACKH(fm0ts,fm0ts));
          STORE(F+s,A+B+fHts);
          STORE(G+s,CONJ(A-B)+fHts);
        });
      Complex *Zetaqr=Zetaqp+p2*r;
      for(unsigned int t=1; t < p2s1; ++t) {
        unsigned int tm=t*m;
        Complex *Ft=F+tm;
        Complex *Gt=G+tm;
        Complex *fm0t=fm0+tm;
        Complex *fHt=fH+tm;
        Vec Zeta=LOAD(Zetaqr+t); // zeta_q^{tr}
        Vec Zetam=ZMULT(Zeta,Zetanr);
        PARALLEL(
          for(unsigned int s=0; s < m; ++s) {
            Vec fm0ts=LOAD(fm0t+s);
            Vec fHts=LOAD(fHt+s);
            Vec A=Zetam*UNPACKL(fm0ts,fm0ts)+Zeta*UNPACKL(fHts,fHts);
            Vec B=ZMULTI(Zetam*UNPACKH(fm0ts,fm0ts)+Zeta*UNPACKH(fHts,fHts));
            STORE(Ft+s,A+B);
            STORE(Gt+s,CONJ(A-B));
          });
      }
      Complex *Ft=F+p2s1m;
      Complex *Gt=G+p2s1m;
      Complex *fm0t=fm0+p2s1m;
      Complex *fHt=fH+p2s1m;
      Vec Zeta=LOAD(Zetaqr+p2s1);
      Vec Zetam=ZMULT(Zeta,Zetanr);
      PARALLEL(
        for(unsigned int s=0; s < m1; ++s) {
          Vec fm0ts=LOAD(fm0t+s);
          Vec fHts=LOAD(fHt+s);
          Vec A=Zetam*UNPACKL(fm0ts,fm0ts)+Zeta*UNPACKL(fHts,fHts);
          Vec B=ZMULTI(Zetam*UNPACKH(fm0ts,fm0ts)+Zeta*UNPACKH(fHts,fHts));
          STORE(Ft+s,A+B);
          STORE(Gt+s,CONJ(A-B));
        });
      PARALLEL(
        for(unsigned int s=m1; s < m; ++s) {
          Vec fm0ts=LOAD(fm0t+s);
          Vec A=Zetam*UNPACKL(fm0ts,fm0ts);
          Vec B=ZMULTI(Zetam*UNPACKH(fm0ts,fm0ts));
          STORE(Ft+s,A+B);
          STORE(Gt+s,CONJ(A-B));
        });

      fftp->fft(F);
      fftp->fft(G);

      unsigned int mr=m*r;
      Complex *Zetar=Zetaqm+mr;
      PARALLEL(
        for(unsigned int s=1; s < m; ++s) {
          Complex Zeta=Zetar[s];
          F[s] *= Zeta;
          G[s] *= conj(Zeta);
        });
      for(unsigned int u=1; u < p2; ++u) {
        unsigned int mu=m*u;
        Complex *Zetar0=Zetaqm+n*mu;
        Complex *Zetar=Zetar0+mr;
        Complex *Wu=F+mu;
        PARALLEL(
          for(unsigned int s=1; s < m; ++s)
            Wu[s] *= Zetar[s];
          );
        Complex *Zetar2=Zetar0-mr;
        Complex *Vu=G+mu;
        PARALLEL(
          for(unsigned int s=1; s < m; ++s)
            Vu[s] *= Zetar2[s];
          );
      }
    }
  }

  fftm1->fft(W0,F0);
}

void fftPadCentered::forwardInnerManyAll(Complex *f, Complex *F, unsigned int,
                                         Complex *)
{
  unsigned int p2=p/2;
  Vec Zetanr=CONJ(LOAD(Zetaqp+p)); // zeta_n^-r
  Complex *g=f+b;
  PARALLEL(
    for(unsigned int s=0; s < m; ++s) {
      unsigned int Ss=S*s;
      Complex *v0=f+Ss;
      Complex *v1=g+Ss;
      Complex *v2=F+Ss;
      for(unsigned int c=0; c < C; ++c) {
        Complex *v0c=v0+c;
        Complex *v1c=v1+c;
        Vec V0=LOAD(v0c);
        Vec V1=LOAD(v1c);
        Vec A=Zetanr*UNPACKL(V0,V0);
        Vec B=ZMULTI(Zetanr*UNPACKH(V0,V0));
        STORE(v0c,V0+V1);
        STORE(v1c,A+B+V1);
        STORE(v2+c,CONJ(A-B)+V1);
      }
    });
  Complex *Zetaqr=Zetaqp+p2;
  for(unsigned int t=1; t < p2; ++t) {
    unsigned int tm=t*Sm;
    Complex *v0=f+tm;
    Complex *v1=g+tm;
    Complex *v2=F+tm;
    Vec Zeta=LOAD(Zetaqr+t); // zeta_q^{tr}
    Vec Zetam=ZMULT(Zeta,Zetanr);
    PARALLEL(
      for(unsigned int s=0; s < m; ++s) {
        unsigned int Ss=S*s;
        Complex *v0s=v0+Ss;
        Complex *v1s=v1+Ss;
        Complex *v2s=v2+Ss;
        for(unsigned int c=0; c < C; ++c) {
          Complex *v0c=v0s+c;
          Complex *v1c=v1s+c;
          Vec V0=LOAD(v0c);
          Vec V1=LOAD(v1c);
          Vec A=Zetam*UNPACKL(V0,V0)+Zeta*UNPACKL(V1,V1);
          Vec B=ZMULTI(Zetam*UNPACKH(V0,V0)+Zeta*UNPACKH(V1,V1));
          STORE(v0c,V0+V1);
          STORE(v1c,A+B);
          STORE(v2s+c,CONJ(A-B));
        }
      });
  }

  if(S == C) {
    fftp->fft(f);
    fftp->fft(g);
    fftp->fft(F);
  } else {
    for(unsigned int s=0; s < m; ++s) {
      unsigned int Ss=S*s;
      fftp->fft(f+Ss);
      fftp->fft(g+Ss);
      fftp->fft(F+Ss);
    }
  }

  Complex *Zetar=Zetaqm+m;
  PARALLEL(
    for(unsigned int s=1; s < m; ++s) {
      Complex Zeta=Zetar[s];
      unsigned int Ss=S*s;
      Complex *gs=g+Ss;
      Complex *Fs=F+Ss;
      for(unsigned int c=0; c < C; ++c) {
        gs[c] *= Zeta;
        Fs[c] *= conj(Zeta);
      }
    });
  unsigned int nm=n*m;
  for(unsigned int u=1; u < p2; ++u) {
    unsigned int Smu=Sm*u;
    Complex *Zetar0=Zetaqm+nm*u;
    Complex *fu=f+Smu;
    PARALLEL(
      for(unsigned int s=1; s < m; ++s) {
        Complex Zeta0=Zetar0[s];
        Complex *fus=fu+S*s;
        for(unsigned int c=0; c < C; ++c)
          fus[c] *= Zeta0;
      });
    Complex *Wu=g+Smu;
    Complex *Zetar=Zetar0+m;
    PARALLEL(
      for(unsigned int s=1; s < m; ++s) {
        Complex Zeta=Zetar[s];
        Complex *Wus=Wu+S*s;
        for(unsigned int c=0; c < C; ++c)
          Wus[c] *= Zeta;
      });
    Complex *Zetar2=Zetar0-m;
    Complex *Vu=F+Smu;
    PARALLEL(
      for(unsigned int s=1; s < m; ++s) {
        Complex Zeta2=Zetar2[s];
        Complex *Vus=Vu+S*s;
        for(unsigned int c=0; c < C; ++c)
          Vus[c] *= Zeta2;
      });
  }
  for(unsigned int t=0; t < p2; ++t) {
    unsigned int Smt=Sm*t;
    fftm->fft(f+Smt);
    fftm->fft(g+Smt);
    fftm->fft(F+Smt);
  }
}

void fftPadCentered::forwardInnerMany(Complex *f, Complex *F, unsigned int r,
                                      Complex *W)
{
  if(W == NULL) W=F;

  unsigned int p2=p/2;
  unsigned int H=L/2;
  unsigned int p2s1=p2-1;
  unsigned int p2s1m=p2s1*m;
  unsigned int p2m=p2*m;
  unsigned int m0=p2m-H;
  unsigned int m1=L-H-p2s1m;
  Complex *fm0=f-S*m0;
  Complex *fH=f+S*H;

  if(r == 0) {
    PARALLEL(
      for(unsigned int s=0; s < m0; ++s) {
        unsigned int Ss=S*s;
        Complex *Ws=W+Ss;
        Complex *fHs=fH+Ss;
        for(unsigned int c=0; c < C; ++c)
          Ws[c]=fHs[c];
      });
    PARALLEL(
      for(unsigned int s=m0; s < m; ++s) {
        unsigned int Ss=S*s;
        Complex *Ws=W+Ss;
        Complex *fm0s=fm0+Ss;
        Complex *fHs=fH+Ss;
        for(unsigned int c=0; c < C; ++c)
          Ws[c]=fm0s[c]+fHs[c];
      });
    for(unsigned int t=1; t < p2s1; ++t) {
      unsigned int tm=t*Sm;
      Complex *Wt=W+tm;
      Complex *fm0t=fm0+tm;
      Complex *fHt=fH+tm;
      PARALLEL(
        for(unsigned int s=0; s < m; ++s) {
          unsigned int Ss=S*s;
          Complex *Wts=Wt+Ss;
          Complex *fm0ts=fm0t+Ss;
          Complex *fHts=fHt+Ss;
          for(unsigned int c=0; c < C; ++c)
            Wts[c]=fm0ts[c]+fHts[c];
        });
    }
    unsigned int p2s1Sm=S*p2s1m;
    Complex *Wt=W+p2s1Sm;
    Complex *fm0t=fm0+p2s1Sm;
    Complex *fHt=fH+p2s1Sm;
    PARALLEL(
      for(unsigned int s=0; s < m1; ++s) {
        unsigned int Ss=S*s;
        Complex *Wts=Wt+Ss;
        Complex *fm0ts=fm0t+Ss;
        Complex *fHts=fHt+Ss;
        for(unsigned int c=0; c < C; ++c)
          Wts[c]=fm0ts[c]+fHts[c];
      });
    PARALLEL(
      for(unsigned int s=m1; s < m; ++s) {
        unsigned int Ss=S*s;
        Complex *Wts=Wt+Ss;
        Complex *fm0ts=fm0t+Ss;
        for(unsigned int c=0; c < C; ++c)
          Wts[c]=fm0ts[c];
      });

    if(S == C)
      fftp->fft(W);
    else
      for(unsigned int s=0; s < m; ++s)
        fftp->fft(W+S*s);

    unsigned int mn=m*n;
    for(unsigned int u=1; u < p2; ++u) {
      Complex *Wu=W+u*Sm;
      Complex *Zeta0=Zetaqm+mn*u;
      PARALLEL(
        for(unsigned int s=1; s < m; ++s) {
          Complex *Wus=Wu+S*s;
          Vec Zeta=LOAD(Zeta0+s);
          Vec X=UNPACKL(Zeta,Zeta);
          Vec Y=UNPACKH(Zeta,-Zeta);
          for(unsigned int c=0; c < C; ++c)
            STORE(Wus+c,ZMULT(X,Y,LOAD(Wus+c)));
        });
    }
  } else {
    Complex *Zetaqr=Zetaqp+p2*r;
    Vec Zetanr=CONJ(LOAD(Zetaqr+p2)); // zeta_n^-r

    PARALLEL(
      for(unsigned int s=0; s < m0; ++s) {
        unsigned int Ss=S*s;
        Complex *Ws=W+Ss;
        Complex *fHs=fH+Ss;
        for(unsigned int c=0; c < C; ++c)
          Ws[c]=fHs[c];
      });

    Vec X=UNPACKL(Zetanr,Zetanr);
    Vec Y=UNPACKH(Zetanr,-Zetanr);
    PARALLEL(
      for(unsigned int s=m0; s < m; ++s) {
        unsigned int Ss=S*s;
        Complex *Ws=W+Ss;
        Complex *fm0s=fm0+Ss;
        Complex *fHs=fH+Ss;
        for(unsigned int c=0; c < C; ++c)
          STORE(Ws+c,ZMULT(X,Y,LOAD(fm0s+c))+LOAD(fHs+c));
      });
    for(unsigned int t=1; t < p2s1; ++t) {
      unsigned int tm=t*Sm;
      Complex *Ft=W+tm;
      Complex *fm0t=fm0+tm;
      Complex *fHt=fH+tm;
      Vec Zeta=LOAD(Zetaqr+t); // zeta_q^{tr}
      Vec Zetam=ZMULT(Zeta,Zetanr);

      Vec Xm=UNPACKL(Zetam,Zetam);
      Vec Ym=UNPACKH(Zetam,-Zetam);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(Zeta,-Zeta);
      PARALLEL(
        for(unsigned int s=0; s < m; ++s) {
          unsigned int Ss=S*s;
          Complex *Fts=Ft+Ss;
          Complex *fm0ts=fm0t+Ss;
          Complex *fHts=fHt+Ss;
          for(unsigned int c=0; c < C; ++c)
            STORE(Fts+c,ZMULT(Xm,Ym,LOAD(fm0ts+c))+ZMULT(X,Y,LOAD(fHts+c)));
        });
    }
    unsigned int p2s1Sm=S*p2s1m;
    Complex *Ft=W+p2s1Sm;
    Complex *fm0t=fm0+p2s1Sm;
    Complex *fHt=fH+p2s1Sm;
    Vec Zeta=LOAD(Zetaqr+p2s1);
    Vec Zetam=ZMULT(Zeta,Zetanr);
    Vec Xm=UNPACKL(Zetam,Zetam);
    Vec Ym=UNPACKH(Zetam,-Zetam);
    X=UNPACKL(Zeta,Zeta);
    Y=UNPACKH(Zeta,-Zeta);
    PARALLEL(
      for(unsigned int s=0; s < m1; ++s) {
        unsigned int Ss=S*s;
        Complex *Fts=Ft+Ss;
        Complex *fm0ts=fm0t+Ss;
        Complex *fHts=fHt+Ss;
        for(unsigned int c=0; c < C; ++c)
          STORE(Fts+c,ZMULT(Xm,Ym,LOAD(fm0ts+c))+ZMULT(X,Y,LOAD(fHts+c)));
      });
    PARALLEL(
      for(unsigned int s=m1; s < m; ++s) {
        unsigned int Ss=S*s;
        Complex *Fts=Ft+Ss;
        Complex *fm0ts=fm0t+Ss;
        for(unsigned int c=0; c < C; ++c)
          STORE(Fts+c,ZMULT(Xm,Ym,LOAD(fm0ts+c)));
      });

    if(S == C)
      fftp->fft(W);
    else
      for(unsigned int s=0; s < m; ++s)
        fftp->fft(W+S*s);

    unsigned int mr=m*r;
    for(unsigned int u=0; u < p2; ++u) {
      unsigned int mu=m*u;
      Complex *Zetar0=Zetaqm+n*mu;
      Complex *Zetar=Zetar0+mr;
      Complex *Wu=W+S*mu;
      PARALLEL(
        for(unsigned int s=1; s < m; ++s) {
          Complex *Wus=Wu+S*s;
          Vec Zetars=LOAD(Zetar+s);
          Vec X=UNPACKL(Zetars,Zetars);
          Vec Y=UNPACKH(Zetars,-Zetars);
          for(unsigned int c=0; c < C; ++c)
            STORE(Wus+c,ZMULT(X,Y,LOAD(Wus+c)));
        });
    }
  }
  for(unsigned int t=0; t < p2; ++t) {
    unsigned int Smt=Sm*t;
    fftm->fft(W+Smt,F+Smt);
  }
}

void fftPadCentered::backwardInnerAll(Complex *F, Complex *f, unsigned int,
                                      Complex *)
{
  Complex *g=f+b;

  ifftm->fft(f);
  ifftm->fft(g);
  ifftm->fft(F);

  Vec Zetanr=LOAD(Zetaqp+p); // zeta_n^r

  Complex *Zetar=Zetaqm+m;
  PARALLEL(
    for(unsigned int s=1; s < m; ++s) {
      Complex Zeta=Zetar[s];
      g[s] *= conj(Zeta);
      F[s] *= Zeta;
    });
  unsigned int p2=p/2;
  for(unsigned int u=1; u < p2; ++u) {
    unsigned int mu=m*u;
    Complex *Zetar0=Zetaqm+n*mu;
    Complex *fu=f+mu;
    PARALLEL(
      for(unsigned int s=1; s < m; ++s)
        fu[s] *= conj(Zetar0[s]);
      );
    Complex *Wu=g+mu;
    Complex *Zetar1=Zetar0+m;
    PARALLEL(
      for(unsigned int s=1; s < m; ++s)
        Wu[s] *= conj(Zetar1[s]);
      );
    Complex *Zetar2=Zetar0-m;
    Complex *Vu=F+mu;
    PARALLEL(
      for(unsigned int s=1; s < m; ++s)
        Vu[s] *= conj(Zetar2[s]);
      );
  }

  ifftp->fft(f);
  ifftp->fft(g);
  ifftp->fft(F);

  Vec Xm=UNPACKL(Zetanr,Zetanr);
  Vec Ym=UNPACKH(Zetanr,-Zetanr);
  PARALLEL(
    for(unsigned int s=0; s < m; ++s) {
      Complex *v0s=f+s;
      Complex *v1s=g+s;
      Vec V0=LOAD(v0s);
      Vec V1=LOAD(v1s);
      Vec V2=LOAD(F+s);
      STORE(v0s,V0+ZMULT2(Xm,Ym,V1,V2));
      STORE(v1s,V0+V1+V2);
    });
  Complex *Zetaqr=Zetaqp+p2;
  for(unsigned int t=1; t < p2; ++t) {
    unsigned int tm=t*m;
    Complex *v0=f+tm;
    Complex *v1=g+tm;
    Complex *v2=F+tm;
    Vec Zeta=LOAD(Zetaqr+t);
    Vec X=UNPACKL(Zeta,Zeta);
    Vec Y=UNPACKH(Zeta,-Zeta);
    Vec Zeta2=ZMULT(X,-Y,Zetanr);
    Vec Xm=UNPACKL(Zeta2,Zeta2);
    Vec Ym=UNPACKH(Zeta2,-Zeta2);
    PARALLEL(
      for(unsigned int s=0; s < m; ++s) {
        Complex *v0s=v0+s;
        Complex *v1s=v1+s;
        Vec V0=LOAD(v0s);
        Vec V1=LOAD(v1s);
        Vec V2=LOAD(v2+s);
        STORE(v0s,V0+ZMULT2(Xm,Ym,V1,V2));
        STORE(v1s,V0+ZMULT2(X,Y,V2,V1));
      });
  }
}

void fftPadCentered::backwardInner(Complex *F0, Complex *f, unsigned int r0,
                                   Complex *W)
{
  if(W == NULL) W=F0;

  (r0 > 0 || D0 == D ? ifftm : ifftm0)->fft(F0,W);

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
      unsigned int mn=m*n;
      for(unsigned int u=1; u < p2; ++u) {
        Complex *Wu=W+u*m;
        Complex *Zeta0=Zetaqm+mn*u;
        PARALLEL(
          for(unsigned int s=1; s < m; ++s)
            Wu[s] *= conj(Zeta0[s]);
          );
      }

      ifftp->fft(W);

      PARALLEL(
        for(unsigned int s=0; s < m0; ++s)
          fH[s]=W[s];
        );

      PARALLEL(
        for(unsigned int s=m0; s < m; ++s)
          fH[s]=fm0[s]=W[s];
        );

      for(unsigned int t=1; t < p2s1; ++t) {
        unsigned int tm=t*m;
        Complex *Wt=W+tm;
        Complex *fm0t=fm0+tm;
        Complex *fHt=fH+tm;
        PARALLEL(
          for(unsigned int s=0; s < m; ++s)
            fHt[s]=fm0t[s]=Wt[s];
          );
      }
      Complex *Wt=W+p2s1m;
      Complex *fm0t=fm0+p2s1m;
      Complex *fHt=fH+p2s1m;
      PARALLEL(
        for(unsigned int s=0; s < m1; ++s)
          fHt[s]=fm0t[s]=Wt[s];
        );

      PARALLEL(
        for(unsigned int s=m1; s < m; ++s)
          fm0t[s]=Wt[s];
        );

    } else { // n even, r=0,n/2
      residues=2;
      Complex *V=W+b;
      unsigned int mn=m*n;
      unsigned int mn2=m*n2;

      Complex *Zetan2=Zetaqm+mn2;
      PARALLEL(
        for(unsigned int s=1; s < m; ++s)
          V[s] *= conj(Zetan2[s]);
        );

      for(unsigned int u=1; u < p2; ++u) {
        unsigned int mu=m*u;
        Complex *Wu=W+mu;
        Complex *Zeta0=Zetaqm+mn*u;
        PARALLEL(
          for(unsigned int s=1; s < m; ++s)
            Wu[s] *= conj(Zeta0[s]);
          );
        Complex *Vu=V+mu;
        Complex *Zetan2=Zeta0+mn2;
        PARALLEL(
          for(unsigned int s=1; s < m; ++s)
            Vu[s] *= conj(Zetan2[s]);
          );
      }
      ifftp->fft(W);
      ifftp->fft(V);
      PARALLEL(
        for(unsigned int s=0; s < m0; ++s)
          fH[s]=W[s]+V[s];
        );

      PARALLEL(
        for(unsigned int s=m0; s < m; ++s) {
          Complex Wts=W[s];
          Complex Vts=V[s];
          fm0[s]=Wts-Vts;
          fH[s]=Wts+Vts;
        });
      Complex *Zetaqn2=Zetaqp+p2*n2;
      for(unsigned int t=1; t < p2s1; ++t) {
        unsigned int tm=t*m;
        Complex *Wt=W+tm;
        Complex *Vt=V+tm;
        Complex *fm0t=fm0+tm;
        Complex *fHt=fH+tm;
        Complex Zeta=conj(Zetaqn2[t]);
        PARALLEL(
          for(unsigned int s=0; s < m; ++s) {
            Complex Wts=Wt[s];
            Complex Vts=Zeta*Vt[s];
            fm0t[s]=Wts-Vts;
            fHt[s]=Wts+Vts;
          });
      }
      Complex *Wt=W+p2s1m;
      Complex *Vt=V+p2s1m;
      Complex *fm0t=fm0+p2s1m;
      Complex *fHt=fH+p2s1m;
      Complex Zeta=conj(Zetaqn2[p2s1]);
      PARALLEL(
        for(unsigned int s=0; s < m1; ++s) {
          Complex Wts=Wt[s];
          Complex Vts=Zeta*Vt[s];
          fm0t[s]=Wts-Vts;
          fHt[s]=Wts+Vts;
        });
      PARALLEL(
        for(unsigned int s=m1; s < m; ++s)
          fm0t[s]=Wt[s]-Zeta*Vt[s];
        );
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
      PARALLEL(
        for(unsigned int s=1; s < m; ++s)
          W[s] *= conj(Zetar[s]);
        );
      for(unsigned int u=1; u < p2; ++u) {
        unsigned int mu=m*u;
        Complex *Zetar0=Zetar+n*mu;
        Complex *Wu=W+mu;
        PARALLEL(
          for(unsigned int s=1; s < m; ++s)
            Wu[s] *= conj(Zetar0[s]);
          );
      }

      ifftp->fft(W);

      PARALLEL(
        for(unsigned int s=0; s < m0; ++s)
          fH[s] += W[s];
        );

      Vec Xm=UNPACKL(Zetanr,Zetanr);
      Vec Ym=UNPACKH(Zetanr,-Zetanr);
      PARALLEL(
        for(unsigned int s=m0; s < m; ++s) {
          Vec Fts=LOAD(W+s);
          STORE(fm0+s,LOAD(fm0+s)+ZMULT(Xm,Ym,Fts));
          STORE(fH+s,LOAD(fH+s)+Fts);
        });
      Complex *Zetaqr=Zetaqp+p2*r0;
      for(unsigned int t=1; t < p2s1; ++t) {
        unsigned int tm=t*m;
        Complex *Ft=W+tm;
        Complex *fm0t=fm0+tm;
        Complex *fHt=fH+tm;
        Vec Zeta=LOAD(Zetaqr+t);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(-Zeta,Zeta);
        Vec Zeta2=ZMULT(X,Y,Zetanr);
        Vec Xm=UNPACKL(Zeta2,Zeta2);
        Vec Ym=UNPACKH(Zeta2,-Zeta2);
        PARALLEL(
          for(unsigned int s=0; s < m; ++s) {
            Vec Fts=LOAD(Ft+s);
            STORE(fm0t+s,LOAD(fm0t+s)+ZMULT(Xm,Ym,Fts));
            STORE(fHt+s,LOAD(fHt+s)+ZMULT(X,Y,Fts));
          });
      }
      Complex *Ft=W+p2s1m;
      Complex *fm0t=fm0+p2s1m;
      Complex *fHt=fH+p2s1m;
      Vec Zeta=LOAD(Zetaqr+p2s1);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(-Zeta,Zeta);
      Vec Zeta2=ZMULT(X,Y,Zetanr);
      Xm=UNPACKL(Zeta2,Zeta2);
      Ym=UNPACKH(Zeta2,-Zeta2);
      PARALLEL(
        for(unsigned int s=0; s < m1; ++s) {
          Vec Fts=LOAD(Ft+s);
          STORE(fm0t+s,LOAD(fm0t+s)+ZMULT(Xm,Ym,Fts));
          STORE(fHt+s,LOAD(fHt+s)+ZMULT(X,Y,Fts));
        });
      PARALLEL(
        for(unsigned int s=m1; s < m; ++s)
          STORE(fm0t+s,LOAD(fm0t+s)+ZMULT(Xm,Ym,LOAD(Ft+s)));
        );
    }
  } else {
    for(unsigned int d=0; d < dr0; ++d) {
      Complex *F=W+2*b*d;
      unsigned int r=r0+d;
      Vec Zetanr=LOAD(Zetaqp+p2*r+p2); // zeta_n^r
      Complex *G=F+b;

      unsigned int mr=m*r;
      Complex *Zetar=Zetaqm+mr;
      PARALLEL(
        for(unsigned int s=1; s < m; ++s) {
          Complex Zeta=Zetar[s];
          F[s] *= conj(Zeta);
          G[s] *= Zeta;
        });
      for(unsigned int u=1; u < p2; ++u) {
        unsigned int mu=m*u;
        Complex *Zetar0=Zetaqm+n*mu;
        Complex *Zetar1=Zetar0+mr;
        Complex *Wu=F+mu;
        PARALLEL(
          for(unsigned int s=1; s < m; ++s)
            Wu[s] *= conj(Zetar1[s]);
          );
        Complex *Zetar2=Zetar0-mr;
        Complex *Vu=G+mu;
        PARALLEL(
          for(unsigned int s=1; s < m; ++s)
            Vu[s] *= conj(Zetar2[s]);
          );
      }

      ifftp->fft(F);
      ifftp->fft(G);

      PARALLEL(
        for(unsigned int s=0; s < m0; ++s)
          fH[s] += F[s]+G[s];
        );

      Vec Xm=UNPACKL(Zetanr,Zetanr);
      Vec Ym=UNPACKH(Zetanr,-Zetanr);
      PARALLEL(
        for(unsigned int s=m0; s < m; ++s) {
          Vec Fts=LOAD(F+s);
          Vec Gts=LOAD(G+s);
          STORE(fm0+s,LOAD(fm0+s)+ZMULT2(Xm,Ym,Fts,Gts));
          STORE(fH+s,LOAD(fH+s)+Fts+Gts);
        });
      Complex *Zetaqr=Zetaqp+p2*r;
      for(unsigned int t=1; t < p2s1; ++t) {
        unsigned int tm=t*m;
        Complex *Ft=F+tm;
        Complex *Gt=G+tm;
        Complex *fm0t=fm0+tm;
        Complex *fHt=fH+tm;
        Vec Zeta=LOAD(Zetaqr+t);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(Zeta,-Zeta);
        Vec Zeta2=ZMULT(X,-Y,Zetanr);
        Vec Xm=UNPACKL(Zeta2,Zeta2);
        Vec Ym=UNPACKH(Zeta2,-Zeta2);
        PARALLEL(
          for(unsigned int s=0; s < m; ++s) {
            Vec Fts=LOAD(Ft+s);
            Vec Gts=LOAD(Gt+s);
            STORE(fm0t+s,LOAD(fm0t+s)+ZMULT2(Xm,Ym,Fts,Gts));
            STORE(fHt+s,LOAD(fHt+s)+ZMULT2(X,Y,Gts,Fts));
          });
      }
      Complex *Ft=F+p2s1m;
      Complex *Gt=G+p2s1m;
      Complex *fm0t=fm0+p2s1m;
      Complex *fHt=fH+p2s1m;
      Vec Zeta=LOAD(Zetaqr+p2s1);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(Zeta,-Zeta);
      Vec Zeta2=ZMULT(X,-Y,Zetanr);
      Xm=UNPACKL(Zeta2,Zeta2);
      Ym=UNPACKH(Zeta2,-Zeta2);
      PARALLEL(
        for(unsigned int s=0; s < m1; ++s) {
          Vec Fts=LOAD(Ft+s);
          Vec Gts=LOAD(Gt+s);
          STORE(fm0t+s,LOAD(fm0t+s)+ZMULT2(Xm,Ym,Fts,Gts));
          STORE(fHt+s,LOAD(fHt+s)+ZMULT2(X,Y,Gts,Fts));
        });
      PARALLEL(
        for(unsigned int s=m1; s < m; ++s)
          STORE(fm0t+s,LOAD(fm0t+s)+ZMULT2(Xm,Ym,LOAD(Ft+s),LOAD(Gt+s)));
        );
    }
  }
}

void fftPadCentered::backwardInnerManyAll(Complex *F, Complex *f, unsigned int,
                                          Complex *)
{
  Complex *g=f+b;

  unsigned int p2=p/2;
  for(unsigned int t=0; t < p2; ++t) {
    unsigned int Smt=Sm*t;
    ifftm->fft(f+Smt);
    ifftm->fft(g+Smt);
    ifftm->fft(F+Smt);
  }

  Vec Zetanr=LOAD(Zetaqp+p); // zeta_n^r

  Complex *Zetar=Zetaqm+m;
  PARALLEL(
    for(unsigned int s=1; s < m; ++s) {
      Complex Zeta=Zetar[s];
      unsigned int Ss=S*s;
      Complex *gs=g+Ss;
      Complex *Fs=F+Ss;
      for(unsigned int c=0; c < C; ++c) {
        gs[c] *= conj(Zeta);
        Fs[c] *= Zeta;
      }
    });
  unsigned int nm=n*m;
  for(unsigned int u=1; u < p2; ++u) {
    unsigned int Smu=Sm*u;
    Complex *Zetar0=Zetaqm+nm*u;
    Complex *fu=f+Smu;
    PARALLEL(
      for(unsigned int s=1; s < m; ++s) {
        Complex Zeta0=conj(Zetar0[s]);
        Complex *fus=fu+S*s;
        for(unsigned int c=0; c < C; ++c)
          fus[c] *= Zeta0;
      });
    Complex *Wu=g+Smu;
    Complex *Zetar=Zetar0+m;
    PARALLEL(
      for(unsigned int s=1; s < m; ++s) {
        Complex Zetar1=conj(Zetar[s]);
        Complex *Wus=Wu+S*s;
        for(unsigned int c=0; c < C; ++c)
          Wus[c] *= Zetar1;
      });
    Complex *Zetar2=Zetar0-m;
    Complex *Vu=F+Smu;
    PARALLEL(
      for(unsigned int s=1; s < m; ++s) {
        Complex Zeta2=conj(Zetar2[s]);
        Complex *Vus=Vu+S*s;
        for(unsigned int c=0; c < C; ++c)
          Vus[c] *= Zeta2;
      });
  }

  if(S == C) {
    ifftp->fft(f);
    ifftp->fft(g);
    ifftp->fft(F);
  } else {
    for(unsigned int s=0; s < m; ++s) {
      unsigned int Ss=S*s;
      ifftp->fft(f+Ss);
      ifftp->fft(g+Ss);
      ifftp->fft(F+Ss);
    }
  }

  Vec Xm=UNPACKL(Zetanr,Zetanr);
  Vec Ym=UNPACKH(Zetanr,-Zetanr);
  PARALLEL(
    for(unsigned int s=0; s < m; ++s) {
      unsigned int Ss=S*s;
      Complex *v0=f+Ss;
      Complex *v1=g+Ss;
      Complex *v2=F+Ss;
      for(unsigned int c=0; c < C; ++c) {
        Complex *v0c=v0+c;
        Complex *v1c=v1+c;
        Vec V0=LOAD(v0c);
        Vec V1=LOAD(v1c);
        Vec V2=LOAD(v2+c);
        STORE(v0c,V0+ZMULT2(Xm,Ym,V1,V2));
        STORE(v1c,V0+V1+V2);
      }
    });
  Complex *Zetaqr=Zetaqp+p2;
  for(unsigned int t=1; t < p2; ++t) {
    unsigned int tm=t*Sm;
    Complex *v0=f+tm;
    Complex *v1=g+tm;
    Complex *v2=F+tm;
    Vec Zeta=LOAD(Zetaqr+t);
    Vec X=UNPACKL(Zeta,Zeta);
    Vec Y=UNPACKH(Zeta,-Zeta);
    Vec Zeta2=ZMULT(X,-Y,Zetanr);
    Vec Xm=UNPACKL(Zeta2,Zeta2);
    Vec Ym=UNPACKH(Zeta2,-Zeta2);
    PARALLEL(
      for(unsigned int s=0; s < m; ++s) {
        unsigned int Ss=S*s;
        Complex *v0s=v0+Ss;
        Complex *v1s=v1+Ss;
        Complex *v2s=v2+Ss;
        for(unsigned int c=0; c < C; ++c) {
          Complex *v0c=v0s+c;
          Complex *v1c=v1s+c;
          Vec V0=LOAD(v0c);
          Vec V1=LOAD(v1c);
          Vec V2=LOAD(v2s+c);
          STORE(v0c,V0+ZMULT2(Xm,Ym,V1,V2));
          STORE(v1c,V0+ZMULT2(X,Y,V2,V1));
        }
      });
  }
}

void fftPadCentered::backwardInnerMany(Complex *F, Complex *f, unsigned int r,
                                       Complex *W)
{
  if(W == NULL) W=F;

  unsigned int p2=p/2;

  for(unsigned int t=0; t < p2; ++t) {
    unsigned int Smt=Sm*t;
    ifftm->fft(F+Smt,W+Smt);
  }

  unsigned int H=L/2;
  unsigned int p2s1=p2-1;
  unsigned int p2s1m=p2s1*m;
  unsigned int p2m=p2*m;
  unsigned int m0=p2m-H;
  unsigned int m1=L-H-p2s1m;
  Complex *fm0=f-S*m0;
  Complex *fH=f+S*H;

  if(r == 0) {
    unsigned int mn=m*n;
    for(unsigned int u=1; u < p2; ++u) {
      Complex *Wu=W+u*Sm;
      Complex *Zeta0=Zetaqm+mn*u;
      PARALLEL(
        for(unsigned int s=1; s < m; ++s) {
          Complex *Wus=Wu+S*s;
          Vec Zeta=LOAD(Zeta0+s);
          Vec X=UNPACKL(Zeta,Zeta);
          Vec Y=UNPACKH(-Zeta,Zeta);
          for(unsigned int c=0; c < C; ++c)
            STORE(Wus+c,ZMULT(X,Y,LOAD(Wus+c)));
        });
    }

    if(S == C)
      ifftp->fft(W);
    else
      for(unsigned int s=0; s < m; ++s)
        ifftp->fft(W+S*s);

    PARALLEL(
      for(unsigned int s=0; s < m0; ++s) {
        unsigned int Ss=S*s;
        Complex *Ws=W+Ss;
        Complex *fHs=fH+Ss;
        for(unsigned int c=0; c < C; ++c)
          fHs[c]=Ws[c];
      });

    PARALLEL(
      for(unsigned int s=m0; s < m; ++s) {
        unsigned int Ss=S*s;
        Complex *Ws=W+Ss;
        Complex *fm0s=fm0+Ss;
        Complex *fHs=fH+Ss;
        for(unsigned int c=0; c < C; ++c)
          fHs[c]=fm0s[c]=Ws[c];
      });

    for(unsigned int t=1; t < p2s1; ++t) {
      unsigned int tm=t*Sm;
      Complex *Wt=W+tm;
      Complex *fm0t=fm0+tm;
      Complex *fHt=fH+tm;
      PARALLEL(
        for(unsigned int s=0; s < m; ++s) {
          unsigned int Ss=S*s;
          Complex *Wts=Wt+Ss;
          Complex *fm0ts=fm0t+Ss;
          Complex *fHts=fHt+Ss;
          for(unsigned int c=0; c < C; ++c)
            fHts[c]=fm0ts[c]=Wts[c];
        });
    }
    unsigned int p2s1Sm=S*p2s1m;
    Complex *Wt=W+p2s1Sm;
    Complex *fm0t=fm0+p2s1Sm;
    Complex *fHt=fH+p2s1Sm;
    PARALLEL(
      for(unsigned int s=0; s < m1; ++s) {
        unsigned int Ss=S*s;
        Complex *Wts=Wt+Ss;
        Complex *fm0ts=fm0t+Ss;
        Complex *fHts=fHt+Ss;
        for(unsigned int c=0; c < C; ++c)
          fHts[c]=fm0ts[c]=Wts[c];
      });

    PARALLEL(
      for(unsigned int s=m1; s < m; ++s) {
        unsigned int Ss=S*s;
        Complex *Wts=Wt+Ss;
        Complex *fm0ts=fm0t+Ss;
        for(unsigned int c=0; c < C; ++c)
          fm0ts[c]=Wts[c];
      });
  } else {
    Vec Zetanr=LOAD(Zetaqp+p2*r+p2); // zeta_n^r
    unsigned int mr=m*r;
    Complex *Zetar=Zetaqm+mr;
    for(unsigned int u=0; u < p2; ++u) {
      unsigned int mu=m*u;
      Complex *Zetar0=Zetar+n*mu;
      Complex *Wu=W+u*Sm;
      PARALLEL(
        for(unsigned int s=1; s < m; ++s) {
          Complex *Wus=Wu+S*s;
          Vec Zeta=LOAD(Zetar0+s);
          Vec X=UNPACKL(Zeta,Zeta);
          Vec Y=UNPACKH(-Zeta,Zeta);
          for(unsigned int c=0; c < C; ++c)
            STORE(Wus+c,ZMULT(X,Y,LOAD(Wus+c)));
        });
    }

    if(S == C)
      ifftp->fft(W);
    else
      for(unsigned int s=0; s < m; ++s)
        ifftp->fft(W+S*s);

    PARALLEL(
      for(unsigned int s=0; s < m0; ++s) {
        unsigned int Ss=S*s;
        Complex *Ws=W+Ss;
        Complex *fHs=fH+Ss;
        for(unsigned int c=0; c < C; ++c)
          fHs[c] += Ws[c];
      });

    Vec Xm=UNPACKL(Zetanr,Zetanr);
    Vec Ym=UNPACKH(Zetanr,-Zetanr);
    PARALLEL(
      for(unsigned int s=m0; s < m; ++s) {
        unsigned int Ss=S*s;
        Complex *Ws=W+Ss;
        Complex *fm0s=fm0+Ss;
        Complex *fHs=fH+Ss;
        for(unsigned int c=0; c < C; ++c) {
          Vec Ftsc=LOAD(Ws+c);
          STORE(fm0s+c,LOAD(fm0s+c)+ZMULT(Xm,Ym,Ftsc));
          STORE(fHs+c,LOAD(fHs+c)+Ftsc);
        }
      });
    Complex *Zetaqr=Zetaqp+p2*r;
    for(unsigned int t=1; t < p2s1; ++t) {
      unsigned int tm=t*Sm;
      Complex *Ft=W+tm;
      Complex *fm0t=fm0+tm;
      Complex *fHt=fH+tm;
      Vec Zeta=LOAD(Zetaqr+t);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(-Zeta,Zeta);
      Vec Zeta2=ZMULT(X,Y,Zetanr);
      Vec Xm=UNPACKL(Zeta2,Zeta2);
      Vec Ym=UNPACKH(Zeta2,-Zeta2);
      PARALLEL(
        for(unsigned int s=0; s < m; ++s) {
          unsigned int Ss=S*s;
          Complex *Fts=Ft+Ss;
          Complex *fm0ts=fm0t+Ss;
          Complex *fHts=fHt+Ss;
          for(unsigned int c=0; c < C; ++c) {
            Vec Ftsc=LOAD(Fts+c);
            STORE(fm0ts+c,LOAD(fm0ts+c)+ZMULT(Xm,Ym,Ftsc));
            STORE(fHts+c,LOAD(fHts+c)+ZMULT(X,Y,Ftsc));
          }
        });
    }
    unsigned int p2s1Sm=S*p2s1m;
    Complex *Ft=W+p2s1Sm;
    Complex *fm0t=fm0+p2s1Sm;
    Complex *fHt=fH+p2s1Sm;
    Vec Zeta=LOAD(Zetaqr+p2s1);
    Vec X=UNPACKL(Zeta,Zeta);
    Vec Y=UNPACKH(-Zeta,Zeta);
    Vec Zeta2=ZMULT(X,Y,Zetanr);
    Xm=UNPACKL(Zeta2,Zeta2);
    Ym=UNPACKH(Zeta2,-Zeta2);
    PARALLEL(
      for(unsigned int s=0; s < m1; ++s) {
        unsigned int Ss=S*s;
        Complex *Fts=Ft+Ss;
        Complex *fm0ts=fm0t+Ss;
        Complex *fHts=fHt+Ss;
        for(unsigned int c=0; c < C; ++c) {
          Vec Ftsc=LOAD(Fts+c);
          STORE(fm0ts+c,LOAD(fm0ts+c)+ZMULT(Xm,Ym,Ftsc));
          STORE(fHts+c,LOAD(fHts+c)+ZMULT(X,Y,Ftsc));
        }
      });
    PARALLEL(
      for(unsigned int s=m1; s < m; ++s) {
        unsigned int Ss=S*s;
        Complex *Fts=Ft+Ss;
        Complex *fm0ts=fm0t+Ss;
        for(unsigned int c=0; c < C; ++c)
          STORE(fm0ts+c,LOAD(fm0ts+c)+ZMULT(Xm,Ym,LOAD(Fts+c)));
      });
  }
}

void fftPadHermitian::init()
{
  common();
  S=C; // Strides are not implemented for Hermitian transforms
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

    Complex *G=ComplexAlign(Ce1);
    double *H=inplace ? (double *) G : doubleAlign(C*m);

    crfftm=new mcrfft1d(m,C, C,C, 1,1, G,H);
    rcfftm=new mrcfft1d(m,C, C,C, 1,1, H,G);

    if(!inplace)
      deleteAlign(H);
    deleteAlign(G);
    dr=D0=R=Q=1;
  } else {
    dr=Dr();

    double twopibyq=twopi/q;
    unsigned int p2=p/2;

    // Output block size
    b=align(ceilquotient(p2*Cm,2));
    // Work block size
    B=align(p2*Ce1);

    if(inplace) b=B;

    Complex *G=ComplexAlign(B);
    double *H=inplace ? (double *) G :
      doubleAlign(C == 1 ? 2*B : Cm*p2);

    if(p > 2) { // p must be even, only C=1 implemented
      Forward=&fftBase::forwardInner;
      Backward=&fftBase::backwardInner;
      Q=n=q/p2;

      Zetaqp0=ComplexAlign((n-1)*p2);
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

    R=residueBlocks();
    D0=Q % D;
    if(D0 == 0) D0=D;

    if(C == 1) {
      crfftm=new mcrfft1d(m,p2, 1,1, e1,m, G,H);
      rcfftm=new mrcfft1d(m,p2, 1,1, m,e1, H,G);
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
  if(p > 2) {
    delete fftp;
    delete ifftp;
    deleteAlign(Zetaqp0);
  }
}

void fftPadHermitian::forwardExplicit(Complex *f, Complex *F, unsigned int, Complex *W)
{
  if(W == NULL) W=F;

  unsigned int H=ceilquotient(L,2);
  if(W != f) {
    PARALLEL(
      for(unsigned int s=0; s < H; ++s)
        W[s]=f[s];
      );
  }
  PARALLEL(
    for(unsigned int s=H; s <= e; ++s)
      W[s]=0.0;
    );

  crfftm->fft(W,F);
}

void fftPadHermitian::forwardExplicitMany(Complex *f, Complex *F, unsigned int,
                                          Complex *W)
{
  if(W == NULL) W=F;

  unsigned int H=ceilquotient(L,2);
  if(W != f) {
    PARALLEL(
      for(unsigned int s=0; s < H; ++s) {
        unsigned int Cs=C*s;
        Complex *Fs=W+Cs;
        Complex *fs=f+Cs;
        for(unsigned int c=0; c < C; ++c)
          Fs[c]=fs[c];
      });
  }
  PARALLEL(
    for(unsigned int s=H; s <= e; ++s) {
      Complex *Fs=W+C*s;
      for(unsigned int c=0; c < C; ++c)
        Fs[c]=0.0;
    });

  crfftm->fft(W,F);
}

void fftPadHermitian::backwardExplicit(Complex *F, Complex *f, unsigned int,
                                       Complex *W)
{
  if(W == NULL) W=F;

  rcfftm->fft(F,W);
  if(W != f) {
    unsigned int H=ceilquotient(L,2);
    PARALLEL(
      for(unsigned int s=0; s < H; ++s)
        f[s]=W[s];
      );
  }
}

void fftPadHermitian::backwardExplicitMany(Complex *F, Complex *f,
                                           unsigned int, Complex *W)
{
  if(W == NULL) W=F;

  rcfftm->fft(F,W);
  if(W != f) {
    unsigned int H=ceilquotient(L,2);
    PARALLEL(
      for(unsigned int s=0; s < H; ++s) {
        unsigned int Cs=C*s;
        Complex *fs=f+Cs;
        Complex *Fs=W+Cs;
        for(unsigned int c=0; c < C; ++c)
          fs[c]=Fs[c];
      });
  }
}

void fftPadHermitian::forward2(Complex *f, Complex *F, unsigned int r,
                               Complex *W)
{
  if(W == NULL) W=F;

  Complex *fm=f+m;
  unsigned int H=ceilquotient(L,2);
  unsigned int mH1=m-H+1;

  if(r == 0) {
    unsigned int q2=q/2;
    if(2*q2 < q) { // q odd, r=0
      PARALLEL(
        for(unsigned int s=0; s < mH1; ++s)
          W[s]=f[s];
        );
      PARALLEL(
        for(unsigned int s=mH1; s <= e; ++s)
          W[s]=f[s]+conj(*(fm-s));
        );

      crfftm->fft(W,F);
    } else { // q even, r=0,q/2
      Complex *G=F+b;
      Complex *V=W == F ? G : W+B;

      Complex *Zetar=Zetaqm+m*q2;

      V[0]=W[0]=f[0];

      PARALLEL(
        for(unsigned int s=1; s < mH1; ++s) {
          Complex fs=f[s];
          W[s]=fs;
          V[s]=Zetar[s]*fs;
        });

      PARALLEL(
        for(unsigned int s=mH1; s <= e; ++s) {
          Complex fs=f[s];
          Complex fms=conj(*(fm-s));
          W[s]=fs+fms;
          V[s]=Zetar[s]*(fs-fms);
        });

      crfftm->fft(W,F);
      crfftm->fft(V,G);
    }
  } else {
    Complex *G=F+b;
    Complex *V=W == F ? G : W+B;
    V[0]=W[0]=f[0];
    Complex *Zetar=Zetaqm+m*r;
    PARALLEL(
      for(unsigned int s=1; s < mH1; ++s) {
//      W[s]=Zetar[s]*f[s];
//      V[s]=conj(Zetar[s])*f[s];
        Vec Zeta=LOAD(Zetar+s);
        Vec fs=LOAD(f+s);
        Vec A=Zeta*UNPACKL(fs,fs);
        Vec B=ZMULTI(Zeta*UNPACKH(fs,fs));
        STORE(W+s,A+B);
        STORE(V+s,CONJ(A-B));
      });
    Complex *Zetarm=Zetar+m;
    PARALLEL(
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
      });

    crfftm->fft(W,F);
    crfftm->fft(V,G);
  }
}

void fftPadHermitian::forward2Many(Complex *f, Complex *F, unsigned int r,
                                   Complex *W)
{
  if(W == NULL) W=F;

  Complex *fm=f+Cm;
  unsigned int H=ceilquotient(L,2);
  unsigned int mH1=m-H+1;
  if(r == 0) {
    unsigned int q2=q/2;
    if(2*q2 < q) { // q odd
      PARALLEL(
        for(unsigned int s=0; s < mH1; ++s) {
          unsigned int Cs=C*s;
          Complex *Ws=W+Cs;
          Complex *fs=f+Cs;
          for(unsigned int c=0; c < C; ++c)
            Ws[c]=fs[c];
        });
      PARALLEL(
        for(unsigned int s=mH1; s <= e; ++s) {
          unsigned int Cs=C*s;
          Complex *Ws=W+Cs;
          Complex *fs=f+Cs;
          Complex *fms=fm-Cs;
          for(unsigned int c=0; c < C; ++c)
            Ws[c]=fs[c]+conj(fms[c]);
        });

      crfftm->fft(W,F);
    } else { // q even, r=0,q/2
      Complex *G=F+b;
      Complex *V=W == F ? G : W+B;

      Complex *Zetar=Zetaqm+m*q2;

      for(unsigned int c=0; c < C; ++c)
        V[c]=W[c]=f[c];

      PARALLEL(
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
        });

      PARALLEL(
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
        });

      crfftm->fft(W,F);
      crfftm->fft(V,G);
    }
  } else {
    Complex *G=F+b;
    Complex *V=W == F ? G : W+B;
    Complex *Zetar=Zetaqm+m*r;

    for(unsigned int c=0; c < C; ++c)
      V[c]=W[c]=f[c];

    PARALLEL(
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
      });
    Complex *Zetarm=Zetar+m;
    PARALLEL(
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
      });

    crfftm->fft(W,F);
    crfftm->fft(V,G);
  }
}

void fftPadHermitian::backward2(Complex *F, Complex *f, unsigned int r,
                                Complex *W)
{
  if(W == NULL) W=F;

  bool even=m == 2*e;

  rcfftm->fft(F,W);

  Complex *fm=f+m;
  unsigned int me=m-e;
  unsigned int H=ceilquotient(L,2);
  unsigned int mH1=m-H+1;
  if(r == 0) {
    unsigned int q2=q/2;
    if(2*q2 < q) { // q odd, r=0
      PARALLEL(
        for(unsigned int s=0; s < mH1; ++s)
          f[s]=W[s];
        );
      PARALLEL(
        for(unsigned int s=mH1; s < me; ++s) {
          Complex A=W[s];
          f[s]=A;
          *(fm-s)=conj(A);
        });
      if(even)
        f[e]=W[e];

    } else { // q even, r=0,q/2
      Complex *G=F+b;
      Complex *V=W == F ? G : W+B;

      rcfftm->fft(G,V);

      f[0]=W[0]+V[0];

      Complex *Zetar=Zetaqm+m*q2;

      PARALLEL(
        for(unsigned int s=1; s < mH1; ++s)
          f[s]=W[s]+conj(Zetar[s])*V[s];
        );

      PARALLEL(
        for(unsigned int s=mH1; s < me; ++s) {
          Complex A=W[s];
          Complex B=conj(Zetar[s])*V[s];
          f[s]=A+B;
          *(fm-s)=conj(A-B);
        });

      if(even)
        f[e]=W[e]-I*V[e];
    }
  } else {
    Complex *G=F+b;
    Complex *V=W == F ? G : W+B;

    rcfftm->fft(G,V);

    f[0] += W[0]+V[0];

    Complex *Zetar=Zetaqm+m*r;
    PARALLEL(
      for(unsigned int s=1; s < mH1; ++s)
//      f[s] += conj(Zeta)*W[s]+Zeta*V[s];
//      Vec Zeta=LOAD(Zetar+s);
        STORE(f+s,LOAD(f+s)+ZMULT2(LOAD(Zetar+s),LOAD(V+s),LOAD(W+s)));
      );

    Complex *Zetarm=Zetar+m;
    PARALLEL(
      for(unsigned int s=mH1; s < me; ++s) {
//    f[s] += conj(Zeta)*W[s]+Zeta*V[s];
//    *(fm-s) += conj(Zetam*W[s])+Zetam*conj(V[s]);
        Vec Zeta=LOAD(Zetar+s);
        Vec Zetam=LOAD(Zetarm-s);
        Vec Ws=LOAD(W+s);
        Vec Vs=LOAD(V+s);
        STORE(f+s,LOAD(f+s)+ZMULT2(Zeta,Vs,Ws));
        STORE(fm-s,LOAD(fm-s)+CONJ(ZMULT2(Zetam,Ws,Vs)));
      });
    if(even) {
//      f[e] += conj(Zetar[e])*W[e]+Zetar[e]*V[e];
      STORE(f+e,LOAD(f+e)+ZMULT2(LOAD(Zetar+e),LOAD(V+e),LOAD(W+e)));
    }
  }
}

void fftPadHermitian::backward2Many(Complex *F, Complex *f, unsigned int r,
                                    Complex *W)
{
  if(W == NULL) W=F;

  bool even=m == 2*e;

  rcfftm->fft(F,W);

  Complex *fm=f+Cm;
  unsigned int me=m-e;
  unsigned int H=ceilquotient(L,2);
  unsigned int mH1=m-H+1;
  unsigned int Ce=C*e;
  if(r == 0) {
    unsigned int q2=q/2;
    if(2*q2 < q) { // q odd, r=0
      PARALLEL(
        for(unsigned int s=0; s < mH1; ++s) {
          unsigned int Cs=C*s;
          Complex *fs=f+Cs;
          Complex *Ws=W+Cs;
          for(unsigned int c=0; c < C; ++c)
            fs[c]=Ws[c];
        });
      PARALLEL(
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
        });
      if(even) {
        Complex *fe=f+Ce;
        Complex *WCe=W+Ce;
        for(unsigned int c=0; c < C; ++c)
          fe[c]=WCe[c];
      }
    } else { // q even, r=0,q/2
      Complex *G=F+b;
      Complex *V=W == F ? G : W+B;

      rcfftm->fft(G,V);

      for(unsigned int c=0; c < C; ++c)
        f[c]=W[c]+V[c];

      Complex *Zetar=Zetaqm+m*q2;

      PARALLEL(
        for(unsigned int s=1; s < mH1; ++s) {
          unsigned int Cs=C*s;
          Complex *fs=f+Cs;
          Complex *Ws=W+Cs;
          Complex *Vs=V+Cs;
          Complex Zetars=conj(Zetar[s]);
          for(unsigned int c=0; c < C; ++c)
            fs[c]=Ws[c]+Zetars*Vs[c];
        });

      PARALLEL(
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
        });

      if(even) {
        Complex *fe=f+Ce;
        Complex *We=W+Ce;
        Complex *Ve=V+Ce;
        for(unsigned int c=0; c < C; ++c)
          fe[c]=We[c]-I*Ve[c];
      }
    }
  } else {
    Complex *G=F+b;
    Complex *V=W == F ? G : W+B;

    Complex We[C];

    rcfftm->fft(G,V);

    for(unsigned int c=0; c < C; ++c)
      f[c] += W[c]+V[c];

    Complex *Zetar=Zetaqm+m*r;
    PARALLEL(
      for(unsigned int s=1; s < mH1; ++s) {
        unsigned int Cs=C*s;
        Complex *fs=f+Cs;
        Complex *Ws=W+Cs;
        Complex *Vs=V+Cs;
        Vec Zeta=LOAD(Zetar+s);
        for(unsigned int c=0; c < C; ++c)
//        fs[c] += conj(Zetars)*Ws[c]+Zetars*Vs[c];
          STORE(fs+c,LOAD(fs+c)+ZMULT2(Zeta,LOAD(Vs+c),LOAD(Ws+c)));
      });

    Complex *Zetarm=Zetar+m;
    PARALLEL(
      for(unsigned int s=mH1; s < me; ++s) {
        unsigned int Cs=C*s;
        Complex *fs=f+Cs;
        Complex *fms=fm-Cs;
        Complex *Ws=W+Cs;
        Complex *Vs=V+Cs;
        Vec Zeta=LOAD(Zetar+s);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(Zeta,-Zeta);
        Vec Zetam=LOAD(Zetarm-s);
        Vec Xm=UNPACKL(Zetam,-Zetam);
        Vec Ym=UNPACKH(Zetam,Zetam);
        for(unsigned int c=0; c < C; ++c) {
//        fs[c] += conj(Zeta)*Ws[c]+Zeta*Vs[c];
//        fms[c] += conj(Zetam*Ws[c])+Zetam*conj(Vs[c]);
          Vec Wsc=LOAD(Ws+c);
          Vec Vsc=LOAD(Vs+c);
          STORE(fs+c,LOAD(fs+c)+ZMULT2(X,Y,Vsc,Wsc));
          STORE(fms+c,LOAD(fms+c)+ZMULT2(Xm,Ym,Wsc,Vsc));
        }
      });
    if(even) {
      Complex *fe=f+Ce;
      Complex *We=W+Ce;
      Complex *Ve=V+Ce;
      Vec Zeta=LOAD(Zetar+e);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(Zeta,-Zeta);
      for(unsigned int c=0; c < C; ++c)
//        fe[c] += conj(Zetare)*We[c]+Zetare*Ve[c];
        STORE(fe+c,LOAD(fe+c)+ZMULT2(X,Y,LOAD(Ve+c),LOAD(We+c)));
    }
  }
}

void fftPadHermitian::forwardInner(Complex *f, Complex *F, unsigned int r,
                                   Complex *W)
{
  if(W == NULL) W=F;

  unsigned int p2=p/2;
  unsigned int H=L/2;
  unsigned int e1=e+1;
  unsigned int p2m=p2*m;
  unsigned int p2mH=p2m+H-L;
  unsigned int m0=min(p2mH+1,e1);
  Complex *fm=f+p2m;
  if(r == 0) {
    unsigned int n2=n/2;
    if(2*n2 < n) { // n odd, r=0
      W[0]=f[0];
      PARALLEL(
        for(unsigned int s=1; s < m0; ++s)
          W[s]=f[s];
        );
      PARALLEL(
        for(unsigned int s=m0; s < e1; ++s)
          W[s]=conj(*(fm-s))+f[s];
        );
      for(unsigned int t=1; t < p2; ++t) {
        unsigned int tm=t*m;
        unsigned int te1=t*e1;
        Complex *fmt=fm-tm;
        Complex *ft=f+tm;
        Complex *Wt=W+te1;
        PARALLEL(
          for(unsigned int s=0; s < e1; ++s)
            Wt[s]=conj(*(fmt-s))+ft[s];
          );
      }

      fftp->fft(W);

      unsigned int mn=m*n;
      for(unsigned int u=1; u < p2; ++u) {
        Complex *Wu=W+u*e1;
        Complex *Zeta0=Zetaqm+mn*u;
        PARALLEL(
          for(unsigned int s=1; s < e1; ++s)
            Wu[s] *= Zeta0[s];
          );
      }
      crfftm->fft(W,F);
    } else { // n even, r=0,n/2
      Complex *V=W+B;
      V[0]=W[0]=f[0];
      PARALLEL(
        for(unsigned int s=1; s < m0; ++s)
          V[s]=W[s]=f[s];
        );
      PARALLEL(
        for(unsigned int s=m0; s < e1; ++s) {
          Complex fms=conj(*(fm-s));
          Complex fs=f[s];
          W[s]=fms+fs;
          V[s]=-fms+fs;
        });
      Complex *Zetaqn2=Zetaqp+p2*n2;
      for(unsigned int t=1; t < p2; ++t) {
        unsigned int tm=t*m;
        unsigned int te1=t*e1;
        Complex *Wt=W+te1;
        Complex *Vt=V+te1;
        Complex *fmt=fm-tm;
        Complex *ft=f+tm;
        Vec Zeta=LOAD(Zetaqn2+t); //*zeta_q^{tn/2}
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(Zeta,-Zeta);
        PARALLEL(
          for(unsigned int s=0; s < e1; ++s) {
            Vec fmts=CONJ(LOAD(fmt-s));
            Vec fts=LOAD(ft+s);
            STORE(Wt+s,fmts+fts);
            STORE(Vt+s,ZMULT(X,Y,fts-fmts));
          });
      }

      fftp->fft(W);
      fftp->fft(V);

      unsigned int mn=m*n;
      unsigned int mn2=m*n2;
      Complex *Zetan2=Zetaqm+mn2;
      PARALLEL(
        for(unsigned int s=0; s < e1; ++s)
          V[s] *= Zetan2[s];
        );
      for(unsigned int u=1; u < p2; ++u) {
        unsigned int ue1=u*e1;
        Complex *Wu=W+ue1;
        Complex *Zeta0=Zetaqm+mn*u;
        PARALLEL(
          for(unsigned int s=1; s < e1; ++s)
            Wu[s] *= Zeta0[s];
          );
        Complex *Vu=V+ue1;
        Complex *Zetan2=Zeta0+mn2;
        PARALLEL(
          for(unsigned int s=1; s < e1; ++s)
            Vu[s] *= Zetan2[s];
          );
      }

      crfftm->fft(W,F);
      crfftm->fft(V,F+b);
    }
  } else {
    Vec Zetanr=CONJ(LOAD(Zetaqp+p2*r+p2)); // zeta_n^-r
    Complex *G=F+b;
    Complex *V=W == F ? G : W+B;
    V[0]=W[0]=f[0];
    PARALLEL(
      for(unsigned int s=1; s < m0; ++s)
        V[s]=W[s]=f[s];
      );
    PARALLEL(
      for(unsigned int s=m0; s < e1; ++s) {
        Vec fms=CONJ(LOAD(fm-s));
        Vec fs=LOAD(f+s);
        Vec A=Zetanr*UNPACKL(fms,fms);
        Vec B=ZMULTI(Zetanr*UNPACKH(fms,fms));
        STORE(W+s,A+B+fs);
        STORE(V+s,CONJ(A-B)+fs);
      });
    Complex *Zetaqr=Zetaqp+p2*r;
    for(unsigned int t=1; t < p2; ++t) {
      unsigned int tm=t*m;
      unsigned int te1=t*e1;
      Complex *Wt=W+te1;
      Complex *Vt=V+te1;
      Complex *fmt=fm-tm;
      Complex *ft=f+tm;
      Vec Zeta=LOAD(Zetaqr+t); // zeta_q^{tr}
      Vec Zetam=ZMULT(Zeta,Zetanr);
      PARALLEL(
        for(unsigned int s=0; s < e1; ++s) {
          Vec fmts=CONJ(LOAD(fmt-s));
          Vec fts=LOAD(ft+s);
          Vec A=Zetam*UNPACKL(fmts,fmts)+Zeta*UNPACKL(fts,fts);
          Vec B=ZMULTI(Zetam*UNPACKH(fmts,fmts)+Zeta*UNPACKH(fts,fts));
          STORE(Wt+s,A+B);
          STORE(Vt+s,CONJ(A-B));
        });
    }

    fftp->fft(W);
    fftp->fft(V);

    unsigned int mr=m*r;
    Complex *Zetar=Zetaqm+mr;
    PARALLEL(
      for(unsigned int s=1; s < e1; ++s) {
        Complex Zeta=Zetar[s];
        W[s] *= Zeta;
        V[s] *= conj(Zeta);
      });

    for(unsigned int u=1; u < p2; ++u) {
      unsigned int mu=m*u;
      unsigned int e1u=e1*u;
      Complex *Zetar0=Zetaqm+n*mu;
      Complex *Zetar1=Zetar0+mr;
      Complex *Wu=W+e1u;
      PARALLEL(
        for(unsigned int s=1; s < e1; ++s)
          Wu[s] *= Zetar1[s];
        );
      Complex *Zetar2=Zetar0-mr;
      Complex *Vu=V+e1u;
      PARALLEL(
        for(unsigned int s=1; s < e1; ++s)
          Vu[s] *= Zetar2[s];
        );
    }
    crfftm->fft(W,F);
    crfftm->fft(V,G);
  }
}

void fftPadHermitian::backwardInner(Complex *F, Complex *f, unsigned int r,
                                    Complex *W)
{
  if(W == NULL) W=F;

  unsigned int p2=p/2;
  unsigned int H=L/2;
  unsigned int e1=e+1;
  unsigned int p2m=p2*m;
  unsigned int p2mH=p2m+H-L;
  unsigned int S=m-e;
  unsigned int m0=min(p2mH+1,e1);
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
        PARALLEL(
          for(unsigned int s=1; s < e1; ++s)
            Wu[s] *= conj(Zeta0[s]);
          );
      }

      ifftp->fft(W);

      PARALLEL(
        for(unsigned int s=0; s < m0; ++s)
          f[s]=W[s];
        );

      PARALLEL(
        for(unsigned int s=m0; s < S; ++s) {
          Complex Ws=W[s];
          f[s]=Ws;
          *(fm-s)=conj(Ws);
        });

      for(unsigned int t=1; t < p2; ++t) {
        unsigned int tm=t*m;
        unsigned int te1=t*e1;
        Complex *Wt=W+te1;
        Complex *ft=f+tm;
        // s=0 case (this is important to avoid overlap)
        ft[0]=Wt[0];
        Complex *fmt=fm-tm;
        PARALLEL(
          for(unsigned int s=1; s < S; ++s) {
            Complex Wts=Wt[s];
            ft[s]=Wts;
            *(fmt-s)=conj(Wts);
          });
      }

      if(S < e1) {
        f[e]=W[e];
        for(unsigned int t=1; t < p2; ++t)
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
      PARALLEL(
        for(unsigned int s=1; s < e1; ++s)
          V[s] *= conj(Zetan2[s]);
        );

      for(unsigned int u=1; u < p2; ++u) {
        unsigned int e1u=u*e1;
        Complex *Wu=W+e1u;
        Complex *Zeta0=Zetaqm+mn*u;
        PARALLEL(
          for(unsigned int s=1; s < e1; ++s)
            Wu[s] *= conj(Zeta0[s]);
          );
        Complex *Vu=V+e1u;
        Complex *Zetan2=Zeta0+mn2;
        PARALLEL(
          for(unsigned int s=1; s < e1; ++s)
            Vu[s] *= conj(Zetan2[s]);
          );
      }
      ifftp->fft(W);
      ifftp->fft(V);

      f[0]=W[0]+V[0];
      PARALLEL(
        for(unsigned int s=1; s < m0; ++s)
          f[s]=W[s]+V[s];
        );

      PARALLEL(
        for(unsigned int s=m0; s < S; ++s) {
          Complex Wts=W[s];
          Complex Vts=V[s];
          *(fm-s)=conj(Wts-Vts);
          f[s]=Wts+Vts;
        });

      Complex *Zetaqn2=Zetaqp+p2*n2;
      for(unsigned int t=1; t < p2; ++t) {
        unsigned int tm=t*m;
        unsigned int te1=t*e1;
        Complex *Wt=W+te1;
        Complex *Vt=V+te1;
        Complex *ft=f+tm;
        Vec Zeta=LOAD(Zetaqn2+t);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(-Zeta,Zeta);
        STORE(ft,LOAD(Wt)+ZMULT(X,Y,LOAD(Vt)));
        Complex *fmt=fm-tm;
        PARALLEL(
          for(unsigned int s=1; s < S; ++s) {
            Vec Wts=LOAD(Wt+s);
            Vec Vts=ZMULT(X,Y,LOAD(Vt+s));
            STORE(fmt-s,CONJ(Wts-Vts));
            STORE(ft+s,Wts+Vts);
          });
      }

      if(S < e1) {
        f[e]=W[e]+V[e];
        for(unsigned int t=1; t < p2; ++t) {
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
    PARALLEL(
      for(unsigned int s=1; s < e1; ++s) {
        Complex Zeta=Zetar[s];
        W[s] *= conj(Zeta);
        V[s] *= Zeta;
      });
    for(unsigned int u=1; u < p2; ++u) {
      unsigned int mu=m*u;
      unsigned int e1u=u*e1;
      Complex *Zetar0=Zetaqm+n*mu;
      Complex *Zetar1=Zetar0+mr;
      Complex *Wu=W+e1u;
      PARALLEL(
        for(unsigned int s=1; s < e1; ++s)
          Wu[s] *= conj(Zetar1[s]);
        );
      Complex *Zetar2=Zetar0-mr;
      Complex *Vu=V+e1u;
      PARALLEL(
        for(unsigned int s=1; s < e1; ++s)
          Vu[s] *= conj(Zetar2[s]);
        );
    }

    ifftp->fft(W);
    ifftp->fft(V);

    f[0] += W[0]+V[0];
    PARALLEL(
      for(unsigned int s=1; s < m0; ++s)
        f[s] += W[s]+V[s];
      );

    Vec Xm=UNPACKL(Zetanr,Zetanr);
    Vec Ym=UNPACKH(Zetanr,-Zetanr);
    PARALLEL(
      for(unsigned int s=m0; s < S; ++s) {
        Vec Wts=LOAD(W+s);
        Vec Vts=LOAD(V+s);
        STORE(fm-s,LOAD(fm-s)+CONJ(ZMULT2(Xm,Ym,Wts,Vts)));
        STORE(f+s,LOAD(f+s)+Wts+Vts);
      });

    Complex *Zetaqr=Zetaqp+p2*r;
    for(unsigned int t=1; t < p2; ++t) {
      unsigned int tm=t*m;
      unsigned int te1=t*e1;
      Complex *Wt=W+te1;
      Complex *Vt=V+te1;
      Complex *ft=f+tm;
      Vec Zeta=LOAD(Zetaqr+t);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(Zeta,-Zeta);
      STORE(ft,LOAD(ft)+ZMULT2(X,Y,LOAD(Vt),LOAD(Wt)));
      Complex *fmt=fm-tm;
      Vec Zeta2=ZMULT(X,-Y,Zetanr); // zeta_q^(-rt)z_n^r
      Vec Xm=UNPACKL(Zeta2,Zeta2);
      Vec Ym=UNPACKH(Zeta2,-Zeta2);
      PARALLEL(
        for(unsigned int s=1; s < S; ++s) {
          Vec Wts=LOAD(Wt+s);
          Vec Vts=LOAD(Vt+s);
          STORE(fmt-s,LOAD(fmt-s)+CONJ(ZMULT2(Xm,Ym,Wts,Vts)));
          STORE(ft+s,LOAD(ft+s)+ZMULT2(X,Y,Vts,Wts));
        });
    }

    if(S < e1) {
      f[e]+=W[e]+V[e];
      for(unsigned int t=1; t < p2; ++t) {
        unsigned int te1e=t*e1+e;
        unsigned int tme=t*m+e;
        Vec Zeta=LOAD(Zetaqr+t);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(Zeta,-Zeta);
        STORE(f+tme,LOAD(f+tme)+ZMULT2(X,Y,LOAD(V+te1e),LOAD(W+te1e)));
      }
    }
  }
}

Convolution::~Convolution()
{
  if(allocateW)
    deleteAlign(W);

  if(q > 1) {

    if(loop2)
      delete [] Fp;

    if(allocateV) {
      for(unsigned int i=0; i < B; ++i)
        deleteAlign(V[i]);
    }
    if(V)
      delete [] V;
  }

  if(allocateF) {
    utils::deleteAlign(F[0]);
    delete [] F;
  }

  if(allocate)
    delete fft;
}

// f is an array of max(A,B) pointers to distinct data blocks
// each of size fft->length()
// offset is applied to each input and output component
void Convolution::convolveRaw(Complex **f, unsigned int offset,
                              Indices *indices2)
{
  Complex **g;
  Complex *G[A];

  for(unsigned int t=0; t < threads; ++t)
    indices.copy(indices2,0);
  indices.fft=fft;

  if(offset) {
    g=G;
    for(unsigned int a=0; a < A; ++a)
      g[a]=f[a]+offset;
  } else g=f;

  if(q == 1) {
    forward(g,F,0,0,A);
    (*mult)(F,blocksize,&indices,threads);
    backward(F,g,0,0,B,W);
  } else {
    if(overwrite) {
      forward(g,F,0,0,A);
      unsigned int final=fft->n-1;
      for(unsigned int r=0; r < final; ++r) {
        Complex *h[A];
        for(unsigned int a=0; a < A; ++a)
          h[a]=g[a]+r*blocksize;
        indices.r=r;
        (*mult)(h,blocksize,&indices,threads);
      }
      indices.r=final;
      (*mult)(F,blocksize,&indices,threads);
      backward(F,g,0,0,B);
    } else {
      if(loop2) {
        forward(g,F,0,0,A);
        operate(F,0,&indices);
        unsigned int C=A-B;
        unsigned int a=0;
        for(; a+C <= B; a += C) {
          forward(g,Fp,r,a,a+C);
          backwardPad(F,g,0,a,a+C,W0);
        }
        forward(g,Fp,r,a,A);
        operate(Fp,r,&indices);
        backwardPad(Fp,g,r,0,B,W0);
      } else {
        Complex **h0;
        if(nloops > 1) {
          if(!V) initV();
          h0=V;
        } else
          h0=g;

        for(unsigned int r=0; r < R; r += fft->increment(r)) {
          forward(g,F,r,0,A);
          operate(F,r,&indices);
          backwardPad(F,h0,r,0,B,W0);
        }

        if(nloops > 1) {
          for(unsigned int b=0; b < B; ++b) {
            Complex *gb=g[b];
            Complex *hb=h0[b];
            for(unsigned int i=0; i < inputSize; ++i)
              gb[i]=hb[i];
          }
        }
      }
    }
  }
}

}
