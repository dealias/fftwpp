#include "convolve.h"
#include "align.h"
#include "cmult-sse2.h"

using namespace std;
using namespace utils;
using namespace Array;

namespace fftwpp {

bool showOptTimes=false;
bool showRoutines=false;

#ifdef __SSE2__
const union uvec sse2_pm={
  {0x00000000,0x00000000,0x00000000,0x80000000}
};

const union uvec sse2_mm={
  {0x00000000,0x80000000,0x00000000,0x80000000}
};
#endif

const double twopi=2.0*M_PI;

void multNone(Complex **F, size_t n, Indices *indices,
              size_t threads)
{
}

// This multiplication routine is for binary convolutions and takes
// two Complex inputs of size n and outputs one Complex value.
void multBinary(Complex **F, size_t n, Indices *indices,
                size_t threads)
{
  Complex *F0=F[0];
  Complex *F1=F[1];

#if 0 // Transformed indices are available. Requires overwrite=false.
  size_t N=indices->size;
  fftBase *fft=indices->fft;
  size_t r=indices->r;
  for(size_t j=0; j < n; ++j) {
    for(size_t d=0; d < N; ++d)
      cout << indices->index[N-1-d] << ",";
    cout << fft->index(r,j) << endl;
  }
#endif

  PARALLELIF(
    n > threshold,
    for(size_t j=0; j < n; ++j)
      F0[j] *= F1[j];
    );
}

// This multiplication routine is for binary convolutions and takes
// two real inputs of size n.
void realMultBinary(Complex **F, size_t n, Indices *indices,
                    size_t threads)
{
  double *F0=(double *) F[0];
  double *F1=(double *) F[1];

#if 0 // Transformed indices are available, if needed.
  size_t N=indices->size;
  fftBase *fft=indices->fft;
  size_t r=indices->r;
  for(size_t j=0; j < n; ++j) {
    for(size_t d=0; d < N; ++d)
      cout << indices->index[N-1-d] << ",";
    cout << fft->index(r,j) << endl;
  }
#endif

  PARALLELIF(
    n > 2*threshold,
    for(size_t j=0; j < n; ++j)
      F0[j] *= F1[j];
    );
}

// This multiplication routine is for binary correlations and takes
// two Complex inputs of size n and outputs one Complex value.
void multcorrelation(Complex **F, size_t n, Indices *indices,
                     size_t threads)
{
  Complex *F0=F[0];
  Complex *F1=F[1];

#if 0 // Transformed indices are available, if needed.
  size_t N=indices->size;
  fftBase *fft=indices->fft;
  size_t r=indices->r;
  for(size_t j=0; j < n; ++j) {
    for(size_t d=0; d < N; ++d)
      cout << indices->index[N-1-d] << ",";
    cout << fft->index(r,j) << endl;
  }
#endif

  PARALLELIF(
    n > threshold,
    for(size_t j=0; j < n; ++j)
      F0[j] *= conj(F1[j]);
    );
}

// Returns the smallest natural number greater than a positive number
// m of the form 2^a 3^b 5^c 7^d for some nonnegative integers a, b, c, and d.
size_t nextfftsize(size_t m)
{
  size_t N=ceilpow2(m);
  if(m == N)
    return m;
  for(size_t ni=1; ni < N; ni *= 7)
    for(size_t nj=ni; nj < N; nj *= 5)
      for(size_t nk=nj; nk < N; nk *= 3)
        N=min(N,nk*ceilpow2(ceilquotient(m,nk)));
  return N;
}

void fftBase::initZetaqm(size_t q, size_t m)
{
  double twopibyM=twopi/M;
  Zetaqm0=ComplexAlign((q-1)*m);
  Zetaqm=Zetaqm0-m;
  for(size_t r=1; r < q; ++r) {
    size_t mr=m*r;
    Zetaqm[mr]=1.0;
    PARALLELIF(
      m > threshold,
      for(size_t s=1; s < m; ++s)
        Zetaqm[mr+s]=expi(r*s*twopibyM);
      );
  }
}

// Returns the smallest natural number greater than m that is a
// power of 2, 3, 5, or 7.
size_t nextpuresize(size_t m)
{
  size_t M=ceilpow2(m);
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

// Returns true iff m is a power of 2 (assumes m > 0).
bool ispow2(size_t m) {
  return (m & (m-1)) == 0;
}

// Returns true iff m is a power of 2, 3, 5, or 7 (assumes m > 0).
bool ispure(size_t m)
{
  if(ispow2(m))
    return true;
  if(m == ceilpow(3,m))
    return true;
  if(m == ceilpow(5,m))
    return true;
  if(m == ceilpow(7,m))
    return true;
  return false;
}

double time(fftBase *fft, double &threshold)
{
  size_t threads=fft->app.threads == 1 ? fft->app.maxthreads : 1;

  size_t N=max(fft->app.A,fft->app.B);
  size_t doubles=fft->doubles();
  Complex **f=(Complex **) doubleAlign(N*threads,doubles);

  // Initialize entire array to 0 to avoid overflow when timing.
  for(size_t t=0; t < threads; ++t) {
    for(size_t a=0; a < fft->app.A; ++a) {
      double *fa=(double *) (f[N*t+a]);
      for(size_t j=0; j < doubles; ++j)
        fa[j]=0.0;
    }
  }

  Convolution **Convolve=new Convolution*[threads];
  for(size_t t=0; t < threads; ++t)
    Convolve[t]=new Convolution(fft);

  statistics Stats(true);
  statistics medianStats(false);
  double eps=0.02;

  do {
    if(threads > 1) {
      cpuTimer C;
#pragma omp parallel for num_threads(threads)
      for(size_t t=0; t < threads; ++t)
        Convolve[t]->convolveRaw(f+N*t);
      Stats.add(C.nanoseconds());
    } else {
      Convolution *Convolve0=Convolve[0];
      cpuTimer C;
      Convolve0->convolveRaw(f);
      Stats.add(C.nanoseconds());
    }
    if(Stats.min() >= 2.0*threshold) break;
    if(Stats.count() >= 4 && Stats.min() >= threshold) break;
    medianStats.add(Stats.median());
  } while(Stats.count() < 5 || medianStats.stderror() > eps*medianStats.mean());

  for(size_t t=0; t < threads; ++t)
    delete Convolve[t];
  delete [] Convolve;

  threshold=min(threshold,Stats.max());
  deleteAlign(f[0]);
  delete [] f;
  return Stats.median();
}

double timePad(fftBase *fft, double& threshold)
{
  return time(fft,threshold);
}

void fftBase::OptBase::optloop(size_t& m, size_t L,
                               size_t M, Application& app,
                               size_t C, size_t S,
                               bool centered, size_t itmax,
                               bool useTimer, bool Explicit,
                               size_t (*nextInnerSize)(size_t))
{

  bool inner=ceilquotient(L,m) > 2;
  size_t i=(inner ? m : 0);
  // If inner == true, i is an m value and itmax is the largest m value that
  // we consider. If inner == false, i is a counter starting at zero, and
  // itmax is maximum number of m values we consider before exiting optloop.

  while(i < itmax) {
    size_t p,n,q;
    parameters(L,M,m,centered,p,n,q);
    if(!Explicit && mForced && m < M && centered && p%2 != 0) {
      cerr << "Odd values of p are incompatible with the centered and Hermitian routines." << endl;
      cerr << "Using explicit routines with m=" << M << ", D=1, and I=0 instead." << endl;
    }

    // In the inner loop we must have the following:
    // p must be a power of 2, 3, 5, or 7.
    // p must be even in the centered case.
    // p != q.
    if(inner && (((!ispure(p) || p == q) && !mForced) || (centered && p%2 != 0)))
      i=m=nextInnerSize(m+1);
    else {
      size_t Dstart=DForced ? app.D : 1;
      size_t Dstop=DForced ? app.D : maxD(n);
      size_t Dstop2=2*Dstop;

      // Check inplace and out-of-place unless C > 1.
      size_t Istart=app.I == -1 ? C > 1 : app.I;

      size_t Istop=app.I == -1 ? 2 : app.I+1;

      for(size_t D=Dstart; D < Dstop2; D *= 2) {
        if(D > Dstop) D=Dstop;
        for(size_t inplace=Istart; inplace < Istop; ++inplace)
          check(L,M,C,S,m,p,q,n,D,inplace,app,useTimer);
      }
      if(mForced) break;
      if(inner) {
        m=nextInnerSize(m+1);
        i=m;
      } else {
        if(ispure(m) and i < itmax) {
          m=nextfftsize(m+1);
          break;
        }
        i++;
        if(i < itmax) {
          m=nextfftsize(m+1);
        }
      }
    }
  }
}

void fftBase::OptBase::opt(size_t L, size_t M, Application& app,
                           size_t C, size_t S,
                           size_t minsize, size_t itmax,
                           bool Explicit, bool centered, bool useTimer)
{
  if(!Explicit) {
    size_t (*nextInnerSize)(size_t);
    if(ispow2(L)) {
      nextInnerSize =&ceilpow2;
    } else {
      nextInnerSize =&nextpuresize;
    }
    size_t H=ceilquotient(L,2);
    if(mForced) {
      if(app.m >= H)
        optloop(app.m,L,M,app,C,S,centered,1,useTimer,false);
      else
        optloop(app.m,L,M,app,C,S,centered,app.m+1,useTimer,false,nextInnerSize);
    } else {
      size_t m=nextInnerSize(minsize);
      optloop(m,L,M,app,C,S,centered,H,useTimer,false,nextInnerSize);

      m=nextfftsize(H);

      optloop(m,L,M,app,C,S,centered,itmax,useTimer,false);

      size_t Mhalf=ceilquotient(M,2);
      if(L > Mhalf) {
        m=nextfftsize(max(Mhalf,m+1));
        optloop(m,L,M,app,C,S,centered,itmax,useTimer,false);

        m=nextfftsize(max(L,m+1));
        optloop(m,L,M,app,C,S,centered,itmax,useTimer,false);
      } else {
        m=nextfftsize(max(L,m+1));
        optloop(m,L,M,app,C,S,centered,itmax,useTimer,false);
        if(M > 16 && m < 16) {
          m=16;
          optloop(m,L,M,app,C,S,centered,1,useTimer,false);
        }
        if(M > 32 && m < 32) {
          m=32;
          optloop(m,L,M,app,C,S,centered,1,useTimer,false);
        }
      }
      m=nextfftsize(max(M,m));
      optloop(m,L,M,app,C,S,centered,itmax,useTimer,false);

      // Check next explict power of 2 (only when C==1)
      size_t ceilpow2M=ceilpow2(M);
      if(ceilpow2M > m && C == 1)
        optloop(ceilpow2M,L,M,app,C,S,centered,1,useTimer,false);
    }
  } else {
    size_t m=nextfftsize(M);
    optloop(m,L,m,app,C,S,centered,itmax,useTimer,true);

    size_t ceilpow2M=ceilpow2(M);
    if(ceilpow2M > m && C == 1)
      optloop(ceilpow2M,L,M,app,C,S,centered,1,useTimer,true);
  }
}

void fftBase::OptBase::check(size_t L, size_t M,
                             size_t C, size_t S, size_t m,
                             size_t p, size_t q, size_t n, size_t D,
                             bool inplace, Application& app, bool useTimer)
{
//  cout << "m=" << m << ", p=" << p << ", q=" << q << ", n=" << n << ", D=" << D << ", I=" << inplace << ", C=" << C << ", S=" << S << endl;
  //cout << "valid=" << valid(m,p,q,n,D,S) << endl << endl;
  if(valid(m,p,q,n,D,S)) {
    if(useTimer) {
      double t=time(L,M,app,C,S,m,D,inplace);
      if(showOptTimes)
        cout << "m=" << m << ", p=" << p << ", q=" << q << ", C=" << C << ", S=" << S << ", D=" << D << ", I=" << inplace << ": t=" << t*1.0e-9 << endl;
      if(t < T) {
        this->m=m;
        this->D=D;
        this->inplace=inplace;
        T=t;
      }
    } else {
      counter += 1;
      if(m != mlist.back())
        mlist.push_back(m);
      if(counter == 1) {
        this->m=m;
        this->D=D;
        this->inplace=inplace;
      }
    }
  }
}

void fftBase::parameters(size_t L, size_t M, size_t m, bool centered,
                         size_t &p, size_t& n, size_t& q)
{
  p=ceilquotient(L,m);
  size_t P=(centered && p == 2*(p/2)) || p == 2 ? (p/2) : p; // effective p
  n=ceilquotient(M,P*m);
  q=P*n;
}

void fftBase::OptBase::scan(size_t L, size_t M, Application& app,
                            size_t C, size_t S, bool Explicit,
                            bool centered)
{
  m=M;
  D=1;
  inplace=false;
  T=DBL_MAX;
  threshold=T;
  mForced=app.m >= 1;
  DForced=app.D > 0;
  counter=0;
  bool sR=showRoutines;

  showRoutines=false;
  size_t mStart=2;
  size_t itmax=3;
  opt(L,M,app,C,S,mStart,itmax,Explicit,centered,false);

  if(counter == 0) {
    cerr << "Optimizer found no valid cases with specified parameters." << endl;
    cerr << "Using explicit routines with m=" << M << ", D=1, and I=0 instead." << endl << endl;
  } else if(counter > 1) {
    if(showOptTimes) cout << endl << "Timing " << counter << " algorithms:" << endl;
    mForced=true;
    Application App(app);
    for(mList::reverse_iterator r=mlist.rbegin(); r != mlist.rend(); ++r) {
      App.m=*r;
      opt(L,M,App,C,S,mStart,itmax,Explicit,centered);
    }
    if(showOptTimes) {
      cout << endl << "Optimal time: t=" << T*1.0e-9 << endl << endl;
    }
  }

  size_t p,q,n;
  parameters(L,M,m,centered,p,n,q);
  showRoutines=sR;

  if(app.verbose) {
    size_t mpL=m*p-L;
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
    cout << "threads=" << app.threads << endl;
    cout << endl;
    cout << "Padding: " << mpL << endl;
  }
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
  parameters(L,M,m,centered,p,n,q);

  if(q*m < M) {
    cerr << "Invalid parameters: " << endl
         << " q=" << q << " m=" << m << " M=" << M << endl;
    exit(-1);
  }
  Cm=C*(size_t) m;
  Sm=S*(size_t) m;
  M=m*q;
  Pad=&fftBase::padNone;
  Zetaqp=NULL;
  ZetaqmS=NULL;
  overwrite=false;
}

void fftPad::init()
{
  common();

  char const *FR;
  char const *BR;

  Complex *G=ComplexAlign(Sm);
  Complex *H=inplace ? G : ComplexAlign(Sm);

  if(q == 1) {
    if(S == 1) {
      Forward=&fftBase::forwardExplicit;
      Backward=&fftBase::backwardExplicit;
      FR="forwardExplicit";
      BR="backwardExplicit";
      fftm1=new fft1d(m,1,G,H,threads);
      ifftm1=new fft1d(m,-1,H,G,threads);
    } else {
      Forward=&fftBase::forwardExplicitMany;
      Backward=&fftBase::backwardExplicitMany;
      FR="forwardExplicitMany";
      BR="backwardExplicitMany";
      fftm=new mfft1d(m,1,C, S,1, G,H,threads);
      ifftm=new mfft1d(m,-1,C, S,1, H,G,threads);

      if(fftm->Threads() > 1)
        threads=fftm->Threads();
    }

    deleteAlign(G);
    if(!inplace)
      deleteAlign(H);
    dr=D=D0=R=1;
    l=M;
    b=S*l;
  } else {
    double twopibyN=twopi/M;
    double twopibyq=twopi/q;

    size_t P,P1;

    if(p == 2) {
      P1=P=1;
    } else if(centered) {
      P=p/2;
      P1=P+1;
    } else
      P1=P=p;

    l=m*P;
    b=S*l;
    size_t d=C*D*P;

    Complex *G,*H;
    size_t size=b*D;

    G=ComplexAlign(size);
    H=inplace ? G : ComplexAlign(size);

    overwrite=inplace && L == p*m && n == (centered ? 3 : p+1) && D == 1 &&
      app.A >= app.B && app.overwrite;

    if(S == 1) {
      fftm=new mfft1d(m,1,d, 1,m, G,H,threads);
      ifftm=new mfft1d(m,-1,d, 1,m, H,G,threads);
    } else {
      fftm=new mfft1d(m,1,C, S,1, G,H,threads);
      ifftm=new mfft1d(m,-1,C, S,1, H,G,threads);
    }

    if(fftm->Threads() > 1)
      threads=fftm->Threads();

    if(p > 2) { // Implies L > 2m
      if(!centered) overwrite=false;
      if(S == 1) {
        if(overwrite) {
          Forward=&fftBase::forwardInnerAll;
          Backward=&fftBase::backwardInnerAll;
          FR="forwardInnerAll";
          BR="backwardInnerAll";
        } else {
          Forward=&fftBase::forwardInner;
          Backward=&fftBase::backwardInner;
          FR="forwardInner";
          BR="backwardInner";
        }
      } else {
        if(overwrite) {
          Forward=&fftBase::forwardInnerManyAll;
          Backward=&fftBase::backwardInnerManyAll;
          FR="forwardInnerManyAll";
          BR="backwardInnerManyAll";
        } else {
          Forward=&fftBase::forwardInnerMany;
          Backward=&fftBase::backwardInnerMany;
          FR="forwardInnerMany";
          BR="backwardInnerMany";
        }
      }

      Zetaqp0=ComplexAlign((n-1)*(P1-1));
      Zetaqp=Zetaqp0-P1;
      for(size_t r=1; r < n; ++r)
        for(size_t t=1; t < P1; ++t)
          Zetaqp[(P1-1)*r+t]=expi(r*t*twopibyq);

      // L'=p, M'=q, m'=p, p'=1, q'=n
      if(S == C) {
        fftp=new mfft1d(P,1,Cm, Cm,1, G,G,threads);
        ifftp=new mfft1d(P,-1,Cm, Cm,1, G,G,threads);
      } else {
        fftp=new mfft1d(P,1,C, Sm,1, G,G,threads);
        ifftp=new mfft1d(P,-1,C, Sm,1, G,G,threads);
      }
    } else {
      if(p == 2) {
        if(!centered) {
          size_t Lm=L-m;
          ZetaqmS0=ComplexAlign((q-1)*Lm);
          ZetaqmS=ZetaqmS0-L;
          for(size_t r=1; r < q; ++r)
            for(size_t s=m; s < L; ++s)
              ZetaqmS[Lm*r+s]=expi(r*s*twopibyN);
        }

        if(S == 1) {
          if(overwrite) {
            Forward=&fftBase::forward2All;
            Backward=&fftBase::backward2All;
            FR="forward2All";
            BR="backward2All";
          } else {
            Forward=&fftBase::forward2;
            Backward=&fftBase::backward2;
            FR="forward2";
            BR="backward2";
          }
        } else {
          if(overwrite) {
            Forward=&fftBase::forward2ManyAll;
            Backward=&fftBase::backward2ManyAll;
            FR="forward2ManyAll";
            BR="backward2ManyAll";
          } else {
            Forward=&fftBase::forward2Many;
            Backward=&fftBase::backward2Many;
            FR="forward2Many";
            BR="backward2Many";
          }
        }
      } else { // p == 1
        if(S == 1) {
          if(overwrite) {
            Forward=&fftBase::forward1All;
            Backward=&fftBase::backward1All;
            FR="forward1All";
            BR="backward1All";
          } else {
            Forward=&fftBase::forward1;
            Backward=&fftBase::backward1;
            FR="forward1";
            BR="backward1";
          }
          if(repad())
            Pad=&fftBase::padSingle;
        } else {
          if(overwrite) {
            Forward=&fftBase::forward1ManyAll;
            Backward=&fftBase::backward1ManyAll;
            FR="forward1ManyAll";
            BR="backward1ManyAll";
          } else {
            Forward=&fftBase::forward1Many;
            Backward=&fftBase::backward1Many;
            FR="forward1Many";
            BR="backward1Many";
          }
          if(repad())
            Pad=&fftBase::padMany;
        }
      }
    }

    dr=Dr();
    R=residueBlocks();
    D0=n % D;
    if(D0 == 0) D0=D;

    if(D0 != D) {
      size_t x=D0*P;
      fftm0=new mfft1d(m,1,x, 1,m, G,H,threads);
      ifftm0=new mfft1d(m,-1,x, 1,m, H,G,threads);
    } else
      fftm0=NULL;

    if(!inplace)
      deleteAlign(H);
    deleteAlign(G);

    initZetaqm(q,centered && p == 2 ? m+1 : m);
  }
  if(showRoutines && (q != 1 || !centered)) {
    char const* cent=centered ? "Centered" : "";
    cout << endl;
    cout << "Forwards Routine: " << "fftPad" << cent << "::" << FR << endl;
    cout << "Backwards Routine: " << "fftPad" << cent << "::" << BR << endl;
  }
}

fftPad::~fftPad() {
  if(q == 1) {
    if(S == 1) {
      delete fftm1;
      delete ifftm1;
    } else {
      delete fftm;
      delete ifftm;
    }
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
  size_t mp=m*p;
  for(size_t d=0; d < D; ++d) {
    Complex *F=W+m*d;
    PARALLELIF(
      mp > L+threshold,
      for(size_t s=L; s < mp; ++s)
        F[s]=0.0;
      );
  }
}

void fftPad::padMany(Complex *W)
{
  size_t mp=m*p;
  PARALLELIF(
    (mp-L)*C > threshold,
    for(size_t s=L; s < mp; ++s) {
      Complex *F=W+S*s;
      for(size_t c=0; c < C; ++c)
        F[c]=0.0;
    }
    );
}

void fftPad::forwardExplicit(Complex *f, Complex *F, size_t, Complex *W)
{
  if(W == NULL) W=F;

  PARALLELIF(
    L > threshold,
    for(size_t s=0; s < L; ++s)
      W[s]=f[s];
    );
  PARALLELIF(
    M-L > threshold,
    for(size_t s=L; s < M; ++s)
      W[s]=0.0;
    );
  fftm1->fft(W,F);
}

void fftPad::backwardExplicit(Complex *F, Complex *f, size_t, Complex *W)
{
  if(W == NULL) W=F;

  ifftm1->fft(F,W);
  PARALLELIF(
    L > threshold,
    for(size_t s=0; s < L; ++s)
      f[s]=W[s];
    );
}

void fftPad::forwardExplicitMany(Complex *f, Complex *F, size_t,
                                 Complex *W)
{
  if(W == NULL) W=F;

  PARALLELIF(
    L*C > threshold,
    for(size_t s=0; s < L; ++s) {
      size_t Ss=S*s;
      Complex *Ws=W+Ss;
      Complex *fs=f+Ss;
      for(size_t c=0; c < C; ++c)
        Ws[c]=fs[c];
    });

  padMany(W);
  fftm->fft(W,F);
}

void fftPad::backwardExplicitMany(Complex *F, Complex *f, size_t,
                                  Complex *W)
{
  if(W == NULL) W=F;

  ifftm->fft(F,W);
  PARALLELIF(
    L*C > threshold,
    for(size_t s=0; s < L; ++s) {
      size_t Ss=S*s;
      Complex *fs=f+Ss;
      Complex *Ws=W+Ss;
      for(size_t c=0; c < C; ++c)
        fs[c]=Ws[c];
    });
}

void fftPad::forward1All(Complex *f, Complex *F, size_t, Complex *)
{
  F[0]=f[0];
  Complex *Zetar=Zetaqm+m;
  PARALLELIF(
    m > threshold,
    for(size_t s=1; s < m; ++s)
      F[s]=Zetar[s]*f[s];
    );
  fftm->fft(f);
  fftm->fft(F);
}

void fftPad::forward1(Complex *f, Complex *F0, size_t r0, Complex *W)
{
  if(W == NULL) W=F0;

  Complex *W0=W;
  size_t dr0=dr;
  mfft1d *fftm1;

  if(r0 == 0) {
    size_t residues;
    size_t q2=q/2;
    if(D == 1 || 2*q2 < q) { // q odd, r=0
      if(!inplace && D == 1 && L >= m)
        return fftm->fft(f,F0);
      residues=1;
      if(W != f) {
        PARALLELIF(
          L > threshold,
          for(size_t s=0; s < L; ++s)
            W[s]=f[s];
          );
      }
    } else { // q even, r=0,q/2
      residues=2;
      Complex *V=W+m;
      V[0]=W[0]=f[0];
      Complex *Zetar=Zetaqm+m*q2;
      PARALLELIF(
        L > threshold,
        for(size_t s=1; s < L; ++s) {
          Complex fs=f[s];
          W[s]=fs;
          V[s]=Zetar[s]*fs;
        });
      if(inplace) {
        PARALLELIF(
          m > L+threshold,
          for(size_t s=L; s < m; ++s)
            V[s]=0.0;
          );
      }
    }
    if(inplace) {
      PARALLELIF(
        m > L+threshold,
        for(size_t s=L; s < m; ++s)
          W[s]=0.0;
        );
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
      PARALLELIF(
        L > threshold,
        for(size_t s=1; s < L; ++s)
          W[s]=Zetar[s]*f[s];
        );
      if(inplace) {
        PARALLELIF(
          m > L+threshold,
          for(size_t s=L; s < m; ++s)
            W[s]=0.0;
          );
      }
    }
  } else {
    for(size_t d=0; d < dr0; ++d) {
      Complex *F=W+2*m*d;
      Complex *G=F+m;
      size_t r=r0+d;
      F[0]=G[0]=f[0];
      Complex *Zetar=Zetaqm+m*r;
      PARALLELIF(
        L > threshold,
        for(size_t s=1; s < L; ++s) {
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
      for(size_t d=0; d < dr0; ++d) {
        Complex *F=W+2*m*d;
        Complex *G=F+m;
        PARALLELIF(
          m > L+threshold,
          for(size_t s=L; s < m; ++s)
            F[s]=0.0;
          for(size_t s=L; s < m; ++s)
            G[s]=0.0;
          );
      }
    }
  }

  fftm1->fft(W0,F0);
}

void fftPad::forward1ManyAll(Complex *f, Complex *F, size_t, Complex *)
{
  Complex *Zetar=Zetaqm+m;
  PARALLELIF(
    C > threshold,
    for(size_t c=0; c < C; ++c)
      F[c]=f[c];
    );
  PARALLELIF(
    (m-1)*C > threshold,
    for(size_t s=1; s < m; ++s) {
      size_t Ss=S*s;
      Complex *fs=f+Ss;
      Complex *Fs=F+Ss;
      Vec Zeta=LOAD(Zetar+s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(Zeta,-Zeta);
      for(size_t c=0; c < C; ++c)
        STORE(Fs+c,ZMULT(X,Y,LOAD(fs+c)));
    });
  fftm->fft(f);
  fftm->fft(F);
}

void fftPad::forward1Many(Complex *f, Complex *F, size_t r, Complex *W)
{
  if(W == NULL) W=F;

  if(inplace) {
    PARALLELIF(
      m*C > L*C+threshold,
      for(size_t s=L; s < m; ++s) {
        Complex *Ws=W+S*s;
        for(size_t c=0; c < C; ++c)
          Ws[c]=0.0;
      });
  }
  if(r == 0) {
    if(!inplace && L >= m && S == C)
      return fftm->fft(f,F);
    PARALLELIF(
      L*C > threshold,
      for(size_t s=0; s < L; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        Complex *fs=f+Ss;
        for(size_t c=0; c < C; ++c)
          Ws[c]=fs[c];
      });
  } else {
    PARALLELIF(
      C > threshold,
      for(size_t c=0; c < C; ++c)
        W[c]=f[c];
      );
    Complex *Zetar=Zetaqm+m*r;
    PARALLELIF(
      (L-1)*C > threshold,
      for(size_t s=1; s < L; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        Complex *fs=f+Ss;
        Vec Zeta=LOAD(Zetar+s);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(Zeta,-Zeta);
        for(size_t c=0; c < C; ++c)
          STORE(Ws+c,ZMULT(X,Y,LOAD(fs+c)));
      });
  }
  fftm->fft(W,F);
}

void fftPad::forward2All(Complex *f, Complex *F, size_t, Complex *)
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
  PARALLELIF(
    m > threshold,
    for(size_t s=1; s < m; ++s) {
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

void fftPad::forward2(Complex *f, Complex *F0, size_t r0, Complex *W)
{
  if(W == NULL) W=F0;

  Complex *W0=W;
  size_t dr0=dr;
  mfft1d *fftm1;

  size_t Lm=L-m;
  Complex *fm=f+m;
  if(r0 == 0) {
    size_t residues;
    size_t q2=q/2;
    if(D == 1 || 2*q2 < q) { // q odd, r=0
      residues=1;
      PARALLELIF(
        Lm > threshold,
        for(size_t s=0; s < Lm; ++s)
          W[s]=f[s]+fm[s];
        );
      PARALLELIF(
        m > Lm+threshold,
        for(size_t s=Lm; s < m; ++s)
          W[s]=f[s];
        );
    } else {
      residues=2;
      Complex *V=W+m;
      Complex f0=f[0];
      Complex fm0=fm[0];
      W[0]=f0+fm0;
      V[0]=f0-fm0;
      Complex *Zetar=Zetaqm+m*q2;
      PARALLELIF(
        Lm > threshold,
        for(size_t s=1; s < Lm; ++s) {
          Complex fs=f[s];
          Complex fms=fm[s];
          W[s]=fs+fms;
          V[s]=conj(Zetar[s])*(fs-fms);
        });
      PARALLELIF(
        m > Lm+threshold,
        for(size_t s=Lm; s < m; ++s) {
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
      W[0]=f[0]+Zetar2[0]*fm[0];
      Complex *Zetar=Zetaqm+m*r0;
      PARALLELIF(
        Lm > threshold,
        for(size_t s=1; s < Lm; ++s) {
//        W[s]=Zetar[s]*f[s]+Zetar2[s]*fm[s];
          Vec Zeta=LOAD(Zetar+s);
          Vec Zeta2=LOAD(Zetar2+s);
          Vec fs=LOAD(f+s);
          Vec fms=LOAD(fm+s);
          STORE(W+s,ZMULT(Zeta,fs)+ZMULT(Zeta2,fms));
        });
      PARALLELIF(
        m > Lm+threshold,
        for(size_t s=Lm; s < m; ++s)
          W[s]=Zetar[s]*f[s];
        );
    }
  } else {
    for(size_t d=0; d < dr0; ++d) {
      Complex *F=W+2*m*d;
      Complex *G=F+m;
      size_t r=r0+d;
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
      PARALLELIF(
        Lm > threshold,
        for(size_t s=1; s < Lm; ++s) {
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
      PARALLELIF(
        m > Lm+threshold,
        for(size_t s=Lm; s < m; ++s) {
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

void fftPad::forward2ManyAll(Complex *f, Complex *F, size_t, Complex *)
{
  Complex *g=f+Sm;
  Complex *Zetar2=ZetaqmS+L;
  Vec Zetam=LOAD(Zetar2);
  PARALLELIF(
    C > threshold,
    for(size_t c=0; c < C; ++c) {
      Complex *v0=f+c;
      Complex *v1=g+c;
      Vec V1=LOAD(v1);
      Vec A=Zetam*UNPACKL(V1,V1);
      Vec B=ZMULTI(Zetam*UNPACKH(V1,V1));
      Vec V0=LOAD(v0);
      STORE(v0,V0+V1);
      STORE(v1,V0+A+B);
      STORE(F+c,V0+CONJ(A-B));
    });
  Complex *Zetar=Zetaqm+m;
  PARALLELIF(
    (m-1)*C > threshold,
    for(size_t s=1; s < m; ++s) {
      size_t Ss=S*s;
      Complex *v0=f+Ss;
      Complex *v1=g+Ss;
      Complex *v2=F+Ss;
      Vec Zeta=LOAD(Zetar+s);
      Vec Zetam=LOAD(Zetar2+s);
      for(size_t c=0; c < C; ++c) {
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

void fftPad::forward2Many(Complex *f, Complex *F, size_t r, Complex *W)
{
  if(W == NULL) W=F;

  size_t Lm=L-m;
  if(r == 0) {
    PARALLELIF(
      Lm*C > threshold,
      for(size_t s=0; s < Lm; ++s) {
        size_t Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fs=f+Ss;
        Complex *fms=f+Sm+Ss;
        for(size_t c=0; c < C; ++c)
          Fs[c]=fs[c]+fms[c];
      });
    PARALLELIF(
      m*C > Lm*C+threshold,
      for(size_t s=Lm; s < m; ++s) {
        size_t Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fs=f+Ss;
        for(size_t c=0; c < C; ++c)
          Fs[c]=fs[c];
      });
  } else {
    Complex *Zetar2=ZetaqmS+Lm*r+m;
    Complex Zeta2=Zetar2[0];
    Complex *fm=f+Sm;
    PARALLELIF(
      C > threshold,
      for(size_t c=0; c < C; ++c)
        W[c]=f[c]+Zeta2*fm[c];
      );
    Complex *Zetar=Zetaqm+m*r;
    PARALLELIF(
      Lm*C > C+threshold,
      for(size_t s=1; s < Lm; ++s) {
        size_t Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fs=f+Ss;
        Complex *fms=f+Sm+Ss;
        Complex Zetars=Zetar[s];
        Complex Zetarms=Zetar2[s];
        for(size_t c=0; c < C; ++c)
          Fs[c]=Zetars*fs[c]+Zetarms*fms[c];
      });
    PARALLELIF(
      m*C > Lm*C+threshold,
      for(size_t s=Lm; s < m; ++s) {
        size_t Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fs=f+Ss;
        Complex Zetars=Zetar[s];
        for(size_t c=0; c < C; ++c)
          Fs[c]=Zetars*fs[c];
      });
  }
  fftm->fft(W,F);
}

void fftPad::forwardInner(Complex *f, Complex *F0, size_t r0, Complex *W)
{
  if(W == NULL) W=F0;

  Complex *W0=W;
  size_t dr0=dr;
  mfft1d *fftm1;

  size_t pm1=p-1;
  size_t mpm1=m*pm1;
  size_t stop=L-mpm1;

  if(r0 == 0) {
    PARALLELIF(
      pm1*m > threshold,
      for(size_t t=0; t < pm1; ++t) {
        size_t mt=m*t;
        Complex *Ft=W+mt;
        Complex *ft=f+mt;
        for(size_t s=0; s < m; ++s)
          Ft[s]=ft[s];
      });

    Complex *Ft=W+mpm1;
    Complex *ft=f+mpm1;
    PARALLELIF(
      stop > threshold,
      for(size_t s=0; s < stop; ++s)
        Ft[s]=ft[s];
      );
    PARALLELIF(
      m > stop+threshold,
      for(size_t s=stop; s < m; ++s)
        Ft[s]=0.0;
      );

    fftp->fft(W);
    PARALLELIF(
      pm1*(m-1) > threshold,
      for(size_t t=1; t < p; ++t) {
        size_t R=n*t;
        Complex *Ft=W+m*t;
        Complex *Zetar=Zetaqm+m*R;
        for(size_t s=1; s < m; ++s)
          Ft[s] *= Zetar[s];
      });

    W += b;
    r0=1;
    dr0=D0-1;
    fftm1=D0 == D ? fftm : fftm0;
  } else
    fftm1=fftm;

  for(size_t d=0; d < dr0; ++d) {
    Complex *F=W+b*d;
    size_t r=r0+d;
    PARALLELIF(
      m > threshold,
      for(size_t s=0; s < m; ++s)
        F[s]=f[s];
      );
    Complex *Zetaqr=Zetaqp+pm1*r;
    PARALLELIF(
      (pm1-1)*m > threshold,
      for(size_t t=1; t < pm1; ++t) {
        size_t mt=m*t;
        Complex *Ft=F+mt;
        Complex *ft=f+mt;
        Complex Zeta=Zetaqr[t]; // TODO: Vectorize
        for(size_t s=0; s < m; ++s)
          Ft[s]=Zeta*ft[s];
      });
    Complex *Ft=F+mpm1;
    Complex *ft=f+mpm1;
    Complex Zeta=Zetaqr[pm1];
    PARALLELIF(
      stop > threshold,
      for(size_t s=0; s < stop; ++s)
        Ft[s]=Zeta*ft[s];
      );
    PARALLELIF(
      m > stop+threshold,
      for(size_t s=stop; s < m; ++s)
        Ft[s]=0.0;
      );

    fftp->fft(F);
    PARALLELIF(
      p*(m-1) > threshold,
      for(size_t t=0; t < p; ++t) {
        size_t R=n*t+r;
        Complex *Ft=F+m*t;
        Complex *Zetar=Zetaqm+m*R;
        for(size_t s=1; s < m; ++s)
          Ft[s] *= Zetar[s];
      });
  }

  fftm1->fft(W0,F0);
}

void fftPad::forwardInnerMany(Complex *f, Complex *F, size_t r,
                              Complex *W)
{
  if(W == NULL) W=F;

  size_t pm1=p-1;
  size_t stop=L-m*pm1;

  if(r == 0) {
    PARALLELIF(
      pm1*m*C > threshold,
      for(size_t t=0; t < pm1; ++t) {
        size_t Smt=Sm*t;
        Complex *Ft=W+Smt;
        Complex *ft=f+Smt;
        for(size_t s=0; s < m; ++s) {
          size_t Ss=S*s;
          Complex *Fts=Ft+Ss;
          Complex *fts=ft+Ss;
          for(size_t c=0; c < C; ++c)
            Fts[c]=fts[c];
        }
      });
    size_t Smt=Sm*pm1;
    Complex *Ft=W+Smt;
    Complex *ft=f+Smt;
    PARALLELIF(
      stop*C > threshold,
      for(size_t s=0; s < stop; ++s) {
        size_t Ss=S*s;
        Complex *Fts=Ft+Ss;
        Complex *fts=ft+Ss;
        for(size_t c=0; c < C; ++c)
          Fts[c]=fts[c];
      });
    PARALLELIF(
      m*C > stop*C+threshold,
      for(size_t s=stop; s < m; ++s) {
        Complex *Fts=Ft+S*s;
        for(size_t c=0; c < C; ++c)
          Fts[c]=0.0;
      });

    if(S == C)
      fftp->fft(W);
    else
      for(size_t s=0; s < m; ++s)
        fftp->fft(W+S*s);

    PARALLELIF(
      (p-1)*(m-1)*C > threshold,
      for(size_t t=1; t < p; ++t) {
        size_t R=n*t;
        Complex *Wt=W+Sm*t;
        Complex *Zetar=Zetaqm+m*R;
        for(size_t s=1; s < m; ++s) {
          Complex *Wts=Wt+S*s;
          Complex Zetars=Zetar[s];
          for(size_t c=0; c < C; ++c)
            Wts[c] *= Zetars;
        }
      });
  } else {
    PARALLELIF(
      m*C > threshold,
      for(size_t s=0; s < m; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        Complex *fs=f+Ss;
        for(size_t c=0; c < C; ++c)
          Ws[c]=fs[c];
      });
    Complex *Zetaqr=Zetaqp+pm1*r;
    PARALLELIF(
      (pm1-1)*m*C > threshold,
      for(size_t t=1; t < pm1; ++t) {
        size_t Smt=Sm*t;
        Complex *Wt=W+Smt;
        Complex *ft=f+Smt;
        Complex Zeta=Zetaqr[t];
        for(size_t s=0; s < m; ++s) {
          size_t Ss=S*s;
          Complex *Wts=Wt+Ss;
          Complex *fts=ft+Ss;
          for(size_t c=0; c < C; ++c)
            Wts[c]=Zeta*fts[c];
        }
      });
    size_t Smpm1=Sm*pm1;
    Complex *Wt=W+Smpm1;
    Complex *ft=f+Smpm1;
    Vec Zeta=LOAD(Zetaqr+pm1);
    Vec X=UNPACKL(Zeta,Zeta);
    Vec Y=UNPACKH(Zeta,-Zeta);
    PARALLELIF(
      stop*C > threshold,
      for(size_t s=0; s < stop; ++s) {
        size_t Ss=S*s;
        Complex *Wts=Wt+Ss;
        Complex *fts=ft+Ss;
        for(size_t c=0; c < C; ++c)
          STORE(Wts+c,ZMULT(X,Y,LOAD(fts+c)));
      });
    PARALLELIF(
      m*C > stop*C+threshold,
      for(size_t s=stop; s < m; ++s) {
        Complex *Wts=Wt+S*s;
        for(size_t c=0; c < C; ++c)
          Wts[c]=0.0;
      });

    if(S == C)
      fftp->fft(W);
    else
      for(size_t s=0; s < m; ++s)
        fftp->fft(W+S*s);

    PARALLELIF(
      p*(m-1)*C > threshold,
      for(size_t t=0; t < p; ++t) {
        size_t R=n*t+r;
        Complex *Ft=W+Sm*t;
        Complex *Zetar=Zetaqm+m*R;
        for(size_t s=1; s < m; ++s) {
          Complex *Fts=Ft+S*s;
          Vec Zetars=LOAD(Zetar+s);
          Vec X=UNPACKL(Zetars,Zetars);
          Vec Y=UNPACKH(Zetars,-Zetars);
          for(size_t c=0; c < C; ++c)
            STORE(Fts+c,ZMULT(X,Y,LOAD(Fts+c)));
        }
      });
  }
  for(size_t t=0; t < p; ++t) {
    size_t Smt=Sm*t;
    fftm->fft(W+Smt,F+Smt);
  }
}

void fftPad::backward1All(Complex *F, Complex *f, size_t, Complex *)
{
  ifftm->fft(f);
  ifftm->fft(F);

  f[0] += F[0];
  Complex *Zetar=Zetaqm+m;
  PARALLELIF(
    (m-1) > threshold,
    for(size_t s=1; s < m; ++s)
      f[s] += conj(Zetar[s])*F[s];
    );
}

void fftPad::backward1(Complex *F0, Complex *f, size_t r0, Complex *W)
{
  if(W == NULL) W=F0;

  if(r0 == 0 && !inplace && D == 1 && L >= m)
    return ifftm->fft(F0,f);

  (r0 > 0 || D0 == D ? ifftm : ifftm0)->fft(F0,W);

  size_t dr0=dr;

  if(r0 == 0) {
    size_t residues;
    size_t q2=q/2;
    if(D == 1 || 2*q2 < q) { // q odd, r=0
      residues=1;
      PARALLELIF(
        L > threshold,
        for(size_t s=0; s < L; ++s)
          f[s]=W[s];
        );
    } else { // q even, r=0,q/2
      residues=2;
      Complex *V=W+m;
      f[0]=W[0]+V[0];
      Complex *Zetar=Zetaqm+m*q2;
      PARALLELIF(
        L > threshold,
        for(size_t s=1; s < L; ++s)
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
      PARALLELIF(
        L > threshold,
        for(size_t s=1; s < L; ++s)
          f[s] += conj(Zetar[s])*W[s];
        );
    }
  } else {
    for(size_t d=0; d < dr0; ++d) {
      Complex *U=W+2*m*d;
      Complex *V=U+m;
      f[0] += U[0]+V[0];
      size_t r=r0+d;
      Complex *Zetar=Zetaqm+m*r;
      PARALLELIF(
        L > threshold,
        for(size_t s=1; s < L; ++s) {
          Vec Zeta=LOAD(Zetar+s);
          Vec Us=LOAD(U+s);
          Vec Vs=LOAD(V+s);
          STORE(f+s,LOAD(f+s)+ZMULT2(Zeta,Vs,Us));
        });
    }
  }
}

void fftPad::backward1ManyAll(Complex *F, Complex *f, size_t, Complex *)
{
  ifftm->fft(f);
  ifftm->fft(F);

  Complex *Zetar=Zetaqm+m;
  PARALLELIF(
    C > threshold,
    for(size_t c=0; c < C; ++c)
      f[c] += F[c];
    );
  PARALLELIF(
    (m-1)*C > threshold,
    for(size_t s=1; s < m; ++s) {
      size_t Ss=S*s;
      Complex *fs=f+Ss;
      Complex *Fs=F+Ss;
      Vec Zeta=LOAD(Zetar+s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(-Zeta,Zeta);
      for(size_t c=0; c < C; ++c)
        STORE(fs+c,LOAD(fs+c)+ZMULT(X,Y,LOAD(Fs+c)));
    });
}

void fftPad::backward1Many(Complex *F, Complex *f, size_t r, Complex *W)
{
  if(W == NULL) W=F;

  if(r == 0 && !inplace && L >= m)
    return ifftm->fft(F,f);

  ifftm->fft(F,W);

  if(r == 0) {
    PARALLELIF(
      L*C > threshold,
      for(size_t s=0; s < L; ++s) {
        size_t Ss=S*s;
        Complex *fs=f+Ss;
        Complex *Fs=W+Ss;
        for(size_t c=0; c < C; ++c)
          fs[c]=Fs[c];
      });
  } else {
    PARALLELIF(
      C > threshold,
      for(size_t c=0; c < C; ++c)
        f[c] += W[c];
      );
    Complex *Zetar=Zetaqm+m*r;
    PARALLELIF(
      (L-1)*C > threshold,
      for(size_t s=1; s < L; ++s) {
        size_t Ss=S*s;
        Complex *fs=f+Ss;
        Complex *Fs=W+Ss;
        Vec Zeta=LOAD(Zetar+s);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(-Zeta,Zeta);
        for(size_t c=0; c < C; ++c)
          STORE(fs+c,LOAD(fs+c)+ZMULT(X,Y,LOAD(Fs+c)));
      }
      );
  }
}

void fftPad::backward2All(Complex *F, Complex *f, size_t, Complex *)
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
  PARALLELIF(
    m > threshold,
    for(size_t s=1; s < m; ++s) {
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

void fftPad::backward2(Complex *F0, Complex *f, size_t r0, Complex *W)
{
  if(W == NULL) W=F0;

  (r0 > 0 || D0 == D ? ifftm : ifftm0)->fft(F0,W);

  size_t dr0=dr;

  size_t Lm=L-m;
  if(r0 == 0) {
    size_t residues;
    size_t q2=q/2;
    if(D == 1 || 2*q2 < q) { // q odd, r=0
      residues=1;
      PARALLELIF(
        m > threshold,
        for(size_t s=0; s < m; ++s)
          f[s]=W[s];
        );
      Complex *Wm=W-m;
      PARALLELIF(
        L > m+threshold,
        for(size_t s=m; s < L; ++s)
          f[s]=Wm[s];
        );
    } else { // q even, r=0,q/2
      residues=2;
      Complex *V=W+m;
      f[0]=W[0]+V[0];
      Complex *Zetar=Zetaqm+m*q2;
      PARALLELIF(
        m > threshold,
        for(size_t s=1; s < m; ++s)
          f[s]=W[s]+Zetar[s]*V[s];
        );
      Complex *Zetar2=ZetaqmS+Lm*q2;
      Complex *Wm=W-m;
      PARALLELIF(
        L > m+threshold,
        for(size_t s=m; s < L; ++s)
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
      PARALLELIF(
        m > threshold,
        for(size_t s=1; s < m; ++s)
          f[s] += conj(Zetar1[s])*W[s];
        );
      Complex *Zetar2=ZetaqmS+Lm*r0;
      Complex *Wm=W-m;
      PARALLELIF(
        L > m+threshold,
        for(size_t s=m; s < L; ++s)
          f[s] += conj(Zetar2[s])*Wm[s];
        );
    }
  } else {
    for(size_t d=0; d < dr0; ++d) {
      Complex *U=W+2*m*d;
      Complex *V=U+m;
      size_t r=r0+d;
      f[0] += U[0]+V[0];
      Complex *Zetar=Zetaqm+m*r;
      PARALLELIF(
        m > threshold,
        for(size_t s=1; s < m; ++s)
//      f[s] += conj(Zetar[s])*U[s]+Zetar[s]*V[s];
          STORE(f+s,LOAD(f+s)+ZMULT2(LOAD(Zetar+s),LOAD(V+s),LOAD(U+s)));
        );
      Complex *Zetar2=ZetaqmS+Lm*r;
      Complex *Um=U-m;
      Complex *Vm=V-m;
      PARALLELIF(
        L > m+threshold,
        for(size_t s=m; s < L; ++s)
//      f[s] += conj(Zetar2[s])*Um[s]+Zetar2[s]*Vm[s];
          STORE(f+s,LOAD(f+s)+ZMULT2(LOAD(Zetar2+s),LOAD(Vm+s),LOAD(Um+s)));
        );
    }
  }
}

void fftPad::backward2ManyAll(Complex *F, Complex *f, size_t, Complex *)
{
  Complex *g=f+Sm;

  ifftm->fft(f);
  ifftm->fft(g);
  ifftm->fft(F);

  Complex *Zetar2=ZetaqmS+L;
  Vec Zetam=LOAD(Zetar2);
  Vec Xm=UNPACKL(Zetam,Zetam);
  Vec Ym=UNPACKH(Zetam,-Zetam);
  PARALLELIF(
    C > threshold,
    for(size_t c=0; c < C; ++c) {
      Complex *v0=f+c;
      Complex *v1=g+c;
      Vec V0=LOAD(v0);
      Vec V1=LOAD(v1);
      Vec V2=LOAD(F+c);
      STORE(v0,V0+V1+V2);
      STORE(v1,V0+ZMULT2(Xm,Ym,V2,V1));
    });
  Complex *Zetar=Zetaqm+m;
  PARALLELIF(
    (m-1)*C > threshold,
    for(size_t s=1; s < m; ++s) {
      size_t Ss=S*s;
      Complex *v0=f+Ss;
      Complex *v1=g+Ss;
      Complex *v2=F+Ss;
      Vec Zeta=LOAD(Zetar+s);
      Vec Zetam=LOAD(Zetar2+s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(Zeta,-Zeta);
      Vec Xm=UNPACKL(Zetam,Zetam);
      Vec Ym=UNPACKH(Zetam,-Zetam);
      for(size_t c=0; c < C; ++c) {
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

void fftPad::backward2Many(Complex *F, Complex *f, size_t r, Complex *W)
{
  if(W == NULL) W=F;

  ifftm->fft(F,W);

  if(r == 0) {
    PARALLELIF(
      m*C > threshold,
      for(size_t s=0; s < m; ++s) {
        size_t Ss=S*s;
        Complex *fs=f+Ss;
        Complex *Fs=W+Ss;
        for(size_t c=0; c < C; ++c)
          fs[c]=Fs[c];
      });
    Complex *WSm=W-Sm;
    PARALLELIF(
      L*C > m*C+threshold,
      for(size_t s=m; s < L; ++s) {
        size_t Ss=S*s;
        Complex *fs=f+Ss;
        Complex *Fs=WSm+Ss;
        for(size_t c=0; c < C; ++c)
          fs[c]=Fs[c];
      });
  } else {
    size_t Lm=L-m;
    PARALLELIF(
      C > threshold,
      for(size_t c=0; c < C; ++c)
        f[c] += W[c];
      );
    Complex *Zetar=Zetaqm+m*r;
    PARALLELIF(
      (m-1)*C > threshold,
      for(size_t s=1; s < m; ++s) {
        size_t Ss=S*s;
        Complex *fs=f+Ss;
        Complex *Fs=W+Ss;
        Complex Zetars=conj(Zetar[s]);
        for(size_t c=0; c < C; ++c)
          fs[c] += Zetars*Fs[c];
      });
    Complex *Zetar2=ZetaqmS+Lm*r;
    Complex *WSm=W-Sm;
    PARALLELIF(
      L*C > m*C+threshold,
      for(size_t s=m; s < L; ++s) {
        size_t Ss=S*s;
        Complex *fs=f+Ss;
        Complex *Fs=WSm+Ss;
        Complex Zetars2=conj(Zetar2[s]);
        for(size_t c=0; c < C; ++c)
          fs[c] += Zetars2*Fs[c];
      });
  }
}

void fftPad::backwardInner(Complex *F0, Complex *f, size_t r0,
                           Complex *W)
{
  if(W == NULL) W=F0;

  (r0 > 0 || D0 == D ? ifftm : ifftm0)->fft(F0,W);

  size_t dr0=dr;

  size_t pm1=p-1;
  size_t stop=L-m*pm1;

  if(r0 == 0) {
    PARALLELIF(
      pm1*(m-1) > threshold,
      for(size_t t=1; t < p; ++t) {
        size_t R=n*t;
        Complex *Ft=W+m*t;
        Complex *Zetar=Zetaqm+m*R;
        for(size_t s=1; s < m; ++s)
          Ft[s] *= conj(Zetar[s]);
      });
    ifftp->fft(W);
    PARALLELIF(
      pm1*m > threshold,
      for(size_t t=0; t < pm1; ++t) {
        size_t mt=m*t;
        Complex *ft=f+mt;
        Complex *Ft=W+mt;
        for(size_t s=0; s < m; ++s)
          ft[s]=Ft[s];
      });
    size_t mt=m*pm1;
    Complex *ft=f+mt;
    Complex *Ft=W+mt;
    PARALLELIF(
      stop > threshold,
      for(size_t s=0; s < stop; ++s)
        ft[s]=Ft[s];
      );

    W += b;
    r0=1;
    dr0=D0-1;
  }

  for(size_t d=0; d < dr0; ++d) {
    Complex *F=W+b*d;
    size_t r=r0+d;
    PARALLELIF(
      p*(m-1) > threshold,
      for(size_t t=0; t < p; ++t) {
        size_t R=n*t+r;
        Complex *Ft=F+m*t;
        Complex *Zetar=Zetaqm+m*R;
        for(size_t s=1; s < m; ++s)
          Ft[s] *= conj(Zetar[s]);
      });
    ifftp->fft(F);
    PARALLELIF(
      m > threshold,
      for(size_t s=0; s < m; ++s)
        f[s] += F[s];
      );
    Complex *Zetaqr=Zetaqp+pm1*r;
    PARALLELIF(
      (pm1-1)*m > threshold,
      for(size_t t=1; t < pm1; ++t) {
        size_t mt=m*t;
        Complex *ft=f+mt;
        Complex *Ft=F+mt;
        Complex Zeta=conj(Zetaqr[t]);
        for(size_t s=0; s < m; ++s)
          ft[s] += Zeta*Ft[s];
      });
    size_t mt=m*pm1;
    Complex *Ft=F+mt;
    Complex *ft=f+mt;
    Complex Zeta=conj(Zetaqr[pm1]);
    PARALLELIF(
      stop > threshold,
      for(size_t s=0; s < stop; ++s)
        ft[s] += Zeta*Ft[s];
      );
  }
}

void fftPad::backwardInnerMany(Complex *F, Complex *f, size_t r,
                               Complex *W)
{
  if(W == NULL) W=F;

  for(size_t t=0; t < p; ++t) {
    size_t Smt=Sm*t;
    ifftm->fft(F+Smt,W+Smt);
  }
  size_t pm1=p-1;
  size_t stop=L-m*pm1;

  if(r == 0) {
    PARALLELIF(
      pm1*(m-1)*C > threshold,
      for(size_t t=1; t < p; ++t) {
        size_t R=n*t;
        Complex *Ft=W+Sm*t;
        Complex *Zetar=Zetaqm+m*R;
        for(size_t s=1; s < m; ++s) {
          Complex *Fts=Ft+S*s;
          Complex Zetars=Zetar[s];
          for(size_t c=0; c < C; ++c)
            Fts[c] *= conj(Zetar[s]);
        }
      });

    if(S == C)
      ifftp->fft(W);
    else
      for(size_t s=0; s < m; ++s)
        ifftp->fft(W+S*s);

    PARALLELIF(
      pm1*m*C > threshold,
      for(size_t t=0; t < pm1; ++t) {
        Complex *ft=f+Sm*t;
        Complex *Ft=W+Sm*t;
        for(size_t s=0; s < m; ++s) {
          Complex *fts=ft+S*s;
          Complex *Fts=Ft+S*s;
          for(size_t c=0; c < C; ++c)
            fts[c]=Fts[c];
        }
      });
    Complex *ft=f+Sm*pm1;
    Complex *Ft=W+Sm*pm1;
    PARALLELIF(
      stop*C > threshold,
      for(size_t s=0; s < stop; ++s) {
        Complex *fts=ft+S*s;
        Complex *Fts=Ft+S*s;
        for(size_t c=0; c < C; ++c)
          fts[c]=Fts[c];
      });
  } else {
    PARALLELIF(
      p*(m-1)*C > threshold,
      for(size_t t=0; t < p; ++t) {
        size_t R=n*t+r;
        Complex *Ft=W+Sm*t;
        Complex *Zetar=Zetaqm+m*R;
        for(size_t s=1; s < m; ++s) {
          Complex *Fts=Ft+S*s;
          Complex Zetars=conj(Zetar[s]);
          for(size_t c=0; c < C; ++c)
            Fts[c] *= Zetars;
        }
      });

    if(S == C)
      ifftp->fft(W);
    else
      for(size_t s=0; s < m; ++s)
        ifftp->fft(W+S*s);
    PARALLELIF(
      m*C > threshold,
      for(size_t s=0; s < m; ++s) {
        size_t Ss=S*s;
        Complex *fs=f+Ss;
        Complex *Fs=W+Ss;
        for(size_t c=0; c < C; ++c)
          fs[c] += Fs[c];
      });
    Complex *Zetaqr=Zetaqp+pm1*r;
    PARALLELIF(
      (pm1-1)*m*C > threshold,
      for(size_t t=1; t < pm1; ++t) {
        size_t Smt=Sm*t;
        Complex *ft=f+Smt;
        Complex *Ft=W+Smt;
        Complex Zeta=conj(Zetaqr[t]);
        for(size_t s=0; s < m; ++s) {
          size_t Ss=S*s;
          Complex *fts=ft+Ss;
          Complex *Fts=Ft+Ss;
          for(size_t c=0; c < C; ++c)
            fts[c] += Zeta*Fts[c];
        }
      });
    Complex *ft=f+Sm*pm1;
    Complex *Ft=W+Sm*pm1;
    Complex Zeta=conj(Zetaqr[pm1]);
    PARALLELIF(
      stop*C > threshold,
      for(size_t s=0; s < stop; ++s) {
        size_t Ss=S*s;
        Complex *fts=ft+Ss;
        Complex *Fts=Ft+Ss;
        for(size_t c=0; c < C; ++c)
          fts[c] += Zeta*Fts[c];
      });
  }
}

void fftPadCentered::init()
{
  ZetaShift=NULL;
  char const *FR;
  char const *BR;
  if(q == 1) {
    if(M % 2 == 0) {
      if(S == 1) {
        Forward=&fftBase::forwardExplicitFast;
        Backward=&fftBase::backwardExplicitFast;
        FR="forwardExplicitFast";
        BR="backwardExplicitFast";
      } else {
        Forward=&fftBase::forwardExplicitManyFast;
        Backward=&fftBase::backwardExplicitManyFast;
        FR="forwardExplicitManyFast";
        BR="backwardExplicitManyFast";
      }
    } else {
      initShift();
      if(S == 1) {
        Forward=&fftBase::forwardExplicitSlow;
        Backward=&fftBase::backwardExplicitSlow;
        FR="forwardExplicitSlow";
        BR="backwardExplicitSlow";
      } else {
        Forward=&fftBase::forwardExplicitManySlow;
        Backward=&fftBase::backwardExplicitManySlow;
        FR="forwardExplicitManySlow";
        BR="backwardExplicitManySlow";
      }
    }
    if(showRoutines) {
      cout << endl << "Forwards Routine: " << "fftPadCentered::" << FR << endl;
      cout << "Backwards Routine: " << "fftPadCentered::" << BR << endl;
    }
  }
}

void fftPadCentered::forwardExplicitFast(Complex *f, Complex *F,
                                         size_t, Complex *W)
{
  if(W == NULL) W=F;

  size_t H=ceilquotient(M-L,2);
  PARALLELIF(
    H > threshold,
    for(size_t s=0; s < H; ++s)
      W[s]=0.0;
    );
  Complex *FH=W+S*H;
  PARALLELIF(
    L > threshold,
    for(size_t s=0; s < L; ++s)
      FH[s]=f[s];
    );
  PARALLELIF(
    M > H+L+threshold,
    for(size_t s=H+L; s < M; ++s)
      W[s]=0.0;
    );

  fftm1->fft(W,F);

  PARALLELIF(
    m > 2*threshold,
    for(size_t s=1; s < m; s += 2)
      F[s] *= -1;
    );
}

void fftPadCentered::backwardExplicitFast(Complex *F, Complex *f,
                                          size_t, Complex *W)
{
  if(W == NULL) W=F;

  PARALLELIF(
    m > 2*threshold,
    for(size_t s=1; s < m; s += 2)
      F[s] *= -1;
    );

  ifftm1->fft(F,W);

  size_t H=ceilquotient(M-L,2);
  Complex *FH=W+S*H;
  PARALLELIF(
    L > threshold,
    for(size_t s=0; s < L; ++s)
      f[s]=FH[s];
    );
}

void fftPadCentered::forwardExplicitManyFast(Complex *f, Complex *F,
                                             size_t, Complex *W)
{
  if(W == NULL) W=F;

  size_t H=ceilquotient(M-L,2);
  PARALLELIF(
    H*C > threshold,
    for(size_t s=0; s < H; ++s) {
      size_t Ss=S*s;
      Complex *Fs=W+Ss;
      for(size_t c=0; c < C; ++c)
        Fs[c]=0.0;
    });
  Complex *FH=W+S*H;
  PARALLELIF(
    L*C > threshold,
    for(size_t s=0; s < L; ++s) {
      size_t Ss=S*s;
      Complex *fs=f+Ss;
      Complex *FHs=FH+Ss;
      for(size_t c=0; c < C; ++c)
        FHs[c]=fs[c];
    });
  PARALLELIF(
    M*C > (H+L)*C+threshold,
    for(size_t s=H+L; s < M; ++s) {
      size_t Ss=S*s;
      Complex *Fs=W+Ss;
      for(size_t c=0; c < C; ++c)
        Fs[c]=0.0;
    });

  fftm->fft(W,F);

  PARALLELIF(
    (m-1)*C > 2*threshold,
    for(size_t s=1; s < m; s += 2) {
      size_t Ss=S*s;
      Complex *Fs=F+Ss;
      for(size_t c=0; c < C; ++c)
        Fs[c] *= -1;
    });
}

void fftPadCentered::backwardExplicitManyFast(Complex *F, Complex *f,
                                              size_t, Complex *W)
{
  if(W == NULL) W=F;

  PARALLELIF(
    (m-1)*C > 2*threshold,
    for(size_t s=1; s < m; s += 2) {
      size_t Ss=S*s;
      Complex *Fs=F+Ss;
      for(size_t c=0; c < C; ++c)
        Fs[c] *= -1;
    });

  ifftm->fft(F,W);

  size_t H=ceilquotient(M-L,2);
  Complex *FH=W+S*H;
  PARALLELIF(
    L*C > threshold,
    for(size_t s=0; s < L; ++s) {
      size_t Ss=S*s;
      Complex *fs=f+Ss;
      Complex *FHs=FH+Ss;
      for(size_t c=0; c < C; ++c)
        fs[c]=FHs[c];
    });
}

void fftPadCentered::initShift()
{
  ZetaShift=ComplexAlign(M);
  double factor=L/2*twopi/M;
  for(size_t r=0; r < q; ++r) {
    Complex *Zetar=ZetaShift+r;
    PARALLELIF(
      m > threshold,
      for(size_t s=0; s < m; ++s)
        Zetar[q*s]=expi(factor*(q*s+r));
      );
  }
}

void fftPadCentered::forwardExplicitSlow(Complex *f, Complex *F,
                                         size_t, Complex *W)
{
  fftPad::forwardExplicit(f,F,0,W);
  PARALLELIF(
    m > threshold,
    for(size_t s=0; s < m; ++s)
      F[s] *= conj(ZetaShift[s]);
    );
}

void fftPadCentered::backwardExplicitSlow(Complex *F, Complex *f,
                                          size_t, Complex *W)
{
  PARALLELIF(
    m > threshold,
    for(size_t s=0; s < m; ++s)
      F[s] *= ZetaShift[s];
    );
  fftPad::backwardExplicit(F,f,0,W);
}

void fftPadCentered::forwardExplicitManySlow(Complex *f, Complex *F,
                                             size_t, Complex *W)
{
  fftPad::forwardExplicitMany(f,F,0,W);
  PARALLELIF(
    m*C > threshold,
    for(size_t s=0; s < m; ++s) {
      Complex *Fs=F+S*s;
      Complex Zeta=conj(ZetaShift[s]);
      for(size_t c=0; c < C; ++c)
        Fs[c] *= Zeta;
    });
}

void fftPadCentered::backwardExplicitManySlow(Complex *F, Complex *f,
                                              size_t, Complex *W)
{
  PARALLELIF(
    m*C > threshold,
    for(size_t s=0; s < m; ++s) {
      Complex *Fs=F+S*s;
      Complex Zeta=ZetaShift[s];
      for(size_t c=0; c < C; ++c)
        Fs[c] *= Zeta;
    });
  fftPad::backwardExplicitMany(F,f,0,W);
}

void fftPadCentered::forward2All(Complex *f, Complex *F, size_t,
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
  PARALLELIF(
    m > threshold,
    for(size_t s=1; s < m; ++s) {
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

void fftPadCentered::forward2(Complex *f, Complex *F0, size_t r0,
                              Complex *W)
{
  if(W == NULL) W=F0;
  Complex *W0=W;
  size_t dr0=dr;
  mfft1d *fftm1;

  size_t H=L/2;
  size_t mH=m-H;
  size_t LH=L-H;
  Complex *fmH=f-mH;
  Complex *fH=f+H;
  if(r0 == 0) {
    size_t residues;
    size_t q2=q/2;
    if(D == 1 || 2*q2 < q) { // q odd, r=0
      residues=1;
      PARALLELIF(
        mH > threshold,
        for(size_t s=0; s < mH; ++s)
          W[s]=fH[s];
        );
      PARALLELIF(
        LH > mH+threshold,
        for(size_t s=mH; s < LH; ++s)
          W[s]=fmH[s]+fH[s];
        );
      PARALLELIF(
        m > LH+threshold,
        for(size_t s=LH; s < m; ++s)
          W[s]=fmH[s];
        );
    } else {
      residues=2;
      Complex *V=W+m;
      Complex *Zetar=Zetaqm+(m+1)*q2;
      PARALLELIF(
        mH > threshold,
        for(size_t s=0; s < mH; ++s) {
          Complex A=fH[s];
          W[s]=A;
          V[s]=Zetar[s]*A;
        });
      PARALLELIF(
        LH > mH+threshold,
        for(size_t s=mH; s < LH; ++s) {
          Complex B=fmH[s];
          Complex A=fH[s];
          W[s]=A+B;
          V[s]=Zetar[s]*(A-B);
        });
      PARALLELIF(
        m > LH+threshold,
        for(size_t s=LH; s < m; ++s) {
          Complex B=fmH[s];
          W[s]=B;
          V[s]=-Zetar[s]*B;
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
      PARALLELIF(
        mH > threshold,
        for(size_t s=0; s < mH; ++s)
          W[s]=Zetar[s]*fH[s];
        );
      Complex *Zetarm=Zetar+m;
      PARALLELIF(
        LH > mH+threshold,
        for(size_t s=mH; s < LH; ++s) {
//        W[s]=conj(*(Zetarm-s))*fmH[s]+Zetar[s]*fH[s];
          Vec Zeta=LOAD(Zetar+s);
          Vec Zetam=LOAD(Zetarm-s);
          Vec fs=LOAD(fH+s);
          Vec fms=LOAD(fmH+s);
          STORE(W+s,ZCMULT(Zetam,fms)+ZMULT(Zeta,fs));
        });
      PARALLELIF(
        m > LH+threshold,
        for(size_t s=LH; s < m; ++s)
          W[s]=conj(*(Zetarm-s))*fmH[s];
        );
    }
  } else {
    for(size_t d=0; d < dr0; ++d) {
      Complex *F=W+2*m*d;
      Complex *G=F+m;
      size_t r=r0+d;
      Complex *Zetar=Zetaqm+(m+1)*r;
      PARALLELIF(
        mH > threshold,
        for(size_t s=0; s < mH; ++s) {
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
      PARALLELIF(
        LH > mH+threshold,
        for(size_t s=mH; s < LH; ++s) {
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
      PARALLELIF(
        m > LH+threshold,
        for(size_t s=LH; s < m; ++s) {
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

void fftPadCentered::forward2ManyAll(Complex *f, Complex *F, size_t,
                                     Complex *)
{
  Complex *g=f+Sm;
  Complex *Zetar=Zetaqm+(m+1);
  Complex *Zetarm=Zetar+m;

  Vec Zetam=CONJ(LOAD(Zetarm));
  PARALLELIF(
    C > threshold,
    for(size_t c=0; c < C; ++c) {
      Complex *v0=f+c;
      Complex *v1=g+c;
      Vec V0=LOAD(v0);
      Vec A=Zetam*UNPACKL(V0,V0);
      Vec B=ZMULTI(Zetam*UNPACKH(V0,V0));
      Vec V1=LOAD(v1);
      STORE(v0,V0+V1);
      STORE(v1,V1+A+B);
      STORE(F+c,V1+CONJ(A-B));
    });

  PARALLELIF(
    (m-1)*C > threshold,
    for(size_t s=1; s < m; ++s) {
      size_t Ss=S*s;
      Complex *v0=f+Ss;
      Complex *v1=g+Ss;
      Complex *v2=F+Ss;
      Vec Zeta=LOAD(Zetar+s);
      Vec Zetam=LOAD(Zetarm-s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(Zeta,-Zeta);
      Vec Xm=UNPACKL(Zetam,Zetam);
      Vec Ym=UNPACKH(-Zetam,Zetam);
      for(size_t c=0; c < C; ++c) {
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

void fftPadCentered::forward2Many(Complex *f, Complex *F, size_t r,
                                  Complex *W)
{
  if(W == NULL) W=F;

  size_t H=L/2;
  size_t mH=m-H;
  size_t LH=L-H;
  Complex *fH=f+S*H;
  Complex *fmH=f-S*mH;
  if(r == 0) {
    PARALLELIF(
      mH*C > threshold,
      for(size_t s=0; s < mH; ++s) {
        size_t Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fHs=fH+Ss;
        for(size_t c=0; c < C; ++c)
          Fs[c]=fHs[c];
      });
    PARALLELIF(
      LH*C > mH*C+threshold,
      for(size_t s=mH; s < LH; ++s) {
        size_t Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fmHs=fmH+Ss;
        Complex *fHs=fH+Ss;
        for(size_t c=0; c < C; ++c)
          Fs[c]=fmHs[c]+fHs[c];
      });
    PARALLELIF(
      m*C > LH*C+threshold,
      for(size_t s=LH; s < m; ++s) {
        size_t Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fmHs=fmH+Ss;
        for(size_t c=0; c < C; ++c)
          Fs[c]=fmHs[c];
      });
  } else {
    Complex *Zetar=Zetaqm+(m+1)*r;
    PARALLELIF(
      mH*C > threshold,
      for(size_t s=0; s < mH; ++s) {
        size_t Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fHs=fH+Ss;
//      Complex Zeta=Zetar[s];
        Vec Zeta=LOAD(Zetar+s);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(Zeta,-Zeta);
        for(size_t c=0; c < C; ++c)
//        Fs[c]=Zeta*fHs[c];
          STORE(Fs+c,ZMULT(X,Y,LOAD(fHs+c)));
      });
    Complex *Zetarm=Zetar+m;
    PARALLELIF(
      LH*C > mH*C+threshold,
      for(size_t s=mH; s < LH; ++s) {
        size_t Ss=S*s;
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
        for(size_t c=0; c < C; ++c)
//        Fs[c]=Zetarms*fmHs[c]+Zetars*fHs[c];
          STORE(Fs+c,ZMULT(Xm,Ym,LOAD(fmHs+c))+ZMULT(X,Y,LOAD(fHs+c)));
      });
    PARALLELIF(
      m*C > LH*C+threshold,
      for(size_t s=LH; s < m; ++s) {
        size_t Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fmHs=fmH+Ss;
//      Complex Zetam=conj(*(Zetarm-s));
        Vec Zetam=LOAD(Zetarm-s);
        Vec Xm=UNPACKL(Zetam,Zetam);
        Vec Ym=UNPACKH(-Zetam,Zetam);
        for(size_t c=0; c < C; ++c)
//        Fs[c]=Zetam*fmHs[c];
          STORE(Fs+c,ZMULT(Xm,Ym,LOAD(fmHs+c)));
      });
  }
  fftm->fft(W,F);
}

void fftPadCentered::backward2All(Complex *F, Complex *f, size_t,
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

  PARALLELIF(
    m > threshold,
    for(size_t s=1; s < m; ++s) {
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

void fftPadCentered::backward2(Complex *F0, Complex *f, size_t r0,
                               Complex *W)
{
  if(W == NULL) W=F0;

  (r0 > 0 || D0 == D ? ifftm : ifftm0)->fft(F0,W);

  size_t dr0=dr;

  size_t H=L/2;
  size_t odd=L-2*H;
  size_t mH=m-H;
  size_t LH=L-H;
  Complex *fmH=f-mH;
  Complex *fH=f+H;
  if(r0 == 0) {
    size_t residues;
    size_t q2=q/2;
    if(D == 1 || 2*q2 < q) { // q odd, r=0
      residues=1;
      PARALLELIF(
        m > mH+threshold,
        for(size_t s=mH; s < m; ++s)
          fmH[s]=W[s];
        );
      PARALLELIF(
        LH > threshold,
        for(size_t s=0; s < LH; ++s)
          fH[s]=W[s];
        );
    } else { // q even, r=0,q/2
      residues=2;
      Complex *V=W+m;
      Complex *Zetar=Zetaqm+(m+1)*q2;
      Complex *Zetarm=Zetar+m;
      PARALLELIF(
        m > mH+threshold,
        for(size_t s=mH; s < m; ++s)
          fmH[s]=W[s]+*(Zetarm-s)*V[s];
        );
      PARALLELIF(
        LH > threshold,
        for(size_t s=0; s < LH; ++s)
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
      PARALLELIF(
        m > mH+threshold,
        for(size_t s=mH; s < m; ++s)
          fmH[s] += *(Zetarm-s)*W[s];
        );
      PARALLELIF(
        LH > threshold,
        for(size_t s=0; s < LH; ++s)
          fH[s] += conj(Zetar[s])*W[s];
        );
    }
  } else {
    for(size_t d=0; d < dr0; ++d) {
      Complex *U=W+2*m*d;
      Complex *V=U+m;
      size_t r=r0+d;
      Complex *Zetar=Zetaqm+(m+1)*r;
      Complex *Zetarm=Zetar+m;
      PARALLELIF(
        m > H+threshold,
        for(size_t s=H; s < m; ++s)
          STORE(fmH+s,LOAD(fmH+s)+ZMULT2(LOAD(Zetarm-s),LOAD(U+s),LOAD(V+s)));
        );
      PARALLELIF(
        H > mH+threshold,
        for(size_t s=mH; s < H; ++s) {
//        fmH[s] += *(Zetarm-s)*Us+conj(*(Zetarm-s))*Vs;
//        fH[s] += conj(Zetar[s])*Us+Zetar[s]*Vs;
          Vec Us=LOAD(U+s);
          Vec Vs=LOAD(V+s);
          Vec Zeta=LOAD(Zetar+s);
          Vec Zetam=LOAD(Zetarm-s);
          STORE(fmH+s,LOAD(fmH+s)+ZMULT2(Zetam,Us,Vs));
          STORE(fH+s,LOAD(fH+s)+ZMULT2(Zeta,Vs,Us));
        });
      PARALLELIF(
        mH > threshold,
        for(size_t s=0; s < mH; ++s)
          STORE(fH+s,LOAD(fH+s)+ZMULT2(LOAD(Zetar+s),LOAD(V+s),LOAD(U+s)));
        );
      if(odd)
        STORE(fH+H,LOAD(fH+H)+ZMULT2(LOAD(Zetar+H),LOAD(V+H),LOAD(U+H)));
    }
  }
}

void fftPadCentered::backward2ManyAll(Complex *F, Complex *f, size_t,
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
  PARALLELIF(
    C > threshold,
    for(size_t c=0; c < C; ++c) {
      Complex *v0=f+c;
      Complex *v1=g+c;
      Vec V0=LOAD(v0);
      Vec V1=LOAD(v1);
      Vec V2=LOAD(F+c);
      STORE(v0,V0+ZMULT2(Xm,Ym,V1,V2));
      STORE(v1,V0+V1+V2);
    });

  PARALLELIF(
    (m-1)*C > threshold,
    for(size_t s=1; s < m; ++s) {
      size_t Ss=S*s;
      Complex *v0=f+Ss;
      Complex *v1=g+Ss;
      Complex *v2=F+Ss;
      Vec Zetam=LOAD(Zetarm-s);
      Vec Zeta=LOAD(Zetar+s);
      Vec Xm=UNPACKL(Zetam,Zetam);
      Vec Ym=UNPACKH(Zetam,-Zetam);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(-Zeta,Zeta);
      for(size_t c=0; c < C; ++c) {
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

void fftPadCentered::backward2Many(Complex *F, Complex *f, size_t r,
                                   Complex *W)
{
  if(W == NULL) W=F;

  ifftm->fft(F,W);

  size_t H=L/2;
  size_t odd=L-2*H;
  size_t mH=m-H;
  Complex *fmH=f-S*mH;
  Complex *fH=f+S*H;
  if(r == 0) {
    PARALLELIF(
      m*C > H*C+threshold,
      for(size_t s=H; s < m; ++s) {
        size_t Ss=S*s;
        Complex *fmHs=fmH+Ss;
        Complex *Fs=W+Ss;
        for(size_t c=0; c < C; ++c)
          fmHs[c]=Fs[c];
      });
    PARALLELIF(
      H*C > mH*C+threshold,
      for(size_t s=mH; s < H; ++s) {
        size_t Ss=S*s;
        Complex *fmHs=fmH+Ss;
        Complex *fHs=fH+Ss;
        Complex *Fs=W+Ss;
        for(size_t c=0; c < C; ++c)
          fHs[c]=fmHs[c]=Fs[c];
      });
    PARALLELIF(
      mH*C > threshold,
      for(size_t s=0; s < mH; ++s) {
        size_t Ss=S*s;
        Complex *fHs=fH+Ss;
        Complex *Fs=W+Ss;
        for(size_t c=0; c < C; ++c)
          fHs[c]=Fs[c];
      });
    if(odd) {
      size_t SH=S*H;
      Complex *fHs=fH+SH;
      Complex *Fs=W+SH;
      PARALLELIF(
        C > threshold,
        for(size_t c=0; c < C; ++c)
          fHs[c]=Fs[c];
        );
    }
  } else {
    Complex *Zetar=Zetaqm+(m+1)*r;
    Complex *Zetarm=Zetar+m;
    PARALLELIF(
      m*C > H*C+threshold,
      for(size_t s=H; s < m; ++s) {
        size_t Ss=S*s;
        Complex *fmHs=fmH+Ss;
        Complex *Fs=W+Ss;
        Vec Zetam=LOAD(Zetarm-s);
        Vec Xm=UNPACKL(Zetam,Zetam);
        Vec Ym=UNPACKH(Zetam,-Zetam);
        for(size_t c=0; c < C; ++c)
          STORE(fmHs+c,LOAD(fmHs+c)+ZMULT(Xm,Ym,LOAD(Fs+c)));
      });
    PARALLELIF(
      H*C > mH*C+threshold,
      for(size_t s=mH; s < H; ++s) {
        size_t Ss=S*s;
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
        for(size_t c=0; c < C; ++c) {
//        fmHs[c] += Zetam*Fsc;
//        fHs[c] += Zeta*Fsc;
          Vec Fsc=LOAD(Fs+c);
          STORE(fmHs+c,LOAD(fmHs+c)+ZMULT(Xm,Ym,Fsc));
          STORE(fHs+c,LOAD(fHs+c)+ZMULT(X,Y,Fsc));
        }
      });
    PARALLELIF(
      mH*C > threshold,
      for(size_t s=0; s < mH; ++s) {
        size_t Ss=S*s;
        Complex *fHs=fH+Ss;
        Complex *Fs=W+Ss;
        Vec Zeta=LOAD(Zetar+s);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(-Zeta,Zeta);
        for(size_t c=0; c < C; ++c)
          STORE(fHs+c,LOAD(fHs+c)+ZMULT(X,Y,LOAD(Fs+c)));
      });
    if(odd) {
      size_t SH=S*H;
      Complex *fHs=fH+SH;
      Complex *Fs=W+SH;
      Vec Zeta=LOAD(Zetar+H);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(-Zeta,Zeta);
      PARALLELIF(
        C > threshold,
        for(size_t c=0; c < C; ++c)
          STORE(fHs+c,LOAD(fHs+c)+ZMULT(X,Y,LOAD(Fs+c)));
        );
    }
  }
}

void fftPadCentered::forwardInnerAll(Complex *f, Complex *F, size_t,
                                     Complex *)
{
  size_t p2=p/2;
  Vec Zetanr=CONJ(LOAD(Zetaqp+p)); // zeta_n^{-r}
  Complex *g=f+b;
  PARALLELIF(
    m > threshold,
    for(size_t s=0; s < m; ++s) {
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
  PARALLELIF(
    (p2-1)*m > threshold,
    for(size_t t=1; t < p2; ++t) {
      size_t tm=t*m;
      Complex *v0=f+tm;
      Complex *v1=g+tm;
      Complex *v2=F+tm;
      Vec Zeta=LOAD(Zetaqr+t); // zeta_q^{tr}
      Vec Zetam=ZMULT(Zeta,Zetanr);
      for(size_t s=0; s < m; ++s) {
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
      }
    });

  fftp->fft(f);
  fftp->fft(g);
  fftp->fft(F);

  Complex *Zetar=Zetaqm+m;
  PARALLELIF(
    m > threshold,
    for(size_t s=1; s < m; ++s) {
      Complex Zeta=Zetar[s];
      g[s] *= Zeta;
      F[s] *= conj(Zeta);
    });
  PARALLELIF(
    (p2-1)*3*(m-1) > threshold,
    for(size_t u=1; u < p2; ++u) {
      size_t mu=m*u;
      Complex *Zetar0=Zetaqm+n*mu;
      Complex *fu=f+mu;
      for(size_t s=1; s < m; ++s)
        fu[s] *= Zetar0[s];
      Complex *Wu=g+mu;
      Complex *Zeta=Zetar0+m;
      for(size_t s=1; s < m; ++s)
        Wu[s] *= Zeta[s];
      Complex *Zeta2=Zetar0-m;
      Complex *Vu=F+mu;
      for(size_t s=1; s < m; ++s)
        Vu[s] *= Zeta2[s];
    });

  fftm->fft(f);
  fftm->fft(g);
  fftm->fft(F);
}

void fftPadCentered::forwardInner(Complex *f, Complex *F0, size_t r0,
                                  Complex *W)
{
  if(W == NULL) W=F0;

  Complex *W0=W;
  size_t dr0=dr;
  mfft1d *fftm1;

  size_t p2=p/2;
  size_t H=L/2;
  size_t p2s1=p2-1;
  size_t p2s1m=p2s1*m;
  size_t p2m=p2*m;
  size_t m0=p2m-H;
  size_t m1=L-H-p2s1m;
  Complex *fm0=f-m0;
  Complex *fH=f+H;
  if(r0 == 0) {
    size_t residues;
    size_t n2=n/2;
    if(D == 1 || 2*n2 < n) { // n odd, r=0
      residues=1;
      PARALLELIF(
        m0 > threshold,
        for(size_t s=0; s < m0; ++s)
          W[s]=fH[s];
        );
      PARALLELIF(
        m > m0+threshold,
        for(size_t s=m0; s < m; ++s)
          W[s]=fm0[s]+fH[s];
        );
      PARALLELIF(
        (p2s1-1)*m > threshold,
        for(size_t t=1; t < p2s1; ++t) {
          size_t tm=t*m;
          Complex *Wt=W+tm;
          Complex *fm0t=fm0+tm;
          Complex *fHt=fH+tm;
          for(size_t s=0; s < m; ++s)
            Wt[s]=fm0t[s]+fHt[s];
        });
      Complex *Wt=W+p2s1m;
      Complex *fm0t=fm0+p2s1m;
      Complex *fHt=fH+p2s1m;
      PARALLELIF(
        m1 > threshold,
        for(size_t s=0; s < m1; ++s)
          Wt[s]=fm0t[s]+fHt[s];
        );
      PARALLELIF(
        m > m1+threshold,
        for(size_t s=m1; s < m; ++s)
          Wt[s]=fm0t[s];
        );

      fftp->fft(W);

      size_t mn=m*n;
      for(size_t u=1; u < p2; ++u) {
        Complex *Wu=W+u*m;
        Complex *Zeta0=Zetaqm+mn*u;
        PARALLELIF(
          m > threshold,
          for(size_t s=1; s < m; ++s)
            Wu[s] *= Zeta0[s];
          );
      }
    } else { // n even, r=0,n/2
      residues=2;
      Complex *V=W+b;
      PARALLELIF(
        m0 > threshold,
        for(size_t s=0; s < m0; ++s)
          V[s]=W[s]=fH[s];
        );
      PARALLELIF(
        m > m0+threshold,
        for(size_t s=m0; s < m; ++s) {
          Complex fm0s=fm0[s];
          Complex fHs=fH[s];
          W[s]=fm0s+fHs;
          V[s]=-fm0s+fHs;
        });
      Complex *Zetaqn2=Zetaqp+p2*n2;
      PARALLELIF(
        (p2s1-1)*m > threshold,
        for(size_t t=1; t < p2s1; ++t) {
          size_t tm=t*m;
          Complex *Wt=W+tm;
          Complex *Vt=V+tm;
          Complex *fm0t=fm0+tm;
          Complex *fHt=fH+tm;
          Complex Zeta=Zetaqn2[t]; //*zeta_q^{tn/2}
          for(size_t s=0; s < m; ++s) {
            Complex fm0ts=fm0t[s];
            Complex fHts=fHt[s];
            Wt[s]=fm0ts+fHts;
            Vt[s]=Zeta*(fHts-fm0ts);
          }
        });
      Complex *Wt=W+p2s1m;
      Complex *Vt=V+p2s1m;
      Complex *fm0t=fm0+p2s1m;
      Complex *fHt=fH+p2s1m;
      Complex Zeta=Zetaqn2[p2s1];
      PARALLELIF(
        m1 > threshold,
        for(size_t s=0; s < m1; ++s) {
          Complex fm0ts=fm0t[s];
          Complex fHts=fHt[s];
          Wt[s]=fm0ts+fHts;
          Vt[s]=Zeta*(fHts-fm0ts);
        });
      Complex mZeta=-Zeta;
      PARALLELIF(
        m > m1+threshold,
        for(size_t s=m1; s < m; ++s) {
          Complex fm0ts=fm0t[s];
          Wt[s]=fm0ts;
          Vt[s]=mZeta*fm0ts;
        });
      fftp->fft(W);
      fftp->fft(V);

      size_t mn=m*n;
      size_t mn2=m*n2;
      Complex *Zetan2=Zetaqm+mn2;
      PARALLELIF(
        m > threshold,
        for(size_t s=0; s < m; ++s)
          V[s] *= Zetan2[s];
        );
      PARALLELIF(
        (p2-1)*2*(m-1) > threshold,
        for(size_t u=1; u < p2; ++u) {
          size_t um=u*m;
          Complex *Wu=W+um;
          Complex *Zeta0=Zetaqm+mn*u;
          for(size_t s=1; s < m; ++s)
            Wu[s] *= Zeta0[s];
          Complex *Vu=V+um;
          Complex *Zetan2=Zeta0+mn2;
          for(size_t s=1; s < m; ++s)
            Vu[s] *= Zetan2[s];
        });
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
      Vec Zetanr=CONJ(LOAD(Zetaqr+p2)); // zeta_n^{-r}

      PARALLELIF(
        m0 > threshold,
        for(size_t s=0; s < m0; ++s)
          W[s]=fH[s];
        );

      Vec X=UNPACKL(Zetanr,Zetanr);
      Vec Y=UNPACKH(Zetanr,-Zetanr);
      PARALLELIF(
        m > m0+threshold,
        for(size_t s=m0; s < m; ++s)
          STORE(W+s,ZMULT(X,Y,LOAD(fm0+s))+LOAD(fH+s));
        );
      PARALLELIF(
        (p2s1-1)*m > threshold,
        for(size_t t=1; t < p2s1; ++t) {
          size_t tm=t*m;
          Complex *Ft=W+tm;
          Complex *fm0t=fm0+tm;
          Complex *fHt=fH+tm;
          Vec Zeta=LOAD(Zetaqr+t); // zeta_q^{tr}
          Vec Zetam=ZMULT(Zeta,Zetanr);

          Vec Xm=UNPACKL(Zetam,Zetam);
          Vec Ym=UNPACKH(Zetam,-Zetam);
          Vec X=UNPACKL(Zeta,Zeta);
          Vec Y=UNPACKH(Zeta,-Zeta);
          for(size_t s=0; s < m; ++s)
            STORE(Ft+s,ZMULT(Xm,Ym,LOAD(fm0t+s))+ZMULT(X,Y,LOAD(fHt+s)));
        });
      Complex *Ft=W+p2s1m;
      Complex *fm0t=fm0+p2s1m;
      Complex *fHt=fH+p2s1m;
      Vec Zeta=LOAD(Zetaqr+p2s1);
      Vec Zetam=ZMULT(Zeta,Zetanr); // zeta_q^{-r}
      Vec Xm=UNPACKL(Zetam,Zetam);
      Vec Ym=UNPACKH(Zetam,-Zetam);
      X=UNPACKL(Zeta,Zeta);
      Y=UNPACKH(Zeta,-Zeta);
      PARALLELIF(
        m1 > threshold,
        for(size_t s=0; s < m1; ++s)
          STORE(Ft+s,ZMULT(Xm,Ym,LOAD(fm0t+s))+ZMULT(X,Y,LOAD(fHt+s)));
        );
      PARALLELIF(
        m > m1+threshold,
        for(size_t s=m1; s < m; ++s)
          STORE(Ft+s,ZMULT(Xm,Ym,LOAD(fm0t+s)));
        );

      fftp->fft(W);

      size_t mr=m*r0;
      Complex *Zetar=Zetaqm+mr;
      PARALLELIF(
        m > threshold,
        for(size_t s=1; s < m; ++s)
          W[s] *= Zetar[s];
        );
      PARALLELIF(
        (p2-1)*(m-1) > threshold,
        for(size_t u=1; u < p2; ++u) {
          size_t mu=m*u;
          Complex *Zetar0=Zetar+n*mu;
          Complex *Wu=W+mu;
          for(size_t s=1; s < m; ++s)
            Wu[s] *= Zetar0[s];
        });
    }
  } else {
    for(size_t d=0; d < dr0; ++d) {
      Complex *F=W+2*b*d;
      size_t r=r0+d;
      Vec Zetanr=CONJ(LOAD(Zetaqp+p2*r+p2)); // zeta_n^{-r}
      Complex *G=F+b;

      PARALLELIF(
        m0 > threshold,
        for(size_t s=0; s < m0; ++s)
          G[s]=F[s]=fH[s];
        );
      PARALLELIF(
        m > m0+threshold,
        for(size_t s=m0; s < m; ++s) {
          Vec fm0ts=LOAD(fm0+s);
          Vec fHts=LOAD(fH+s);
          Vec A=Zetanr*UNPACKL(fm0ts,fm0ts);
          Vec B=ZMULTI(Zetanr*UNPACKH(fm0ts,fm0ts));
          STORE(F+s,A+B+fHts);
          STORE(G+s,CONJ(A-B)+fHts);
        });
      Complex *Zetaqr=Zetaqp+p2*r;
      PARALLELIF(
        (p2s1-1)*m > threshold,
        for(size_t t=1; t < p2s1; ++t) {
          size_t tm=t*m;
          Complex *Ft=F+tm;
          Complex *Gt=G+tm;
          Complex *fm0t=fm0+tm;
          Complex *fHt=fH+tm;
          Vec Zeta=LOAD(Zetaqr+t); // zeta_q^{tr}
          Vec Zetam=ZMULT(Zeta,Zetanr);
          for(size_t s=0; s < m; ++s) {
            Vec fm0ts=LOAD(fm0t+s);
            Vec fHts=LOAD(fHt+s);
            Vec A=Zetam*UNPACKL(fm0ts,fm0ts)+Zeta*UNPACKL(fHts,fHts);
            Vec B=ZMULTI(Zetam*UNPACKH(fm0ts,fm0ts)+Zeta*UNPACKH(fHts,fHts));
            STORE(Ft+s,A+B);
            STORE(Gt+s,CONJ(A-B));
          }
        });
      Complex *Ft=F+p2s1m;
      Complex *Gt=G+p2s1m;
      Complex *fm0t=fm0+p2s1m;
      Complex *fHt=fH+p2s1m;
      Vec Zeta=LOAD(Zetaqr+p2s1);
      Vec Zetam=ZMULT(Zeta,Zetanr);
      PARALLELIF(
        m1 > threshold,
        for(size_t s=0; s < m1; ++s) {
          Vec fm0ts=LOAD(fm0t+s);
          Vec fHts=LOAD(fHt+s);
          Vec A=Zetam*UNPACKL(fm0ts,fm0ts)+Zeta*UNPACKL(fHts,fHts);
          Vec B=ZMULTI(Zetam*UNPACKH(fm0ts,fm0ts)+Zeta*UNPACKH(fHts,fHts));
          STORE(Ft+s,A+B);
          STORE(Gt+s,CONJ(A-B));
        });
      PARALLELIF(
        m > m1+threshold,
        for(size_t s=m1; s < m; ++s) {
          Vec fm0ts=LOAD(fm0t+s);
          Vec A=Zetam*UNPACKL(fm0ts,fm0ts);
          Vec B=ZMULTI(Zetam*UNPACKH(fm0ts,fm0ts));
          STORE(Ft+s,A+B);
          STORE(Gt+s,CONJ(A-B));
        });

      fftp->fft(F);
      fftp->fft(G);

      size_t mr=m*r;
      Complex *Zetar=Zetaqm+mr;
      PARALLELIF(
        m > threshold,
        for(size_t s=1; s < m; ++s) {
          Complex Zeta=Zetar[s];
          F[s] *= Zeta;
          G[s] *= conj(Zeta);
        });
      PARALLELIF(
        (p2-1)*2*(m-1)> threshold,
        for(size_t u=1; u < p2; ++u) {
          size_t mu=m*u;
          Complex *Zetar0=Zetaqm+n*mu;
          Complex *Zetar=Zetar0+mr;
          Complex *Wu=F+mu;
          for(size_t s=1; s < m; ++s)
            Wu[s] *= Zetar[s];
          Complex *Zetar2=Zetar0-mr;
          Complex *Vu=G+mu;
          for(size_t s=1; s < m; ++s)
            Vu[s] *= Zetar2[s];
        });
    }
  }

  fftm1->fft(W0,F0);
}

void fftPadCentered::forwardInnerManyAll(Complex *f, Complex *F, size_t,
                                         Complex *)
{
  size_t p2=p/2;
  Vec Zetanr=CONJ(LOAD(Zetaqp+p)); // zeta_n^{-r}
  Complex *g=f+b;
  PARALLELIF(
    m*C > threshold,
    for(size_t s=0; s < m; ++s) {
      size_t Ss=S*s;
      Complex *v0=f+Ss;
      Complex *v1=g+Ss;
      Complex *v2=F+Ss;
      for(size_t c=0; c < C; ++c) {
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
  PARALLELIF(
    (p2-1)*m*C > threshold,
    for(size_t t=1; t < p2; ++t) {
      size_t tm=t*Sm;
      Complex *v0=f+tm;
      Complex *v1=g+tm;
      Complex *v2=F+tm;
      Vec Zeta=LOAD(Zetaqr+t); // zeta_q^{tr}
      Vec Zetam=ZMULT(Zeta,Zetanr);
      for(size_t s=0; s < m; ++s) {
        size_t Ss=S*s;
        Complex *v0s=v0+Ss;
        Complex *v1s=v1+Ss;
        Complex *v2s=v2+Ss;
        for(size_t c=0; c < C; ++c) {
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
      }
    });

  if(S == C) {
    fftp->fft(f);
    fftp->fft(g);
    fftp->fft(F);
  } else {
    for(size_t s=0; s < m; ++s) {
      size_t Ss=S*s;
      fftp->fft(f+Ss);
      fftp->fft(g+Ss);
      fftp->fft(F+Ss);
    };
  }

  Complex *Zetar=Zetaqm+m;
  PARALLELIF(
    (m-1)*C > threshold,
    for(size_t s=1; s < m; ++s) {
      Complex Zeta=Zetar[s];
      size_t Ss=S*s;
      Complex *gs=g+Ss;
      Complex *Fs=F+Ss;
      for(size_t c=0; c < C; ++c) {
        gs[c] *= Zeta;
        Fs[c] *= conj(Zeta);
      }
    });
  size_t nm=n*m;
  PARALLELIF(
    (p2-1)*3*(m-1)*C > threshold,
    for(size_t u=1; u < p2; ++u) {
      size_t Smu=Sm*u;
      Complex *Zetar0=Zetaqm+nm*u;
      Complex *fu=f+Smu;
      for(size_t s=1; s < m; ++s) {
        Complex Zeta0=Zetar0[s];
        Complex *fus=fu+S*s;
        for(size_t c=0; c < C; ++c)
          fus[c] *= Zeta0;
      }    Complex *Wu=g+Smu;
      Complex *Zetar=Zetar0+m;
      for(size_t s=1; s < m; ++s) {
        Complex Zeta=Zetar[s];
        Complex *Wus=Wu+S*s;
        for(size_t c=0; c < C; ++c)
          Wus[c] *= Zeta;
      }
      Complex *Zetar2=Zetar0-m;
      Complex *Vu=F+Smu;
      for(size_t s=1; s < m; ++s) {
        Complex Zeta2=Zetar2[s];
        Complex *Vus=Vu+S*s;
        for(size_t c=0; c < C; ++c)
          Vus[c] *= Zeta2;
      }
    });
  for(size_t t=0; t < p2; ++t) {
    size_t Smt=Sm*t;
    fftm->fft(f+Smt);
    fftm->fft(g+Smt);
    fftm->fft(F+Smt);
  }
}

void fftPadCentered::forwardInnerMany(Complex *f, Complex *F, size_t r,
                                      Complex *W)
{
  if(W == NULL) W=F;

  size_t p2=p/2;
  size_t H=L/2;
  size_t p2s1=p2-1;
  size_t p2s1m=p2s1*m;
  size_t p2m=p2*m;
  size_t m0=p2m-H;
  size_t m1=L-H-p2s1m;
  Complex *fm0=f-S*m0;
  Complex *fH=f+S*H;

  if(r == 0) {
    PARALLELIF(
      m0*C > threshold,
      for(size_t s=0; s < m0; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        Complex *fHs=fH+Ss;
        for(size_t c=0; c < C; ++c)
          Ws[c]=fHs[c];
      });
    PARALLELIF(
      m*C > m0*C+threshold,
      for(size_t s=m0; s < m; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        Complex *fm0s=fm0+Ss;
        Complex *fHs=fH+Ss;
        for(size_t c=0; c < C; ++c)
          Ws[c]=fm0s[c]+fHs[c];
      });
    PARALLELIF(
      (p2s1-1)*m*C > threshold,
      for(size_t t=1; t < p2s1; ++t) {
        size_t tm=t*Sm;
        Complex *Wt=W+tm;
        Complex *fm0t=fm0+tm;
        Complex *fHt=fH+tm;
        for(size_t s=0; s < m; ++s) {
          size_t Ss=S*s;
          Complex *Wts=Wt+Ss;
          Complex *fm0ts=fm0t+Ss;
          Complex *fHts=fHt+Ss;
          for(size_t c=0; c < C; ++c)
            Wts[c]=fm0ts[c]+fHts[c];
        }
      });
    size_t p2s1Sm=S*p2s1m;
    Complex *Wt=W+p2s1Sm;
    Complex *fm0t=fm0+p2s1Sm;
    Complex *fHt=fH+p2s1Sm;
    PARALLELIF(
      m1*C > threshold,
      for(size_t s=0; s < m1; ++s) {
        size_t Ss=S*s;
        Complex *Wts=Wt+Ss;
        Complex *fm0ts=fm0t+Ss;
        Complex *fHts=fHt+Ss;
        for(size_t c=0; c < C; ++c)
          Wts[c]=fm0ts[c]+fHts[c];
      });
    PARALLELIF(
      m*C > m1*C+threshold,
      for(size_t s=m1; s < m; ++s) {
        size_t Ss=S*s;
        Complex *Wts=Wt+Ss;
        Complex *fm0ts=fm0t+Ss;
        for(size_t c=0; c < C; ++c)
          Wts[c]=fm0ts[c];
      });

    if(S == C)
      fftp->fft(W);
    else
      for(size_t s=0; s < m; ++s)
        fftp->fft(W+S*s);
    size_t mn=m*n;
    for(size_t u=1; u < p2; ++u) {
      Complex *Wu=W+u*Sm;
      Complex *Zeta0=Zetaqm+mn*u;
      PARALLELIF(
        (m-1)*C > threshold,
        for(size_t s=1; s < m; ++s) {
          Complex *Wus=Wu+S*s;
          Vec Zeta=LOAD(Zeta0+s);
          Vec X=UNPACKL(Zeta,Zeta);
          Vec Y=UNPACKH(Zeta,-Zeta);
          for(size_t c=0; c < C; ++c)
            STORE(Wus+c,ZMULT(X,Y,LOAD(Wus+c)));
        });
    }
  } else {
    Complex *Zetaqr=Zetaqp+p2*r;
    Vec Zetanr=CONJ(LOAD(Zetaqr+p2)); // zeta_n^{-r}

    PARALLELIF(
      m0*C > threshold,
      for(size_t s=0; s < m0; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        Complex *fHs=fH+Ss;
        for(size_t c=0; c < C; ++c)
          Ws[c]=fHs[c];
      });

    Vec X=UNPACKL(Zetanr,Zetanr);
    Vec Y=UNPACKH(Zetanr,-Zetanr);
    PARALLELIF(
      m*C > m0*C+threshold,
      for(size_t s=m0; s < m; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        Complex *fm0s=fm0+Ss;
        Complex *fHs=fH+Ss;
        for(size_t c=0; c < C; ++c)
          STORE(Ws+c,ZMULT(X,Y,LOAD(fm0s+c))+LOAD(fHs+c));
      });
    PARALLELIF(
      (p2s1-1)*m*C > threshold,
      for(size_t t=1; t < p2s1; ++t) {
        size_t tm=t*Sm;
        Complex *Ft=W+tm;
        Complex *fm0t=fm0+tm;
        Complex *fHt=fH+tm;
        Vec Zeta=LOAD(Zetaqr+t); // zeta_q^{tr}
        Vec Zetam=ZMULT(Zeta,Zetanr);

        Vec Xm=UNPACKL(Zetam,Zetam);
        Vec Ym=UNPACKH(Zetam,-Zetam);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(Zeta,-Zeta);
        for(size_t s=0; s < m; ++s) {
          size_t Ss=S*s;
          Complex *Fts=Ft+Ss;
          Complex *fm0ts=fm0t+Ss;
          Complex *fHts=fHt+Ss;
          for(size_t c=0; c < C; ++c)
            STORE(Fts+c,ZMULT(Xm,Ym,LOAD(fm0ts+c))+ZMULT(X,Y,LOAD(fHts+c)));
        }
      });
    size_t p2s1Sm=S*p2s1m;
    Complex *Ft=W+p2s1Sm;
    Complex *fm0t=fm0+p2s1Sm;
    Complex *fHt=fH+p2s1Sm;
    Vec Zeta=LOAD(Zetaqr+p2s1);
    Vec Zetam=ZMULT(Zeta,Zetanr);
    Vec Xm=UNPACKL(Zetam,Zetam);
    Vec Ym=UNPACKH(Zetam,-Zetam);
    X=UNPACKL(Zeta,Zeta);
    Y=UNPACKH(Zeta,-Zeta);
    PARALLELIF(
      m1*C > threshold,
      for(size_t s=0; s < m1; ++s) {
        size_t Ss=S*s;
        Complex *Fts=Ft+Ss;
        Complex *fm0ts=fm0t+Ss;
        Complex *fHts=fHt+Ss;
        for(size_t c=0; c < C; ++c)
          STORE(Fts+c,ZMULT(Xm,Ym,LOAD(fm0ts+c))+ZMULT(X,Y,LOAD(fHts+c)));
      });
    PARALLELIF(
      m1*C > m*C+threshold,
      for(size_t s=m1; s < m; ++s) {
        size_t Ss=S*s;
        Complex *Fts=Ft+Ss;
        Complex *fm0ts=fm0t+Ss;
        for(size_t c=0; c < C; ++c)
          STORE(Fts+c,ZMULT(Xm,Ym,LOAD(fm0ts+c)));
      });

    if(S == C)
      fftp->fft(W);
    else
      for(size_t s=0; s < m; ++s)
        fftp->fft(W+S*s);
    size_t mr=m*r;
    for(size_t u=0; u < p2; ++u) {
      size_t mu=m*u;
      Complex *Zetar0=Zetaqm+n*mu;
      Complex *Zetar=Zetar0+mr;
      Complex *Wu=W+S*mu;
      PARALLELIF(
        (m-1)*C > threshold,
        for(size_t s=1; s < m; ++s) {
          Complex *Wus=Wu+S*s;
          Vec Zetars=LOAD(Zetar+s);
          Vec X=UNPACKL(Zetars,Zetars);
          Vec Y=UNPACKH(Zetars,-Zetars);
          for(size_t c=0; c < C; ++c)
            STORE(Wus+c,ZMULT(X,Y,LOAD(Wus+c)));
        });
    }
  }
  for(size_t t=0; t < p2; ++t) {
    size_t Smt=Sm*t;
    fftm->fft(W+Smt,F+Smt);
  }
}

void fftPadCentered::backwardInnerAll(Complex *F, Complex *f, size_t,
                                      Complex *)
{
  Complex *g=f+b;

  ifftm->fft(f);
  ifftm->fft(g);
  ifftm->fft(F);

  Vec Zetanr=LOAD(Zetaqp+p); // zeta_n^r

  Complex *Zetar=Zetaqm+m;
  PARALLELIF(
    m > threshold,
    for(size_t s=1; s < m; ++s) {
      Complex Zeta=Zetar[s];
      g[s] *= conj(Zeta);
      F[s] *= Zeta;
    });
  size_t p2=p/2;
  PARALLELIF(
    (p2-1)*3*(m-1) > threshold,
    for(size_t u=1; u < p2; ++u) {
      size_t mu=m*u;
      Complex *Zetar0=Zetaqm+n*mu;
      Complex *fu=f+mu;
      for(size_t s=1; s < m; ++s)
        fu[s] *= conj(Zetar0[s]);
      Complex *Wu=g+mu;
      Complex *Zetar1=Zetar0+m;
      for(size_t s=1; s < m; ++s)
        Wu[s] *= conj(Zetar1[s]);
      Complex *Zetar2=Zetar0-m;
      Complex *Vu=F+mu;
      for(size_t s=1; s < m; ++s)
        Vu[s] *= conj(Zetar2[s]);
    });


  ifftp->fft(f);
  ifftp->fft(g);
  ifftp->fft(F);

  Vec Xm=UNPACKL(Zetanr,Zetanr);
  Vec Ym=UNPACKH(Zetanr,-Zetanr);
  PARALLELIF(
    m > threshold,
    for(size_t s=0; s < m; ++s) {
      Complex *v0s=f+s;
      Complex *v1s=g+s;
      Vec V0=LOAD(v0s);
      Vec V1=LOAD(v1s);
      Vec V2=LOAD(F+s);
      STORE(v0s,V0+ZMULT2(Xm,Ym,V1,V2));
      STORE(v1s,V0+V1+V2);
    });
  Complex *Zetaqr=Zetaqp+p2;
  PARALLELIF(
    (p2-1)*m > threshold,
    for(size_t t=1; t < p2; ++t) {
      size_t tm=t*m;
      Complex *v0=f+tm;
      Complex *v1=g+tm;
      Complex *v2=F+tm;
      Vec Zeta=LOAD(Zetaqr+t);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(Zeta,-Zeta);
      Vec Zeta2=ZMULT(X,-Y,Zetanr);
      Vec Xm=UNPACKL(Zeta2,Zeta2);
      Vec Ym=UNPACKH(Zeta2,-Zeta2);
      for(size_t s=0; s < m; ++s) {
        Complex *v0s=v0+s;
        Complex *v1s=v1+s;
        Vec V0=LOAD(v0s);
        Vec V1=LOAD(v1s);
        Vec V2=LOAD(v2+s);
        STORE(v0s,V0+ZMULT2(Xm,Ym,V1,V2));
        STORE(v1s,V0+ZMULT2(X,Y,V2,V1));
      }
    });
}

void fftPadCentered::backwardInner(Complex *F0, Complex *f, size_t r0,
                                   Complex *W)
{
  if(W == NULL) W=F0;

  (r0 > 0 || D0 == D ? ifftm : ifftm0)->fft(F0,W);

  size_t dr0=dr;

  size_t p2=p/2;
  size_t H=L/2;
  size_t p2s1=p2-1;
  size_t p2s1m=p2s1*m;
  size_t p2m=p2*m;
  size_t m0=p2m-H;
  size_t m1=L-H-p2s1m;
  Complex *fm0=f-m0;
  Complex *fH=f+H;

  if(r0 == 0) {
    size_t residues;
    size_t n2=n/2;
    if(D == 1 || 2*n2 < n) { // n odd, r=0
      residues=1;
      size_t mn=m*n;
      for(size_t u=1; u < p2; ++u) {
        Complex *Wu=W+u*m;
        Complex *Zeta0=Zetaqm+mn*u;
        PARALLELIF(
          m > threshold,
          for(size_t s=1; s < m; ++s)
            Wu[s] *= conj(Zeta0[s]);
          );
      }

      ifftp->fft(W);

      PARALLELIF(
        m0 > threshold,
        for(size_t s=0; s < m0; ++s)
          fH[s]=W[s];
        );

      PARALLELIF(
        m > m0+threshold,
        for(size_t s=m0; s < m; ++s)
          fH[s]=fm0[s]=W[s];
        );

      for(size_t t=1; t < p2s1; ++t) {
        size_t tm=t*m;
        Complex *Wt=W+tm;
        Complex *fm0t=fm0+tm;
        Complex *fHt=fH+tm;
        PARALLELIF(
          m > threshold,
          for(size_t s=0; s < m; ++s)
            fHt[s]=fm0t[s]=Wt[s];
          );
      }
      Complex *Wt=W+p2s1m;
      Complex *fm0t=fm0+p2s1m;
      Complex *fHt=fH+p2s1m;
      PARALLELIF(
        m1 > threshold,
        for(size_t s=0; s < m1; ++s)
          fHt[s]=fm0t[s]=Wt[s];
        );

      PARALLELIF(
        m > m1+threshold,
        for(size_t s=m1; s < m; ++s)
          fm0t[s]=Wt[s];
        );

    } else { // n even, r=0,n/2
      residues=2;
      Complex *V=W+b;
      size_t mn=m*n;
      size_t mn2=m*n2;

      Complex *Zetan2=Zetaqm+mn2;
      PARALLELIF(
        m > threshold,
        for(size_t s=1; s < m; ++s)
          V[s] *= conj(Zetan2[s]);
        );

      PARALLELIF(
        (p2-1)*2*(m-1) > threshold,
        for(size_t u=1; u < p2; ++u) {
          size_t mu=m*u;
          Complex *Wu=W+mu;
          Complex *Zeta0=Zetaqm+mn*u;
          for(size_t s=1; s < m; ++s)
            Wu[s] *= conj(Zeta0[s]);
          Complex *Vu=V+mu;
          Complex *Zetan2=Zeta0+mn2;
          for(size_t s=1; s < m; ++s)
            Vu[s] *= conj(Zetan2[s]);
        });
      ifftp->fft(W);
      ifftp->fft(V);
      PARALLELIF(
        m0 > threshold,
        for(size_t s=0; s < m0; ++s)
          fH[s]=W[s]+V[s];
        );

      PARALLELIF(
        m > m0+threshold,
        for(size_t s=m0; s < m; ++s) {
          Complex Wts=W[s];
          Complex Vts=V[s];
          fm0[s]=Wts-Vts;
          fH[s]=Wts+Vts;
        });
      Complex *Zetaqn2=Zetaqp+p2*n2;
      PARALLELIF(
        (p2s1-1)*m > threshold,
        for(size_t t=1; t < p2s1; ++t) {
          size_t tm=t*m;
          Complex *Wt=W+tm;
          Complex *Vt=V+tm;
          Complex *fm0t=fm0+tm;
          Complex *fHt=fH+tm;
          Complex Zeta=conj(Zetaqn2[t]);
          for(size_t s=0; s < m; ++s) {
            Complex Wts=Wt[s];
            Complex Vts=Zeta*Vt[s];
            fm0t[s]=Wts-Vts;
            fHt[s]=Wts+Vts;
          }
        });
      Complex *Wt=W+p2s1m;
      Complex *Vt=V+p2s1m;
      Complex *fm0t=fm0+p2s1m;
      Complex *fHt=fH+p2s1m;
      Complex Zeta=conj(Zetaqn2[p2s1]);
      PARALLELIF(
        m1 > threshold,
        for(size_t s=0; s < m1; ++s) {
          Complex Wts=Wt[s];
          Complex Vts=Zeta*Vt[s];
          fm0t[s]=Wts-Vts;
          fHt[s]=Wts+Vts;
        });
      PARALLELIF(
        m > m1+threshold,
        for(size_t s=m1; s < m; ++s)
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
      size_t mr=m*r0;
      Complex *Zetar=Zetaqm+mr;
      PARALLELIF(
        m > threshold,
        for(size_t s=1; s < m; ++s)
          W[s] *= conj(Zetar[s]);
        );
      PARALLELIF(
        (p2-1)*(m-1) > threshold,
        for(size_t u=1; u < p2; ++u) {
          size_t mu=m*u;
          Complex *Zetar0=Zetar+n*mu;
          Complex *Wu=W+mu;
          for(size_t s=1; s < m; ++s)
            Wu[s] *= conj(Zetar0[s]);
        });

      ifftp->fft(W);

      PARALLELIF(
        m0 > threshold,
        for(size_t s=0; s < m0; ++s)
          fH[s] += W[s];
        );

      Vec Xm=UNPACKL(Zetanr,Zetanr);
      Vec Ym=UNPACKH(Zetanr,-Zetanr);
      PARALLELIF(
        m > m0+threshold,
        for(size_t s=m0; s < m; ++s) {
          Vec Fts=LOAD(W+s);
          STORE(fm0+s,LOAD(fm0+s)+ZMULT(Xm,Ym,Fts));
          STORE(fH+s,LOAD(fH+s)+Fts);
        });
      Complex *Zetaqr=Zetaqp+p2*r0;
      PARALLELIF(
        (p2s1-1)*m > threshold,
        for(size_t t=1; t < p2s1; ++t) {
          size_t tm=t*m;
          Complex *Ft=W+tm;
          Complex *fm0t=fm0+tm;
          Complex *fHt=fH+tm;
          Vec Zeta=LOAD(Zetaqr+t);
          Vec X=UNPACKL(Zeta,Zeta);
          Vec Y=UNPACKH(-Zeta,Zeta);
          Vec Zeta2=ZMULT(X,Y,Zetanr);
          Vec Xm=UNPACKL(Zeta2,Zeta2);
          Vec Ym=UNPACKH(Zeta2,-Zeta2);
          for(size_t s=0; s < m; ++s) {
            Vec Fts=LOAD(Ft+s);
            STORE(fm0t+s,LOAD(fm0t+s)+ZMULT(Xm,Ym,Fts));
            STORE(fHt+s,LOAD(fHt+s)+ZMULT(X,Y,Fts));
          }
        });
      Complex *Ft=W+p2s1m;
      Complex *fm0t=fm0+p2s1m;
      Complex *fHt=fH+p2s1m;
      Vec Zeta=LOAD(Zetaqr+p2s1);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(-Zeta,Zeta);
      Vec Zeta2=ZMULT(X,Y,Zetanr);
      Xm=UNPACKL(Zeta2,Zeta2);
      Ym=UNPACKH(Zeta2,-Zeta2);
      PARALLELIF(
        m1 > threshold,
        for(size_t s=0; s < m1; ++s) {
          Vec Fts=LOAD(Ft+s);
          STORE(fm0t+s,LOAD(fm0t+s)+ZMULT(Xm,Ym,Fts));
          STORE(fHt+s,LOAD(fHt+s)+ZMULT(X,Y,Fts));
        });
      PARALLELIF(
        m > m1+threshold,
        for(size_t s=m1; s < m; ++s)
          STORE(fm0t+s,LOAD(fm0t+s)+ZMULT(Xm,Ym,LOAD(Ft+s)));
        );
    }
  } else {
    for(size_t d=0; d < dr0; ++d) {
      Complex *F=W+2*b*d;
      size_t r=r0+d;
      Vec Zetanr=LOAD(Zetaqp+p2*r+p2); // zeta_n^r
      Complex *G=F+b;

      size_t mr=m*r;
      Complex *Zetar=Zetaqm+mr;
      PARALLELIF(
        m > threshold,
        for(size_t s=1; s < m; ++s) {
          Complex Zeta=Zetar[s];
          F[s] *= conj(Zeta);
          G[s] *= Zeta;
        });
      PARALLELIF(
        (p2-1)*(m-1) > threshold,
        for(size_t u=1; u < p2; ++u) {
          size_t mu=m*u;
          Complex *Zetar0=Zetaqm+n*mu;
          Complex *Zetar1=Zetar0+mr;
          Complex *Wu=F+mu;
          for(size_t s=1; s < m; ++s)
            Wu[s] *= conj(Zetar1[s]);
          Complex *Zetar2=Zetar0-mr;
          Complex *Vu=G+mu;
          for(size_t s=1; s < m; ++s)
            Vu[s] *= conj(Zetar2[s]);
        });

      ifftp->fft(F);
      ifftp->fft(G);

      PARALLELIF(
        m0 > threshold,
        for(size_t s=0; s < m0; ++s)
          fH[s] += F[s]+G[s];
        );

      Vec Xm=UNPACKL(Zetanr,Zetanr);
      Vec Ym=UNPACKH(Zetanr,-Zetanr);
      PARALLELIF(
        m > m0+threshold,
        for(size_t s=m0; s < m; ++s) {
          Vec Fts=LOAD(F+s);
          Vec Gts=LOAD(G+s);
          STORE(fm0+s,LOAD(fm0+s)+ZMULT2(Xm,Ym,Fts,Gts));
          STORE(fH+s,LOAD(fH+s)+Fts+Gts);
        });
      Complex *Zetaqr=Zetaqp+p2*r;
      PARALLELIF(
        (p2s1-1)*m > threshold,
        for(size_t t=1; t < p2s1; ++t) {
          size_t tm=t*m;
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
          for(size_t s=0; s < m; ++s) {
            Vec Fts=LOAD(Ft+s);
            Vec Gts=LOAD(Gt+s);
            STORE(fm0t+s,LOAD(fm0t+s)+ZMULT2(Xm,Ym,Fts,Gts));
            STORE(fHt+s,LOAD(fHt+s)+ZMULT2(X,Y,Gts,Fts));
          }
        });
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
      PARALLELIF(
        m1 > threshold,
        for(size_t s=0; s < m1; ++s) {
          Vec Fts=LOAD(Ft+s);
          Vec Gts=LOAD(Gt+s);
          STORE(fm0t+s,LOAD(fm0t+s)+ZMULT2(Xm,Ym,Fts,Gts));
          STORE(fHt+s,LOAD(fHt+s)+ZMULT2(X,Y,Gts,Fts));
        });
      PARALLELIF(
        m > m1+threshold,
        for(size_t s=m1; s < m; ++s)
          STORE(fm0t+s,LOAD(fm0t+s)+ZMULT2(Xm,Ym,LOAD(Ft+s),LOAD(Gt+s)));
        );
    }
  }
}

void fftPadCentered::backwardInnerManyAll(Complex *F, Complex *f, size_t,
                                          Complex *)
{
  Complex *g=f+b;

  size_t p2=p/2;
  for(size_t t=0; t < p2; ++t) {
    size_t Smt=Sm*t;
    ifftm->fft(f+Smt);
    ifftm->fft(g+Smt);
    ifftm->fft(F+Smt);
  }

  Vec Zetanr=LOAD(Zetaqp+p); // zeta_n^r

  Complex *Zetar=Zetaqm+m;
  PARALLELIF(
    (m-1)*C > threshold,
    for(size_t s=1; s < m; ++s) {
      Complex Zeta=Zetar[s];
      size_t Ss=S*s;
      Complex *gs=g+Ss;
      Complex *Fs=F+Ss;
      for(size_t c=0; c < C; ++c) {
        gs[c] *= conj(Zeta);
        Fs[c] *= Zeta;
      }
    });
  size_t nm=n*m;
  PARALLELIF(
    (p2-1)*3*(m-1)*C > threshold,
    for(size_t u=1; u < p2; ++u) {
      size_t Smu=Sm*u;
      Complex *Zetar0=Zetaqm+nm*u;
      Complex *fu=f+Smu;
      for(size_t s=1; s < m; ++s) {
        Complex Zeta0=conj(Zetar0[s]);
        Complex *fus=fu+S*s;
        for(size_t c=0; c < C; ++c)
          fus[c] *= Zeta0;
      }
      Complex *Wu=g+Smu;
      Complex *Zetar=Zetar0+m;
      for(size_t s=1; s < m; ++s) {
        Complex Zetar1=conj(Zetar[s]);
        Complex *Wus=Wu+S*s;
        for(size_t c=0; c < C; ++c)
          Wus[c] *= Zetar1;
      }
      Complex *Zetar2=Zetar0-m;
      Complex *Vu=F+Smu;
      for(size_t s=1; s < m; ++s) {
        Complex Zeta2=conj(Zetar2[s]);
        Complex *Vus=Vu+S*s;
        for(size_t c=0; c < C; ++c)
          Vus[c] *= Zeta2;
      }
    });

  if(S == C) {
    ifftp->fft(f);
    ifftp->fft(g);
    ifftp->fft(F);
  } else {
    for(size_t s=0; s < m; ++s) {
      size_t Ss=S*s;
      ifftp->fft(f+Ss);
      ifftp->fft(g+Ss);
      ifftp->fft(F+Ss);
    }
  }

  Vec Xm=UNPACKL(Zetanr,Zetanr);
  Vec Ym=UNPACKH(Zetanr,-Zetanr);
  PARALLELIF(
    m*C > threshold,
    for(size_t s=0; s < m; ++s) {
      size_t Ss=S*s;
      Complex *v0=f+Ss;
      Complex *v1=g+Ss;
      Complex *v2=F+Ss;
      for(size_t c=0; c < C; ++c) {
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
  PARALLELIF(
    (p2-1)*m*C > threshold,
    for(size_t t=1; t < p2; ++t) {
      size_t tm=t*Sm;
      Complex *v0=f+tm;
      Complex *v1=g+tm;
      Complex *v2=F+tm;
      Vec Zeta=LOAD(Zetaqr+t);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(Zeta,-Zeta);
      Vec Zeta2=ZMULT(X,-Y,Zetanr);
      Vec Xm=UNPACKL(Zeta2,Zeta2);
      Vec Ym=UNPACKH(Zeta2,-Zeta2);
      for(size_t s=0; s < m; ++s) {
        size_t Ss=S*s;
        Complex *v0s=v0+Ss;
        Complex *v1s=v1+Ss;
        Complex *v2s=v2+Ss;
        for(size_t c=0; c < C; ++c) {
          Complex *v0c=v0s+c;
          Complex *v1c=v1s+c;
          Vec V0=LOAD(v0c);
          Vec V1=LOAD(v1c);
          Vec V2=LOAD(v2s+c);
          STORE(v0c,V0+ZMULT2(Xm,Ym,V1,V2));
          STORE(v1c,V0+ZMULT2(X,Y,V2,V1));
        }
      }
    });
}

void fftPadCentered::backwardInnerMany(Complex *F, Complex *f, size_t r,
                                       Complex *W)
{
  if(W == NULL) W=F;

  size_t p2=p/2;

  for(size_t t=0; t < p2; ++t) {
    size_t Smt=Sm*t;
    ifftm->fft(F+Smt,W+Smt);
  }

  size_t H=L/2;
  size_t p2s1=p2-1;
  size_t p2s1m=p2s1*m;
  size_t p2m=p2*m;
  size_t m0=p2m-H;
  size_t m1=L-H-p2s1m;
  Complex *fm0=f-S*m0;
  Complex *fH=f+S*H;

  if(r == 0) {
    size_t mn=m*n;
    for(size_t u=1; u < p2; ++u) {
      Complex *Wu=W+u*Sm;
      Complex *Zeta0=Zetaqm+mn*u;
      PARALLELIF(
        (m-1)*C > threshold,
        for(size_t s=1; s < m; ++s) {
          Complex *Wus=Wu+S*s;
          Vec Zeta=LOAD(Zeta0+s);
          Vec X=UNPACKL(Zeta,Zeta);
          Vec Y=UNPACKH(-Zeta,Zeta);
          for(size_t c=0; c < C; ++c)
            STORE(Wus+c,ZMULT(X,Y,LOAD(Wus+c)));
        });
    }

    if(S == C)
      ifftp->fft(W);
    else
      for(size_t s=0; s < m; ++s)
        ifftp->fft(W+S*s);

    PARALLELIF(
      m0*C > threshold,
      for(size_t s=0; s < m0; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        Complex *fHs=fH+Ss;
        for(size_t c=0; c < C; ++c)
          fHs[c]=Ws[c];
      });

    PARALLELIF(
      m*C > m0*C+threshold,
      for(size_t s=m0; s < m; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        Complex *fm0s=fm0+Ss;
        Complex *fHs=fH+Ss;
        for(size_t c=0; c < C; ++c)
          fHs[c]=fm0s[c]=Ws[c];
      });

    PARALLELIF(
      (p2s1-1)*m*C > threshold,
      for(size_t t=1; t < p2s1; ++t) {
        size_t tm=t*Sm;
        Complex *Wt=W+tm;
        Complex *fm0t=fm0+tm;
        Complex *fHt=fH+tm;
        for(size_t s=0; s < m; ++s) {
          size_t Ss=S*s;
          Complex *Wts=Wt+Ss;
          Complex *fm0ts=fm0t+Ss;
          Complex *fHts=fHt+Ss;
          for(size_t c=0; c < C; ++c)
            fHts[c]=fm0ts[c]=Wts[c];
        }
      });
    size_t p2s1Sm=S*p2s1m;
    Complex *Wt=W+p2s1Sm;
    Complex *fm0t=fm0+p2s1Sm;
    Complex *fHt=fH+p2s1Sm;
    PARALLELIF(
      m1*C > threshold,
      for(size_t s=0; s < m1; ++s) {
        size_t Ss=S*s;
        Complex *Wts=Wt+Ss;
        Complex *fm0ts=fm0t+Ss;
        Complex *fHts=fHt+Ss;
        for(size_t c=0; c < C; ++c)
          fHts[c]=fm0ts[c]=Wts[c];
      });

    PARALLELIF(
      m*C > m1*C+threshold,
      for(size_t s=m1; s < m; ++s) {
        size_t Ss=S*s;
        Complex *Wts=Wt+Ss;
        Complex *fm0ts=fm0t+Ss;
        for(size_t c=0; c < C; ++c)
          fm0ts[c]=Wts[c];
      });
  } else {
    Vec Zetanr=LOAD(Zetaqp+p2*r+p2); // zeta_n^r
    size_t mr=m*r;
    Complex *Zetar=Zetaqm+mr;
    for(size_t u=0; u < p2; ++u) {
      size_t mu=m*u;
      Complex *Zetar0=Zetar+n*mu;
      Complex *Wu=W+u*Sm;
      PARALLELIF(
        (m-1)*C > threshold,
        for(size_t s=1; s < m; ++s) {
          Complex *Wus=Wu+S*s;
          Vec Zeta=LOAD(Zetar0+s);
          Vec X=UNPACKL(Zeta,Zeta);
          Vec Y=UNPACKH(-Zeta,Zeta);
          for(size_t c=0; c < C; ++c)
            STORE(Wus+c,ZMULT(X,Y,LOAD(Wus+c)));
        });
    }

    if(S == C)
      ifftp->fft(W);
    else
      for(size_t s=0; s < m; ++s)
        ifftp->fft(W+S*s);

    PARALLELIF(
      m0*C > threshold,
      for(size_t s=0; s < m0; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        Complex *fHs=fH+Ss;
        for(size_t c=0; c < C; ++c)
          fHs[c] += Ws[c];
      });

    Vec Xm=UNPACKL(Zetanr,Zetanr);
    Vec Ym=UNPACKH(Zetanr,-Zetanr);
    PARALLELIF(
      m*C > m0*C+threshold,
      for(size_t s=m0; s < m; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        Complex *fm0s=fm0+Ss;
        Complex *fHs=fH+Ss;
        for(size_t c=0; c < C; ++c) {
          Vec Ftsc=LOAD(Ws+c);
          STORE(fm0s+c,LOAD(fm0s+c)+ZMULT(Xm,Ym,Ftsc));
          STORE(fHs+c,LOAD(fHs+c)+Ftsc);
        }
      });
    Complex *Zetaqr=Zetaqp+p2*r;
    PARALLELIF(
      (p2s1-1)*m*C > threshold,
      for(size_t t=1; t < p2s1; ++t) {
        size_t tm=t*Sm;
        Complex *Ft=W+tm;
        Complex *fm0t=fm0+tm;
        Complex *fHt=fH+tm;
        Vec Zeta=LOAD(Zetaqr+t);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(-Zeta,Zeta);
        Vec Zeta2=ZMULT(X,Y,Zetanr);
        Vec Xm=UNPACKL(Zeta2,Zeta2);
        Vec Ym=UNPACKH(Zeta2,-Zeta2);
        for(size_t s=0; s < m; ++s) {
          size_t Ss=S*s;
          Complex *Fts=Ft+Ss;
          Complex *fm0ts=fm0t+Ss;
          Complex *fHts=fHt+Ss;
          for(size_t c=0; c < C; ++c) {
            Vec Ftsc=LOAD(Fts+c);
            STORE(fm0ts+c,LOAD(fm0ts+c)+ZMULT(Xm,Ym,Ftsc));
            STORE(fHts+c,LOAD(fHts+c)+ZMULT(X,Y,Ftsc));
          }
        }
      });
    size_t p2s1Sm=S*p2s1m;
    Complex *Ft=W+p2s1Sm;
    Complex *fm0t=fm0+p2s1Sm;
    Complex *fHt=fH+p2s1Sm;
    Vec Zeta=LOAD(Zetaqr+p2s1);
    Vec X=UNPACKL(Zeta,Zeta);
    Vec Y=UNPACKH(-Zeta,Zeta);
    Vec Zeta2=ZMULT(X,Y,Zetanr);
    Xm=UNPACKL(Zeta2,Zeta2);
    Ym=UNPACKH(Zeta2,-Zeta2);
    PARALLELIF(
      m1*C > threshold,
      for(size_t s=0; s < m1; ++s) {
        size_t Ss=S*s;
        Complex *Fts=Ft+Ss;
        Complex *fm0ts=fm0t+Ss;
        Complex *fHts=fHt+Ss;
        for(size_t c=0; c < C; ++c) {
          Vec Ftsc=LOAD(Fts+c);
          STORE(fm0ts+c,LOAD(fm0ts+c)+ZMULT(Xm,Ym,Ftsc));
          STORE(fHts+c,LOAD(fHts+c)+ZMULT(X,Y,Ftsc));
        }
      });
    PARALLELIF(
      m*C > m1*C+threshold,
      for(size_t s=m1; s < m; ++s) {
        size_t Ss=S*s;
        Complex *Fts=Ft+Ss;
        Complex *fm0ts=fm0t+Ss;
        for(size_t c=0; c < C; ++c)
          STORE(fm0ts+c,LOAD(fm0ts+c)+ZMULT(Xm,Ym,LOAD(Fts+c)));
      });
  }
}

void fftPadHermitian::init()
{
  common();
  S=C; // Stride gaps are not implemented for Hermitian transforms
  e=m/2+1;
  size_t Ce=C*e;
  char const *FR;
  char const *BR;

  if(q == 1) {
    B=b=Ce;
    Complex *G=ComplexAlign(Ce);
    double *H=inplace ? (double *) G : doubleAlign(C*m);

    if(C == 1) {
      Forward=&fftBase::forwardExplicit;
      Backward=&fftBase::backwardExplicit;
      FR="forwardExplicit";
      BR="backwardExplicit";
      crfftm1=new crfft1d(m,G,H,threads);
      rcfftm1=new rcfft1d(m,H,G,threads);
    } else {
      Forward=&fftBase::forwardExplicitMany;
      Backward=&fftBase::backwardExplicitMany;
      FR="forwardExplicitMany";
      BR="backwardExplicitMany";
      crfftm=new mcrfft1d(m,C, C,C, 1,1, G,H,threads);
      rcfftm=new mrcfft1d(m,C, C,C, 1,1, H,G,threads);

      if(crfftm->Threads() > 1)
        threads=crfftm->Threads();
    }

    if(!inplace)
      deleteAlign(H);
    deleteAlign(G);
    dr=D0=R=1;
  } else {
    dr=Dr();

    double twopibyq=twopi/q;
    size_t p2=p/2;

    // Output block size
    b=align(ceilquotient(p2*Cm,2));
    // Work block size
    B=align(p2*Ce);

    if(inplace) b=B;

    Complex *G=ComplexAlign(B);
    double *H=inplace ? (double *) G :
      doubleAlign(C == 1 ? 2*B : Cm*p2);

    if(p > 2) { // p must be even, only C=1 implemented
      Forward=&fftBase::forwardInner;
      Backward=&fftBase::backwardInner;
      FR="forwardInner";
      BR="backwardInner";

      Zetaqp0=ComplexAlign((n-1)*p2);
      Zetaqp=Zetaqp0-p2-1;

      PARALLELIF(
        (n-1)*p2 > threshold,
        for(size_t r=1; r < n; ++r)
          for(size_t t=1; t <= p2; ++t)
            Zetaqp[p2*r+t]=expi(r*t*twopibyq);
        );

      fftp=new mfft1d(p2,1,Ce, Ce,1, G,G,threads);
      ifftp=new mfft1d(p2,-1,Ce, Ce,1, G,G,threads);
    } else { // p=2
      if(C == 1) {
        Forward=&fftBase::forward2;
        Backward=&fftBase::backward2;
        FR="forward2";
        BR="backward2";
      } else {
        Forward=&fftBase::forward2Many;
        Backward=&fftBase::backward2Many;
        FR="forward2Many";
        BR="backward2Many";
      }
    }

    R=residueBlocks();
    D0=n % D;
    if(D0 == 0) D0=D;

    if(C == 1) {
      crfftm=new mcrfft1d(m,p2, 1,1, e,m, G,H,threads);
      rcfftm=new mrcfft1d(m,p2, 1,1, m,e, H,G,threads);
    } else {
      size_t d=C*p2;
      crfftm=new mcrfft1d(m,d, C,C, 1,1, G,H,threads);
      rcfftm=new mrcfft1d(m,d, C,C, 1,1, H,G,threads);
    }

    if(crfftm->Threads() > 1)
      threads=crfftm->Threads();

    if(!inplace)
      deleteAlign(H);
    deleteAlign(G);

    initZetaqm(p == 2 ? q/2+1 : q,m);
  }
  if(showRoutines) {
    cout << endl << "Forwards Routine: " << "fftPadHermitian::" << FR << endl;
    cout << "Backwards Routine: " << "fftPadHermitian::" << BR << endl;
  }
}

fftPadHermitian::~fftPadHermitian()
{
  if(C == 1 && q == 1) {
    delete crfftm1;
    delete rcfftm1;
  } else {
    delete crfftm;
    delete rcfftm;
  }
  if(p > 2) {
    delete fftp;
    delete ifftp;
    deleteAlign(Zetaqp0);
  }
}

void fftPadHermitian::forwardExplicit(Complex *f, Complex *F, size_t, Complex *W)
{
  if(W == NULL) W=F;

  size_t H=ceilquotient(L,2);
  PARALLELIF(
    H > threshold,
    for(size_t s=0; s < H; ++s)
      W[s]=f[s];
    );
  PARALLELIF(
    e > H+threshold,
    for(size_t s=H; s < e; ++s)
      W[s]=0.0;
    );

  crfftm1->fft(W,F);
}

void fftPadHermitian::forwardExplicitMany(Complex *f, Complex *F, size_t,
                                          Complex *W)
{
  if(W == NULL) W=F;

  size_t H=ceilquotient(L,2);
  PARALLELIF(
    H*C > threshold,
    for(size_t s=0; s < H; ++s) {
      size_t Cs=C*s;
      Complex *Fs=W+Cs;
      Complex *fs=f+Cs;
      for(size_t c=0; c < C; ++c)
        Fs[c]=fs[c];
    });
  PARALLELIF(
    e*C > H*C+threshold,
    for(size_t s=H; s < e; ++s) {
      Complex *Fs=W+C*s;
      for(size_t c=0; c < C; ++c)
        Fs[c]=0.0;
    });

  crfftm->fft(W,F);
}

void fftPadHermitian::backwardExplicit(Complex *F, Complex *f, size_t,
                                       Complex *W)
{
  if(W == NULL) W=F;

  rcfftm1->fft(F,W);

  size_t H=ceilquotient(L,2);
  PARALLELIF(
    H > threshold,
    for(size_t s=0; s < H; ++s)
      f[s]=W[s];
    );
}

void fftPadHermitian::backwardExplicitMany(Complex *F, Complex *f,
                                           size_t, Complex *W)
{
  if(W == NULL) W=F;

  rcfftm->fft(F,W);
  size_t H=ceilquotient(L,2);
  PARALLELIF(
    H*C > threshold,
    for(size_t s=0; s < H; ++s) {
      size_t Cs=C*s;
      Complex *fs=f+Cs;
      Complex *Fs=W+Cs;
      for(size_t c=0; c < C; ++c)
        fs[c]=Fs[c];
    });
}

void fftPadHermitian::forward2(Complex *f, Complex *F, size_t r,
                               Complex *W)
{
  if(W == NULL) W=F;

  Complex *fm=f+m;
  size_t H=ceilquotient(L,2);
  size_t mH1=m-H+1;

  if(r == 0) {
    size_t q2=q/2;
    if(2*q2 < q) { // q odd, r=0
      PARALLELIF(
        mH1 > threshold,
        for(size_t s=0; s < mH1; ++s)
          W[s]=f[s];
        );
      PARALLELIF(
        e > mH1+threshold,
        for(size_t s=mH1; s < e; ++s)
          W[s]=f[s]+conj(*(fm-s));
        );

      crfftm->fft(W,F);
    } else { // q even, r=0,q/2
      Complex *G=F+b;
      Complex *V=W == F ? G : W+B;

      Complex *Zetar=Zetaqm+m*q2;

      V[0]=W[0]=f[0];

      PARALLELIF(
        mH1 > threshold,
        for(size_t s=1; s < mH1; ++s) {
          Complex fs=f[s];
          W[s]=fs;
          V[s]=Zetar[s]*fs;
        });

      PARALLELIF(
        e > mH1+threshold,
        for(size_t s=mH1; s < e; ++s) {
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
    PARALLELIF(
      mH1 > threshold,
      for(size_t s=1; s < mH1; ++s) {
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
    PARALLELIF(
      e > mH1+threshold,
      for(size_t s=mH1; s < e; ++s) {
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

void fftPadHermitian::forward2Many(Complex *f, Complex *F, size_t r,
                                   Complex *W)
{
  if(W == NULL) W=F;

  Complex *fm=f+Cm;
  size_t H=ceilquotient(L,2);
  size_t mH1=m-H+1;
  if(r == 0) {
    size_t q2=q/2;
    if(2*q2 < q) { // q odd
      PARALLELIF(
        mH1*C > threshold,
        for(size_t s=0; s < mH1; ++s) {
          size_t Cs=C*s;
          Complex *Ws=W+Cs;
          Complex *fs=f+Cs;
          for(size_t c=0; c < C; ++c)
            Ws[c]=fs[c];
        });
      PARALLELIF(
        e*C > mH1*C+threshold,
        for(size_t s=mH1; s < e; ++s) {
          size_t Cs=C*s;
          Complex *Ws=W+Cs;
          Complex *fs=f+Cs;
          Complex *fms=fm-Cs;
          for(size_t c=0; c < C; ++c)
            Ws[c]=fs[c]+conj(fms[c]);
        });

      crfftm->fft(W,F);
    } else { // q even, r=0,q/2
      Complex *G=F+b;
      Complex *V=W == F ? G : W+B;

      Complex *Zetar=Zetaqm+m*q2;

      PARALLELIF(
        C > threshold,
        for(size_t c=0; c < C; ++c)
          V[c]=W[c]=f[c];
        );

      PARALLELIF(
        mH1*C > threshold,
        for(size_t s=1; s < mH1; ++s) {
          size_t Cs=C*s;
          Complex *fs=f+Cs;
          Complex *Ws=W+Cs;
          Complex *Vs=V+Cs;
          Complex Zetars=Zetar[s];
          for(size_t c=0; c < C; ++c) {
            Complex fsc=fs[c];
            Ws[c]=fsc;
            Vs[c]=Zetars*fsc;
          }
        });

      PARALLELIF(
        e*C > mH1*C+threshold,
        for(size_t s=mH1; s < e; ++s) {
          size_t Cs=C*s;
          Complex *fs=f+Cs;
          Complex *fms=fm-Cs;
          Complex *Ws=W+Cs;
          Complex *Vs=V+Cs;
          Complex Zetars=Zetar[s];
          for(size_t c=0; c < C; ++c) {
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

    PARALLELIF(
      C > threshold,
      for(size_t c=0; c < C; ++c)
        V[c]=W[c]=f[c];
      );

    PARALLELIF(
      mH1*C > threshold,
      for(size_t s=1; s < mH1; ++s) {
        size_t Cs=C*s;
        Complex *Ws=W+Cs;
        Complex *Vs=V+Cs;
        Complex *fs=f+Cs;
        Vec Zeta=LOAD(Zetar+s);
        for(size_t c=0; c < C; ++c) {
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
    PARALLELIF(
      e*C > mH1*C+threshold,
      for(size_t s=mH1; s < e; ++s) {
        size_t Cs=C*s;
        Complex *Ws=W+Cs;
        Complex *Vs=V+Cs;
        Complex *fs=f+Cs;
        Complex *fms=fm-Cs;
        Vec Zeta=LOAD(Zetar+s);
        Vec Zetam=CONJ(LOAD(Zetarm-s));
        for(size_t c=0; c < C; ++c) {
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

void fftPadHermitian::backward2(Complex *F, Complex *f, size_t r,
                                Complex *W)
{
  if(W == NULL) W=F;

  bool even=m == 2*(e-1);

  rcfftm->fft(F,W);

  Complex *fm=f+m;
  size_t me=m-e;
  size_t H=ceilquotient(L,2);
  size_t mH1=m-H+1;
  if(r == 0) {
    size_t q2=q/2;
    if(2*q2 < q) { // q odd, r=0
      PARALLELIF(
        mH1 > threshold,
        for(size_t s=0; s < mH1; ++s)
          f[s]=W[s];
        );
      PARALLELIF(
        me > mH1+threshold,
        for(size_t s=mH1; s <= me; ++s) {
          Complex A=W[s];
          f[s]=A;
          *(fm-s)=conj(A);
        });
      if(even)
        f[e-1]=W[e-1];

    } else { // q even, r=0,q/2
      Complex *G=F+b;
      Complex *V=W == F ? G : W+B;

      rcfftm->fft(G,V);

      f[0]=W[0]+V[0];

      Complex *Zetar=Zetaqm+m*q2;

      PARALLELIF(
        mH1 > threshold,
        for(size_t s=1; s < mH1; ++s)
          f[s]=W[s]+conj(Zetar[s])*V[s];
        );

      PARALLELIF(
        me > mH1+threshold,
        for(size_t s=mH1; s <= me; ++s) {
          Complex A=W[s];
          Complex B=conj(Zetar[s])*V[s];
          f[s]=A+B;
          *(fm-s)=conj(A-B);
        });

      if(even)
        f[e-1]=W[e-1]-I*V[e-1];
    }
  } else {
    Complex *G=F+b;
    Complex *V=W == F ? G : W+B;

    rcfftm->fft(G,V);

    f[0] += W[0]+V[0];

    Complex *Zetar=Zetaqm+m*r;
    PARALLELIF(
      mH1 > threshold,
      for(size_t s=1; s < mH1; ++s)
//      f[s] += conj(Zeta)*W[s]+Zeta*V[s];
//      Vec Zeta=LOAD(Zetar+s);
        STORE(f+s,LOAD(f+s)+ZMULT2(LOAD(Zetar+s),LOAD(V+s),LOAD(W+s)));
      );

    Complex *Zetarm=Zetar+m;
    PARALLELIF(
      me > mH1+threshold,
      for(size_t s=mH1; s <= me; ++s) {
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
//      f[e-1] += conj(Zetar[e-1])*W[e-1]+Zetar[e-1]*V[e-1];
      STORE(f+e-1,LOAD(f+e-1)+ZMULT2(LOAD(Zetar+e-1),LOAD(V+e-1),LOAD(W+e-1)));
    }
  }
}

void fftPadHermitian::backward2Many(Complex *F, Complex *f, size_t r,
                                    Complex *W)
{
  if(W == NULL) W=F;

  bool even=m == 2*(e-1);

  rcfftm->fft(F,W);

  Complex *fm=f+Cm;
  size_t me=m-e;
  size_t H=ceilquotient(L,2);
  size_t mH1=m-H+1;
  size_t Ce1=C*(e-1);
  if(r == 0) {
    size_t q2=q/2;
    if(2*q2 < q) { // q odd, r=0
      PARALLELIF(
        mH1*C > threshold,
        for(size_t s=0; s < mH1; ++s) {
          size_t Cs=C*s;
          Complex *fs=f+Cs;
          Complex *Ws=W+Cs;
          for(size_t c=0; c < C; ++c)
            fs[c]=Ws[c];
        });
      PARALLELIF(
        (me+1)*C > mH1*C+threshold,
        for(size_t s=mH1; s <= me; ++s) {
          size_t Cs=C*s;
          Complex *fs=f+Cs;
          Complex *fms=fm-Cs;
          Complex *Ws=W+Cs;
          for(size_t c=0; c < C; ++c) {
            Complex A=Ws[c];
            fs[c]=A;
            fms[c]=conj(A);
          }
        });
      if(even) {
        Complex *fe=f+Ce1;
        Complex *We=W+Ce1;
        PARALLELIF(
          C > threshold,
          for(size_t c=0; c < C; ++c)
            fe[c]=We[c];
          );
      }
    } else { // q even, r=0,q/2
      Complex *G=F+b;
      Complex *V=W == F ? G : W+B;

      rcfftm->fft(G,V);

      PARALLELIF(
        C > threshold,
        for(size_t c=0; c < C; ++c)
          f[c]=W[c]+V[c];
        );

      Complex *Zetar=Zetaqm+m*q2;

      PARALLELIF(
        mH1*C > threshold,
        for(size_t s=1; s < mH1; ++s) {
          size_t Cs=C*s;
          Complex *fs=f+Cs;
          Complex *Ws=W+Cs;
          Complex *Vs=V+Cs;
          Complex Zetars=conj(Zetar[s]);
          for(size_t c=0; c < C; ++c)
            fs[c]=Ws[c]+Zetars*Vs[c];
        });

      PARALLELIF(
        (me+1)*C > mH1*C+threshold,
        for(size_t s=mH1; s <= me; ++s) {
          size_t Cs=C*s;
          Complex *fs=f+Cs;
          Complex *fms=fm-Cs;
          Complex *Ws=W+Cs;
          Complex *Vs=V+Cs;
          Complex Zetars=conj(Zetar[s]);
          for(size_t c=0; c < C; ++c) {
            Complex A=Ws[c];
            Complex B=Zetars*Vs[c];
            fs[c]=A+B;
            fms[c]=conj(A-B);
          }
        });

      if(even) {
        Complex *fe=f+Ce1;
        Complex *We=W+Ce1;
        Complex *Ve=V+Ce1;
        PARALLELIF(
          C > threshold,
          for(size_t c=0; c < C; ++c)
            fe[c]=We[c]-I*Ve[c];
          );
      }
    }
  } else {
    Complex *G=F+b;
    Complex *V=W == F ? G : W+B;

    Complex We[C];

    rcfftm->fft(G,V);

    PARALLELIF(
      C > threshold,
      for(size_t c=0; c < C; ++c)
        f[c] += W[c]+V[c];
      );

    Complex *Zetar=Zetaqm+m*r;
    PARALLELIF(
      mH1*C > threshold,
      for(size_t s=1; s < mH1; ++s) {
        size_t Cs=C*s;
        Complex *fs=f+Cs;
        Complex *Ws=W+Cs;
        Complex *Vs=V+Cs;
        Vec Zeta=LOAD(Zetar+s);
        for(size_t c=0; c < C; ++c)
//        fs[c] += conj(Zetars)*Ws[c]+Zetars*Vs[c];
          STORE(fs+c,LOAD(fs+c)+ZMULT2(Zeta,LOAD(Vs+c),LOAD(Ws+c)));
      });

    Complex *Zetarm=Zetar+m;
    PARALLELIF(
      (me+1)*C > mH1*C+threshold,
      for(size_t s=mH1; s <= me; ++s) {
        size_t Cs=C*s;
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
        for(size_t c=0; c < C; ++c) {
//        fs[c] += conj(Zeta)*Ws[c]+Zeta*Vs[c];
//        fms[c] += conj(Zetam*Ws[c])+Zetam*conj(Vs[c]);
          Vec Wsc=LOAD(Ws+c);
          Vec Vsc=LOAD(Vs+c);
          STORE(fs+c,LOAD(fs+c)+ZMULT2(X,Y,Vsc,Wsc));
          STORE(fms+c,LOAD(fms+c)+ZMULT2(Xm,Ym,Wsc,Vsc));
        }
      });
    if(even) {
      Complex *fe=f+Ce1;
      Complex *We=W+Ce1;
      Complex *Ve=V+Ce1;
      Vec Zeta=LOAD(Zetar+e-1);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(Zeta,-Zeta);
      PARALLELIF(
        C > threshold,
        for(size_t c=0; c < C; ++c)
//        fe[c] += conj(Zetare)*We[c]+Zetare*Ve[c];
          STORE(fe+c,LOAD(fe+c)+ZMULT2(X,Y,LOAD(Ve+c),LOAD(We+c)));
        );
    }
  }
}

void fftPadHermitian::forwardInner(Complex *f, Complex *F, size_t r,
                                   Complex *W)
{
  if(W == NULL) W=F;

  size_t p2=p/2;
  size_t H=ceilquotient(L,2);
  size_t p2m=p2*m;
  size_t p2mH=p2m-H;
  size_t m0=min(p2mH+1,e);
  Complex *fm=f+p2m;
  if(r == 0) {
    size_t n2=n/2;
    if(2*n2 < n) { // n odd, r=0
      PARALLELIF(
        m0 > threshold,
        for(size_t s=0; s < m0; ++s)
          W[s]=f[s];
        );
      PARALLELIF(
        e > m0+threshold,
        for(size_t s=m0; s < e; ++s)
          W[s]=conj(*(fm-s))+f[s];
        );
      PARALLELIF(
        (p2-1)*e > threshold,
        for(size_t t=1; t < p2; ++t) {
          size_t tm=t*m;
          size_t te=t*e;
          Complex *fmt=fm-tm;
          Complex *ft=f+tm;
          Complex *Wt=W+te;
          for(size_t s=0; s < e; ++s)
            Wt[s]=conj(*(fmt-s))+ft[s];
        });

      fftp->fft(W);

      size_t mn=m*n;
      PARALLELIF(
        (p2-1)*(e-1) > threshold,
        for(size_t u=1; u < p2; ++u) {
          Complex *Wu=W+u*e;
          Complex *Zeta0=Zetaqm+mn*u;
          for(size_t s=1; s < e; ++s)
            Wu[s] *= Zeta0[s];
        });
      crfftm->fft(W,F);
    } else { // n even, r=0,n/2
      Complex *V=W+B;
      PARALLELIF(
        m0 > threshold,
        for(size_t s=0; s < m0; ++s)
          V[s]=W[s]=f[s];
        );
      PARALLELIF(
        e > m0+threshold,
        for(size_t s=m0; s < e; ++s) {
          Complex fms=conj(*(fm-s));
          Complex fs=f[s];
          W[s]=fms+fs;
          V[s]=-fms+fs;
        });
      Complex *Zetaqn2=Zetaqp+p2*n2;
      PARALLELIF(
        (p2-1)*e > threshold,
        for(size_t t=1; t < p2; ++t) {
          size_t tm=t*m;
          size_t te=t*e;
          Complex *Wt=W+te;
          Complex *Vt=V+te;
          Complex *fmt=fm-tm;
          Complex *ft=f+tm;
          Vec Zeta=LOAD(Zetaqn2+t); //*zeta_q^{tn/2}
          Vec X=UNPACKL(Zeta,Zeta);
          Vec Y=UNPACKH(Zeta,-Zeta);
          for(size_t s=0; s < e; ++s) {
            Vec fmts=CONJ(LOAD(fmt-s));
            Vec fts=LOAD(ft+s);
            STORE(Wt+s,fmts+fts);
            STORE(Vt+s,ZMULT(X,Y,fts-fmts));
          }
        });

      fftp->fft(W);
      fftp->fft(V);

      size_t mn=m*n;
      size_t mn2=m*n2;
      Complex *Zetan2=Zetaqm+mn2;
      PARALLELIF(
        e > threshold,
        for(size_t s=0; s < e; ++s)
          V[s] *= Zetan2[s];
        );
      PARALLELIF(
        (p2-1)*2*(e-1) > threshold,
        for(size_t u=1; u < p2; ++u) {
          size_t ue=u*e;
          Complex *Wu=W+ue;
          Complex *Zeta0=Zetaqm+mn*u;
          for(size_t s=1; s < e; ++s)
            Wu[s] *= Zeta0[s];
          Complex *Vu=V+ue;
          Complex *Zetan2=Zeta0+mn2;
          for(size_t s=1; s < e; ++s)
            Vu[s] *= Zetan2[s];
        });


      crfftm->fft(W,F);
      crfftm->fft(V,F+b);
    }
  } else {
    Vec Zetanr=CONJ(LOAD(Zetaqp+p2*r+p2)); // zeta_n^{-r}
    Complex *G=F+b;
    Complex *V=W == F ? G : W+B;
    PARALLELIF(
      m0 > threshold,
      for(size_t s=0; s < m0; ++s)
        V[s]=W[s]=f[s];
      );
    PARALLELIF(
      e > m0+threshold,
      for(size_t s=m0; s < e; ++s) {
        Vec fms=CONJ(LOAD(fm-s));
        Vec fs=LOAD(f+s);
        Vec A=Zetanr*UNPACKL(fms,fms);
        Vec B=ZMULTI(Zetanr*UNPACKH(fms,fms));
        STORE(W+s,A+B+fs);
        STORE(V+s,CONJ(A-B)+fs);
      });
    Complex *Zetaqr=Zetaqp+p2*r;
    PARALLELIF(
      (p2-1)*e > threshold,
      for(size_t t=1; t < p2; ++t) {
        size_t tm=t*m;
        size_t te=t*e;
        Complex *Wt=W+te;
        Complex *Vt=V+te;
        Complex *fmt=fm-tm;
        Complex *ft=f+tm;
        Vec Zeta=LOAD(Zetaqr+t); // zeta_q^{tr}
        Vec Zetam=ZMULT(Zeta,Zetanr);
        for(size_t s=0; s < e; ++s) {
          Vec fmts=CONJ(LOAD(fmt-s));
          Vec fts=LOAD(ft+s);
          Vec A=Zetam*UNPACKL(fmts,fmts)+Zeta*UNPACKL(fts,fts);
          Vec B=ZMULTI(Zetam*UNPACKH(fmts,fmts)+Zeta*UNPACKH(fts,fts));
          STORE(Wt+s,A+B);
          STORE(Vt+s,CONJ(A-B));
        }
      });

    fftp->fft(W);
    fftp->fft(V);

    size_t mr=m*r;
    Complex *Zetar=Zetaqm+mr;
    PARALLELIF(
      e > threshold,
      for(size_t s=1; s < e; ++s) {
        Complex Zeta=Zetar[s];
        W[s] *= Zeta;
        V[s] *= conj(Zeta);
      });

    PARALLELIF(
      (p2-1)*2*(e-1) > threshold,
      for(size_t u=1; u < p2; ++u) {
        size_t mu=m*u;
        size_t eu=e*u;
        Complex *Zetar0=Zetaqm+n*mu;
        Complex *Zetar1=Zetar0+mr;
        Complex *Wu=W+eu;
        for(size_t s=1; s < e; ++s)
          Wu[s] *= Zetar1[s];
        Complex *Zetar2=Zetar0-mr;
        Complex *Vu=V+eu;
        for(size_t s=1; s < e; ++s)
          Vu[s] *= Zetar2[s];
      });
    crfftm->fft(W,F);
    crfftm->fft(V,G);
  }
}

void fftPadHermitian::backwardInner(Complex *F, Complex *f, size_t r,
                                    Complex *W)
{
  if(W == NULL) W=F;

  bool even=m == 2*(e-1);

  size_t p2=p/2;
  size_t H=ceilquotient(L,2);
  size_t p2m=p2*m;
  size_t p2mH=p2m-H;
  size_t me=m-e;
  size_t m0=min(p2mH+1,e);
  Complex *fm=f+p2m;

  if(r == 0) {
    size_t n2=n/2;
    if(2*n2 < n) { // n odd, r=0
      rcfftm->fft(F,W);
      PARALLELIF(
        (p2-1)*(e-1) > threshold,
        for(size_t u=1; u < p2; ++u) {
          size_t mu=m*u;
          size_t eu=u*e;
          Complex *Wu=W+eu;
          Complex *Zeta0=Zetaqm+n*mu;
          for(size_t s=1; s < e; ++s)
            Wu[s] *= conj(Zeta0[s]);
        });

      ifftp->fft(W);

      PARALLELIF(
        m0 > threshold,
        for(size_t s=0; s < m0; ++s)
          f[s]=W[s];
        );

      PARALLELIF(
        me > m0+threshold,
        for(size_t s=m0; s <= me; ++s) {
          Complex Ws=W[s];
          f[s]=Ws;
          *(fm-s)=conj(Ws);
        });

      PARALLELIF(
        (p2-1)*me > threshold,
        for(size_t t=1; t < p2; ++t) {
          size_t tm=t*m;
          size_t te=t*e;
          Complex *Wt=W+te;
          Complex *ft=f+tm;
          // s=0 case (this is important to avoid overlap)
          ft[0]=Wt[0];
          Complex *fmt=fm-tm;
          for(size_t s=1; s <= me; ++s) {
            Complex Wts=Wt[s];
            ft[s]=Wts;
            *(fmt-s)=conj(Wts);
          }
        });

      if(even) {
        Complex *fe1=f+e-1;
        Complex *We1=W+e-1;
        PARALLELIF(
          p2 > threshold,
          for(size_t t=0; t < p2; ++t)
            fe1[t*m]=We1[t*e];
          );
      }

    } else { // n even, r=0,n/2
      Complex *V=W+B;
      Complex *G=F+b;
      rcfftm->fft(F,W);
      rcfftm->fft(G,V);
      size_t mn=m*n;
      size_t mn2=m*n2;

      Complex *Zetan2=Zetaqm+mn2;
      PARALLELIF(
        e > threshold,
        for(size_t s=1; s < e; ++s)
          V[s] *= conj(Zetan2[s]);
        );

      PARALLELIF(
        (p2-1)*2*(e-1) > threshold,
        for(size_t u=1; u < p2; ++u) {
          size_t eu=u*e;
          Complex *Wu=W+eu;
          Complex *Zeta0=Zetaqm+mn*u;
          for(size_t s=1; s < e; ++s)
            Wu[s] *= conj(Zeta0[s]);
          Complex *Vu=V+eu;
          Complex *Zetan2=Zeta0+mn2;
          for(size_t s=1; s < e; ++s)
            Vu[s] *= conj(Zetan2[s]);
        });

      ifftp->fft(W);
      ifftp->fft(V);

      PARALLELIF(
        m0 > threshold,
        for(size_t s=0; s < m0; ++s)
          f[s]=W[s]+V[s];
        );

      PARALLELIF(
        me > m0+threshold,
        for(size_t s=m0; s <= me; ++s) {
          Complex Wts=W[s];
          Complex Vts=V[s];
          *(fm-s)=conj(Wts-Vts);
          f[s]=Wts+Vts;
        });

      Complex *Zetaqn2=Zetaqp+p2*n2;
      PARALLELIF(
        (p2-1)*me > threshold,
        for(size_t t=1; t < p2; ++t) {
          size_t tm=t*m;
          size_t te=t*e;
          Complex *Wt=W+te;
          Complex *Vt=V+te;
          Complex *ft=f+tm;
          Vec Zeta=LOAD(Zetaqn2+t);
          Vec X=UNPACKL(Zeta,Zeta);
          Vec Y=UNPACKH(-Zeta,Zeta);
          STORE(ft,LOAD(Wt)+ZMULT(X,Y,LOAD(Vt)));
          Complex *fmt=fm-tm;
          for(size_t s=1; s <= me; ++s) {
            Vec Wts=LOAD(Wt+s);
            Vec Vts=ZMULT(X,Y,LOAD(Vt+s));
            STORE(fmt-s,CONJ(Wts-Vts));
            STORE(ft+s,Wts+Vts);
          }
        });

      if(even) {
        size_t e1=e-1;
        Complex *fe1=f+e1;
        Complex *We1=W+e1;
        Complex *Ve1=V+e1;
        fe1[0]=We1[0]+Ve1[0];
        PARALLELIF(
          p2 > threshold,
          for(size_t t=1; t < p2; ++t)
            fe1[t*m]=We1[t*e]+conj(Zetaqn2[t])*Ve1[t*e];
          );
      }
    }
  } else { // r > 0
    Complex *V=W+B;
    Complex *G=F+b;
    rcfftm->fft(F,W);
    rcfftm->fft(G,V);
    Vec Zetanr=LOAD(Zetaqp+p2*r+p2); // zeta_n^r

    size_t mr=m*r;
    Complex *Zetar=Zetaqm+mr;
    PARALLELIF(
      e > threshold,
      for(size_t s=1; s < e; ++s) {
        Complex Zeta=Zetar[s];
        W[s] *= conj(Zeta);
        V[s] *= Zeta;
      });
    PARALLELIF(
      (p2-1)*2*(e-1) > threshold,
      for(size_t u=1; u < p2; ++u) {
        size_t eu=u*e;
        Complex *Zetar0=Zetaqm+n*m*u;
        Complex *Zetar1=Zetar0+mr;
        Complex *Wu=W+eu;
        for(size_t s=1; s < e; ++s)
          Wu[s] *= conj(Zetar1[s]);
        Complex *Zetar2=Zetar0-mr;
        Complex *Vu=V+eu;
        for(size_t s=1; s < e; ++s)
          Vu[s] *= conj(Zetar2[s]);
      });

    ifftp->fft(W);
    ifftp->fft(V);

    PARALLELIF(
      m0 > threshold,
      for(size_t s=0; s < m0; ++s)
        f[s] += W[s]+V[s];
      );

    Vec Xm=UNPACKL(Zetanr,Zetanr);
    Vec Ym=UNPACKH(Zetanr,-Zetanr);
    PARALLELIF(
      me > m0+threshold,
      for(size_t s=m0; s <= me; ++s) {
        Vec Wts=LOAD(W+s);
        Vec Vts=LOAD(V+s);
        STORE(fm-s,LOAD(fm-s)+CONJ(ZMULT2(Xm,Ym,Wts,Vts)));
        STORE(f+s,LOAD(f+s)+Wts+Vts);
      });

    Complex *Zetaqr=Zetaqp+p2*r;
    PARALLELIF(
      (p2-1)*me > threshold,
      for(size_t t=1; t < p2; ++t) {
        size_t tm=t*m;
        size_t te=t*e;
        Complex *Wt=W+te;
        Complex *Vt=V+te;
        Complex *ft=f+tm;
        Vec Zeta=LOAD(Zetaqr+t);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(Zeta,-Zeta);
        STORE(ft,LOAD(ft)+ZMULT2(X,Y,LOAD(Vt),LOAD(Wt)));
        Complex *fmt=fm-tm;
        Vec Zeta2=ZMULT(X,-Y,Zetanr); // zeta_q^(-rt)z_n^r
        Vec Xm=UNPACKL(Zeta2,Zeta2);
        Vec Ym=UNPACKH(Zeta2,-Zeta2);
        for(size_t s=1; s <= me; ++s) {
          Vec Wts=LOAD(Wt+s);
          Vec Vts=LOAD(Vt+s);
          STORE(fmt-s,LOAD(fmt-s)+CONJ(ZMULT2(Xm,Ym,Wts,Vts)));
          STORE(ft+s,LOAD(ft+s)+ZMULT2(X,Y,Vts,Wts));
        }
      });

    if(even) {
      Complex *fe1=f+e-1;
      Complex *We1=W+e-1;
      Complex *Ve1=V+e-1;
      fe1[0] += We1[0]+Ve1[0];
      PARALLELIF(
        p2 > threshold,
        for(size_t t=1; t < p2; ++t) {
          Vec Zeta=LOAD(Zetaqr+t);
          Vec X=UNPACKL(Zeta,Zeta);
          Vec Y=UNPACKH(Zeta,-Zeta);
          STORE(fe1+t*m,LOAD(fe1+t*m)+ZMULT2(X,Y,LOAD(Ve1+t*e),LOAD(We1+t*e)));
        });
    }
  }
}

void fftPadReal::init()
{
  common();
  e=m/2+1;
  size_t Se=S*e;
  char const *FR;
  char const *BR;

  if(q == 1) {
    l=e;
    b=S*l;
    Complex *G=ComplexAlign(Se);
    double *H=inplace ? (double *) G : doubleAlign(Sm);

    if(S == 1) {
      Forward=&fftBase::forwardExplicit;
      Backward=&fftBase::backwardExplicit;
      FR="forwardExplicit";
      BR="backwardExplicit";
      rcfftm1=new rcfft1d(m,H,G,threads);
      crfftm1=new crfft1d(m,G,H,threads);
    } else {
      Forward=&fftBase::forwardExplicitMany;
      Backward=&fftBase::backwardExplicitMany;
      FR="forwardExplicitMany";
      BR="backwardExplicitMany";
      rcfftm=new mrcfft1d(m,C, S,S, 1,1, H,G,threads);
      crfftm=new mcrfft1d(m,C, S,S, 1,1, G,H,threads);

      if(crfftm->Threads() > 1)
        threads=crfftm->Threads();
    }

    if(!inplace)
      deleteAlign(H);
    deleteAlign(G);
    dr=D0=R=n=1;
  } else {
    size_t P=p == 2 ? 1 : p;
    l=m*P;
    b=S*l;
    size_t d=C*D*P;

    Complex *G,*H;
    size_t size=b*D;

    G=ComplexAlign(size);
    H=inplace ? G : ComplexAlign(size);
    if(p <= 2) {
      rcfftm=new mrcfft1d(m,C, S,S, 1,1, (double *) H,G,threads);
      crfftm=new mcrfft1d(m,C, S,S, 1,1, G,(double *) H,threads);
    } else {
      if(S == 1)
        V=H;
      else
        V=inplace && S > C ? ComplexAlign(outputSize()) : H;
    }
    if (S == 1 && n > 2) {
      fftm=new mfft1d(m,1,d, 1,m, G,H,threads);
      ifftm=new mfft1d(m,-1,d, 1,m, H,G,threads);
    } else if (S > 1 && (n%2 == 0 || n > 2)) {
      fftm=new mfft1d(m,1,C, S,1, G,H,threads);
      ifftm=new mfft1d(m,-1,C, S,1, H,G,threads);
    }

    if(q % 2 == 0) {
      size_t h=e-1;
      if(p <= 2) {
        ffth=new mfft1d(h,1,C, S,S, 1,1, G,H,threads);
        iffth=new mfft1d(h,-1,C, S,S, 1,1, H,G,threads);
      } else {
        ffth=new mfft1d(h,1,C, C,S, 1,1, G,V,threads);
        iffth=new mfft1d(h,-1,C, S,C, 1,1, V,G,threads);
      }
    }

    if(p == 1) {
      if(S == 1) {
        Forward=&fftBase::forward1;
        Backward=&fftBase::backward1;
        FR="forward1";
        BR="backward1";
        if(repad())
          Pad=&fftBase::padSingle;
      } else {
        Forward=&fftBase::forward1Many;
        Backward=&fftBase::backward1Many;
        FR="forward1Many";
        BR="backward1Many";
        if(repad())
          Pad=&fftBase::padMany;
      }
    } else if(p == 2) {
      if(S == 1) {
        Forward=&fftBase::forward2;
        Backward=&fftBase::backward2;
        FR="forward2";
        BR="backward2";
      } else {
        Forward=&fftBase::forward2Many;
        Backward=&fftBase::backward2Many;
        FR="forward2Many";
        BR="backward2Many";
      }
      size_t Lm=L-m;
      double twopibyN=twopi/M;
      ZetaqmS0=ComplexAlign((q-1)*Lm);
      ZetaqmS=ZetaqmS0-L;
      for(size_t r=1; r < q; ++r)
        for(size_t s=m; s < L; ++s)
          ZetaqmS[Lm*r+s]=expi(r*s*twopibyN);
    } else { // p > 2
      if(S == 1) {
        Forward=&fftBase::forwardInner;
        Backward=&fftBase::backwardInner;
        FR="forwardInner";
        BR="backwardInner";
      } else {
        Forward=&fftBase::forwardInnerMany;
        Backward=&fftBase::backwardInnerMany;
        FR="forwardInnerMany";
        BR="backwardInnerMany";
      }

      double twopibyq=twopi/q;
      Zetaqp0=ComplexAlign((n-1)*(p-1));
      Zetaqp=Zetaqp0-p;
      for(size_t r=1; r < n; ++r)
        for(size_t t=1; t < p; ++t)
          Zetaqp[(p-1)*r+t]=expi(r*t*twopibyq);

      rcfftp=new mrcfft1d(p,Cm, Cm,Cm, 1,1, (double *) G,G,threads);
      crfftp=new mcrfft1d(p,Cm, Cm,Cm, 1,1, G,(double *) G,threads);

      size_t p2=ceilquotient(p,2);
      size_t Cp2=C*p2;
      if(S == 1) {
        fftmp2=new mfft1d(m,1,Cp2, 1,m, G,H,threads);
        ifftmp2=new mfft1d(m,-1,Cp2, 1,m, H,G,threads);
      } else {
        fftmr0=new mfft1d(m,1,C, C,S, 1,1, G,V,threads);
        ifftmr0=new mfft1d(m,-1,C, S,C, 1,1, V,G,threads);
        if(!inplace || S == C)
          V=NULL;
      }

      if(n > 2) {
        if(S == C) {
          fftp=new mfft1d(p,1,Cm, Sm,1, G,G,threads);
          ifftp=new mfft1d(p,-1,Cm, Sm,1, G,G,threads);
        } else {
          fftp=new mfft1d(p,1,C, Sm,1, G,G,threads);
          ifftp=new mfft1d(p,-1,C, Sm,1, G,G,threads);
        }
      }
      if(n%2 == 0) {
        if(S == C) {
          fftp2=new mfft1d(p2,1,Cm, Sm,1, G,G,threads);
          ifftp2=new mfft1d(p2,-1,Cm, Sm,1, G,G,threads);
        } else {
          fftp2=new mfft1d(p2,1,C, Sm,1, G,G,threads);
          ifftp2=new mfft1d(p2,-1,C, Sm,1, G,G,threads);
        }
      }
    }

    dr=Dr();
    R=residueBlocks();
    D0=(n-1)/2 % D;
    if(D0 == 0) D0=D;
    if(D0 != D) {
      size_t x=D0*P;
      fftm0=new mfft1d(m,1,x,1,m,G,H,threads);
      ifftm0=new mfft1d(m,-1,x,1,m,H,G,threads);
    } else
      fftm0=NULL;

    if(!inplace)
      deleteAlign(H);
    deleteAlign(G);

    initZetaqm(q,m);
  }

  if(showRoutines) {
    cout << endl << "Forwards Routine: " << "fftPadReal::" << FR << endl;
    cout << "Backwards Routine: " << "fftPadReal::" << BR << endl;
  }
}

fftPadReal::~fftPadReal()
{
  if(q == 1) {
    if(S == 1) {
      delete crfftm1;
      delete rcfftm1;
    } else {
      delete crfftm;
      delete rcfftm;
    }
  } else {

    if(fftm0) {
      delete fftm0;
      delete ifftm0;
    }

    if(p > 2) {
      if(S == 1) {
        delete fftmp2;
        delete ifftmp2;
      } else {
        delete fftmr0;
        delete ifftmr0;
      }
      delete rcfftp;
      delete crfftp;
      if(n > 2) {
        delete fftp;
        delete ifftp;
      }
      if(n%2 == 0) {
        delete fftp2;
        delete ifftp2;
      }
      if(inplace && S > C)
        deleteAlign(V);
    }

    if(p <= 2) {
      delete crfftm;
      delete rcfftm;
    }

    if(n > 2 || (S > 1 && n%2 == 0)) {
      delete fftm;
      delete ifftm;
    }

    if(q % 2 == 0) {
      delete ffth;
      delete iffth;
    }
  }
}

void fftPadReal::padSingle(Complex *W)
{
  // If q=2, W is allocated as half the size.
  size_t m=this->m;
  if(q == 2) m /= 2;
  size_t mp=m*p;
  for(size_t d=0; d < D; ++d) {
    Complex *F=W+m*d;
    PARALLELIF(
      mp > L+threshold,
      for(size_t s=L; s < mp; ++s)
        F[s]=0.0;
      );
  }
}

void fftPadReal::padMany(Complex *W)
{
  // If q=2, W is allocated as half the size.
  size_t m=this->m;
  if(q == 2) m /= 2;
  size_t mp=m*p;
  PARALLELIF(
    mp*C > L*C+threshold,
    for(size_t s=L; s < mp; ++s) {
      Complex *Ws=W+S*s;
      for(size_t c=0; c < C; ++c)
        Ws[c]=0.0;
    });
}

void fftPadReal::forwardExplicit(Complex *f, Complex *F, size_t, Complex *W)
{
  if(W == NULL) W=F;

  double *Wr=(double *) W;
  double *fr=(double *) f;
  PARALLELIF(
    L > threshold,
    for(size_t s=0; s < L; ++s)
      Wr[s]=fr[s];
    );
  PARALLELIF(
    m > L+threshold,
    for(size_t s=L; s < m; ++s)
      Wr[s]=0.0;
    );
  rcfftm1->fft(W,F);
}

void fftPadReal::forwardExplicitMany(Complex *f, Complex *F, size_t,
                                     Complex *W)
{
  if(W == NULL) W=F;

  double *Wr=(double *) W;
  double *fr=(double *) f;
  PARALLELIF(
    L*C > threshold,
    for(size_t s=0; s < L; ++s) {
      size_t Ss=S*s;
      double *Wrs=Wr+Ss;
      double *frs=fr+Ss;
      for(size_t c=0; c < C; ++c)
        Wrs[c]=frs[c];
    });
  PARALLELIF(
    m*C > L*C+threshold,
    for(size_t s=L; s < m; ++s) {
      double *Wrs=Wr+S*s;
      for(size_t c=0; c < C; ++c)
        Wrs[c]=0.0;
    });
  rcfftm->fft(W,F);
}

void fftPadReal::forward1(Complex *f, Complex *F, size_t r0, Complex *W)
{
  if(W == NULL) W=F;

  double *fr=(double *) f;

  if(r0 == 0) {
    double *Wr=(double *) W;
    bool repad=inplace || q == 2 || L%2;
    // Make use of existing zero padding, respecting minimal alignment
    if(!repad) Wr += L;
    PARALLELIF(
      L > threshold,
      for(size_t s=0; s < L; ++s)
        Wr[s]=fr[s];
      );
    if(repad) {
      PARALLELIF(
        m > L+threshold,
        for(size_t s=L; s < m; ++s)
          Wr[s]=0.0;
        );
    }
    rcfftm->fft(Wr,F);
  } else if(2*r0 < q) {
    bool remainder=r0 == 1 && D0 != D;
    size_t Dstop=remainder ? D0 : D;
    for(size_t d=0; d < Dstop; ++d) {
      size_t r=r0+d;
      Complex *U=W+m*d;
      U[0]=fr[0];
      Complex *Zetar=Zetaqm+m*r;
      PARALLELIF(
        L > threshold,
        for(size_t s=1; s < L; ++s)
          U[s]=Zetar[s]*fr[s];
        );
      if(inplace) {
        PARALLELIF(
          m > L+threshold,
          for(size_t s=L; s < m; ++s)
            U[s]=0.0;
          );
      }
    }
    remainder ? fftm0->fft(W,F) : fftm->fft(W,F);
  } else {
    size_t h=e-1;
    Complex *Zetar=Zetaqm+m*r0;
    double *frh=fr+h;
    size_t Lmh,stop;
    if(L <= h) {
      Lmh=0;
      stop=L;
      W[0]=fr[0];
    } else {
      Lmh=L-h;
      stop=h;
      W[0]=Complex(fr[0],frh[0]);
    }
    PARALLELIF(
      Lmh > threshold,
      for(size_t s=1; s < Lmh; ++s)
        W[s]=Zetar[s]*Complex(fr[s],frh[s]);
      );
    PARALLELIF(
      stop > Lmh+threshold,
      for(size_t s=Lmh; s < stop; ++s)
        W[s]=Zetar[s]*fr[s];
      );
    if(inplace) {
      PARALLELIF(
        h > L+threshold,
        for(size_t s=L; s < h; ++s)
          W[s]=0.0;
        );
    }
    ffth->fft(W,F);
  }
}

void fftPadReal::forward1Many(Complex *f, Complex *F, size_t r, Complex *W)
{
  if(W == NULL) W=F;

  double *fr=(double *) f;

  if(r == 0) {
    double *Wr=(double *) W;
    bool repad=inplace || q == 2 || L%2 || S > C;
    // Make use of existing zero padding, respecting minimal alignment
    if(!repad) Wr += L*S;
    PARALLELIF(
      L*C > threshold,
      for(size_t s=0; s < L; ++s) {
        size_t Ss=S*s;
        double *Wrs=Wr+Ss;
        double *frs=fr+Ss;
        for(size_t c=0; c < C; ++c)
          Wrs[c]=frs[c];
      });
    if(repad) {
      PARALLELIF(
        m*C > L*C+threshold,
        for(size_t s=L; s < m; ++s) {
          double *Wrs=Wr+S*s;
          for(size_t c=0; c < C; ++c)
            Wrs[c]=0.0;
        });
    }
    rcfftm->fft(Wr,F);
  } else if(2*r < q) {
    PARALLELIF(
      C > threshold,
      for(size_t c=0; c < C; ++c)
        W[c]=fr[c];
      );
    Complex *Zetar=Zetaqm+m*r;
    PARALLELIF(
      L*C > threshold,
      for(size_t s=1; s < L; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        double *frs=fr+Ss;
        Complex zeta=Zetar[s];
        for(size_t c=0; c < C; ++c) // TODO: Vectorize
          Ws[c]=zeta*frs[c];
      });
    if(inplace) {
      PARALLELIF(
        m*C > L*C+threshold,
        for(size_t s=L; s < m; ++s) {
          Complex *Ws=W+S*s;
          for(size_t c=0; c < C; ++c)
            Ws[c]=0.0;
        });
    }
    fftm->fft(W,F);
  } else {
    size_t h=e-1;
    Complex *Zetar=Zetaqm+m*r;
    double *frh=fr+S*h;
    size_t Lmh,stop;
    if(L <= h) {
      Lmh=0;
      stop=L;
      PARALLELIF(
        C > threshold,
        for(size_t c=0; c < C; ++c)
          W[c]=fr[c];
        );
    } else {
      Lmh=L-h;
      stop=h;
      PARALLELIF(
        C > threshold,
        for(size_t c=0; c < C; ++c)
          W[c]=Complex(fr[c],frh[c]);
        );
    }

    PARALLELIF(
      Lmh*C > C+threshold,
      for(size_t s=1; s < Lmh; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        double *frs=fr+Ss;
        double *frhs=frh+Ss;
        Complex zeta=Zetar[s];
        for(size_t c=0; c < C; ++c)
          Ws[c]=zeta*Complex(frs[c],frhs[c]);
      });
    PARALLELIF(
      stop*C > Lmh*C+threshold,
      for(size_t s=Lmh; s < stop; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        double *frs=fr+Ss;
        Complex zeta=Zetar[s];
        for(size_t c=0; c < C; ++c)
          Ws[c]=zeta*frs[c];
      });
    if(inplace) {
      PARALLELIF(
        h*C > L*C+threshold,
        for(size_t s=L; s < h; ++s) {
          Complex *Ws=W+S*s;
          for(size_t c=0; c < C; ++c)
            Ws[c]=0.0;
        });
    }
    ffth->fft(W,F);
  }
}

void fftPadReal::forward2(Complex *f, Complex *F, size_t r0, Complex *W)
{
  if(W == NULL) W=F;

  double *fr=(double *) f;
  double *frm=fr+m;

  if(r0 == 0) {
    size_t Lm=L-m;
    double *Wr=(double *) W;
    PARALLELIF(
      Lm > threshold,
      for(size_t s=0; s < Lm; ++s)
        Wr[s]=fr[s]+frm[s];
      );
    PARALLELIF(
      m > Lm+threshold,
      for(size_t s=Lm; s < m; ++s)
        Wr[s]=fr[s];
      );
    rcfftm->fft(Wr,F);
  } else if(2*r0 < q) {
    size_t Lm=L-m;
    bool remainder=r0 == 1 && D0 != D;
    size_t Dstop=remainder ? D0 : D;
    for(size_t d=0; d < Dstop; ++d) {
      size_t r=r0+d;
      Complex *U=W+m*d;
      Complex *Zetar2=ZetaqmS+Lm*r+m;
      U[0]=fr[0]+Zetar2[0]*frm[0];
      Complex *Zetar=Zetaqm+m*r;
      PARALLELIF(
        Lm > threshold,
        for(size_t s=1; s < Lm; ++s)
          U[s]=Zetar[s]*fr[s]+Zetar2[s]*frm[s];
        );
      PARALLELIF(
        m > Lm+threshold,
        for(size_t s=Lm; s < m; ++s)
          U[s]=Zetar[s]*fr[s];
        );
    }
    remainder ? fftm0->fft(W,F) : fftm->fft(W,F);
  } else {
    size_t h=e-1;
    Complex *Zetar=Zetaqm+m*r0;
    double *frh=fr+h;
    double *frhm=frh+m;
    size_t mh=m+h;
    size_t stop1, stop2;
    if(mh < L) {
      stop1=L-mh;
      stop2=h;
      W[0]=Complex(fr[0]-frm[0],frh[0]-frhm[0]);
    } else {
      stop1=1;
      stop2=L-m;
      W[0]=Complex(fr[0]-frm[0],frh[0]);
    }
    PARALLELIF(
      stop1 > threshold,
      for(size_t s=1; s < stop1; ++s)
        W[s]=Zetar[s]*Complex(fr[s]-frm[s],frh[s]-frhm[s]);
      );
    PARALLELIF(
      stop2 > stop1+threshold,
      for(size_t s=stop1; s < stop2; ++s)
        W[s]=Zetar[s]*Complex(fr[s]-frm[s],frh[s]);
      );
    PARALLELIF(
      h > stop2+threshold,
      for(size_t s=stop2; s < h; ++s)
        W[s]=Zetar[s]*Complex(fr[s],frh[s]);
      );
    ffth->fft(W,F);
  }
}

void fftPadReal::forward2Many(Complex *f, Complex *F, size_t r, Complex *W)
{
  if(W == NULL) W=F;

  double *fr=(double *) f;
  double *frm=fr+Sm;

  if(r == 0) {
    size_t Lm=L-m;
    double *Wr=(double *) W;
    PARALLELIF(
      Lm*C > threshold,
      for(size_t s=0; s < Lm; ++s) {
        size_t Ss=S*s;
        double *Wrs=Wr+Ss;
        double *frs=fr+Ss;
        double *frms=frm+Ss;
        for(size_t c=0; c < C; ++c)
          Wrs[c]=frs[c]+frms[c];
      });
    PARALLELIF(
      m*C > Lm*C+threshold,
      for(size_t s=Lm; s < m; ++s) {
        size_t Ss=S*s;
        double *Wrs=Wr+Ss;
        double *frs=fr+Ss;
        for(size_t c=0; c < C; ++c)
          Wrs[c]=frs[c];
      });
    rcfftm->fft(Wr,F);
  } else if(2*r < q) {
    size_t Lm=L-m;
    Complex *Zetar2=ZetaqmS+Lm*r+m;
    Complex zeta2=Zetar2[0];
    for(size_t c=0; c < C; ++c)
      W[c]=fr[c]+zeta2*frm[c];
    Complex *Zetar=Zetaqm+m*r;
    PARALLELIF(
      Lm*C > C+threshold,
      for(size_t s=1; s < Lm; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        double *frs=fr+Ss;
        double *frms=frm+Ss;
        Complex zeta=Zetar[s];
        Complex zeta2=Zetar2[s];
        for(size_t c=0; c < C; ++c)
          Ws[c]=zeta*frs[c]+zeta2*frms[c];
      });
    PARALLELIF(
      m*C > Lm*C+threshold,
      for(size_t s=Lm; s < m; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        double *frs=fr+Ss;
        Complex zeta=Zetar[s]; // TODO: Vectorize
        for(size_t c=0; c < C; ++c)
          Ws[c]=zeta*frs[c];
      });
    fftm->fft(W,F);
  } else {
    size_t h=e-1;
    Complex *Zetar=Zetaqm+m*r;
    double *frh=fr+S*h;
    double *frhm=frh+Sm;
    size_t mh=m+h;
    size_t stop1, stop2;
    if(mh < L) {
      stop1=L-mh;
      stop2=h;
      PARALLELIF(
        C > threshold,
        for(size_t c=0; c < C; ++c)
          W[c]=Complex(fr[c]-frm[c],frh[c]-frhm[c]);
        );
    } else {
      stop1=1;
      stop2=L-m;
      PARALLELIF(
        C > threshold,
        for(size_t c=0; c < C; ++c)
          W[c]=Complex(fr[c]-frm[c],frh[c]);
        );
    }
    PARALLELIF(
      stop1 > threshold,
      for(size_t s=1; s < stop1; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        double *frs=fr+Ss;
        double *frms=frm+Ss;
        double *frhs=frh+Ss;
        double *frhms=frhm+Ss;
        Complex zeta=Zetar[s]; // TODO: Vectorize
        for(size_t c=0; c < C; ++c)
          Ws[c]=zeta*Complex(frs[c]-frms[c],frhs[c]-frhms[c]);
      });
    PARALLELIF(
      stop2 > stop1+threshold,
      for(size_t s=stop1; s < stop2; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        double *frs=fr+Ss;
        double *frms=frm+Ss;
        double *frhs=frh+Ss;
        Complex zeta=Zetar[s]; // TODO: Vectorize
        for(size_t c=0; c < C; ++c)
          Ws[c]=zeta*Complex(frs[c]-frms[c],frhs[c]);
      });
    PARALLELIF(
      h > stop2+threshold,
      for(size_t s=stop2; s < h; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        double *frs=fr+Ss;
        double *frhs=frh+Ss;
        Complex zeta=Zetar[s]; // TODO: Vectorize
        for(size_t c=0; c < C; ++c)
          Ws[c]=zeta*Complex(frs[c],frhs[c]);
      });
    ffth->fft(W,F);
  }
}

void fftPadReal::forwardInner(Complex *f, Complex *F0, size_t r0, Complex *W)
{
  if(W == NULL) W=F0;

  double *fr=(double *) f;

  size_t n2=ceilquotient(n,2);
  size_t pm1=p-1;
  size_t mpm1=m*pm1;
  size_t stop=L-mpm1;

  if(r0 == 0) {
    double *Wr=(double *) W;
    size_t p2=ceilquotient(p,2);
    size_t p2m=p2*m;
    PARALLELIF(
      pm1*m > threshold,
      for(size_t t=0; t < pm1; ++t) {
        size_t mt=m*t;
        double *Ft=Wr+mt;
        double *ft=fr+mt;
        for(size_t s=0; s < m; ++s)
          Ft[s]=ft[s];
      });

    double *Ft=Wr+mpm1;
    double *ft=fr+mpm1;
    PARALLELIF(
      stop > threshold,
      for(size_t s=0; s < stop; ++s)
        Ft[s]=ft[s];
      );
    PARALLELIF(
      m > stop+threshold,
      for(size_t s=stop; s < m; ++s)
        Ft[s]=0.0;
      );

    rcfftp->fft(W);

    PARALLELIF(
      (p2-1)*(m-1) > threshold,
      for(size_t t=1; t < p2; ++t) {
        size_t R=n*t;
        Complex *Ft=W+m*t;
        Complex *Zetar=Zetaqm+m*R;
        for(size_t s=1; s < m; ++s)
          Ft[s] *= conj(Zetar[s]); // Compensate for conjugate transform sign.
      });

    fftmp2->fft(W,F0);

    if(p == 2*p2) {
      size_t h=e-1;
      Complex *Zetar=Zetaqm+m*n*p2;
      Complex *Ft=W+p2m;
      double *Wt=Wr+2*p2m;
      double *Wth=Wt+2*h;

      Ft[0]=Complex(Wt[0],Wth[0]);
      PARALLELIF(
        h > threshold,
        for(size_t s=1; s < h; ++s) {
          size_t s2=2*s;
          Ft[s]=Zetar[s]*Complex(Wt[s2],Wth[s2]);
        });
      ffth->fft(Ft,F0+p2m);
    }

  } else if (r0 < n2) {
    bool remainder=r0 == 1 && D0 != D;
    size_t Dstop=remainder ? D0 : D;
    for(size_t d=0; d < Dstop; ++d) {
      Complex *F=W+b*d;
      size_t r=r0+d;
      PARALLELIF(
        m > threshold,
        for(size_t s=0; s < m; ++s)
          F[s]=fr[s];
        );
      Complex *Zetaqr=Zetaqp+pm1*r;
      PARALLELIF(
        pm1*m > m+threshold,
        for(size_t t=1; t < pm1; ++t) {
          size_t mt=m*t;
          Complex *Ft=F+mt;
          double *ft=fr+mt;
          Complex Zeta=Zetaqr[t];
          for(size_t s=0; s < m; ++s)
            Ft[s]=Zeta*ft[s];
        });
      Complex *Ft=F+mpm1;
      double *ft=fr+mpm1;
      Complex Zeta=Zetaqr[pm1];
      PARALLELIF(
        stop > threshold,
        for(size_t s=0; s < stop; ++s)
          Ft[s]=Zeta*ft[s];
        );
      PARALLELIF(
        m > stop+threshold,
        for(size_t s=stop; s < m; ++s)
          Ft[s]=0.0;
        );

      fftp->fft(F);
      PARALLELIF(
        p*(m-1) > threshold,
        for(size_t t=0; t < p; ++t) {
          size_t R=n*t+r;
          Complex *Ft=F+m*t;
          Complex *Zetar=Zetaqm+m*R;
          for(size_t s=1; s < m; ++s)
            Ft[s] *= Zetar[s];
        });
    }
    remainder ? fftm0->fft(W,F0) : fftm->fft(W,F0);
  } else { // 2*r0 == n
    size_t p2=ceilquotient(p,2);
    size_t p2m=p2*m;
    size_t p2m1=p2-1;
    Complex *Zetaqr=Zetaqp+pm1*r0;

    Complex *Ft=W;
    double *ft=fr;
    double *ftp2=ft+p2m;

    PARALLELIF(
      m > threshold,
      for(size_t s=0; s < m; ++s)
        Ft[s]=Complex(ft[s],ftp2[s]);
      );

    PARALLELIF(
      (p2m1-1)*m > threshold,
      for(size_t t=1; t < p2m1; ++t) {
        size_t mt=m*t;
        Complex *Ft=W+mt;
        double *ft=fr+mt;
        double *ftp2=ft+p2m;
        Complex Zeta=Zetaqr[t];
        for(size_t s=0; s < m; ++s)
          Ft[s]=Zeta*Complex(ft[s],ftp2[s]);
      });

    size_t mp2m1=m*p2m1;
    Ft=W+mp2m1;
    ft=fr+mp2m1;
    ftp2=ft+p2m;
    Complex Zeta=Zetaqr[p2m1];

    PARALLELIF(
      stop > threshold,
      for(size_t s=0; s < stop; ++s)
        Ft[s]=Zeta*Complex(ft[s],ftp2[s]);
      );

    PARALLELIF(
      m > stop+threshold,
      for(size_t s=stop; s < m; ++s)
        Ft[s]=Zeta*ft[s];
      );

    fftp2->fft(W);

    PARALLELIF(
      p2*(m-1) > threshold,
      for(size_t t=0; t < p2; ++t) {
        size_t mt=m*t;
        Complex *Ft=W+mt;
        Complex *Zetar=Zetaqm+m*(2*n*t+r0);
        for(size_t s=1; s < m; ++s)
          Ft[s] *= Zetar[s];
      });

    fftmp2->fft(W,F0);
  }
}

void fftPadReal::forwardInnerMany(Complex *f, Complex *F, size_t r, Complex *W)
{
  if(W == NULL) W=r == 0 && V ? V : F;

  double *fr=(double *) f;

  size_t n2=ceilquotient(n,2);
  size_t pm1=p-1;

  size_t stop=L-m*pm1;
  size_t Smpm1=Sm*pm1;

  if(r == 0) {
    double *Wr=(double *) W;
    size_t p2=ceilquotient(p,2);
    PARALLELIF(
      pm1*m*C > threshold,
      for(size_t t=0; t < pm1; ++t) {
        double *Wt=Wr+Cm*t;
        double *ft=fr+Sm*t;
        for(size_t s=0; s < m; ++s) {
          double *Wts=Wt+C*s;
          double *fts=ft+S*s;
          for(size_t c=0; c < C; ++c)
            Wts[c]=fts[c];
        }
      });

    double *Wt=Wr+Cm*pm1;
    double *ft=fr+Smpm1;
    PARALLELIF(
      stop*C > threshold,
      for(size_t s=0; s < stop; ++s) {
        double *Wts=Wt+C*s;
        double *fts=ft+S*s;
        for(size_t c=0; c < C; ++c)
          Wts[c]=fts[c];
      });
    PARALLELIF(
      m*C > stop*C+threshold,
      for(size_t s=stop; s < m; ++s) {
        double *Wts=Wt+C*s;
        for(size_t c=0; c < C; ++c)
          Wts[c]=0.0;
      });

    rcfftp->fft(Wr);

    PARALLELIF(
      (p2-1)*(m-1)*C > threshold,
      for(size_t t=1; t < p2; ++t) {
        size_t R=n*t;
        Complex *Wt=W+Cm*t;
        Complex *Zetar=Zetaqm+m*R;
        for(size_t s=1; s < m; ++s) {
          Complex *Wts=Wt+C*s;
          Complex zeta=conj(Zetar[s]);  // Compensate for conjugate transform sign.
          for(size_t c=0; c < C; ++c)
            Wts[c] *= zeta;
        }
      });

    for(size_t t=0; t < p2; ++t)
      fftmr0->fft(W+Cm*t,F+Sm*t);

    if(p == 2*p2) {
      size_t h=e-1;
      Complex *Zetar=Zetaqm+m*n*p2;
      size_t p2Cm=p2*Cm;
      Complex *Ft=W+p2Cm;
      double *Wt=Wr+2*p2Cm;
      double *Wth=Wt+2*C*h;

      PARALLELIF(
        C > threshold,
        for(size_t c=0; c < C; ++c) {
          size_t c2=2*c;
          Ft[c]=Complex(Wt[c2],Wth[c2]);
        });
      PARALLELIF(
        h*C > C+threshold,
        for(size_t s=1; s < h; ++s) {
          double *Wts2=Wt+2*C*s;
          double *Wths2=Wth+2*C*s;
          Complex *Fts=Ft+C*s;
          Complex zeta=Zetar[s];
          for(size_t c=0; c < C; ++c) {
            size_t c2=2*c;
            Fts[c]=zeta*Complex(Wts2[c2],Wths2[c2]);
          }
        });
      ffth->fft(Ft,F+p2*Sm);
    }

  } else if (r < n2) {
    PARALLELIF(
      m*C > threshold,
      for(size_t s=0; s < m; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        double *frs=fr+Ss;
        for(size_t c=0; c < C; ++c)
          Ws[c]=frs[c];
      });
    Complex *Zetaqr=Zetaqp+pm1*r;
    PARALLELIF(
      pm1*m*C > m*C+threshold,
      for(size_t t=1; t < pm1; ++t) {
        size_t Smt=Sm*t;
        Complex *Wt=W+Smt;
        double *ft=fr+Smt;
        Complex Zeta=Zetaqr[t];
        for(size_t s=0; s < m; ++s) {
          size_t Ss=S*s;
          Complex *Wts=Wt+Ss;
          double *fts=ft+Ss;
          for(size_t c=0; c < C; ++c)
            Wts[c]=Zeta*fts[c];
        }
      });
    Complex *Wt=W+Smpm1;
    double *ft=fr+Smpm1;
    Complex Zeta=Zetaqr[pm1];
    PARALLELIF(
      stop > threshold,
      for(size_t s=0; s < stop; ++s) {
        size_t Ss=S*s;
        Complex *Wts=Wt+Ss;
        double *fts=ft+Ss;
        for(size_t c=0; c < C; ++c) // TODO: Vectorize
          Wts[c]=Zeta*fts[c];
      });
    PARALLELIF(
      m*C > stop*C+threshold,
      for(size_t s=stop; s < m; ++s) {
        Complex *Wts=Wt+S*s;
        for(size_t c=0; c < C; ++c)
          Wts[c]=0.0;
      });

    if(S == C)
      fftp->fft(W);
    else
      for(size_t s=0; s < m; ++s)
        fftp->fft(W+S*s);

    PARALLELIF(
      p*(m-1)*C > threshold,
      for(size_t t=0; t < p; ++t) {
        size_t R=n*t+r;
        Complex *Wt=W+Sm*t;
        Complex *Zetar=Zetaqm+m*R;
        for(size_t s=1; s < m; ++s) {
          Complex zeta=Zetar[s]; // TODO: Vectorize
          Complex *Wts=Wt+S*s;
          for(size_t c=0; c < C; ++c)
            Wts[c] *= zeta;
        }
      });
    for(size_t t=0; t < p; ++t) {
      size_t Smt=Sm*t;
      fftm->fft(W+Smt,F+Smt);
    }
  } else { // 2*r == n
    size_t p2=ceilquotient(p,2);
    size_t p2m1=p2-1;
    size_t p2Sm=p2*Sm;
    Complex *Zetaqr=Zetaqp+pm1*r;

    double *frp2=fr+p2Sm;

    PARALLELIF(
      m*C > threshold,
      for(size_t s=0; s < m; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        double *frs=fr+Ss;
        double *frp2s=frp2+Ss;
        for(size_t c=0; c < C; ++c)
          Ws[c]=Complex(frs[c],frp2s[c]);
      });

    PARALLELIF(
      p2m1*m*C > m*C+threshold,
      for(size_t t=1; t < p2m1; ++t) {
        size_t Smt=Sm*t;
        Complex *Wt=W+Smt;
        double *ft=fr+Smt;
        double *ftp2=ft+p2Sm;
        Complex Zeta=Zetaqr[t];
        for(size_t s=0; s < m; ++s) {
          size_t Ss=S*s;
          Complex *Wts=Wt+Ss;
          double *fts=ft+Ss;
          double *ftp2s=ftp2+Ss;
          for(size_t c=0; c < C; ++c)
            Wts[c]=Zeta*Complex(fts[c],ftp2s[c]);
        }
      });

    size_t Smp2m1=Sm*p2m1;
    Complex *Wt=W+Smp2m1;
    double *ft=fr+Smp2m1;
    double *ftp2=ft+p2Sm;
    Complex Zeta=Zetaqr[p2m1];

    PARALLELIF(
      stop*C > threshold,
      for(size_t s=0; s < stop; ++s) {
        size_t Ss=S*s;
        Complex *Wts=Wt+Ss;
        double *fts=ft+Ss;
        double *ftp2s=ftp2+Ss;
        for(size_t c=0; c < C; ++c)
          Wts[c]=Zeta*Complex(fts[c],ftp2s[c]); // TODO: Optimize
      });

    PARALLELIF(
      m*C > stop*C+threshold,
      for(size_t s=stop; s < m; ++s) {
        size_t Ss=S*s;
        Complex *Wts=Wt+Ss;
        double *fts=ft+Ss;
        for(size_t c=0; c < C; ++c)
          Wts[c]=Zeta*fts[c];
      });

    if(S == C)
      fftp2->fft(W);
    else
      for(size_t s=0; s < m; ++s)
        fftp2->fft(W+S*s);

    PARALLELIF(
      p2*(m-1)*C > threshold,
      for(size_t t=0; t < p2; ++t) {
        size_t Smt=Sm*t;
        Complex *Wt=W+Smt;
        Complex *Zetar=Zetaqm+m*(2*n*t+r);
        for(size_t s=1; s < m; ++s) {
          Complex zeta=Zetar[s];
          size_t Ss=S*s;
          Complex *Wts=Wt+Ss;
          for(size_t c=0; c < C; ++c)
            Wts[c] *= zeta;
        }
      });

    for(size_t t=0; t < p2; ++t) {
      size_t Smt=Sm*t;
      fftm->fft(W+Smt,F+Smt);
    }
  }
}

void fftPadReal::backwardExplicit(Complex *F, Complex *f, size_t, Complex *W)
{
  if(W == NULL) W=F;

  crfftm1->fft(F,W);

  double *fr=(double *) f;
  double *Wr=(double *) W;
  PARALLELIF(
    L > threshold,
    for(size_t s=0; s < L; ++s)
      fr[s]=Wr[s];
    );
}

void fftPadReal::backwardExplicitMany(Complex *F, Complex *f, size_t, Complex *W)
{
  if(W == NULL) W=F;

  crfftm->fft(F,W);

  double *fr=(double *) f;
  double *Wr=(double *) W;
  PARALLELIF(
    L*C > threshold,
    for(size_t s=0; s < L; ++s) {
      size_t Ss=S*s;
      double *frs=fr+Ss;
      double *Wrs=Wr+Ss;
      for(size_t c=0; c < C; ++c)
        frs[c]=Wrs[c];
    });
}

void fftPadReal::backward1(Complex *F, Complex *f, size_t r0, Complex *W)
{
  if(W == NULL) W=F;
  double *fr=(double *) f;
  double *Wr=(double *) W;

  if(r0 == 0) {
    crfftm->fft(F,Wr);
    PARALLELIF(
      L > threshold,
      for(size_t s=0; s < L; ++s)
        fr[s]=Wr[s];
      );
  } else if(2*r0 < q) {
    bool remainder=r0 == 1 && D0 != D;
    size_t Dstop;
    if(remainder) {
      ifftm0->fft(F,W);
      Dstop=D0;
    } else {
      ifftm->fft(F,W);
      Dstop=D;
    }
    for(size_t d=0; d < Dstop; ++d) {
      size_t r=r0+d;
      Complex *U=W+m*d;
      fr[0] += 2.0*U[0].re;
      Complex *Zetar=Zetaqm+m*r;
      PARALLELIF(
        L > threshold,
        for(size_t s=1; s < L; ++s)
          fr[s] += 2.0*realProduct(Zetar[s],U[s]);
        );
    }
  } else {
    iffth->fft(F,W);
    Complex *Zetar=Zetaqm+m*r0;
    size_t h=e-1;
    double *frh=fr+h;
    size_t Lmh,stop;
    Complex z=W[0];
    fr[0] += 2.0*z.re;
    if(L <= h) {
      Lmh=1;
      stop=L;
    } else {
      Lmh=L-h;
      stop=h;
      frh[0] += 2.0*z.im;
    }
    PARALLELIF(
      Lmh > threshold,
      for(size_t s=1; s < Lmh; ++s) {
        Complex z=2.0*conj(Zetar[s])*W[s];
        fr[s] += z.re;
        frh[s] += z.im;
      });

    PARALLELIF(
      stop > Lmh+threshold,
      for(size_t s=Lmh; s < stop; ++s) {
        fr[s] += 2.0*realProduct(Zetar[s],W[s]);
      });
  }
}

void fftPadReal::backward1Many(Complex *F, Complex *f, size_t r, Complex *W)
{
  if(W == NULL) W=F;
  double *fr=(double *) f;
  double *Wr=(double *) W;

  if(r == 0) {
    crfftm->fft(F,Wr);
    PARALLELIF(
      L*C > threshold,
      for(size_t s=0; s < L; ++s) {
        size_t Ss=S*s;
        double *Wrs=Wr+Ss;
        double *frs=fr+Ss;
        for(size_t c=0; c < C; ++c)
          frs[c]=Wrs[c];
      });
  } else if(2*r < q) {
    ifftm->fft(F,W);
    PARALLELIF(
      C > threshold,
      for(size_t c=0; c < C; ++c)
        fr[c] += 2.0*W[c].re;
      );
    Complex *Zetar=Zetaqm+m*r;
    PARALLELIF(
      L*C > threshold,
      for(size_t s=1; s < L; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        double *frs=fr+Ss;
        Complex zeta=2.0*Zetar[s];
        for(size_t c=0; c < C; ++c) // TODO: Vectorize
          frs[c] += realProduct(zeta,Ws[c]);
      });
  } else {
    iffth->fft(F,W);
    Complex *Zetar=Zetaqm+m*r;
    size_t h=e-1;
    double *frh=fr+S*h;
    size_t Lmh,stop;
    if(L <= h) {
      PARALLELIF(
        C > threshold,
        for(size_t c=0; c < C; ++c)
          fr[c] += 2.0*W[c].re;
        );
      Lmh=1;
      stop=L;
    } else {
      Lmh=L-h;
      stop=h;
      PARALLELIF(
        C > threshold,
        for(size_t c=0; c < C; ++c) {
          Complex z=W[c];
          fr[c] += 2.0*z.re;
          frh[c] += 2.0*z.im;
        });
    }

    PARALLELIF(
      Lmh*C > C+threshold,
      for(size_t s=1; s < Lmh; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        double *frs=fr+Ss;
        double *frhs=frh+Ss;
        Complex zeta=2.0*conj(Zetar[s]);
        for(size_t c=0; c < C; ++c) {
          Complex z=zeta*Ws[c]; // Optimize
          frs[c] += z.re;
          frhs[c] += z.im;
        }
      });
    PARALLELIF(
      stop*C > Lmh*C+threshold,
      for(size_t s=Lmh; s < stop; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        double *frs=fr+Ss;
        Complex zeta=2.0*Zetar[s];
        for(size_t c=0; c < C; ++c)
          frs[c] += realProduct(zeta,Ws[c]); // TODO Vectorize
      });
  }
}

void fftPadReal::backward2(Complex *F, Complex *f, size_t r0, Complex *W)
{
  if(W == NULL) W=F;
  double *fr=(double *) f;
  double *Wr=(double *) W;

  if(r0 == 0) {
    crfftm->fft(F,Wr);
    PARALLELIF(
      m > threshold,
      for(size_t s=0; s < m; ++s)
        fr[s]=Wr[s];
      );
    double *Wm=Wr-m;
    PARALLELIF(
      L > m+threshold,
      for(size_t s=m; s < L; ++s)
        fr[s]=Wm[s];
      );
  } else if(2*r0 < q) {
    size_t Lm=L-m;
    bool remainder=r0 == 1 && D0 != D;
    size_t Dstop;
    if(remainder) {
      ifftm0->fft(F,W);
      Dstop=D0;
    } else {
      ifftm->fft(F,W);
      Dstop=D;
    }
    for(size_t d=0; d < Dstop; ++d) {
      size_t r=r0+d;
      Complex *U=W+m*d;
      fr[0] += 2.0*U[0].re;
      Complex *Zetar=Zetaqm+m*r;
      PARALLELIF(
        m > threshold,
        for(size_t s=1; s < m; ++s)
          fr[s] += 2.0*realProduct(Zetar[s],U[s]);
        );
      Complex *Zetar2=ZetaqmS+Lm*r;
      Complex *Um=U-m;
      PARALLELIF(
        Lm > threshold,
        for(size_t s=m; s < L; ++s)
          fr[s] += 2.0*realProduct(Zetar2[s],Um[s]);
        );
    }
  } else {
    iffth->fft(F,W);
    Complex *Zetar=Zetaqm+m*r0;
    size_t h=e-1;
    size_t mh=m+h;
    double *frh=fr+h;
    double *frm=fr+m;
    double *frhm=frh+m;
    size_t stop1,stop2;

    Complex z0=2.0*W[0];
    double Rez0=z0.re;
    double Imz0=z0.im;
    fr[0] += Rez0;
    frm[0] -= Rez0;
    frh[0] += Imz0;

    if(mh < L) {
      stop1=L-mh;
      stop2=h;
      frhm[0] -= Imz0;
    } else {
      stop1=1;
      stop2=L-m;
    }
    PARALLELIF(
      stop1 > threshold,
      for(size_t s=1; s < stop1; ++s) {
        Complex z=2.0*conj(Zetar[s])*W[s];
        double Rez=z.re;
        double Imz=z.im;
        // t=0
        fr[s] += Rez;
        frh[s] += Imz;
        // t=1
        frm[s] -= Rez;
        frhm[s] -= Imz;
      });
    PARALLELIF(
      stop2 > stop1+threshold,
      for(size_t s=stop1; s < stop2; ++s) {
        Complex z=2.0*conj(Zetar[s])*W[s];
        double Rez=z.re;
        // t=0
        fr[s] += Rez;
        frh[s] += z.im;
        // t=1
        frm[s] -= Rez;
      });
    PARALLELIF(
      h > stop2+threshold,
      for(size_t s=stop2; s < h; ++s) {
        Complex z=2.0*conj(Zetar[s])*W[s];
        // t=0
        fr[s] += z.re;
        frh[s] += z.im;
      });
  }
}

void fftPadReal::backward2Many(Complex *F, Complex *f, size_t r, Complex *W)
{
  if(W == NULL) W=F;
  double *fr=(double *) f;
  double *Wr=(double *) W;

  if(r == 0) {
    crfftm->fft(F,Wr);
    PARALLELIF(
      m > threshold,
      for(size_t s=0; s < m; ++s) {
        size_t Ss=S*s;
        double *Wrs=Wr+Ss;
        double *frs=fr+Ss;
        for(size_t c=0; c < C; ++c)
          frs[c]=Wrs[c];
      });
    double *Wm=Wr-Sm;
    PARALLELIF(
      L > m+threshold,
      for(size_t s=m; s < L; ++s) {
        size_t Ss=S*s;
        double *Wms=Wm+Ss;
        double *frs=fr+Ss;
        for(size_t c=0; c < C; ++c)
          frs[c]=Wms[c];
      });
  } else if(2*r < q) {
    size_t Lm=L-m;
    ifftm->fft(F,W);
    for(size_t c=0; c < C; ++c)
      fr[c] += 2.0*W[c].re;
    Complex *Zetar=Zetaqm+m*r;
    PARALLELIF(
      m > threshold,
      for(size_t s=1; s < m; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        double *frs=fr+Ss;
        Complex zeta=2.0*Zetar[s];
        for(size_t c=0; c < C; ++c) // TODO: Vectorize
          frs[c] += realProduct(zeta,Ws[c]);
      });
    Complex *Zetar2=ZetaqmS+Lm*r;
    Complex *Wm=W-Sm;
    PARALLELIF(
      Lm > threshold,
      for(size_t s=m; s < L; ++s) {
        size_t Ss=S*s;
        Complex *Wms=Wm+Ss;
        double *frs=fr+Ss;
        Complex zeta2=2.0*Zetar2[s];
        for(size_t c=0; c < C; ++c) // TODO: Vectorize
          frs[c] += realProduct(zeta2,Wms[c]);
      });
  } else {
    iffth->fft(F,W);
    Complex *Zetar=Zetaqm+m*r;
    size_t h=e-1;
    size_t mh=m+h;
    double *frh=fr+S*h;
    double *frm=fr+Sm;
    double *frhm=frh+Sm;
    size_t stop1,stop2;

    if(mh < L) {
      for(size_t c=0; c < C; ++c) {
        Complex z0=2.0*W[c];
        double Rez0=z0.re;
        double Imz0=z0.im;
        fr[c] += Rez0;
        frm[c] -= Rez0;
        frh[c] += Imz0;
        frhm[c] -= Imz0;
      }
      stop1=L-mh;
      stop2=h;
    } else {
      for(size_t c=0; c < C; ++c) {
        Complex z0=2.0*W[c];
        double Rez0=z0.re;
        double Imz0=z0.im;
        fr[c] += Rez0;
        frm[c] -= Rez0;
        frh[c] += Imz0;
      }
      stop1=1;
      stop2=L-m;
    }

    PARALLELIF(
      stop1*C > C+threshold,
      for(size_t s=1; s < stop1; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        double *frs=fr+Ss;
        double *frhs=frh+Ss;
        double *frms=frm+Ss;
        double *frhms=frhm+Ss;
        Complex zeta=2.0*conj(Zetar[s]);
        for(size_t c=0; c < C; ++c) {
          Complex z=zeta*Ws[c];
          double Rez=z.re;
          double Imz=z.im;
          // t=0
          frs[c] += Rez;
          frhs[c] += Imz;
          // t=1
          frms[c] -= Rez;
          frhms[c] -= Imz;
        }
      });
    PARALLELIF(
      stop2*C > stop1*C+threshold,
      for(size_t s=stop1; s < stop2; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        double *frs=fr+Ss;
        double *frhs=frh+Ss;
        double *frms=frm+Ss;
        Complex zeta=2.0*conj(Zetar[s]);
        for(size_t c=0; c < C; ++c) {
          Complex z=zeta*Ws[c];
          double Rez=z.re;
          // t=0
          frs[c] += Rez;
          frhs[c] += z.im;
          // t=1
          frms[c] -= Rez;
        }
      });
    PARALLELIF(
      h*C > stop2*C+threshold,
      for(size_t s=stop2; s < h; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        double *frs=fr+Ss;
        double *frhs=frh+Ss;
        Complex zeta=2.0*conj(Zetar[s]);
        for(size_t c=0; c < C; ++c) {
          Complex z=zeta*Ws[c];
          // t=0
          frs[c] += z.re;
          frhs[c] += z.im;
        }
      });
  }
}

void fftPadReal::backwardInner(Complex *F0, Complex *f, size_t r0, Complex *W)
{
  if(W == NULL) W=F0;

  double *fr=(double *) f;
  double *Wr=(double *) W;

  size_t n2=ceilquotient(n,2);
  size_t pm1=p-1;
  size_t mpm1=m*pm1;
  size_t stop=L-mpm1;

  if(r0 == 0) {
    size_t p2=ceilquotient(p,2);
    size_t p2m=p2*m;

    ifftmp2->fft(F0,W);

    PARALLELIF(
      (p2-1)*(m-1) > threshold,
      for(size_t t=1; t < p2; ++t) {
        size_t R=n*t;
        Complex *Ft=W+m*t;
        Complex *Zetar=Zetaqm+m*R;
        for(size_t s=1; s < m; ++s)
          Ft[s] *= Zetar[s]; // Compensate for conjugate transform sign.
      });
    if(p == 2*p2) {
      iffth->fft(F0+p2m,W+p2m);

      size_t h=e-1;

      Complex *Zetar=Zetaqm+m*n*p2;
      Complex *Wt=W+p2m;
      Complex *Wth=Wt+h;

      Complex z=2.0*Wt[0];
      Wt[0]=z.re;
      Wth[0]=z.im;

      PARALLELIF(
        h > 1+threshold,
        for(size_t s=1; s < h; ++s) {
          Complex z=2.0*conj(Zetar[s])*Wt[s];
          Wt[s]=z.re;
          Wth[s]=z.im;
        });
    }
    crfftp->fft(W);
    PARALLELIF(
      pm1*m > threshold,
      for(size_t t=0; t < pm1; ++t) {
        size_t mt=m*t;
        double *ft=fr+mt;
        double *Ft=Wr+mt;
        for(size_t s=0; s < m; ++s)
          ft[s]=Ft[s];
      });
    size_t mt=m*pm1;
    double *ft=fr+mt;
    double *Ft=Wr+mt;
    PARALLELIF(
      stop > threshold,
      for(size_t s=0; s < stop; ++s)
        ft[s]=Ft[s];
      );

  } else if(r0 < n2) {
    bool remainder=r0 == 1 && D0 != D;
    size_t Dstop;
    if(remainder) {
      ifftm0->fft(F0,W);
      Dstop=D0;
    } else {
      ifftm->fft(F0,W);
      Dstop=D;
    }
    for(size_t d=0; d < Dstop; ++d) {
      Complex *F=W+b*d;
      size_t r=r0+d;
      PARALLELIF(
        p*(m-1) > threshold,
        for(size_t t=0; t < p; ++t) {
          size_t R=n*t+r;
          Complex *Ft=F+m*t;
          Complex *Zetar=Zetaqm+m*R;
          for(size_t s=1; s < m; ++s)
            Ft[s] *= conj(Zetar[s]);
        });
      ifftp->fft(F);

      PARALLELIF(
        m > threshold,
        for(size_t s=0; s < m; ++s)
          fr[s] += 2.0*F[s].re;
        );
      Complex *Zetaqr=Zetaqp+pm1*r;
      PARALLELIF(
        (pm1-1)*m > threshold,
        for(size_t t=1; t < pm1; ++t) {
          size_t mt=m*t;
          double *ft=fr+mt;
          Complex *Ft=F+mt;
          Complex Zeta=2.0*Zetaqr[t];
          for(size_t s=0; s < m; ++s)
            ft[s] += realProduct(Zeta,Ft[s]);
        });
      size_t mt=m*pm1;
      Complex *Ft=F+mt;
      double *ft=fr+mt;
      Complex Zeta=2.0*Zetaqr[pm1];
      PARALLELIF(
        stop > threshold,
        for(size_t s=0; s < stop; ++s)
          ft[s] += realProduct(Zeta,Ft[s]);
        );
    }
  } else { // 2*r0 == n
    size_t p2=ceilquotient(p,2);
    size_t p2m1=p2-1;
    size_t p2m=p2*m;
    Complex *Zetaqr=Zetaqp+pm1*r0;

    ifftmp2->fft(F0,W);

    PARALLELIF(
      p2*(m-1) > threshold,
      for(size_t t=0; t < p2; ++t) {
        size_t mt=m*t;
        Complex *Wt=W+mt;
        Complex *Zetar=Zetaqm+m*(2*n*t+r0);
        for(size_t s=1; s < m; ++s)
          Wt[s] *= conj(Zetar[s]);
      });

    ifftp2->fft(W);

    double *frp2=fr+p2m;

    PARALLELIF(
      m > threshold,
      for(size_t s=0; s < m; ++s) {
        Complex Z=2.0*W[s];
        fr[s] += Z.re;
        frp2[s] += Z.im;
      });

    PARALLELIF(
      (p2m1-1)*m > threshold,
      for(size_t t=1; t < p2m1; ++t) {
        size_t mt=m*t;
        Complex *Wt=W+mt;
        double *ft=fr+mt;
        double *ftp2=frp2+mt;
        Complex Zeta=2.0*conj(Zetaqr[t]);
        for(size_t s=0; s < m; ++s) {
          Complex Z=Zeta*Wt[s];
          ft[s] += Z.re;
          ftp2[s] += Z.im;
        }
      });

    size_t mp2m1=m*p2m1;
    Complex *Wt=W+mp2m1;
    double *ft=fr+mp2m1;
    double *ftp2=frp2+mp2m1;
    Complex ZetaConj=2.0*conj(Zetaqr[p2m1]);

    PARALLELIF(
      stop > threshold,
      for(size_t s=0; s < stop; ++s) {
        Complex Z=ZetaConj*Wt[s];
        ft[s] += Z.re;
        ftp2[s] += Z.im;
      });

    PARALLELIF(
      m > stop+threshold,
      for(size_t s=stop; s < m; ++s) {
        Complex Wts=Wt[s];
        ft[s] += ZetaConj.re*Wts.re-ZetaConj.im*Wts.im;
      });
  }
}

void fftPadReal::backwardInnerMany(Complex *F, Complex *f, size_t r, Complex *W)
{
  if(W == NULL) W=r == 0 && V ? V : F;

  double *fr=(double *) f;
  double *Wr=(double *) W; // TODO: Move

  size_t n2=ceilquotient(n,2);
  size_t pm1=p-1;
  size_t stop=L-m*pm1;
  size_t Smpm1=Sm*pm1;

  if(r == 0) {
    size_t p2=ceilquotient(p,2);

    for(size_t t=0; t < p2; ++t)
      ifftmr0->fft(F+Sm*t,W+Cm*t);

    PARALLELIF(
      (p2-1)*(m-1)*C > threshold,
      for(size_t t=1; t < p2; ++t) {
        size_t R=n*t;
        Complex *Wt=W+Cm*t;
        Complex *Zetar=Zetaqm+m*R;
        for(size_t s=1; s < m; ++s) {
          Complex *Wts=Wt+C*s;
          Complex zeta=Zetar[s];  // Compensate for conjugate transform sign.
          for(size_t c=0; c < C; ++c)
            Wts[c] *= zeta;
        }
      });

    if(p == 2*p2) {
      size_t p2Cm=p2*Cm;
      iffth->fft(F+p2*Sm,W+p2Cm);

      size_t h=e-1;

      Complex *Zetar=Zetaqm+m*n*p2;
      Complex *Wt=W+p2Cm;
      Complex *Wth=Wt+C*h;

      PARALLELIF(
        C > threshold,
        for(size_t c=0; c < C; ++c) {
          Complex z=2.0*Wt[c];
          Wt[c]=z.re;
          Wth[c]=z.im;
        });

      PARALLELIF(
        h*C > C+threshold,
        for(size_t s=1; s < h; ++s) {
          size_t Cs=C*s;
          Complex *Wts=Wt+Cs;
          Complex *Wths=Wth+Cs;
          Complex zeta=2.0*conj(Zetar[s]);
          for(size_t c=0; c < C; ++c) {
            Complex z=zeta*Wts[c];
            Wts[c]=z.re;
            Wths[c]=z.im;
          }
        });
    }

    crfftp->fft(W);

    PARALLELIF(
      pm1*m*C > threshold,
      for(size_t t=0; t < pm1; ++t) {
        double *Wt=Wr+Cm*t;
        double *ft=fr+Sm*t;
        for(size_t s=0; s < m; ++s) {
          double *Wts=Wt+C*s;
          double *fts=ft+S*s;
          for(size_t c=0; c < C; ++c)
            fts[c]=Wts[c];
        }
      });
    double *ft=fr+Smpm1;
    double *Wt=Wr+Cm*pm1;
    PARALLELIF(
      stop*C > threshold,
      for(size_t s=0; s < stop; ++s) {
        double *Wts=Wt+C*s;
        double *fts=ft+S*s;
        for(size_t c=0; c < C; ++c)
          fts[c]=Wts[c];
      });

  } else if(r < n2) {
    for(size_t t=0; t < p; ++t) {
      size_t Smt=Sm*t;
      ifftm->fft(F+Smt,W+Smt);
    }
    PARALLELIF(
      p*(m-1)*C > threshold,
      for(size_t t=0; t < p; ++t) {
        size_t R=n*t+r;
        Complex *Wt=W+Sm*t;
        Complex *Zetar=Zetaqm+m*R;
        for(size_t s=1; s < m; ++s) {
          size_t Ss=S*s;
          Complex *Wts=Wt+Ss;
          Complex zeta=conj(Zetar[s]);
          for(size_t c=0; c < C; ++c)
            Wts[c] *= zeta;
        }
      });

    if(S == C)
      ifftp->fft(W);
    else
      for(size_t s=0; s < m; ++s)
        ifftp->fft(W+S*s);

    PARALLELIF(
      m*C > threshold,
      for(size_t s=0; s < m; ++s) {
        Complex *Ws=W+S*s;
        double *frs=fr+S*s;
        for(size_t c=0; c < C; ++c)
          frs[c] += 2.0*Ws[c].re;
      });
    Complex *Zetaqr=Zetaqp+pm1*r;
    PARALLELIF(
      (pm1-1)*m*C > threshold,
      for(size_t t=1; t < pm1; ++t) {
        size_t Smt=Sm*t;
        double *ft=fr+Smt;
        Complex *Wt=W+Smt;
        Complex Zeta=2.0*Zetaqr[t];
        for(size_t s=0; s < m; ++s) {
          Complex *Wts=Wt+S*s;
          double *fts=ft+S*s;
          for(size_t c=0; c < C; ++c)
            fts[c] += realProduct(Zeta,Wts[c]);
        }
      });
    size_t Smt=Sm*pm1;
    Complex *Wt=W+Smt;
    double *ft=fr+Smt;
    Complex Zeta=2.0*Zetaqr[pm1];
    PARALLELIF(
      stop*C > threshold,
      for(size_t s=0; s < stop; ++s) {
        Complex *Wts=Wt+S*s;
        double *fts=ft+S*s;
        for(size_t c=0; c < C; ++c)
          fts[c] += realProduct(Zeta,Wts[c]);
      });
  } else { // 2*r == n
    size_t p2=ceilquotient(p,2);
    size_t p2m1=p2-1;
    size_t p2Sm=p2*Sm;
    Complex *Zetaqr=Zetaqp+pm1*r;

    for(size_t t=0; t < p2; ++t) {
      size_t Smt=Sm*t;
      ifftm->fft(F+Smt,W+Smt);
    }

    PARALLELIF(
      p2*(m-1)*C > threshold,
      for(size_t t=0; t < p2; ++t) {
        size_t Smt=Sm*t;
        Complex *Wt=W+Smt;
        Complex *Zetar=Zetaqm+m*(2*n*t+r);
        for(size_t s=1; s < m; ++s) {
          Complex *Wts=Wt+S*s;
          Complex zeta=conj(Zetar[s]);
          for(size_t c=0; c < C; ++c)
            Wts[c] *= zeta;
        }
      });

    if(S == C)
      ifftp2->fft(W);
    else
      for(size_t s=0; s < m; ++s)
        ifftp2->fft(W+S*s);

    double *frp2=fr+p2Sm;

    PARALLELIF(
      m*C > threshold,
      for(size_t s=0; s < m; ++s) {
        size_t Ss=S*s;
        Complex *Ws=W+Ss;
        double *frs=fr+Ss;
        double *frp2s=frp2+Ss;
        for(size_t c=0; c < C; ++c) {
          Complex Z=2.0*Ws[c];
          frs[c] += Z.re;
          frp2s[c] += Z.im;
        }
      });

    PARALLELIF(
      (p2m1-1)*m*C > threshold,
      for(size_t t=1; t < p2m1; ++t) {
        size_t Smt=Sm*t;
        Complex *Wt=W+Smt;
        double *ft=fr+Smt;
        double *ftp2=frp2+Smt;
        Complex Zeta=2.0*conj(Zetaqr[t]);
        for(size_t s=0; s < m; ++s) {
          size_t Ss=S*s;
          Complex *Wts=Wt+Ss;
          double *fts=ft+Ss;
          double *ftp2s=ftp2+Ss;
          for(size_t c=0; c < C; ++c) {
            Complex Z=Zeta*Wts[c];
            fts[c] += Z.re;
            ftp2s[c] += Z.im;
          }
        }
      });

    size_t Smp2m1=Sm*p2m1;
    Complex *Wt=W+Smp2m1;
    double *ft=fr+Smp2m1;
    double *ftp2=frp2+Smp2m1;
    Complex ZetaConj=2.0*conj(Zetaqr[p2m1]);

    PARALLELIF(
      stop*C > threshold,
      for(size_t s=0; s < stop; ++s) {
        size_t Ss=S*s;
        Complex *Wts=Wt+Ss;
        double *fts=ft+Ss;
        double *ftp2s=ftp2+Ss;
        for(size_t c=0; c < C; ++c) {
          Complex Z=ZetaConj*Wts[c];
          fts[c] += Z.re;
          ftp2s[c] += Z.im;
        }
      });

    PARALLELIF(
      m*C > stop*C+threshold,
      for(size_t s=stop; s < m; ++s) {
        size_t Ss=S*s;
        Complex *Wts=Wt+Ss;
        double *fts=ft+Ss;
        for(size_t c=0; c < C; ++c) {
          Complex Wtsc=Wts[c];
          fts[c] += ZetaConj.re*Wtsc.re-ZetaConj.im*Wtsc.im;
        }
      });
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
      for(size_t i=0; i < B; ++i)
        deleteAlign(V[i]);
    }
    if(V)
      delete [] V;
  }

  if(allocateF) {
    utils::deleteAlign(F[0]);
    delete [] F;
  }
}

// g is an array of max(A,B) pointers to distinct data blocks
void Convolution::convolveRaw(Complex **g)
{
  if(q == 1) {
    size_t blocksize=fft->blocksize(0);
    forward(g,F,0,0,A);
    (*mult)(F,blocksize,&indices,threads);
    backward(F,g,0,0,B,W);
  } else {
    if(fft->overwrite) {
      forward(g,F,0,0,A);
      size_t final=fft->n-1;
      for(size_t r=0; r < final; ++r) {
        size_t blocksize=fft->blocksize(r);
        Complex *h[A];
        for(size_t a=0; a < A; ++a)
          h[a]=g[a]+r*blocksize;
        indices.r=r;
        (*mult)(h,blocksize,&indices,threads);
      }
      indices.r=final;
      size_t blocksize=fft->blocksize(final);
      (*mult)(F,blocksize,&indices,threads);
      backward(F,g,0,0,B);
    } else {
      if(loop2) {
        forward(g,F,0,0,A);
        operate(F,0,&indices);
        size_t C=A-B;
        size_t a=0;
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
        for(size_t r=0; r < R; r += fft->increment(r)) {
          forward(g,F,r,0,A);
          operate(F,r,&indices);
          backwardPad(F,h0,r,0,B,W0);
        }

        if(nloops > 1) {
          size_t wL=fft->wordSize()*fft->inputLength();
          for(size_t b=0; b < B; ++b) {
            double *gb=(double *) (g[b]);
            double *hb=(double *) (h0[b]);
            for(size_t i=0; i < wL; ++i)
              gb[i]=hb[i];
          }
        }
      }
    }
  }
}

void Convolution::convolveRaw(Complex **f, Indices *indices2)
{
  for(size_t t=0; t < threads; ++t)
    indices.copy(indices2,0);
  indices.fft=fft;
  convolveRaw(f);
}

void Convolution::convolveRaw(Complex **f, size_t offset)
{
  Complex *G[A];
  Complex **g=G;
  for(size_t a=0; a < A; ++a)
    g[a]=f[a]+offset;
  convolveRaw(g);
}

void Convolution::convolveRaw(Complex **f, size_t offset,
                              Indices *indices2)
{
  for(size_t t=0; t < threads; ++t)
    indices.copy(indices2,0);
  indices.fft=fft;
  convolveRaw(f,offset);
}

}
