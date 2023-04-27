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
// F0[j] *= F1[j];
void multbinary(Complex **F, size_t n, Indices *indices,
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
      F0[j] *= F1[j];
    );
}

// This multiplication routine is for binary convolutions and takes
// two real inputs of size n.
// F0[j] *= F1[j];
void realmultbinary(Complex **F, size_t n, Indices *indices,
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

// Returns the smallest natural number greater than a positive number
// m of the form 2^a 3^b 5^c 7^d for some nonnegative integers a, b, c, and d.
size_t nextfftsize(size_t m)
{
  if(m == ceilpow2(m))
    return m;
  size_t N=SIZE_MAX;
  size_t ni=1;
  for(size_t i=0; ni < 7*m; ni=pow(7,i),++i) {
    size_t nj=ni;
    for(size_t j=0; nj < 5*m; nj=ni*pow(5,j),++j) {
      size_t nk=nj;
      for(size_t k=0; nk < 3*m; nk=nj*pow(3,k),++k) {
        N=min(N,nk*ceilpow2(ceilquotient(m,nk)));
      }
      if(N == m)
        return N;
    }
  }
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

// Returns true iff m is a power of 2, 3, 5, or 7.
bool ispure(size_t m)
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

double time(fftBase *fft, double &threshold)
{
  size_t threads=fft->app.threads == 1 ? fft->app.maxthreads : 1;

  size_t N=max(fft->app.A,fft->app.B);
  size_t inputSize=fft->inputSize();
  Complex **f=ComplexAlign(N*threads,inputSize);

  // Initialize entire array to 0 to avoid overflow when timing.
  for(size_t t=0; t < threads; ++t) {
    for(size_t a=0; a < fft->app.A; ++a) {
      Complex *fa=f[N*t+a];
      for(size_t j=0; j < inputSize; ++j)
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
                               bool useTimer, bool Explicit, bool inner)
{
  size_t i=(inner ? m : 0);
  // If inner == true, i is an m value and itmax is the largest m value that
  // we consider. If inner == false, i is a counter starting at zero, and
  // itmax is maximum number of m values we consider before exiting optloop.
  while(i < itmax) {
    size_t p=ceilquotient(L,m);
    // P is the effective p value
    size_t P=(centered && p == 2*(p/2)) || p == 2 ? (p/2) : p;
    size_t n=ceilquotient(M,m*P);

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
      size_t q=(inner ? P*n : ceilquotient(M,m));
      size_t Dstart=forceD ? app.D : 1;
      size_t Dstop=forceD ? app.D : n;
      size_t Dstop2=2*Dstop;

      // Check inplace and out-of-place unless C > 1.
      size_t Istart=app.I == -1 ? C > 1 : app.I;

      size_t Istop=app.I == -1 ? 2 : app.I+1;

      for(size_t D=Dstart; D < Dstop2; D *= 2) {
        if(D > Dstop) D=Dstop;
        for(size_t inplace=Istart; inplace < Istop; ++inplace)
          if((q == 1 || valid(D,p,S)) && D <= n)
            check(L,M,C,S,m,p,q,D,inplace,app,useTimer);
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

void fftBase::OptBase::opt(size_t L, size_t M, Application& app,
                           size_t C, size_t S,
                           size_t minsize, size_t itmax,
                           bool Explicit, bool centered, bool useTimer)
{
  if(!Explicit) {
    size_t H=ceilquotient(L,2);
    if(mForced) {
      if(app.m >= H)
        optloop(app.m,L,M,app,C,S,centered,1,useTimer,false);
      else
        optloop(app.m,L,M,app,C,S,centered,app.m+1,useTimer,false,true);
    } else {
      size_t m=nextfftsize(minsize);

      optloop(m,L,M,app,C,S,centered,max(L/2,32),useTimer,false,true);

      m=nextfftsize(H);

      optloop(m,L,M,app,C,S,centered,itmax,useTimer,false);

      size_t Mhalf=ceilquotient(M,2);
      if(L > Mhalf) {
        m=nextfftsize(max(Mhalf,m));
        optloop(m,L,M,app,C,S,centered,itmax,useTimer,false);

        m=nextfftsize(max(L,m));
        optloop(m,L,M,app,C,S,centered,itmax,useTimer,false);
      } else {
        m=nextfftsize(max(min(L,Mhalf),m));
        optloop(m,L,M,app,C,S,centered,itmax,useTimer,false);
      }
      m=nextfftsize(max(M,m));
      optloop(m,L,M,app,C,S,centered,itmax,useTimer,false);
    }
  } else {
    size_t m=nextfftsize(M);
    optloop(m,L,m,app,C,S,centered,itmax,useTimer,true);
  }
}

void fftBase::OptBase::check(size_t L, size_t M,
                             size_t C, size_t S, size_t m,
                             size_t p, size_t q, size_t D,
                             bool inplace, Application& app, bool useTimer)
{
  //cout << "m=" << m << ", p=" << p << ", q=" << q << ", D=" << D << " I=" << inplace << endl;
  if(useTimer) {
    double t=time(L,M,C,S,m,q,D,inplace,app);
    if(showOptTimes)
      cout << "m=" << m << ", p=" << p << ", q=" << q << ", C=" << C << ", S=" << S << ", D=" << D << ", I=" << inplace << ": t=" << t*1.0e-9 << endl;
    if(t < T) {
      this->m=m;
      this->q=q;
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
      this->q=q;
      this->D=D;
      this->inplace=inplace;
    }
  }
}

void fftBase::OptBase::scan(size_t L, size_t M, Application& app,
                            size_t C, size_t S, bool Explicit,
                            bool centered)
{
  m=M;
  q=1;
  D=1;
  inplace=false;
  T=DBL_MAX;
  threshold=T;
  mForced=app.m >= 1;

  bool sR=showRoutines;

  showRoutines=false;
  size_t mStart=2;
  size_t itmax=3;
  opt(L,M,app,C,S,mStart,itmax,Explicit,centered,false);

  if(counter == 0) {
    cerr << "Optimizer found no valid cases with specified parameters." << endl;
    cerr << "Using explicit routines with m=" << M << " instead." << endl << endl;
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

  size_t p=ceilquotient(L,m);
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

  showRoutines=sR;
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
    dr=D=D0=R=Q=1;
    l=M;
    b=S*l;
  } else {
    double twopibyN=twopi/M;
    double twopibyq=twopi/q;

    size_t d;

    size_t P,P1;
    size_t p2=p/2;

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
    size_t size=b*D;

    G=ComplexAlign(size);
    H=inplace ? G : ComplexAlign(size);

    overwrite=inplace && L == p*m && n == (centered ? 3 : p+1) && D == 1;

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
        if(Overwrite()) {
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
        if(Overwrite()) {
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
      Q=n;

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
          if(Overwrite()) {
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
          if(Overwrite()) {
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
          if(Overwrite()) {
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
          if(Overwrite()) {
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
        Q=q;
      }
    }

    dr=Dr();
    R=residueBlocks();
    D0=Q % D;
    if(D0 == 0) D0=D;

    if(D0 != D) {
      size_t x=D0*P;
      fftm0=new mfft1d(m,1,x, 1,Sm, G,H,threads);
      ifftm0=new mfft1d(m,-1,x, 1,Sm, H,G,threads);
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
      mp-L > threshold,
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
      Complex *Fs=W+Ss;
      Complex *fs=f+Ss;
      for(size_t c=0; c < C; ++c)
        Fs[c]=fs[c];
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
      Complex *Fs=W+Ss;
      for(size_t c=0; c < C; ++c)
        fs[c]=Fs[c];
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
        for(size_t s=L; s < m; ++s)
          V[s]=0.0;
      }
    }
    if(inplace) {
      for(size_t s=L; s < m; ++s)
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
      PARALLELIF(
        L > threshold,
        for(size_t s=1; s < L; ++s)
          W[s]=Zetar[s]*f[s];
        );
      if(inplace) {
        for(size_t s=L; s < m; ++s)
          W[s]=0.0;
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
        for(size_t s=L; s < m; ++s)
          F[s]=0.0;
        for(size_t s=L; s < m; ++s)
          G[s]=0.0;
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
      (m-L)*C > threshold,
      for(size_t s=L; s < m; ++s) {
        Complex *Fs=W+S*s;
        for(size_t c=0; c < C; ++c)
          Fs[c]=0.0;
      });
  }
  if(r == 0) {
    if(!inplace && L >= m && S == C)
      return fftm->fft(f,F);
    PARALLELIF(
      L*C > threshold,
      for(size_t s=0; s < L; ++s) {
        size_t Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fs=f+Ss;
        for(size_t c=0; c < C; ++c)
          Fs[c]=fs[c];
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
        Complex *Fs=W+Ss;
        Complex *fs=f+Ss;
        Vec Zeta=LOAD(Zetar+s);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(Zeta,-Zeta);
        for(size_t c=0; c < C; ++c)
          STORE(Fs+c,ZMULT(X,Y,LOAD(fs+c)));
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
  if(r0 == 0) {
    size_t residues;
    size_t q2=q/2;
    if(D == 1 || 2*q2 < q) { // q odd, r=0
      residues=1;
      PARALLELIF(
        Lm > threshold,
        for(size_t s=0; s < Lm; ++s)
          W[s]=f[s]+f[m+s];
        );
      PARALLELIF(
        m-Lm > threshold,
        for(size_t s=Lm; s < m; ++s)
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
      PARALLELIF(
        Lm > threshold,
        for(size_t s=1; s < Lm; ++s) {
          Complex fs=f[s];
          Complex fms=fm[s];
          W[s]=fs+fms;
          V[s]=conj(Zetar[s])*(fs-fms);
        });
      PARALLELIF(
        m-Lm > threshold,
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
      Complex *fm=f+m;
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
        m-Lm > threshold,
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
        m-Lm > threshold,
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
      (m-Lm)*C > threshold,
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
      (Lm-1)*C > threshold,
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
      (m-Lm)*C > threshold,
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
  size_t stop=L-m*pm1;

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

    size_t mt=m*pm1;
    Complex *Ft=W+mt;
    Complex *ft=f+mt;
    PARALLELIF(
      stop > threshold,
      for(size_t s=0; s < stop; ++s)
        Ft[s]=ft[s];
      );
    PARALLELIF(
      m-stop > threshold,
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
        Complex Zeta=Zetaqr[t];
        for(size_t s=0; s < m; ++s)
          Ft[s]=Zeta*ft[s];
      });
    size_t mt=m*pm1;
    Complex *Ft=F+mt;
    Complex *ft=f+mt;
    Complex Zeta=Zetaqr[pm1];
    PARALLELIF(
      stop > threshold,
      for(size_t s=0; s < stop; ++s)
        Ft[s]=Zeta*ft[s];
      );
    PARALLELIF(
      m-stop > threshold,
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
      (m-stop)*C > threshold,
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
        Complex *Ft=W+Sm*t;
        Complex *Zetar=Zetaqm+m*R;
        for(size_t s=1; s < m; ++s) {
          Complex *Fts=Ft+S*s;
          Complex Zetars=Zetar[s];
          for(size_t c=0; c < C; ++c)
            Fts[c] *= Zetars;
        }
      });
  } else {
    PARALLELIF(
      m*C > threshold,
      for(size_t s=0; s < m; ++s) {
        size_t Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fs=f+Ss;
        for(size_t c=0; c < C; ++c)
          Fs[c]=fs[c];
      });
    Complex *Zetaqr=Zetaqp+pm1*r;
    PARALLELIF(
      (pm1-1)*m*C > threshold,
      for(size_t t=1; t < pm1; ++t) {
        size_t Smt=Sm*t;
        Complex *Ft=W+Smt;
        Complex *ft=f+Smt;
        Complex Zeta=Zetaqr[t];
        for(size_t s=0; s < m; ++s) {
          size_t Ss=S*s;
          Complex *Fts=Ft+Ss;
          Complex *fts=ft+Ss;
          for(size_t c=0; c < C; ++c)
            Fts[c]=Zeta*fts[c];
        }
      });
    Complex *Ft=W+Sm*pm1;
    Complex *ft=f+Sm*pm1;
    Vec Zeta=LOAD(Zetaqr+pm1);
    PARALLELIF(
      stop*C > threshold,
      for(size_t s=0; s < stop; ++s) {
        size_t Ss=S*s;
        Complex *Fts=Ft+Ss;
        Complex *fts=ft+Ss;
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(Zeta,-Zeta);
        for(size_t c=0; c < C; ++c)
          STORE(Fts+c,ZMULT(X,Y,LOAD(fts+c)));
      });
    PARALLELIF(
      (m-stop)*C > threshold,
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
        L-m > threshold,
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
        L-m > threshold,
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
        L-m > threshold,
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
        L-m > threshold,
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
      (L-m)*C > threshold,
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
      (L-m)*C > threshold,
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
        size_t Smt=Sm*t;
        Complex *ft=f+Smt;
        Complex *Ft=W+Smt;
        for(size_t s=0; s < m; ++s) {
          size_t Ss=S*s;
          Complex *fts=ft+Ss;
          Complex *Fts=Ft+Ss;
          for(size_t c=0; c < C; ++c)
            fts[c]=Fts[c];
        }
      });
    size_t Smt=Sm*pm1;
    Complex *ft=f+Smt;
    Complex *Ft=W+Smt;
    PARALLELIF(
      stop*C > threshold,
      for(size_t s=0; s < stop; ++s) {
        size_t Ss=S*s;
        Complex *fts=ft+Ss;
        Complex *Fts=Ft+Ss;
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
    M-H-L > threshold,
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
    (M-H-L)*C > threshold,
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
        LH-mH > threshold,
        for(size_t s=mH; s < LH; ++s)
          W[s]=fmH[s]+fH[s];
        );
      PARALLELIF(
        m-LH > threshold,
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
        LH-mH > threshold,
        for(size_t s=mH; s < LH; ++s) {
          Complex B=fmH[s];
          Complex A=fH[s];
          W[s]=A+B;
          V[s]=Zetar[s]*(A-B);
        });
      PARALLELIF(
        m-LH > threshold,
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
        LH-mH > threshold,
        for(size_t s=mH; s < LH; ++s) {
//        W[s]=conj(*(Zetarm-s))*fmH[s]+Zetar[s]*fH[s];
          Vec Zeta=LOAD(Zetar+s);
          Vec Zetam=LOAD(Zetarm-s);
          Vec fs=LOAD(fH+s);
          Vec fms=LOAD(fmH+s);
          STORE(W+s,ZCMULT(Zetam,fms)+ZMULT(Zeta,fs));
        });
      PARALLELIF(
        m-LH > threshold,
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
        LH-mH > threshold,
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
        m-LH > threshold,
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
      (LH-mH)*C > threshold,
      for(size_t s=mH; s < LH; ++s) {
        size_t Ss=S*s;
        Complex *Fs=W+Ss;
        Complex *fmHs=fmH+Ss;
        Complex *fHs=fH+Ss;
        for(size_t c=0; c < C; ++c)
          Fs[c]=fmHs[c]+fHs[c];
      });
    PARALLELIF(
      (m-LH)*C > threshold,
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
      (LH-mH)*C > threshold,
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
      (m-LH)*C > threshold,
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
        m-mH > threshold,
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
        m-mH > threshold,
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
        m-mH > threshold,
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
        m-H > threshold,
        for(size_t s=H; s < m; ++s)
          STORE(fmH+s,LOAD(fmH+s)+ZMULT2(LOAD(Zetarm-s),LOAD(U+s),LOAD(V+s)));
        );
      PARALLELIF(
        H-mH > threshold,
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
      (m-H)*C > threshold,
      for(size_t s=H; s < m; ++s) {
        size_t Ss=S*s;
        Complex *fmHs=fmH+Ss;
        Complex *Fs=W+Ss;
        for(size_t c=0; c < C; ++c)
          fmHs[c]=Fs[c];
      });
    PARALLELIF(
      (H-mH)*C > threshold,
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
      (m-H)*C > threshold,
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
      (H-mH)*C > threshold,
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
        m-m0 > threshold,
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
        m-m1 > threshold,
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
        m-m0 > threshold,
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
        m-m1 > threshold,
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
        m-m0 > threshold,
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
        m-m1 > threshold,
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
        m-m0 > threshold,
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
        m-m1 > threshold,
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
      (m-m0)*C > threshold,
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
      (m-m1)*C > threshold,
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
      (m-m0)*C > threshold,
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
      (m1-m)*C > threshold,
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
        m-m0 > threshold,
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
        m-m1 > threshold,
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
        m-m0 > threshold,
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
        m-m1 > threshold,
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
        m-m0 > threshold,
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
        m-m1 > threshold,
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
        m-m0 > threshold,
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
        m-m1 > threshold,
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
      (m-m0)*C > threshold,
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
      (m-m1)*C > threshold,
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
      (m-m0)*C > threshold,
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
      (m-m1)*C > threshold,
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
  S=C; // Strides are not implemented for Hermitian transforms
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
    dr=D0=R=Q=1;
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
      Q=n=q/p2;

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
      Q=n=q;
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
    D0=Q % D;
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
    e-H > threshold,
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
    (e-H)*C > threshold,
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
        e-mH1 > threshold,
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
        e-mH1 > threshold,
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
      e-mH1 > threshold,
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
        (e-mH1)*C > threshold,
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
        (e-mH1)*C > threshold,
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
      (e-mH1)*C > threshold,
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
        me-mH1 > threshold,
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
        me-mH1 > threshold,
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
      me-mH1 > threshold,
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
        (me+1-mH1)*C > threshold,
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
        (me+1-mH1)*C > threshold,
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
      (me+1-mH1)*C > threshold,
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
        e-m0 > threshold,
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
        e-m0 > threshold,
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
      e-m0 > threshold,
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
        me-m0 > threshold,
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
        me-m0 > threshold,
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
        Complex *fe1=f+e-1;
        Complex *We1=W+e-1;
        Complex *Ve1=V+e-1;
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
      me-m0 > threshold,
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

  if(allocate)
    delete fft;
}

// g is an array of max(A,B) pointers to distinct data blocks
// each of size fft->inputSize()
void Convolution::convolveRaw(Complex **g)
{
  if(q == 1) {
    forward(g,F,0,0,A);
    (*mult)(F,blocksize,&indices,threads);
    backward(F,g,0,0,B,W);
  } else {
    if(fft->Overwrite()) {
      forward(g,F,0,0,A);
      size_t final=fft->n-1;
      for(size_t r=0; r < final; ++r) {
        Complex *h[A];
        for(size_t a=0; a < A; ++a)
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
          size_t inputSize=fft->inputSize();
          for(size_t b=0; b < B; ++b) {
            Complex *gb=g[b];
            Complex *hb=h0[b];
            for(size_t i=0; i < inputSize; ++i)
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
