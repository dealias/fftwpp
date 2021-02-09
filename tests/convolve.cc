#include "convolve.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

namespace fftwpp {

unsigned int threads=1;

unsigned int mOption=0;
unsigned int DOption=0;

int IOption=-1;

unsigned int A=2; // number of inputs
unsigned int B=1; // number of outputs
unsigned int C=1; // number of copies

unsigned int surplusFFTsizes=25;

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

unsigned int nextfftsize(unsigned int m)
{
  unsigned int N=-1;
  unsigned int ni=1;
  for(unsigned int i=0; ni < 7*m; ni=pow(7,i), ++i) {
    unsigned int nj=ni;
    for(unsigned int j=0; nj < 5*m; nj=ni*pow(5,j), ++j) {
      unsigned int nk=nj;
      for(unsigned int k=0; nk < 3*m; nk=nj*pow(3,k), ++k) {
        N=std::min(N,nk*utils::ceilpow2(utils::ceilquotient(m,nk)));
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
  unsigned int q=utils::ceilquotient(M,m);
  unsigned int p=utils::ceilquotient(L,m);

  if(p > 2) return; // Temporary ***********************************

  if(p == q && p > 1 && !mForced) return;

  if(!fixed) {
    unsigned int n=utils::ceilquotient(M,m*p);
    unsigned int q2=p*n;
    if(q2 != q) {
      unsigned int start=DOption > 0 ? std::min(DOption,n) : 1;
      unsigned int stop=DOption > 0 ? std::min(DOption,n) : n;
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

//      if(p % 2 == 0) q=utils::ceilquotient(2*M,p*m);
  if(p != 2 && q % p != 0) return;

  unsigned int start=DOption > 0 ? std::min(DOption,q) : 1;
  unsigned int stop=DOption > 0 ? std::min(DOption,q) : q;
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
    std::cerr << "L=" << L << " is greater than M=" << M << "." << std::endl;
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

  unsigned int p=utils::ceilquotient(L,m);
  std::cout << std::endl;
  std::cout << "Optimal values:" << std::endl;
  std::cout << "m=" << m << std::endl;
  std::cout << "p=" << p << std::endl;
  std::cout << "q=" << q << std::endl;
  std::cout << "C=" << C << std::endl;
  std::cout << "D=" << D << std::endl;
  std::cout << "Padding:" << m*p-L << std::endl;
}

double fftBase::meantime(Application& app, double *Stdev)
{
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
  std::cout << std::endl;

  double mean=meantime(app,&stdev);

  std::cout << "mean=" << mean << " +/- " << stdev << std::endl;

  return mean;
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
    G=utils::ComplexAlign(Cm);
    fftm=new mfft1d(m,1,C, C,1, G);
    ifftm=new mfft1d(m,-1,C, C,1, G);
    utils::deleteAlign(G);
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
    G=utils::ComplexAlign(m*d);
    H=inplace ? G : utils::ComplexAlign(m*d);

    if(twop) {
      unsigned Lm=L-m;
      Zetaqm2=utils::ComplexAlign((q-1)*Lm)-L;
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
      Zetaqp=utils::ComplexAlign((n-1)*(p-1))-p;
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
      utils::deleteAlign(H);
    utils::deleteAlign(G);

    initZetaqm();
  }

  fftBase::Forward=Forward;
  fftBase::Backward=Backward;
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

}
