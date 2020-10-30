// TODO:
// check results
// optimize memory use
// vectorize and optimize Zeta computations
// use output strides
// use out-of-place transforms

#include <cassert>
#include <cfloat>

#include "convolution.h"
#include "explicit.h"
#include "direct.h"
#include "utils.h"

using namespace std;
using namespace utils;
using namespace fftwpp;

unsigned int K=100; // Number of tests ***TEMP***

// Constants used for initialization and testing.
const Complex I(0.0,1.0);
const double E=exp(1.0);
const Complex iF(sqrt(3.0),sqrt(7.0));
const Complex iG(sqrt(5.0),sqrt(11.0));

bool Test=false;

unsigned int A=2; // number of inputs
unsigned int B=1; // number of outputs

const unsigned int Nsize=1000; // FIXME
unsigned int nsize=1000;
unsigned int size[Nsize];

unsigned int n0=1;//25;

// Search a sorted ordered array a of n elements for key, returning the index i
// if a[i] <= key < a[i+1], -1 if key is less than all elements of a, or
// n-1 if key is greater than or equal to the last element of a.

int search(unsigned int *a, unsigned int n, unsigned int key)
{
  if(n == 0 || key < a[0]) return -1;
  size_t u=n-1;
  if(key >= a[u]) return u;
  size_t l=0;

  while (l < u) {
    size_t i=(l+u)/2;
    if(key < a[i]) u=i;
    else if(key < a[i+1]) return i;
    else l=i+1;
  }
  return 0;
}

void decompose(unsigned int *D, unsigned int& n, unsigned int p,
               unsigned int q)
{
  unsigned int first=search(size,nsize,min(q/2,p));
  unsigned int last=search(size,nsize,n0);
// Return the factors of q in reverse order.
  n=0;
  for(unsigned int i=first; i > last; --i) {
    if(p <= n0) break;
    unsigned int f=size[i];
    if(f <= p && q % f == 0) {
      do {
        D[n++]=f;
        p -= f;
      } while(f <= p);
    }
  }
}

class FFTpad {
protected:
  unsigned int L;
  unsigned int M;
  unsigned int m;
  unsigned int p;
  unsigned int q;
  unsigned int *D; // divisors of q
  unsigned int n; // number of elements in D
  int sign;
  fft1d *fftN;
  fft1d *fftm;
  fft1d **fft;
  mfft1d *fftmo;
  unsigned int S;
  Complex *ZetaH, *ZetaL;
  Complex *g,*h,*G,*e,*E,*H; // Many, many work arrays!
public:

  void init(Complex *f) {
    unsigned int N=q*m;
    if(p == q)
      fftN=new fft1d(N,sign);
    else {
      // Revisit memory allocation
      S=p*N;
      BuildZeta(N,p*N,ZetaH,ZetaL,1,S);//,threads);

      g=ComplexAlign(m);
      G=ComplexAlign(N); // Rewrite so only used for n > 0.
      H=ComplexAlign(M);
      if(n > 0) {
        h=ComplexAlign(q);
        unsigned int D0=D[0];
        E=ComplexAlign(D0);
        e=ComplexAlign(D0);
        fft=new fft1d*[n];
        for(unsigned int i=0; i < n; ++i)
          fft[i]=q < L ? new fft1d(D[i],sign,f) : new fft1d(D[i],sign);
      }
      fftm=m < L ? new fft1d(m,sign,f) : new fft1d(m,sign);
      fftmo=new mfft1d(m,sign,1,1,q,0,0,f,G);
    }
 }

  // Compute an fft padded to N=q*m >= M >= L=f.length
  FFTpad(Complex *f, unsigned int L, unsigned int M,
         unsigned int m, unsigned int q, unsigned int *D=NULL,
         unsigned int n=0, int sign=-1) :
    L(L), M(M), m(m), p(ceilquotient(L,m)), q(q), D(D), n(n),
    sign(sign) {init(f);}

  class Opt {
  public:
    unsigned int m,q;
    unsigned int n; // Number of divisors

    // Determine optimal m,q values for padding L data values to
    // size >= M
    // If fixed=true then an FFT of size M is enforced.
    Opt(Complex *f, unsigned int L, unsigned int M, bool fixed=false)
    {
      assert(L <= M);
      m=M;
      q=1;
      n=0;
      unsigned int stop;
      unsigned int start;

      /*
      FFTpad fft(f,L,M,m,q);
      Complex *F=ComplexAlign(fft.length());
      seconds();
      for(unsigned int i=0; i < K; ++i) {
        for(unsigned int j=0; j < L; ++j)
          f[j]=j;
        fft.forwards(f,F);
      }
      double T=seconds()*100;
      utils::deleteAlign(F);
      */
      double T=DBL_MAX; // Temporary
      unsigned int i=0;

      while(true) {
        unsigned int m=size[i];
        cout << "m=" << m << endl;
        if(fixed && M % m != 0) continue;
        if(m > L) break; // Assume size 2 FFT is in table
        unsigned int p=ceilquotient(L,m);
        start=ceilquotient(M,m);
        if(p <= n0 || fixed)
          stop=start;
        else
          stop=p*ceilquotient(M,p*m);

        for(unsigned int q=ceilquotient(M,m); q <= stop; ++q) {
          cout << "q=" << q << endl;
          unsigned int D[p/(n0+1)];
          unsigned int n=0;
          decompose(D,n,p,q);
          assert(n < p/(n0+1));
          if(n > 0 || q == start) {
            FFTpad fft(f,L,M,m,q,D,n);
            Complex *F=ComplexAlign(fft.length());
            seconds();
            for(unsigned int i=0; i < K; ++i) {
              for(unsigned int j=0; j < L; ++j)
                f[j]=j;
              fft.forwards(f,F);
            }
            double t=seconds();
            utils::deleteAlign(F);

            if(t < T) {
              this->m=m;
              this->q=q;
              this->n=n;
              T=t;
            }
          }
        }
        ++i;
      }
      cout << "Optimal values:" << endl;
      cout << "m=" << m << endl;
      cout << "p=" << ceilquotient(L,m) << endl;
      cout << "q=" << q << endl;
      cout << "n=" << n << endl;
    }
  };

  // Normal entry point.
  // Compute an fft of length L padded to at least M
  // (or exactly M if fixed=true)
  FFTpad(Complex *f, unsigned int L, unsigned int M, int sign=-1,
         bool fixed=false) :
    L(L), M(M), sign(sign) {
    Opt opt=Opt(f,L,M,fixed);
    m=opt.m;
    p=ceilquotient(L,m);
    q=opt.q;
    D=new unsigned int[opt.n];
    decompose(D,n,p,q);
    init(f);
  }

  void forwards(Complex *f, Complex *F) {
    if(p == q) {
      unsigned int N=q*m;
      for(unsigned int i=0; i < L; ++i)
        F[i]=f[i];
      for(unsigned int i=L; i < N; ++i)
        F[i]=0.0;
      return fftN->fft(F);
    }

    unsigned int nsum=0;

    unsigned int pm=p*m;
    for(unsigned int i=0; i < L; ++i)
      H[i]=f[i];
    for(unsigned int i=L; i < pm; ++i)
      H[i]=0.0;

    /*
    unsigned int N=q*m;
    for(unsigned int i=0; i < N; ++i)
      G[i]=0.0;

    for(unsigned int i=0; i < n; ++i) {
      unsigned int n=D[i];
      unsigned int Q=q/n;
      fft1d* ffti=fft[i];
      for(unsigned int s=0; s < m; ++s) {
        for(unsigned int t=0; t < n; ++t)
          e[t]=F[(t+nsum)*m+s];
        for(unsigned int r=0; r < Q; ++r) {
          for(unsigned int t=0; t < n; ++t) {
            unsigned int c=m*r*t;
//            unsigned int a=c/S;
//            Complex Zeta=ZetaH[a]*ZetaL[c-S*a];
            Complex Zeta=ZetaL[c];
//            if(sign < 0) Zeta=conj(Zeta);
            E[t]=Zeta*e[t];
          }
          ffti->fft(E);
          for(unsigned int l=0; l < n; ++l)
            h[l*Q+r]=E[l];
        }
        for(unsigned int r=0; r < q; ++r) {
          unsigned int c=r*(s+m*nsum);
//          unsigned int a=c/S;
//          Complex Zeta=ZetaH[a]*ZetaL[c-S*a];
          Complex Zeta=ZetaL[c];
//          if(sign < 0) Zeta=conj(Zeta);
          G[r*m+s] += Zeta*h[r];
        }
      }
      nsum += n;
    }
    */

    if(p == 1)
      fftmo->fft(H,F);
    else {
      for(unsigned int s=0; s < m; ++s) {
        Complex sum=0.0;
        for(unsigned int t=nsum; t < p; ++t)
          sum += H[t*m+s];
//        g[s]=G[s]+sum;
        g[s]=sum;
      }
      fftmo->fft(g,F);
//      for(unsigned int l=0; l < m; ++l)
//        F[l*q]=g[l];
    }

    for(unsigned int r=1; r < q; ++r) {
      for(unsigned int s=0; s < m; ++s) {
        if(p == 1) {
          unsigned int c=r*s;// % N;
//          unsigned int a=c/S;
//          Complex Zeta=ZetaH[a]*ZetaL[c-S*a];
          Complex Zeta=ZetaL[c];
          if(sign < 0) Zeta=conj(Zeta);
          g[s]=Zeta*H[s];
        } else {
          Complex sum=0.0;
          for(unsigned int t=nsum; t < p; ++t) {
            unsigned int j=t*m+s;
            unsigned int c=r*j;// % N;
//          unsigned int a=c/S;
//          Complex Zeta=ZetaH[a]*ZetaL[c-S*a];
            Complex Zeta=ZetaL[c];
          if(sign < 0) Zeta=conj(Zeta);
            sum += Zeta*H[j];
          }
//        g[s]=G[r*m+s]+sum;
          g[s]=sum;
        }
      }
      fftmo->fft(g,F+r);
//      for(unsigned int l=0; l < m; ++l)
//        F[l*q+r]=g[l];
    }

    return;
  }

  unsigned int padding() {
    return p*m-L;
  }
  unsigned int length() {
    return m*q;
  }
};



inline void init(Complex **F, unsigned int m, unsigned int A)
{
  if(A % 2 == 0) {
    unsigned int M=A/2;
    double factor=1.0/sqrt((double) M);
    for(unsigned int s=0; s < M; ++s) {
      double ffactor=(1.0+s)*factor;
      double gfactor=1.0/(1.0+s)*factor;
      Complex *fs=F[s];
      Complex *gs=F[s+M];
      if(Test) {
        for(unsigned int k=0; k < m; k++) {
          fs[k]=factor*iF*pow(E,k*I);
          gs[k]=factor*iG*pow(E,k*I);
        }
      } else {
        for(unsigned int k=0; k < m; k++) {
          fs[k]=ffactor*Complex(k,k+1);
          gs[k]=gfactor*Complex(k,2*k+1);
        }
      }
    }
  } else {
    for(unsigned int a=0; a < A; ++a) {
      for(unsigned int k=0; k < m; ++k) {
        F[a][k]=(a+1)*Complex(k,k+1);
      }
    }
  }
}

// Pair-wise binary multiply for A=2 or A=4.
// NB: example function, not optimised or threaded.
void multA(Complex **F, unsigned int m,
           const unsigned int indexsize,
           const unsigned int *index,
           unsigned int r, unsigned int threads)
{
  switch(A) {
    case 2: multbinary(F,m,indexsize,index,r,threads); break;
    case 4: multbinary2(F,m,indexsize,index,r,threads); break;
    default:
      cerr << "A=" << A << " is not yet implemented" << endl;
      exit(1);
  }

  for(unsigned int b=1; b < B; ++b) {
    double factor=1.0+b;
    for(unsigned int i=0; i < m; ++i) {
      F[b][i]=factor*F[0][i];
    }
  }
}

unsigned L,M;

double test(FFTpad *fft, Complex *f, Complex *F)
{
  cout << endl;

  unsigned int K=100000;
  utils::statistics S;

  for(unsigned int k=0; k < K; ++k) {
    for(unsigned int i=0; i < L; ++i) f[i]=i;
    double t0=utils::totalseconds();
    fft->forwards(f,F);
    double t=utils::totalseconds();
    S.add(t-t0);
  }

  double mean=S.mean();

  cout << "mean=" << mean << " +/- " << S.stdev() << endl;

  unsigned int N=fft->length();
  if(N < 10) {
    for(unsigned int i=0; i < N; ++i)
      cout << F[i] << endl;
  }

  return mean;
}

int main(int argc, char* argv[])
{
  fftw::maxthreads=1;//get_max_threads();

  const char *name="optimalSorted.dat";
  ifstream fin(name);
  if(!fin) {
    cerr << name << " not found" << endl;
    exit(-1);
  }
  nsize=0;
  while(true) {
    unsigned int i;
    double mean,stdev;
    fin >> i >> mean >> stdev;
    if(fin.eof()) break;
    size[nsize]=i;
    ++nsize;
  }

//  L=683;
//  M=1023;

//  L=512;
  L=1025;
  M=2*L;

  /*
  L=81;
  M=649;
  */

  Complex *f=ComplexAlign(L);

  FFTpad fft(f,L,M);

  cout << "L=" << L << endl;
  cout << "M=" << M << endl;

  cout << "Padding:" << fft.padding() << endl;

  unsigned int N=fft.length();
  Complex *F=ComplexAlign(N);

  double mean1=test(&fft,f,F);

  FFTpad fft0(f,L,M,N,1);

  Complex *F2=ComplexAlign(fft.length());
  double mean2=test(&fft0,f,F2);

  cout << endl;
  if(mean2 > 0)
    cout << "ratio=" << mean1/mean2 << endl;
  cout << endl;

  double error=0.0;
  double norm=0.0;
  for(unsigned int i=0; i < L; i++) {
    error += abs2(F2[i]-F[i]);
    norm += abs2(F[i]);
  }

  if(norm > 0) error=sqrt(error/norm);
  cout << "error=" << error << endl;
  if (error > 1e-12)
    cerr << "Caution! error=" << error << endl;

  return 0;
}
