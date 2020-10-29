// TODO:
// check results
// optimize memory use
// vectorize and optimize Zeta computations
// use output strides
// use out-of-place transforms

#include <cassert>

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
  fft1d *fftM;
  fft1d *fftm;
  fft1d **fft;
  unsigned int S;
  Complex *ZetaH, *ZetaL;
  Complex *g,*h,*G,*e,*E; // Many, many work arrays!
public:

  void init(Complex *f) {
    if(p == q)
      fftM=new fft1d(M,sign);
    else {
      // Revisit memory allocation
      unsigned int N=q*m;
      S=p*N;
      BuildZeta(N,p*N,ZetaH,ZetaL,1,S);//,threads);

      g=ComplexAlign(m);
      G=ComplexAlign(N); // Rewrite so only used for n > 0.
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
//      m=L; q=2; // Temp
      n=0;
      unsigned int stop;
      unsigned int start;

      FFTpad fft(f,L,M,m,q);
      Complex *F=ComplexAlign(fft.length());
      seconds();
      for(unsigned int i=0; i < K; ++i) {
        for(unsigned int j=0; j < L; ++j)
          f[j]=j;
        fft.forwards(f,F);
      }
      double T=seconds();
      utils::deleteAlign(F);
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
    unsigned int pm=p*m;
    for(unsigned int i=0; i < L; ++i)
      F[i]=f[i];
    for(unsigned int i=L; i < pm; ++i)
      F[i]=0.0;

    if(p == q)
      return fftM->fft(F);
    unsigned int nsum=0;


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

    if(p == 1) {
      fftm->fft(F);
      for(unsigned int l=0; l < m; ++l)
        F[l*q]=g[l];
    } else {
      for(unsigned int s=0; s < m; ++s) {
        Complex sum=0.0;
        for(unsigned int t=nsum; t < p; ++t)
          sum += F[t*m+s];
//        g[s]=G[s]+sum;
        g[s]=sum;
      }
      fftm->fft(g);
      for(unsigned int l=0; l < m; ++l)
        F[l*q]=g[l];
    }

    for(unsigned int r=1; r < q; ++r) {
      for(unsigned int s=0; s < m; ++s) {
        Complex sum=0.0;
        for(unsigned int t=nsum; t < p; ++t) {
          unsigned int j=t*m+s;
          unsigned int c=r*j;// % N;
//          unsigned int a=c/S;
//          Complex Zeta=ZetaH[a]*ZetaL[c-S*a];
          Complex Zeta=ZetaL[c];
//          if(sign < 0) Zeta=conj(Zeta);
          sum += Zeta*F[j];
        }
//        g[s]=G[r*m+s]+sum;
        g[s]=sum;
      }
      fftm->fft(g);
      for(unsigned int l=0; l < m; ++l)
        F[l*q+r]=g[l];
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

int main(int argc, char* argv[])
{
  fftw::maxthreads=1;//get_max_threads();

  /*
  bool Direct=false;
  bool Implicit=true;
  bool Explicit=false;

  // Number of iterations.
  unsigned int N0=1000000000;
  unsigned int N=0;

  unsigned int m=11; // Problem size

  int stats=0; // Type of statistics used in timing test.

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

#ifdef __GNUC__
  optind=0;
#endif
  for (;;) {
    int c = getopt(argc,argv,"hdeiptA:B::N:m:n:S:T:");
    if (c == -1) break;

    switch (c) {
      case 0:
        break;
      case 'd':
        Direct=true;
        break;
      case 'e':
        Explicit=true;
        Implicit=false;
        break;
      case 'i':
        Implicit=true;
        Explicit=false;
        break;
      case 'p':
        break;
      case 'A':
        A=atoi(optarg);
        break;
      case 'B':
        B=atoi(optarg);
        break;
      case 'N':
        N=atoi(optarg);
        break;
      case 't':
        Test=true;
        break;
      case 'm':
        m=atoi(optarg);
        break;
      case 'n':
        N0=atoi(optarg);
        break;
      case 'T':
        fftw::maxthreads=max(atoi(optarg),1);
        break;
      case 'S':
        stats=atoi(optarg);
        break;
      case 'h':
      default:
        usage(1);
        usageTest();
        exit(1);
    }
  }
  */

  ifstream fin("optimalSorted.dat");
  nsize=0;
  while(true) {
    unsigned int i;
    double mean,stdev;
    fin >> i >> mean >> stdev;
    if(fin.eof()) break;
    size[nsize]=i;
    ++nsize;
  }

  unsigned L,M;
//  L=683;
//  M=1023;

  L=512;
//  L=2048;
  M=2*L;

  /*
  L=81;
  M=649;
  */

  Complex *f=ComplexAlign(L); // Temp

  FFTpad fft(f,L,M);

  cout << "L=" << L << endl;
  cout << "M=" << M << endl;

  cout << "Padding:" << fft.padding() << endl;

  for(unsigned int i=0; i < L; ++i)
    f[i]=i;

  Complex *F=ComplexAlign(fft.length());

  unsigned int K=1000;
  seconds();
  for(unsigned int i=0; i < K; ++i) {
    for(unsigned int j=0; j < L; ++j)
      f[j]=j;
    fft.forwards(f,F);
  }
  cout << seconds() << endl;


//  for(unsigned int i=0; i < M; ++i)
//    cout << F[i] << endl;

#if 0
  unsigned int n=cpadding(m);

  cout << "n=" << n << endl;
  cout << "m=" << m << endl;

  if(N == 0) {
    N=N0/n;
    N = max(N, 20);
  }
  cout << "N=" << N << endl;

  // Explicit and direct methods are only implemented for binary convolutions.
  if(!Implicit)
    A=2;

  if(B < 1)
    B=1;

  unsigned int np=Explicit ? n : m;
  unsigned int C=max(A,B);
  Complex *f=ComplexAlign(C*np);

  Complex **F=new Complex *[C];
  for(unsigned int s=0; s < C; ++s)
    F[s]=f+s*np;

  Complex *h0=NULL;
  if(Test || Direct)
    h0=ComplexAlign(m*B);

  double *T=new double[N];

  if(Implicit) {
    ImplicitConvolution C(m,A,B);
    cout << "threads=" << C.Threads() << endl << endl;

    multiplier *mult=NULL;
    switch(B) {
      case 1:
        switch(A) {
          case 1: mult=multautoconvolution; break;
          case 2: mult=multbinary; break;
          case 4: mult=multbinary2; break;
          case 6: mult=multbinary3; break;
          case 8: mult=multbinary4; break;
          case 16: mult=multbinary8; break;
          default:
            cerr << "A=" << A << ", B=" << B << " is not yet implemented"
                 << endl;
            exit(1);
        }
        break;
      default:
        mult=multA;
        break;
    }
    if(!mult) {
      cerr << "A=" << A << ", B=" << B << " is not yet implemented"
           << endl;
      exit(1);
    }

    for(unsigned int i=0; i < N; ++i) {
      init(F,m,A);
      seconds();
      C.convolve(F,mult);
      //C.convolve(F[0],F[1]);
      T[i]=seconds();
    }

    timings("Implicit",m,T,N,stats);

    if(m < 100) {
      for(unsigned int b=0; b < B; ++b) {
        for(unsigned int i=0; i < m; i++)
          cout << F[b][i] << endl;
        cout << endl;
      }
    }
    else {
      cout << f[0] << endl;
    }

    if(Test || Direct) {
      for(unsigned int b=0; b < B; ++b) {
        for(unsigned int i=0; i < m; i++) {
          h0[i+b*m]=F[b][i];
        }
      }
    }
  }

  if(Explicit) {
    ExplicitConvolution C(n,m,F[0]);
    for(unsigned int i=0; i < N; ++i) {
      init(F,m,A);
      seconds();
      C.convolve(F[0],F[1]);
      T[i]=seconds();
    }

    cout << endl;
    timings("Explicit",m,T,N,stats);

    if(m < 100)
      for(unsigned int i=0; i < m; i++)
        cout << F[0][i] << endl;
    else cout << F[0][0] << endl;
    cout << endl;
    if(Test || Direct)
      for(unsigned int i=0; i < m; i++)
        h0[i]=F[0][i];
  }

  if(Direct) {
    DirectConvolution C(m);
    if(A % 2 == 0 && A > 2)
      A=2;
    init(F,m,A);
    Complex *h=ComplexAlign(m);
    seconds();
    if(A % 2 == 0)
      C.convolve(h,F[0],F[1]);
    if(A == 1)
      C.autoconvolve(h,F[0]);
    T[0]=seconds();

    cout << endl;
    timings("Direct",m,T,1);

    if(m < 100)
      for(unsigned int i=0; i < m; i++)
        cout << h[i] << endl;
    else
      cout << h[0] << endl;

    { // compare implicit or explicit version with direct verion:
      double error=0.0;
      double norm=0.0;
      for(unsigned int b=0; b < B; ++b) {
        double factor=1.0+b;
        for(unsigned long long k=0; k < m; k++) {
          error += abs2(h0[k+b*m]-factor*h[k]);
          norm += abs2(h[k]);
        }
      }
      if(norm > 0) error=sqrt(error/norm);
      cout << "error=" << error << endl;
      if (error > 1e-12)
        cerr << "Caution! error=" << error << endl;
    }

    if(Test)
      for(unsigned int i=0; i < m; i++)
        h0[i]=h[i];
    deleteAlign(h);
  }

  if(Test) {
    Complex *h=ComplexAlign(n*B);
    // test accuracy of convolution methods:
    double error=0.0;
    cout << endl;
    double norm=0.0;

    bool testok=false;

    // Exact solutions for test case.
    if(A % 2 == 0) {
      testok=true;
      for(unsigned long long k=0; k < m; k++) {
        h[k]=iF*iG*(k+1)*pow(E,k*I);
        //  h[k]=iF*iG*(k*(k+1)/2.0*(k-(2*k+1)/3.0));
      }
    }

    // autoconvolution of f[k]=k
    if(A == 1) {
      testok=true;
      for(unsigned long long k=0; k < m; k++)
        h[k]=k*(0.5*k*(k+1)) - k*(k+1)*(2*k+1)/6.0;
    }

    if(!testok) {
      cout << "ERROR: no test case for A="<<A<<endl;
      exit(1);
    }

    for(unsigned long long k=0; k < m; k++) {
      error += abs2(h0[k]-h[k]);
      norm += abs2(h[k]);
    }
    if(norm > 0) error=sqrt(error/norm);
    cout << "error=" << error << endl;
    if (error > 1e-12)
      cerr << "Caution! error=" << error << endl;
    deleteAlign(h);
  }

  delete [] T;
  delete [] F;
  deleteAlign(f);

#endif
  return 0;
}
