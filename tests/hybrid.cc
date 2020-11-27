// TODO:
// decouple work memory were possible
// optimize memory use
// use out-of-place transforms
// vectorize and optimize Zeta computations
//          unsigned int a=c/S;
//          Complex Zeta=ZetaH[a]*ZetaL[c-S*a];

#include <cfloat>
#include <climits>

#include "convolution.h"
#include "utils.h"

using namespace std;
using namespace utils;
using namespace fftwpp;

// Constants used for initialization and testing.
const Complex I(0.0,1.0);
const double E=exp(1.0);
const Complex iF(sqrt(3.0),sqrt(7.0));
const Complex iG(sqrt(5.0),sqrt(11.0));

bool Test=false;

unsigned int A=2; // number of inputs
unsigned int B=1; // number of outputs

unsigned int surplusFFTsizes=25;

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
  unsigned int m;
  unsigned int p;
  unsigned int q;
  unsigned int n;
protected:
  fft1d *fftM;
  fft1d *ifftM;
  mfft1d *fftm;
  mfft1d *ifftm;
  mfft1d *ifftp;
  mfft1d *fftp;
  Complex *ZetaH,*ZetaL;
  Complex *ZetaHq,*ZetaLq;
  Complex *ZetaLqp;
  Complex *ZetaLqm;
  utils::statistics S;
  bool innerFFT;
  bool modular;
public:

  void init() {
    if(p != q) modular=true;
    if(m > M) M=m;
    if(!modular) {
      fftM=new fft1d(M,1);
      ifftM=new fft1d(M,-1);
    } else {
      unsigned int N=m*q;
      BuildZeta(N,N,ZetaH,ZetaL,1,N);//,threads);
      n=q/p;
      Complex *G=ComplexAlign(N);
      innerFFT=p > 1 && n*p == q;
      if(innerFFT) {
        BuildZeta(q,q,ZetaHq,ZetaLq,1,q);//,threads);
//      p->L, M->q, m->p, q->n
        unsigned int s=n*m;
        fftp=new mfft1d(p,1,m, s,s, 1,1, G,G);
        fftm=new mfft1d(m,1,q, 1,1, m,m, G,G);

        ifftp=new mfft1d(p,-1,n, n,1, 1,p, G,G);
        ifftm=new mfft1d(m,-1,q, 1,q, m,1, G,G);
      } else {
        ZetaLqp=ComplexAlign((q-1)*(p-1));
        for(unsigned int r=1; r < q; ++r)
          for(unsigned int t=1; m*t < L; ++t)
            ZetaLqp[(p-1)*(r-1)+(t-1)]=ZetaL[(m*r*t) % N];

        fftm=new mfft1d(m,1,q, 1,1, m,m, G,G);
        ifftm=new mfft1d(m,-1,q, 1,1, m,m, G,G);
      }

      ZetaLqm=ComplexAlign((q-1)*(m-1));
      for(unsigned int r=1; r < q; ++r)
        for(unsigned int s=1; s < m; ++s)
          ZetaLqm[(m-1)*(r-1)+s-1]=ZetaL[r*s];

      deleteAlign(G);
    }
  }

  // Compute an fft padded to N=m*q >= M >= L=f.length
  FFTpad(unsigned int L, unsigned int M,
         unsigned int m, unsigned int q) :
    L(L), M(M), m(m), p(ceilquotient(L,m)), q(q), modular(true) {
    init();
  }

  ~FFTpad() {
    if(!modular) {
      delete fftM;
      delete ifftM;
    } else {
      deleteAlign(ZetaL);
      deleteAlign(ZetaH);
      if(innerFFT) {
        deleteAlign(ZetaLq);
        deleteAlign(ZetaHq);
        delete fftp;
        delete ifftp;
      } else
        deleteAlign(ZetaLqp);
      deleteAlign(ZetaLqm);
      delete fftm;
      delete ifftm;
    }
  }

  class Opt {
  public:
    unsigned int m,q;
    double T;

    void check(unsigned int L, unsigned int M,
               unsigned int m, bool fixed=false) {
//      cout << "m=" << m << endl;
      unsigned int p=ceilquotient(L,m);
      unsigned int q=ceilquotient(M,m);
//      cout << "q=" << q << endl;

      if(!fixed) {
        unsigned int q2=p*ceilquotient(M,m*p);
        if(q2 != q) {
//            cout << "q2=" << q2 << endl;
          FFTpad fft(L,M,m,q2);
          double t=fft.meantime();
          if(t < T) {
            this->m=m;
            this->q=q2;
            T=t;
          }
        }
      }

      FFTpad fft(L,M,m,q);
      double t=fft.meantime();

      if(t < T) {
        this->m=m;
        this->q=q;
        T=t;
      }
    }

    // Determine optimal m,q values for padding L data values to
    // size >= M
    // If fixed=true then an FFT of size M is enforced.
    Opt(unsigned int L, unsigned int M, bool fixed=false, bool Explicit=false)
    {
      if(L > M) {
        cerr << "L=" << L << " is greater than M=" << M << "." << endl;
        exit(-1);
      }
      m=1;
      q=M;
      T=DBL_MAX;
      unsigned int i=0;

      unsigned int stop=M-1;
      for(unsigned int k=0; k < surplusFFTsizes; ++k)
        stop=nextfftsize(stop+1);

      unsigned int m0=1;

      /*
// Temp
      if(!Explicit)
        check(L,M,L,fixed || Explicit);
      else
      */
      while(true) {
        m0=nextfftsize(m0+1);
        if(Explicit) {
          if(m0 > stop) break;
          if(m0 < M) {++i; continue;}
          M=m0;
        } else if(m0 > stop) break;
//        } else if(m0 > L) break;
        if(!fixed || Explicit || M % m0 == 0)
          check(L,M,m0,fixed || Explicit);
        ++i;
      }

      unsigned int p=ceilquotient(L,m);
      cout << endl;
      cout << "Optimal values:" << endl;
      cout << "m=" << m << endl;
      cout << "p=" << p << endl;
      cout << "q=" << q << endl;
      cout << "Padding:" << m*p-L << endl;
    }
  };

  // Normal entry point.
  // Compute an fft of length L padded to at least M
  // (or exactly M if fixed=true)
  FFTpad(unsigned int L, unsigned int M, bool fixed=false,
         bool Explicit=false) : L(L), M(M), modular(!Explicit) {
    Opt opt=Opt(L,M,fixed,Explicit);
    m=opt.m;
    if(Explicit) this->M=M=m;
    p=ceilquotient(L,m);
    q=opt.q;
    init();
  }

  // TODO: Check for cases when arrays f and F must be distinct
  void forward(Complex *f, Complex *F) {
    if(!modular) {
      for(unsigned int i=0; i < L; ++i)
        F[i]=f[i];
      for(unsigned int i=L; i < M; ++i)
        F[i]=0.0;
      fftM->fft(F);
      return;
    }

    if(innerFFT) {
      for(unsigned int t=0; m*t < L; ++t) {
        unsigned int mt=m*t;
        Complex *Ft=F+n*mt;
        Complex *ft=f+mt;
        unsigned int stop=min(L-mt,m);
        for(unsigned int s=0; s < stop; ++s)
          Ft[s]=ft[s];
        for(unsigned int s=stop; s < m; ++s)
          Ft[s]=0.0;
      }
      fftp->fft(F);
      for(unsigned int r=1; r < n; ++r) {
        Complex *Fr=F+m*r;
        unsigned int stop=min(L,m);
        for(unsigned int s=0; s < stop; ++s)
          Fr[s]=f[s];
        for(unsigned int s=stop; s < m; ++s)
          Fr[s]=0.0;
        for(unsigned int t=1; m*t < L; ++t) {
          unsigned int mt=m*t;
          Complex *Frt=Fr+n*mt;
          Complex *ft=f+mt;
          unsigned int stop=min(L-mt,m);
          Complex Zeta=ZetaLq[r*t];
          for(unsigned int s=0; s < stop; ++s)
            Frt[s]=Zeta*ft[s];
          for(unsigned int s=stop; s < m; ++s)
            Frt[s]=0.0;
        }
        fftp->fft(Fr);
      }
      for(unsigned int r=1; r < q; ++r) {
        Complex *Fr=F+m*r;
        Complex *ZetaLr=ZetaLqm+(m-1)*(r-1)-1;
        for(unsigned int s=1; s < m; ++s)
          Fr[s] *= ZetaLr[s];
      }

      fftm->fft(F);
    } else {
      unsigned int stop=min(m,L);
      if(p == 1) {
        for(unsigned int i=0; i < L; ++i)
          F[i]=f[i];
        for(unsigned int i=L; i < m; ++i)
          F[i]=0.0;
        for(unsigned int r=1; r < q; ++r) {
          Complex *Fmr=F+m*r;
          Fmr[0]=f[0];
          Complex *ZetaLr=ZetaLqm+(m-1)*(r-1)-1;
          for(unsigned int s=1; s < stop; ++s)
            Fmr[s]=ZetaLr[s]*f[s];
          for(unsigned int s=stop; s < m; ++s)
            Fmr[s]=0.0;
        }
        fftm->fft(F);
      } else {
        for(unsigned int s=0; s < m; ++s) {
          Complex sum=0.0;
          for(unsigned int t=s; t < L; t += m)
            sum += f[t];
          F[s]=sum;
        }
        for(unsigned int r=1; r < q; ++r) {
          Complex *Fmr=F+m*r;
          Complex *ZetaLmr=ZetaLqm+(m-1)*(r-1)-1;
          Complex *ZetaLr=ZetaLqp+(p-1)*(r-1)-1;
          Complex sum=f[0];
          for(unsigned int t=1; m*t < L; ++t)
            sum += ZetaLr[t]*f[m*t];
          Fmr[0]=sum;
          for(unsigned int s=1; s < stop; ++s) {
            Complex *fs=f+s;
            Complex sum=fs[0];
            unsigned stop=L-s;
            for(unsigned int t=1; m*t < stop; ++t)
              sum += ZetaLr[t]*fs[m*t];
            Fmr[s]=sum*ZetaLmr[s];
          }
          for(unsigned int s=stop; s < m; ++s)
            Fmr[s]=0.0;
        }
        fftm->fft(F);
      }
    }
  }

// Compute an inverse fft of length N=q*m unpadded back
// to size p*m >= L.
  // Input F destroyed
  void backward(Complex *F, Complex *f) {
    if(!modular) {
      ifftM->fft(F);
      for(unsigned int i=0; i < L; ++i)
        f[i]=F[i];
      return;
    }

    if(F == f) {
      cerr << "input and output arrays must be distinct"
           << endl;
    }

    unsigned int N=m*q;

    ifftm->fft(F);

    if(innerFFT) {
      ifftp->fft(F);
      for(unsigned int s=1; s < m; ++s) {
        Complex *Fqs=F+q*s;
        for(unsigned int r=1; r < q; ++r)
          Fqs[r] *= conj(ZetaL[r*s]);

        ifftp->fft(Fqs);
      }

      for(unsigned int s=0; s < m; ++s)
        f[s]=F[q*s];
      for(unsigned int r=1; r < n; ++r) {
        Complex *Fr=F+r*p;
        for(unsigned int s=0; s < m; ++s)
          f[s] += Fr[q*s];
      }
      for(unsigned int t=1; t < p; ++t) {
        Complex *Ft=F+t;
        Complex *ft=f+m*t;
        for(unsigned int s=0; s < m; ++s)
            ft[s]=Ft[q*s];
        for(unsigned int r=1; r < n; ++r) {
          Complex *Ftr=Ft+r*p;
          Complex Zeta=conj(ZetaLq[r*t]);
          for(unsigned int s=0; s < m; ++s)
            ft[s] += Zeta*Ftr[q*s];
        }
      }
    } else {
      // Direct sum:
      if(p == 1) {
        Complex sum=F[0];
        for(unsigned int r=1; r < q; ++r)
          sum += F[m*r];
        f[0]=sum;
        for(unsigned int s=1; s < m; ++s) {
          Complex *Fs=F+s;
          Complex sum=Fs[0];
          for(unsigned int r=1; r < q; ++r)
            sum += conj(ZetaL[r*s])*Fs[m*r];
          f[s]=sum;
        }
      } else {
        unsigned int mp=m*p;
        for(unsigned int t=0; t < mp; t += m) {
          for(unsigned int s=0; s < m; ++s) {
            unsigned int K=t+s;
            Complex *Fs=F+s;
            Complex sum=Fs[0];
            for(unsigned int r=1; r < q; ++r) {
              unsigned int a=(r*K) % N;
              sum += conj(ZetaL[a])*Fs[m*r];
            }
            f[K]=sum;
          }
        }
      }
    }
  }

  unsigned int inverseLength() {
    return modular ? m*p : L;
  }

  unsigned int length() {
    return modular ? m*q : M;
  }

  double meantime(double *Stdev=NULL) {
    S.clear();
    Complex *F=ComplexAlign(length());
    Complex *f=ComplexAlign(inverseLength());

// Assume f != F (out-of-place)
    for(unsigned int j=0; j < L; ++j)
      f[j]=0.0;
    forward(f,F); // Create wisdom
    backward(F,f); // Create wisdom
    unsigned int K=1;

    double eps=0.1;

    for(;;) {
      double t0=totalseconds();
      for(unsigned int i=0; i < K; ++i) {
        forward(f,F);
        backward(F,f);
      }
      double t=totalseconds();
      S.add((t-t0)/K);

      double mean=S.mean();
      double stdev=S.stdev();
      if(S.count() < 7) continue;
      if(K*mean < 1000.0/CLOCKS_PER_SEC || stdev > eps*mean) {
        K *= 2;
        S.clear();
      } else {
        if(Stdev) *Stdev=stdev;
        deleteAlign(F);
        deleteAlign(f);
        return mean;
      }
    }

    deleteAlign(F);
    deleteAlign(f);

    if(Stdev) *Stdev=0.0;
    return 0.0;
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
  std::cerr << "-S\t\t number of surplus FFT sizes" << std::endl;
  std::cerr << "-L\t\t number of physical data values" << std::endl;
  std::cerr << "-M\t\t minimal number of padded data values" << std::endl;
  std::cerr << "-T\t\t number of threads" << std::endl;
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
    int c = getopt(argc,argv,"hL:M:S:T:");
    if (c == -1) break;

    switch (c) {
      case 0:
        break;
      case 'L':
        L=atoi(optarg);
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
      case 'h':
      default:
        usage();
        exit(1);
    }
  }

  cout << "L=" << L << endl;
  cout << "M=" << M << endl;

  // Minimal explicit padding
  FFTpad fft0(L,M,M,1);
  double mean0=report(fft0);

  // Optimal explicit padding
  FFTpad fft1(L,M,false,true);
  double mean1=report(fft1);

  // Hybrid padding
  FFTpad fft(L,M);
  double mean=report(fft);

  if(mean0 > 0)
    cout << "minimal ratio=" << mean/mean0 << endl;
  cout << endl;

  if(mean1 > 0)
    cout << "optimal ratio=" << mean/mean1 << endl;
  cout << endl;

  unsigned int N=fft.length();

  Complex *f=ComplexAlign(L);
  Complex *F=ComplexAlign(N);

  for(unsigned int j=0; j < L; ++j)
    f[j]=j+1;
  fft.forward(f,F);

  Complex *f0=ComplexAlign(fft.inverseLength());
  Complex *F0=ComplexAlign(N);

  for(unsigned int j=0; j < fft.length(); ++j)
    F0[j]=F[j];

  fft.backward(F0,f0);

  if(L < 30) {
    cout << endl;
    cout << "Inverse:" << endl;
    for(unsigned int j=0; j < L; ++j)
      cout << f0[j]/fft.length() << endl;
    cout << endl;
  }

  Complex *F2=ComplexAlign(N);
  FFTpad fft2(L,N,N,1);
  for(unsigned int j=0; j < L; ++j)
    f[j]=j+1;
  fft2.forward(f,F2);

  double error=0.0;
  double norm=0.0;

  unsigned int i=0;
  unsigned int m=fft.m;
  unsigned int q=fft.q;
  for(unsigned int s=0; s < m; ++s) {
    for(unsigned int r=0; r < q; ++r) {
      error += abs2(F[m*r+s]-F2[i]);
      norm += abs2(F2[i]);
      ++i;
    }
  }

  if(norm > 0) error=sqrt(error/norm);
  if(error > 1e-12)
    cerr << endl << "WARNING: " << endl;
  cout << "error=" << error << endl;

  return 0;
}
