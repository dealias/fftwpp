#include "convolution.h"
#include "explicit.h"
#include "direct.h"
#include "utils.h"
#include "Array.h"

using namespace std;
using namespace utils;
using namespace fftwpp;

// Number of iterations.
unsigned int N0=10000000;
unsigned int N=0;
unsigned int m=12;
unsigned int M=1;
unsigned int B=1;
  
bool Direct=false, Implicit=true, Explicit=false;

inline void init(Complex *e, Complex *f, Complex *g, unsigned int M=1) 
{
  unsigned int m1=m+1;
  unsigned int Mm=M*m1;
  double factor=1.0/cbrt((double) M);
  for(unsigned int i=0; i < Mm; i += m1) {
    double s=sqrt(1.0+i);
    double efactor=1.0/s*factor;
    double ffactor=(1.0+i)*s*factor;
    double gfactor=1.0/(1.0+i)*factor;
    Complex *ei=e+i;
    Complex *fi=f+i;
    Complex *gi=g+i;
    ei[0]=1.0*efactor;
    for(unsigned int k=1; k < m; k++) ei[k]=efactor*Complex(k,k+1);
    fi[0]=1.0*ffactor;
    for(unsigned int k=1; k < m; k++) fi[k]=ffactor*Complex(k,k+1);
    gi[0]=2.0*gfactor;
    for(unsigned int k=1; k < m; k++) gi[k]=gfactor*Complex(k,2*k+1);
  }
}

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

  unsigned int stats=0; // Type of statistics used in timing test.

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  
  
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"hdeipA:B:N:m:n:T:S:");
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
        M=2*atoi(optarg);
        break;
      case 'B':
        B=atoi(optarg);
        break;
      case 'N':
        N=atoi(optarg);
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
        usageExplicit(1);
        exit(0);
    }
  }

  unsigned int n=tpadding(m);

  cout << "n=" << n << endl;
  cout << "m=" << m << endl;
  
  if(N == 0) {
    N=N0/n;
    N = max(N, 20);
  }
  cout << "N=" << N << endl;
  
  Complex *h0=NULL;
  if(Direct && ! Explicit) h0=ComplexAlign(m);

  unsigned int m1=m+1;
  unsigned int np=Explicit ? n/2+1 : m1;
  if(Implicit) np *= M;
    
  if(B != 1) {
    cerr << "B=" << B << " is not yet implemented" << endl;
    exit(1);
  }
    
  Complex *e=ComplexAlign(np);
  Complex *f=ComplexAlign(np);
  Complex *g=ComplexAlign(np);

  double *T=new double[N];

  if(Implicit) {
    ImplicitHTConvolution C(m,M);
    cout << "Using " << C.Threads() << " threads."<< endl;
    Complex **E=new Complex *[M];
    Complex **F=new Complex *[M];
    Complex **G=new Complex *[M];
    for(unsigned int s=0; s < M; ++s) {
      unsigned int sm=s*m1;
      E[s]=e+sm;
      F[s]=f+sm;
      G[s]=g+sm;
    }
    for(unsigned int i=0; i < N; ++i) {
      init(e,f,g,M);
      seconds();
      C.convolve(E,F,G);
//      C.convolve(e,f,g);
      T[i]=seconds();
    }
    
    timings("Implicit",m,T,N,stats);

    if(Direct) for(unsigned int i=0; i < m; i++) h0[i]=e[i];

    if(m < 100) 
      for(unsigned int i=0; i < m; i++) cout << e[i] << endl;
    else cout << e[0] << endl;
    
    delete [] G;
    delete [] F;
    delete [] E;
  }
  
  if(Explicit) {
    ExplicitHTConvolution C(n,m,f);
    for(unsigned int i=0; i < N; ++i) {
      init(e,f,g);
      seconds();
      C.convolve(e,f,g);
      T[i]=seconds();
    }
    
    timings("Explicit",m,T,N,stats);

    if(m < 100) 
      for(unsigned int i=0; i < m; i++) cout << e[i] << endl;
    else cout << e[0] << endl;
  }
  
  if(Direct) {
    DirectHTConvolution C(m);
    init(e,f,g);
    Complex *h=ComplexAlign(m);
    seconds();
    C.convolve(h,e,f,g);
    T[0]=seconds();
    
    timings("Direct",m,T,1);

    if(m < 100) 
      for(unsigned int i=0; i < m; i++) cout << h[i] << endl;
    else cout << h[0] << endl;

    if(Implicit) { // compare implicit or explicit version with direct verion:
      double error=0.0;
      cout << endl;
      double norm=0.0;
      for(unsigned long long k=0; k < m; k++) {
        error += abs2(h0[k]-h[k]);
        norm += abs2(h[k]);
      }
      if(norm > 0) error=sqrt(error/norm);
      cout << "error=" << error << endl;
      if (error > 1e-12) cerr << "Caution! error=" << error << endl;
    }

    deleteAlign(h);
  }

  deleteAlign(g);
  deleteAlign(f);
  deleteAlign(e);
  
  delete [] T;

  return 0;
}
