#include "Complex.h"
#include "convolution.h"
#include "explicit.h"
#include "direct.h"
#include "utils.h"
#include <unistd.h>

using namespace std;
using namespace fftwpp;

// Number of iterations.
unsigned int N0=10000000;
unsigned int N=0;
  
const Complex I(0.0,1.0);
const double E=exp(1.0);
const Complex F(sqrt(3.0),sqrt(7.0));
const Complex G(sqrt(5.0),sqrt(11.0));

unsigned int m=11;
unsigned int n=2*m;
unsigned int M=1;

bool Direct=false, Implicit=true, Explicit=false, Test=false;

inline void init(Complex *f, Complex *g, unsigned int M=1) 
{
  unsigned int Mm=M*m;
  double factor=1.0/sqrt((double) M);
  for(unsigned int i=0; i < Mm; i += m) {
    double ffactor=(1.0+i)*factor;
    double gfactor=1.0/(1.0+i)*factor;
    Complex *fi=f+i;
    Complex *gi=g+i;
    if(Test) {
      for(unsigned int k=0; k < m; k++) fi[k]=factor*F*pow(E,k*I);
      for(unsigned int k=0; k < m; k++) gi[k]=factor*G*pow(E,k*I);
//    for(unsigned int k=0; k < m; k++) fi[k]=factor*F*k;
//    for(unsigned int k=0; k < m; k++) gi[k]=factor*G*k;
  } else {
      for(unsigned int k=0; k < m; k++) fi[k]=ffactor*Complex(k,k+1);
      for(unsigned int k=0; k < m; k++) gi[k]=gfactor*Complex(k,2*k+1);
    }
  }
}

void add(Complex *f, Complex *F) 
{
  for(unsigned int i=0; i < m; ++i)
      f[i] += F[i];
}

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  
  
#ifdef __GNUC__	
  optind=0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"hdeiptM:N:m:n:");
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
      case 'M':
        M=atoi(optarg);
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
      case 'h':
      default:
        usage(1,true);
    }
  }

  n=2*m;
  cout << "min padded buffer=" << n << endl;
  unsigned int log2n;
  // Choose next power of 2 for maximal efficiency.
  for(log2n=0; n > ((unsigned int) 1 << log2n); log2n++);
  n=1 << log2n;
  cout << "n=" << n << endl;
  cout << "m=" << m << endl;
  
  if(N == 0) {
    N=N0/n;
    if(N < 10) N=10;
  }
  cout << "N=" << N << endl;
  
  unsigned int np=Explicit ? n : m;
  if(Implicit) np *= M;
  Complex *f=ComplexAlign(np);
  Complex *g=ComplexAlign(np);

  Complex *h0=NULL;
  if(Test || Direct) h0=ComplexAlign(m);

  double *T=new double[N];

  if(Implicit) {
    ImplicitConvolution C(m,M);
    Complex **F=new Complex *[M];
    Complex **G=new Complex *[M];
    for(unsigned int s=0; s < M; ++s) {
      unsigned int sm=s*m;
      F[s]=f+sm;
      G[s]=g+sm;
    }
    for(unsigned int i=0; i < N; ++i) {
      init(f,g,M);
      seconds();
      C.convolve(F,G);
//      C.convolve(f,g);
      T[i]=seconds();
    }
    
    timings("Implicit",T,N);

    if(m < 100) 
      for(unsigned int i=0; i < m; i++) cout << f[i] << endl;
    else cout << f[0] << endl;
    if(Test || Direct) for(unsigned int i=0; i < m; i++) h0[i]=f[i];
    
    delete [] G;
    delete [] F;
  }
  
  if(Explicit) {
    ExplicitConvolution C(n,m,f);
    for(unsigned int i=0; i < N; ++i) {
      init(f,g);
      seconds();
      C.convolve(f,g);
      T[i]=seconds();
    }
    
    timings("Explicit",T,N);

    if(m < 100) 
      for(unsigned int i=0; i < m; i++) cout << f[i] << endl;
    else cout << f[0] << endl;
    cout << endl;
    if(Test || Direct) for(unsigned int i=0; i < m; i++) h0[i]=f[i];
  }
  
  if(Direct) {
    DirectConvolution C(m);
    init(f,g);
    Complex *h=ComplexAlign(n);
    seconds();
    C.convolve(h,f,g);
    T[0]=seconds();  
    
    timings("Direct",T,1);

    if(m < 100)
      for(unsigned int i=0; i < m; i++) cout << h[i] << endl;
    else cout << h[0] << endl;

    { // compare implicit or explicit version with direct verion:
      double error=0.0;
      cout << endl;
      double norm=0.0;
      for(unsigned long long k=0; k < m; k++) {
	error += abs2(h0[k]-h[k]);
	norm += abs2(h[k]);
      }
      error=sqrt(error/norm);
      cout << "error=" << error << endl;
      if (error > 1e-12) cerr << "Caution! error=" << error << endl;
    }
    
    if(Test) for(unsigned int i=0; i < m; i++) h0[i]=h[i];
    deleteAlign(h);
  }
    
  if(Test) {
    Complex *h=ComplexAlign(n);
    // test accuracy of convolution methods:
    double error=0.0;
    cout << endl;
    double norm=0.0;
    for(unsigned long long k=0; k < m; k++) {
      // exact solution for test case.
      h[k]=F*G*(k+1)*pow(E,k*I);
//      h[k]=F*G*(k*(k+1)/2.0*(k-(2*k+1)/3.0));
      error += abs2(h0[k]-h[k]);
      norm += abs2(h[k]);
    }
    error=sqrt(error/norm);
    cout << "error=" << error << endl;
    if (error > 1e-12)
      cerr << "Caution! error=" << error << endl;
    deleteAlign(h);
  }

  delete [] T;
  deleteAlign(g);
  deleteAlign(f);

  return 0;
}
