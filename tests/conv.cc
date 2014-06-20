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
unsigned int m=11;
unsigned int M=1;
  
const Complex I(0.0,1.0);
const double E=exp(1.0);
const double F=sqrt(3.0);
const double G=sqrt(5.0);

bool Direct=false, Implicit=true, Explicit=false, Test=false;

unsigned int A, B; // number of inputs and outputs

// Pair-wise binary multiply for even A and B=1.
// NB: example function, not optimised or threaded.
void mymult(double ** F, unsigned int m, unsigned int threads)
{
  for(unsigned int i=0; i < m; ++i) {
    F[0][i] *= F[1][i];
    for(unsigned int a=2; a < A; a += 2) 
      F[0][i] += F[a][i]*F[a+1][i];
  }
}

// Pair-wise binary multiply for even A.
// All of the B outputs are identical.
// NB: example function, not optimised or threaded.
void mymultB(double ** F, unsigned int m, unsigned int threads)
{
  mymult(F,m,threads);
  // copy output
  for(unsigned int i=0; i < m; ++i)
    for(unsigned int b=0; b < B; b++) 
      F[b][i]=F[0][i];
}

inline void init(Complex *f, Complex *g, unsigned int A=2) 
{
  if (A%2 != 0) {
    cerr << "A=" << A << " is not yet implemented" << endl; 
    exit(1);
  }
  unsigned int M=A/2;
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
      fi[0]=1.0*ffactor;
      for(unsigned int k=1; k < m; k++) fi[k]=ffactor*Complex(k,k+1);
      gi[0]=2.0*gfactor;
      for(unsigned int k=1; k < m; k++) gi[k]=gfactor*Complex(k,2*k+1);
    }
  }
}

unsigned int padding(unsigned int m)
{
  unsigned int n=3*m-2;
  cout << "min padded buffer=" << n << endl;
  unsigned int log2n;
  // Choose next power of 2 for maximal efficiency.
  for(log2n=0; n > ((unsigned int) 1 << log2n); log2n++);
  return 1 << log2n;
}

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

  A=2; // Number of inputs
  B=1; // Number of outputs
  unsigned int Bcheck=0; // Which output to check

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  
  
#ifdef __GNUC__	
  optind=0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"hdeiptM:A:B:b:N:m:n:T:");
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
	A=2*M; // Number of independent inputs
        break;
      case 'A':
        A=atoi(optarg);
        break;
      case 'B':
        B=atoi(optarg);
        break;
      case 'b':
        Bcheck=atoi(optarg);
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
      case 'h':
      default:
        usage(1,true,true,true);
	usageA();
	usageB();
	exit(1);
    }
  }

  unsigned int n=padding(m);
  
  cout << "n=" << n << endl;
  cout << "m=" << m << endl;
  
  if(N == 0) {
    N=N0/n;
    if(N < 10) N=10;
  }
  cout << "N=" << N << endl;
  
  unsigned int np=Explicit ? n/2+1 : m;
  if(Implicit) np *= A;
    
  Complex *f=ComplexAlign(np);
  Complex *g=ComplexAlign(np);

  Complex *h0=NULL;
  if(Test || Direct) h0=ComplexAlign(m);

  double* T=new double[N];

  if(Implicit) {
    ImplicitHConvolution C(m,A,B);
    cout << "threads=" << C.Threads() << endl << endl;

    if (A%2 != 0) {
      cerr << "A=" << A << " is not yet implemented" << endl; 
      exit(1);
    }
    
    realmultiplier *mult;
    if(B == 1) {
      switch(A) {
      case 2: mult=multbinary; break;
      case 4: mult=multbinary2; break;
      default: mult=mymult;
      }
    } else
      mult=mymultB;
    
    Complex **F=new Complex *[A];
    for(unsigned int s=0; s < A/2; ++s) {
      unsigned int sm=s*m;
      F[2*s]=f+sm;
      F[2*s+1]=g+sm;
    }

    for(unsigned int i=0; i < N; ++i) {
      init(f,g,A);
      seconds();
      C.convolve(F,mult);
//      C.convolve(f,g);
      T[i]=seconds();
    }

    timings("Implicit",m,T,N);
    
    if(m < 100) 
      for(unsigned int i=0; i < m; i++) cout << F[Bcheck][i] << endl;
    else cout << f[0] << endl;
    if(Test || Direct)
      for(unsigned int i=0; i < m; i++) h0[i]=F[Bcheck][i];
    
    delete [] F;
  }
  
  if(Explicit) {
    ExplicitHConvolution C(n,m,f);
    for(unsigned int i=0; i < N; ++i) {
      init(f,g);
      seconds();
      C.convolve(f,g);
      T[i]=seconds();
    }

    timings("Explicit",m,T,N);

    if(m < 100) 
      for(unsigned int i=0; i < m; i++) cout << f[i] << endl;
    else cout << f[0] << endl;
    if(Test || Direct) 
      for(unsigned int i=0; i < m; i++) h0[i]=f[i];
  }
  
  if(Direct) {
    DirectHConvolution C(m);
    init(f,g);
    Complex *h=ComplexAlign(m);
    seconds();
    C.convolve(h,f,g);
    T[0]=seconds();
    
    timings("Direct",m,T,1);

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
      if(norm > 0) error=sqrt(error/norm);
      cout << "error=" << error << endl;
      if (error > 1e-12) cerr << "Caution! error=" << error << endl;
    }

    if(Test) 
      for(unsigned int i=0; i < m; i++) h0[i]=h[i];
    deleteAlign(h);
  }

  if(Test) {
    Complex *h=ComplexAlign(m);
    double error=0.0;
    cout << endl;
    double norm=0.0;
    long long M=m;
    for(long long k=0; k < M; k++) {
      h[k]=F*G*(2*M-1-k)*pow(E,k*I);
//      h[k]=F*G*(4*m*m*m-6*(k+1)*m*m+(6*k+2)*m+3*k*k*k-3*k)/6.0;
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
  deleteAlign(f);
  deleteAlign(g);

  return 0;
}
