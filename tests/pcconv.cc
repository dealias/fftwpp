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
unsigned int A=2; // Number of independent inputs

bool Direct=false, Implicit=true, Explicit=false, Test=false;

inline void init(Complex **f, unsigned int A)
{
  unsigned int M=A/2;
  double factor=1.0/sqrt((double) M);
  for(unsigned int s=0; s < M; ++s) {
    double ffactor=(1.0+s*m)*factor;
    double gfactor=1.0/(1.0+s*m)*factor;
    Complex *fs=f[2*s];
    Complex *gs=f[2*s+1];
    if(Test) {
      for(unsigned int k=0; k < m; k++) fs[k]=factor*F*pow(E,k*I);
      for(unsigned int k=0; k < m; k++) gs[k]=factor*G*pow(E,k*I);
//    for(unsigned int k=0; k < m; k++) fi[k]=factor*F*k;
//    for(unsigned int k=0; k < m; k++) gi[k]=factor*G*k;
    } else {
      for(unsigned int k=0; k < m; k++) fs[k]=ffactor*Complex(k,k+1);
      for(unsigned int k=0; k < m; k++) gs[k]=gfactor*Complex(k,2*k+1);
    }
  }
}

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

  unsigned int returnval=0;

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  
  
#ifdef __GNUC__	
  optind=0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"hdeiptA:N:m:n:T:");
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
        fftw::maxthreads=atoi(optarg);
        break;
      case 'h':
      default:
        usage(1,true);
    }
  }

  cout << "m=" << m << endl;
  
  if(N == 0) {
    
    n=2*m;
    unsigned int log2n;
    for(log2n=0; n > ((unsigned int) 1 << log2n); log2n++);
    n=1 << log2n;
    cout << "n=" << n << endl;
    N=N0/n;
    if(N < 10) N=10;
  }
  cout << "N=" << N << endl;
  
  Complex *h0=NULL;
  if(Test || Direct) h0=ComplexAlign(m);
  
  double *T=new double[N];

  pImplicitConvolution C(m,A);
  Complex **f=new Complex *[A];
  for(unsigned int s=0; s < A; ++s)
    f[s]=ComplexAlign(m);
  
  /*
  init(f,A);
  cout << "input:" << endl;
  for(unsigned int s=0; s < A; ++s) {
    for(unsigned int k=0; k < m; ++k)
      cout << f[s][k] << " ";
    cout << endl;
  }
  */
  void (*pmult)(Complex **,unsigned int,unsigned int)=NULL;
  if(A == 2) pmult=multbinary;
  if(A == 4) pmult=multbinarydot;
  if(A == 6) pmult=multbinarydot6;
  if(A == 8) pmult=multbinarydot8;
  if(A == 16) pmult=multbinarydot16;

  for(unsigned int i=0; i < N; ++i) {
    init(f,A);
    seconds();
    C.convolve(f,pmult);
    T[i]=seconds();
  }
  timings("Implicit",m,T,N);

  // output:
  if(m < 100) 
    for(unsigned int i=0; i < m; i++) cout << f[0][i] << endl;
  else 
    cout << f[0][0] << endl;
  if(Test || Direct) for(unsigned int i=0; i < m; i++) h0[i]=f[0][i];
  
  // Compare with direct convolution:
  if(Direct) {
    DirectConvolution C(m);
    init(f,2);
    Complex *h=ComplexAlign(n);
    seconds();
    C.convolve(h,f[0],f[1]);
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
      error=sqrt(error/norm);
      cout << "error=" << error << endl;
      if (error > 1e-12) {
	cerr << "Caution! error=" << error << endl;
	returnval=1;
      }
    }
  }
   
  // Compare with test-case, for which the exact solution is known:
  if(Test) {
    Complex *h=ComplexAlign(n);
    // test accuracy of convolution methods:
    double error=0.0;
    cout << endl;
    double norm=0.0;
    for(unsigned long long k=0; k < m; k++) {
      // Exact solution for test case
      h[k]=F*G*(k+1)*pow(E,k*I);
//      h[k]=F*G*(k*(k+1)/2.0*(k-(2*k+1)/3.0));
      error += abs2(h0[k]-h[k]);
      norm += abs2(h[k]);
    }
    error=sqrt(error/norm);
    cout << "error=" << error << endl;
    if (error > 1e-12) {
      cerr << "Caution! error=" << error << endl;
      returnval=1;
    }
    deleteAlign(h);
  }

  delete [] T;
  
  for(unsigned int s=0; s < A; ++s) 
    deleteAlign(f[s]);
  delete[] f;

  return returnval;
}
