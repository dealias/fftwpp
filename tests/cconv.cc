#include "convolution.h"
#include "explicit.h"
#include "direct.h"
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

inline void init(Complex **F, unsigned int m, unsigned int A) 
{
  if(A % 2 == 0) { // binary case
    unsigned int M=A/2;
    double factor=1.0/sqrt((double) M);
    for(unsigned int s=0; s < M; ++s) {
      double ffactor=(1.0+s)*factor;
      double gfactor=1.0/(1.0+s)*factor;

      Complex *fs=F[s];
      Complex *gs=F[M+s];
      if(Test) {
        for(unsigned int k=0; k < m; k++) {
	  fs[k]=factor*iF*pow(E,k*I);
	  gs[k]=factor*iG*pow(E,k*I);
	  // fs[k]=factor*iF*k;
	  // gs[k]=factor*iG*k;
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

void multA2(Complex **F, unsigned int m,
	    const unsigned int indexsize,
	    const unsigned int *index,
	    unsigned int r, unsigned int threads)
{
  Complex* F0=F[0];
  Complex* F1=F[1];
  
  for(unsigned int j=0; j < m; ++j)
    F0[j] *= F1[j];
  
  for(unsigned int b=1; b < B; ++b) {
    double factor=1.0+b;
    for(unsigned int i=0; i < m; ++i) {
      F[b][i]=factor*F0[i]+1.0;
    }
  }
}

void multA4(Complex **F, unsigned int m,
	    const unsigned int indexsize,
	    const unsigned int *index,
	    unsigned int r, unsigned int threads)
{
  Complex* F0=F[0];
  Complex* F1=F[1];
  Complex* F2=F[2];
  Complex* F3=F[3];
  
  for(unsigned int j=0; j < m; ++j)
    F0[j]=F0[j]*F2[j]+F1[j]*F3[j];
  
  for(unsigned int b=1; b < B; ++b) {
    double factor=1.0+b;
    for(unsigned int i=0; i < m; ++i) {
      F[b][i]=factor*F0[i]+1.0;
    }
  }
}

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

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
	switch(A) {
	  case 2: mult=multA2; break;
	  case 4: mult=multA4; break;
	}
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
      cout << endl;
      double norm=0.0;
      for(unsigned int b=1; b < B; ++b) {
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

  return 0;
}
