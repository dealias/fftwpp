#include "convolution.h"
#include "explicit.h"
#include "direct.h"
#include "utils.h"
#include "Array.h"
#include <omp.h>

using namespace std;
using namespace Array;
using namespace fftwpp;

// Number of iterations.
unsigned int N0=10000000;
unsigned int N=0;

bool Direct=false, Implicit=true, Explicit=false, Test=false;

void pmult(Complex **f,
           unsigned int m, unsigned int M,
           unsigned int offset) {
  Complex* f0=f[0]+offset;
  Complex* f1=f[1]+offset;
  for(unsigned int j=0; j < m; ++j)
    f0[j] *= f1[j];
}

inline void init(array2<Complex> &f0, array2<Complex> &f1, 
		 unsigned int mx, unsigned int my) 
{
  for(unsigned int i=0; i < mx; ++i) {
    for(unsigned int j=0; j < my; ++j) {
      f0[i][j]=Complex(i,j);
      f1[i][j]=Complex(2*i,j+1);
    }
  }
}

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  
  
  // Problem Size:
  unsigned int m=4;
  unsigned int mx=4;
  unsigned int my=4;

  unsigned int A=2; // Number of inputs
  unsigned int B=1; // Number of outputs

#ifdef __GNUC__	
  optind=0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"hdiptM:N:m:T:");
    if (c == -1) break;
		
    switch (c) {
      case 0:
        break;
      case 'd':
        Direct=true;
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
        mx=my=atoi(optarg);
        break;
      case 'T':
        fftw::maxthreads=atoi(optarg);
        break;
      case 'h':
      default:
        usage(1,true);
    }
  }

  if(N == 0) {
    unsigned int nx=2*mx;
    unsigned int ny=2*my;
    N=N0/nx/ny;
    if(N < 10) N=10;
  }
  cout << "N=" << N << endl;
  
  Complex *h0=NULL;
  if(Test || Direct) h0=ComplexAlign(m);

  double *T=new double[N];

  if(Implicit) {
    
    // Allocate input arrays:
    Complex **f = new Complex *[A];
    for(unsigned int s=0; s < A; ++s)
      f[s]=ComplexAlign(mx*my);
  
    // Set up the input:
    array2<Complex> f0(mx,my,f[0]);
    array2<Complex> f1(mx,my,f[1]);
    init(f0,f1,mx,my);
    cout << "input:" << endl;
    cout << "f[0]:" << endl << f0 << endl;
    cout << "f[1]:" << endl << f1 << endl;

    // Creat a convolution object C:
    pImplicitConvolution2 C(mx,my,A,B);

    for(unsigned int i=0; i < N; ++i) {
      init(f0,f1,mx,my);
      seconds();
      C.convolve(f,pmult,0);
      T[i]=seconds();
    }

    timings("Implicit",mx,T,N);

    // Display output:
    if(m < 100) 
      cout << f0 << endl;
    else 
      cout << f0[0][0] << endl;

    if(Test || Direct) for(unsigned int i=0; i < m; i++) h0[i]=f0[0][i];
    
    for(unsigned int s=0; s < A; ++s) 
      deleteAlign(f[s]);
    delete[] f;
  }
  
  /*
  
  if(Direct) {
    DirectConvolution C(m);
    init(f,g);
    Complex *h=ComplexAlign(n);
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

  */
  delete [] T;


  return 0;
}
