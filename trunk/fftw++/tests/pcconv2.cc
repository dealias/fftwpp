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

unsigned int outlimit=100;

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  
  
  // Problem Size:
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
  
  double *T=new double[N];
  array2<Complex> h0;
  size_t align=sizeof(Complex);
  if(Direct) h0.Allocate(mx,my,align);

  if(Implicit) {
    
    // Allocate input arrays:
    Complex **f = new Complex *[A];
    for(unsigned int s=0; s < A; ++s)
      f[s]=ComplexAlign(mx*my);
  
    // Set up the input:
    array2<Complex> f0(mx,my,f[0]);
    array2<Complex> f1(mx,my,f[1]);
    init(f0,f1,mx,my);
//    cout << "input:" << endl;
//    cout << "f[0]:" << endl << f0 << endl;
//    cout << "f[1]:" << endl << f1 << endl;

    // Creat a convolution object C:
    PImplicitConvolution2 C(mx,my,A,B);

    for(unsigned int i=0; i < N; ++i) {
      init(f0,f1,mx,my);
      seconds();
      C.convolve(f,multbinary);
      T[i]=seconds();
    }

    timings("Implicit",mx,T,N);

    // Display output:
    if(mx*my < outlimit) 
      cout << f0 << endl;
    else 
      cout << f0[0][0] << endl;

    if(Test || Direct) {
      for(unsigned int i=0; i < mx; ++i) {
	for(unsigned int j=0; j < my; ++j) {
	  h0[i][j]=f0[i][j];
	}
      }
    }
    
    for(unsigned int s=0; s < A; ++s) 
      deleteAlign(f[s]);
    delete[] f;
  }
  
  if(Direct) {
    array2<Complex> h(mx,my,align);
    array2<Complex> f(mx,my,align);
    array2<Complex> g(mx,my,align);
    DirectConvolution2 C(mx,my);
    init(f,g,mx,my);
    seconds();
    C.convolve(h,f,g);
    T[0]=seconds();
  
    timings("Direct",mx,T,1);
    
    if(mx*my < outlimit) 
      for(unsigned int i=0; i < mx; i++) {
        for(unsigned int j=0; j < my; j++)
          cout << h[i][j] << "\t";
        cout << endl;
      } else cout << h[0][0] << endl;

    { // compare implicit or explicit version with direct verion:
      double error=0.0;
      cout << endl;
      double norm=0.0;
      for(unsigned int i=0; i < mx; i++) {
        for(unsigned int j=0; j < my; j++) {
	  error += abs2(h0[i][j]-h[i][j]);
	  norm += abs2(h[i][j]);
	}
      }
      error=sqrt(error/norm);
      cout << "error=" << error << endl;
      if (error > 1e-12) cerr << "Caution! error=" << error << endl;
    }

  }
  /*    
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
