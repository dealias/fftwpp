#include "convolution.h"
#include "explicit.h"
#include "direct.h"
#include "utils.h"
#include <unistd.h>

#include "Array.h"

using namespace std;
using namespace Array;
using namespace fftwpp;

// Number of iterations.
unsigned int N0=10000000;
unsigned int N=0;
unsigned int mx=4;
unsigned int my=4;
unsigned int mz=4;
unsigned int M=1;

bool Direct=false, Implicit=true, Explicit=false, Pruned=false;

inline void init(array3<Complex>& f0, array3<Complex>& f1, 
		 unsigned int mx, unsigned int my, unsigned int mz) 
{
  for(unsigned int i=0; i < mx; ++i) {
    for(unsigned int j=0; j < my; j++) {
      for(unsigned int k=0; k < mz; k++) {
	f0[i][j][k]=Complex(i+k,j+k);
	f1[i][j][k]=Complex(2*i+k,j+1+k);
      }
    }
  }
}
unsigned int outlimit=3000;

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  
  
  unsigned int A=2; // Number of inputs
  unsigned int B=1; // Number of outputs

#ifdef __GNUC__	
  optind=0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"hdeiptN:m:x:y:z:n:T:");
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
        Pruned=false;
        break;
      case 'i':
        Implicit=true;
        Explicit=false;
        break;
      case 'p':
        Explicit=true;
        Implicit=false;
        Pruned=true;
        break;
      case 'N':
        N=atoi(optarg);
        break;
      case 'm':
        mx=my=mz=atoi(optarg);
        break;
      case 'x':
        mx=atoi(optarg);
        break;
      case 'y':
        my=atoi(optarg);
        break;
      case 'z':
        mz=atoi(optarg);
        break;
      case 'n':
        N0=atoi(optarg);
        break;
      case 'T':
        fftw::maxthreads=atoi(optarg);
        break;
      case 'h':
      default:
        usage(3);
    }
  }

  if(my == 0) my=mx;

  cout << "mx=" << mx << ", my=" << my << ", mz=" << mz << endl;
  
  if(N == 0) {
    unsigned int nx=2*mx;
    unsigned int ny=2*my;
    unsigned int nz=2*mz;
    N=N0/nx/ny/nz;
    if(N < 10) N=10;
  }
  cout << "N=" << N << endl;
  
  size_t align=sizeof(Complex);
  array3<Complex> h0;
  if(Direct) h0.Allocate(mx,my,mz,align);
  double *T=new double[N];
  
  if(Implicit) {
    pImplicitConvolution3 C(mx,my,mz,A,B);

    // Allocate input arrays:
    Complex **f = new Complex *[A];
    for(unsigned int s=0; s < A; ++s)
      f[s]=ComplexAlign(mx*my*mz);
    array3<Complex> f0(mx,my,mz,f[0]);
    array3<Complex> f1(mx,my,mz,f[1]);
  
    for(unsigned int i=0; i < N; ++i) {
      init(f0,f1,mx,my,mz);
      seconds();
      C.convolve(f,multbinary);
//      C.convolve(f,g);
      T[i]=seconds();
    }
    
    timings("Implicit",mx,T,N);
        
    if(Direct) {
      for(unsigned int i=0; i < mx; i++) 
        for(unsigned int j=0; j < my; j++)
	  for(unsigned int k=0; k < mz; k++)
	    h0[i][j][k]=f0[i][j][k];
    }

    if(mx*my*mz < outlimit) 
      for(unsigned int i=0; i < mx; i++) {
        for(unsigned int j=0; j < my; j++) {
          for(unsigned int k=0; k < mz; k++) 
            cout << f0[i][j][k] << "\t";
          cout << endl;
        }
        cout << endl;
      } else cout << f0[0][0][0] << endl;
    cout << endl;
    
    // Free input arrays:
    for(unsigned int s=0; s < A; ++s) 
      deleteAlign(f[s]);
    delete[] f;

  }
  

  if(Direct) {
    array3<Complex> h(mx,my,mz,align);
    array3<Complex> f(mx,my,mz,align);
    array3<Complex> g(mx,my,mz,align);
    DirectConvolution3 C(mx,my,mz);
    init(f,g,mx,my,mz);
    seconds();
    C.convolve(h,f,g);
    T[0]=seconds();
  
    timings("Direct",mx,T,1);

    if(mx*my*mz < outlimit) {
      for(unsigned int i=0; i < mx; i++) {
        for(unsigned int j=0; j < my; j++) {
          for(unsigned int k=0; k < mz; k++)
            cout << h[i][j][k] << "\t";
          cout << endl;
        }
        cout << endl;
      }
    } else cout << h[0][0][0] << endl;

    { // compare implicit or explicit version with direct verion:
      double error=0.0;
      double norm=0.0;
      for(unsigned int i=0; i < mx; i++) {
        for(unsigned int j=0; j < my; j++) {
	  for(unsigned int k=0; k < mz; k++) {
	    error += abs2(h0[i][j][k]-h[i][j][k]);
	    norm += abs2(h[i][j][k]);
	  }
	}
      }
      error=sqrt(error/norm);
      cout << "error=" << error << endl;
      if (error > 1e-12) cerr << "Caution! error=" << error << endl;
    }

  }


  delete [] T;

  return 0;
}
