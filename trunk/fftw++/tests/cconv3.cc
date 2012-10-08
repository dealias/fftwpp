#include "Complex.h"
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
unsigned int nx=0;
unsigned int ny=0;
unsigned int nz=0;
unsigned int mx=4;
unsigned int my=4;
unsigned int mz=4;
unsigned int M=1;

bool Direct=false, Implicit=true, Explicit=false, Pruned=false;

inline void init(array3<Complex>& f, array3<Complex>& g, unsigned int M=1) 
{
  double factor=1.0/sqrt((double) M);
  for(unsigned int s=0; s < M; ++s) {
    double S=sqrt(1.0+s);
    double ffactor=S*factor;
    double gfactor=1.0/S*factor;
    for(unsigned int i=0; i < mx; ++i) {
      unsigned int I=s*mx+i;
      for(unsigned int j=0; j < my; j++) {
        for(unsigned int k=0; k < mz; k++) {
          f[I][j][k]=ffactor*Complex(i+k,j+k);
          g[I][j][k]=gfactor*Complex(2*i+k,j+1+k);
        }
      }
    }
  }
}

unsigned int outlimit=300;

unsigned int padding(unsigned int m)
{
  unsigned int n=2*m;
  cout << "min padded buffer=" << n << endl;
  unsigned int log2n;
  // Choose next power of 2 for maximal efficiency.
  for(log2n=0; n > ((unsigned int) 1 << log2n); log2n++);
  return 1 << log2n;
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
    int c = getopt(argc,argv,"hdeiptM:N:m:x:y:n:");
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
      case 'M':
        M=atoi(optarg);
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
      case 'h':
      default:
        usage(3);
    }
  }

  if(my == 0) my=mx;

  nx=padding(mx);
  ny=padding(my);
  nz=padding(mz);
  
  cout << "nx=" << nx << ", ny=" << ny << ", nz=" << ny << endl;
  cout << "mx=" << mx << ", my=" << my << ", mz=" << mz << endl;
  
  if(N == 0) {
    N=N0/nx/ny/nz;
    if(N < 10) N=10;
  }
  cout << "N=" << N << endl;
  
  size_t align=sizeof(Complex);
  array3<Complex> h0;
  if(Direct) h0.Allocate(mx,my,mz,align);
  int nxp=Explicit ? nx : mx;
  int nyp=Explicit ? ny : my;
  int nzp=Explicit ? nz : mz;
  if(Implicit) nxp *= M;
  array3<Complex> f(nxp,nyp,nzp,align);
  array3<Complex> g(nxp,nyp,nzp,align);

  double *T=new double[N];
  
  if(Implicit) {
    ImplicitConvolution3 C(mx,my,mz,M);
    unsigned int mxyz=mx*my*mz;
    Complex **F=new Complex *[M];
    Complex **G=new Complex *[M];
    for(unsigned int s=0; s < M; ++s) {
      unsigned int smxyz=s*mxyz;
      F[s]=f+smxyz;
      G[s]=g+smxyz;
    }
    for(unsigned int i=0; i < N; ++i) {
      init(f,g,M);
      seconds();
      C.convolve(F,G);
//      C.convolve(f,g);
      T[i]=seconds();
    }
    
    timings("Implicit",T,N);
    
    if(Direct) {
      for(unsigned int i=0; i < mx; i++) 
        for(unsigned int j=0; j < my; j++)
	  for(unsigned int k=0; k < mz; k++)
	    h0[i][j][k]=f[i][j][k];
    }

    if(mx*my*mz < outlimit) 
      for(unsigned int i=0; i < mx; i++) {
        for(unsigned int j=0; j < my; j++) {
          for(unsigned int k=0; k < mz; k++) 
            cout << f[i][j][k] << "\t";
          cout << endl;
        }
        cout << endl;
      } else cout << f[0][0][0] << endl;
    cout << endl;
    
    delete [] G;
    delete [] F;
  }
  
  if(Explicit) {
    ExplicitConvolution3 C(nx,ny,nz,mx,my,mz,f,Pruned);
    for(unsigned int i=0; i < N; ++i) {
      init(f,g);
      seconds();
      C.convolve(f,g);
      T[i]=seconds();
    }
    
    timings(Pruned ? "Pruned" : "Explicit",T,N);

    if(Direct) {
      for(unsigned int i=0; i < mx; i++) 
        for(unsigned int j=0; j < my; j++)
	  for(unsigned int k=0; k < mz; k++)
	    h0[i][j][k]=f[i][j][k];
    }

    if(mx*my*mz < outlimit) {
      for(unsigned int i=0; i < mx; i++) {
        for(unsigned int j=0; j < my; j++) {
          for(unsigned int k=0; k < mz; k++)
            cout << f[i][j][k] << "\t";
          cout << endl;
        }
        cout << endl;
      }
    } else cout << f[0][0][0] << endl;
  }

  if(Direct) {
    array3<Complex> h(mx,my,mz,align);
    array3<Complex> f(mx,my,mz,align);
    array3<Complex> g(mx,my,mz,align);
    DirectConvolution3 C(mx,my,mz);
    init(f,g);
    seconds();
    C.convolve(h,f,g);
    T[0]=seconds();
  
    timings("Direct",T,1);

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
