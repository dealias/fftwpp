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

bool Direct=false, Implicit=true, Explicit=false, Pruned=false;


inline void init(Complex **F, 
		 unsigned int mx, unsigned int my, unsigned int mz, 
		 unsigned int A) 
{
  if(A %2 == 0) {
    unsigned int M=A/2;
    double factor=1.0/sqrt((double) M);
    for(unsigned int s=0; s < M; ++s) {
      double S=sqrt(1.0+s);
      double ffactor=S*factor;
      double gfactor=1.0/S*factor;
      array3<Complex> f(mx,my,mz,F[s]);
      array3<Complex> g(mx,my,mz,F[M+s]);
      for(unsigned int i=0; i < mx; ++i) {
	for(unsigned int j=0; j < my; j++) {
	  for(unsigned int k=0; k < mz; k++) {
	    f[i][j][k]=ffactor*Complex(i+k,j+k);
	    g[i][j][k]=gfactor*Complex(2*i+k,j+1+k);
	    //cout << f[i][j][k] << " ";
	  }
	  //cout << endl;
	}
	//cout << endl;
      }
    }
  } else {
    cerr << "Init not implemented for A=" << A << endl;
    exit(1);
  }
}

unsigned int outlimit=3000;

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

  unsigned int A=2; // Number of independent inputs
  unsigned int B=1; // Number of independent inputs
  
#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  
  
#ifdef __GNUC__	
  optind=0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"hdeiptA:B:M:N:m:x:y:z:n:T:");
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
      case 'A':
        A=atoi(optarg);
        break;
      case 'B':
        B=atoi(optarg);
        break;
      case 'M':
        A=2*atoi(optarg);
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
        fftw::maxthreads=max(atoi(optarg),1);
        break;
      case 'h':
      default:
        usage(3);
    }
  }

  
  if(my == 0) my=mx;

  nx=cpadding(mx);
  ny=cpadding(my);
  nz=cpadding(mz);
  
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

  if(!Implicit)
    A=2;

  // Allocate input/ouput memory and set up pointers
  Complex **F=new Complex *[A];
  for(unsigned int a=0; a < A; ++a)
    F[a]=ComplexAlign(nxp*nyp*nzp);

  // For easy access of first element
  array3<Complex> f(mx,my,mz,F[0]);

  double *T=new double[N];
  
  if(Implicit) {
    multiplier *mult;
  
    switch(A) {
    case 2: mult=multbinary; break;
    case 4: mult=multbinary2; break;
    case 6: mult=multbinary3; break;
    case 8: mult=multbinary4; break;
    case 10: mult=multbinary8; break;
    default: cout << "mult for A=" << A 
		  << " is not yet implemented" << endl; exit(1);
    }

    ImplicitConvolution3 C(mx,my,mz,A,B);
    cout << "Using " << C.Threads() << " threads."<< endl;
    for(unsigned int i=0; i < N; ++i) {
      init(F,mx,my,mz,A);
      seconds();
      C.convolve(F,mult);
//      C.convolve(F[0],F[1]);
      T[i]=seconds();
    }
    
    timings("Implicit",mx,T,N);
    
    if(Direct)
      for(unsigned int i=0; i < mx; i++) 
        for(unsigned int j=0; j < my; j++)
	  for(unsigned int k=0; k < mz; k++)
	    h0[i][j][k]=f[i][j][k];

    if(mx*my*mz < outlimit) 
      cout << f << endl;
    else 
      cout << f[0][0][0] << endl;

    cout << endl;
  }
  
  if(Explicit) {
    ExplicitConvolution3 C(nx,ny,nz,mx,my,mz,f,Pruned);

    for(unsigned int i=0; i < N; ++i) {
      init(F,nx,ny,nz,2);
      seconds();
      C.convolve(F[0],F[1]);
      T[i]=seconds();
    }
    timings(Pruned ? "Pruned" : "Explicit",mx,T,N);

    if(Direct) {
      for(unsigned int i=0; i < mx; i++) 
        for(unsigned int j=0; j < my; j++)
	  for(unsigned int k=0; k < mz; k++)
	    h0[i][j][k]=F[0][i*nx*ny + j*ny + k];
    }

    if(mx*my*mz < outlimit) {
      for(unsigned int i=0; i < mx; i++) {
        for(unsigned int j=0; j < my; j++) {
          for(unsigned int k=0; k < mz; k++)
            cout << F[0][i*nx*ny + j*ny + k] << " ";
          cout << endl;
        }
        cout << endl;
      }
    } else { 
      cout << F[0][0] << endl;
    }
  }

  if(Direct) {
    array3<Complex> h(mx,my,mz,align);
    DirectConvolution3 C(mx,my,mz);
    init(F,mx,my,mz,2);
    seconds();
    C.convolve(h,F[0],F[1]);
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
      if(norm > 0) error=sqrt(error/norm);
      cout << "error=" << error << endl;
      if (error > 1e-12) 
	cerr << "Caution! error=" << error << endl;
    }

  }
  
  delete [] T;
  for(unsigned int a=0; a < A; ++a)
    deleteAlign(F[a]);
  delete [] F;

  return 0;
}
