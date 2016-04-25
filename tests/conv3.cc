#include "convolution.h"
#include "explicit.h"
#include "direct.h"
#include "utils.h"
#include "Array.h"

using namespace std;
using namespace utils;
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
unsigned int nxp;
unsigned int nyp;
unsigned int nzp;
bool xcompact=true;
bool ycompact=true;
bool zcompact=true;

bool Direct=false, Implicit=true;

unsigned int outlimit=300;

inline void init(Complex **F,
                 unsigned int mx, unsigned int my, unsigned int mz,
                 unsigned int nxp, unsigned int nyp, unsigned int nzp,
                 unsigned int A, bool xcompact, bool ycompact, bool zcompact)
{
  if(A % 2 == 0) {
    unsigned int M=A/2;
    unsigned int nx=2*mx-1;
    unsigned int ny=2*my-1;
    
    double factor=1.0/sqrt((double) M);
    for(unsigned int s=0; s < M; ++s) {
      double S=sqrt(1.0+s);
      double ffactor=S*factor;
      double gfactor=1.0/S*factor;

      array3<Complex> f(nxp,nyp,nzp,F[s]);
      array3<Complex> g(nxp,nyp,nzp,F[M+s]);

      if(!xcompact) {
        for(unsigned int j=0; j < ny+!ycompact; ++j) {
          for(unsigned int k=0; k < mz+!zcompact; ++k) {
            f[0][j][k]=0.0;
            g[0][j][k]=0.0;
          }
        }
      }
      if(!ycompact) {
        for(unsigned int i=0; i < nx+!xcompact; ++i) {
          for(unsigned int k=0; k < mz+!zcompact; ++k) {
            f[i][0][k]=0.0;
            g[i][0][k]=0.0;
          }
        }
      }

      if(!zcompact) {
        for(unsigned int i=0; i < nx+!xcompact; ++i) {
          for(unsigned int j=0; j < ny+!ycompact; ++j) {
            f[i][j][mz]=0.0;
            g[i][j][mz]=0.0;
          }
        }
      }

#pragma omp parallel for
      for(unsigned int i=0; i < nx; ++i) {
        unsigned int I=i+!xcompact;
        for(unsigned int j=0; j < ny; ++j) {
          unsigned int J=j+!ycompact;
          for(unsigned int k=0; k < mz; ++k) {
            f[I][J][k]=ffactor*Complex(i+k,j+k);
            g[I][J][k]=gfactor*Complex(2*i+k,j+1+k);
          }
        }
      }
    }
  } else {
    cerr << "Init not implemented for A=" << A << endl;
    exit(1);
  }
}


inline void init(array3<Complex>& f, array3<Complex>& g, unsigned int M=1,
                 bool xcompact=true, bool ycompact=true)
{
  unsigned int nx=2*mx-1;
  unsigned int ny=2*my-1;
  unsigned int nxoffset=nx+!xcompact;
  double factor=1.0/sqrt((double) M);
  for(unsigned int s=0; s < M; ++s) {
    double S=sqrt(1.0+s);
    double ffactor=S*factor;
    double gfactor=1.0/S*factor;
    for(unsigned int i=0; i < nx; ++i) {
      unsigned int I=s*nxoffset+i+!xcompact;
      for(unsigned int j=0; j < ny; ++j) {
        unsigned int J=j+!ycompact;
        for(unsigned int k=0; k < mz; ++k) {
          f[I][J][k]=ffactor*Complex(i+k,j+k);
          g[I][J][k]=gfactor*Complex(2*i+k,j+1+k);
        }
      }
    }
  }
}


int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();
  
  unsigned int A=2; // Number of independent inputs
  unsigned int B=1; // Number of outputs

  int stats=0; // Type of statistics used in timing test.

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  
  
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"hdeipA:B:N:m:x:y:z:n:T:S:X:Y:Z:");
    if (c == -1) break;
                
    switch (c) {
      case 0:
        break;
      case 'd':
        Direct=true;
        break;
      case 'e':
        Implicit=false;
        break;
      case 'i':
        Implicit=true;
        break;
      case 'p':
        Implicit=false;
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
      case 'S':
        stats=atoi(optarg);
        break;
      case 'X':
        xcompact=atoi(optarg) == 0;
        break;
      case 'Y':
        ycompact=atoi(optarg) == 0;
        break;
      case 'Z':
        zcompact=atoi(optarg) == 0;
        break;
      case 'h':
      default:
        usage(3);
        usageDirect();
        usageCompact(3);
        exit(1);
    }
  }

  nx=hpadding(mx);
  ny=hpadding(my);
  nz=hpadding(mz);
  
  cout << "nx=" << nx << ", ny=" << ny << ", nz=" << ny << endl;
  cout << "mx=" << mx << ", my=" << my << ", mz=" << mz << endl;
  
  if(N == 0) {
    N=N0/nx/ny/nz;
    N = max(N, 20);
  }
  cout << "N=" << N << endl;
    
  size_t align=sizeof(Complex);
  nxp=2*mx-xcompact;
  nyp=2*my-ycompact;
  nzp=mz+!zcompact;

  cout << "nxp=" << nxp << ", nyp" << nyp << ", nzp=" << nzp << endl;
  
  if(B < 1) B=1;
  if(B > A) {
    cerr << "B=" << B << " is not yet implemented for A=" << A << endl;
    exit(1);
  }
  
  Complex **F=new Complex *[A];
  for(unsigned int a=0; a < A; ++a)
    F[a]=ComplexAlign(nxp*nyp*nzp);

  // For easy access of first element
  array3<Complex> f(nxp,nyp,nzp,F[0]);

  array3<Complex> h0;
  if(Direct && Implicit) h0.Allocate(mx,my,mz,align);

  double *T=new double[N];

  if(Implicit) {
    ImplicitHConvolution3 C(mx,my,mz,xcompact,ycompact,zcompact,A,B);
    cout << "threads=" << C.Threads() << endl << endl;
    
    realmultiplier *mult;
    switch(A) {
      case 2: mult=multbinary; break;
      case 4: mult=multbinary2; break;
      default: cerr << "A=" << A << " is not yet implemented" << endl; exit(1);
    }

    for(unsigned int i=0; i < N; ++i) {
      init(F,mx,my,mz,nxp,nyp,nzp,A,xcompact,ycompact,zcompact);
      seconds();
      C.convolve(F,mult);
//      C.convolve(f,g);
      T[i]=seconds();
    }
    
    timings("Implicit",mx,T,N,stats);

    if(Direct) {
      for(unsigned int i=0; i < mx; i++) 
        for(unsigned int j=0; j < my; j++)
          for(unsigned int k=0; k < mz; k++)
            h0[i][j][k]=f[i+!xcompact][j+!ycompact][k];
    }

    if(nxp*nyp*mz < outlimit) {
      for(unsigned int i=!xcompact; i < nxp; ++i) {
        for(unsigned int j=!ycompact; j < nyp; ++j) {
          for(unsigned int k=0; k < mz; ++k)
            cout << f[i][j][k] << "\t";
          cout << endl;
        }
        cout << endl;
      }
    } else cout << f[!xcompact][!ycompact][0] << endl;
    
  }
  
  if(Direct) {
    unsigned int nxp=2*mx-1;
    unsigned int nyp=2*my-1;
      
    array3<Complex> h(nxp,nyp,mz,align);
    array3<Complex> f(nxp,nyp,mz,align);
    array3<Complex> g(nxp,nyp,mz,align);
    DirectHConvolution3 C(mx,my,mz);
    init(f,g);
    seconds();
    C.convolve(h,f,g);
    T[0]=seconds();

    timings("Direct",mx,T,1);

    if(nxp*nyp*mz < outlimit)
      for(unsigned int i=0; i < nxp; ++i) {
        for(unsigned int j=0; j < nyp; ++j) {
          for(unsigned int k=0; k < mz; ++k)
            cout << h[i][j][k] << "\t";
          cout << endl;
        }
        cout << endl;
      }
    else cout << h[0][0][0] << endl;
    
    if(Implicit) { // compare implicit version with direct verion:
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
      if (error > 1e-12) cerr << "Caution! error=" << error << endl;
    }

  }
  
  delete [] T;
  for(unsigned int a=0; a < A; ++a)
    deleteAlign(F[a]);
  delete [] F;
  

  return 0;
}
