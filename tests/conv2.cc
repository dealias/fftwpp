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
unsigned int mx=4;
unsigned int my=4;
unsigned int nxp;
unsigned int nyp;
unsigned int M=1;

bool Direct=false, Implicit=true, Explicit=false, Pruned=false;

unsigned int outlimit=100;

inline void init(array2<Complex>& f, array2<Complex>& g, unsigned int M=1,
                 unsigned int extra=0)
{
  unsigned int offset=Explicit ? nx/2-mx+1 : extra;
  unsigned int stop=2*mx-1;
  unsigned int stopoffset=stop+(Explicit ? 2*offset-1 : extra);
  double factor=1.0/sqrt((double) M);
  for(unsigned int s=0; s < M; ++s) {
    double S=sqrt(1.0+s);
    double ffactor=S*factor;
    double gfactor=1.0/S*factor;
    for(unsigned int i=0; i < stop; ++i) {
      unsigned int I=s*stopoffset+i+offset;
      for(unsigned int j=0; j < my; j++) {
        f[I][j]=ffactor*Complex(i,j);
        g[I][j]=gfactor*Complex(2*i,j+1);
      }
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

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif
  
#ifdef __GNUC__	
  optind=0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"hdeiptM:N:m:x:y:n:T:");
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
        mx=my=atoi(optarg);
        break;
      case 'x':
        mx=atoi(optarg);
        break;
      case 'y':
        my=atoi(optarg);
        break;
      case 'n':
        N0=atoi(optarg);
        break;
      case 'T':
        fftw::maxthreads=max(atoi(optarg),1);
        break;
      case 'h':
      default:
        usage(2);
    }
  }

  unsigned int A=2*M; // Number of independent inputs
  
  nx=padding(mx);
  ny=padding(my);
  
  cout << "nx=" << nx << ", ny=" << ny << endl;
  cout << "mx=" << mx << ", my=" << my << endl;
  
  if(N == 0) {
    N=N0/nx/ny;
    if(N < 10) N=10;
  }
  cout << "N=" << N << endl;
    
  size_t align=sizeof(Complex);

  array2<Complex> h0;
  if(Direct && Implicit) h0.Allocate(mx,my,align);

  nxp=Explicit ? nx : 2*mx;
  nyp=Explicit ? ny/2+1 : my;
  unsigned int nxp0=nxp*M;
  array2<Complex> f(nxp0,nyp,align);
  array2<Complex> g(nxp0,nyp,align);

  double *T=new double[N];

  if(Implicit) {
    ImplicitHConvolution2 C(mx,my,A);
    cout << "threads=" << C.Threads() << endl << endl;

    realmultiplier *mult;
    switch(M) {
      case 1: mult=multbinary; break;
      case 2: mult=multbinary2; break;
        
      default: cerr << "M=" << M << " is not yet implemented" << endl; exit(1);
    }
    
    Complex **F=new Complex *[A];
    unsigned int mf=nxp*nyp;
    for(unsigned int s=0; s < M; ++s) {
      unsigned int smf=s*mf;
      F[2*s]=f+smf;
      F[2*s+1]=g+smf;
    }

    for(unsigned int i=0; i < N; ++i) {
      init(f,g,M,1);
      seconds();
      C.convolve(F,mult);
//      C.convolve(f,g);
      T[i]=seconds();
    }
    
    timings("Implicit",mx,T,N);

    if(Direct) {
      for(unsigned int i=0; i < mx; i++) 
        for(unsigned int j=0; j < my; j++)
	  h0[i][j]=f[i+1][j];
    }
    
    if(nxp*my < outlimit)
      for(unsigned int i=1; i < nxp; i++) {
        for(unsigned int j=0; j < my; j++)
          cout << f[i][j] << "\t";
        cout << endl;
      } else cout << f[1][0] << endl;
    cout << endl;
    
    delete [] F;
  }
  
  if(Explicit) {
    ExplicitHConvolution2 C(nx,ny,mx,my,f,M,Pruned);
    Complex **F=new Complex *[M];
    Complex **G=new Complex *[M];
    unsigned int mf=nxp*nyp;
    for(unsigned int s=0; s < M; ++s) {
      unsigned int smf=s*mf;
      F[s]=f+smf;
      G[s]=g+smf;
    }
    for(unsigned int i=0; i < N; ++i) {
      init(f,g,M);
      seconds();
      C.convolve(F,G);
      T[i]=seconds();
    }

    timings(Pruned ? "Pruned" : "Explicit",mx,T,N);

    unsigned int offset=nx/2-mx+1;

    if(2*(mx-1)*my < outlimit) 
      for(unsigned int i=offset; i < offset+2*mx-1; i++) {
        for(unsigned int j=0; j < my; j++)
          cout << f[i][j] << "\t";
        cout << endl;
      } else cout << f[offset][0] << endl;
    
    delete [] G;
    delete [] F;
  }
  
  if(Direct) {
    Explicit=0;
    unsigned int nxp=2*mx-1;
    array2<Complex> h(nxp,my,align);
    array2<Complex> f(nxp,my,align);
    array2<Complex> g(nxp,my,align);
    DirectHConvolution2 C(mx,my);
    init(f,g);
    seconds();
    C.convolve(h,f,g);
    T[0]=seconds();
  
    timings("Direct",mx,T,1);

    if(nxp*my < outlimit)
      for(unsigned int i=0; i < nxp; i++) {
        for(unsigned int j=0; j < my; j++)
          cout << h[i][j] << "\t";
        cout << endl;
      } else cout << h[0][0] << endl;

    if(Implicit) { // compare implicit or explicit version with direct verion:
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
  
  delete [] T;
  
  return 0;
}
