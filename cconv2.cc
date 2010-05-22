#include "Complex.h"
#include "convolution.h"
#include "utils.h"
#include "Array.h"

using namespace std;
using namespace Array;
using namespace fftwpp;

// g++ -g -O3 -DNDEBUG -fomit-frame-pointer -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse cconv2.cc fftw++.cc -lfftw3 -march=native

// icpc -O3 -ansi-alias -malign-double -fp-model fast=2 cconv2.cc fftw++.cc -lfftw3

// FFTW: 
// configure --enable-sse2 CC=icpc CFLAGS="-O3 -ansi-alias -malign-double -fp-model fast=2"

// Number of iterations.
unsigned int N0=10000000;
unsigned int N=0;
unsigned int nx=0;
unsigned int ny=0;
unsigned int mx=4;
unsigned int my=4;
unsigned int M=1;

bool Direct=false, Implicit=true, Explicit=false, Pruned=false;

inline void init(array2<Complex>& f, array2<Complex>& g, unsigned int M=1) 
{
  double factor=1.0/sqrt((double) M);
  for(unsigned int s=0; s < M; ++s) {
    double S=sqrt(1.0+s);
    double ffactor=S*factor;
    double gfactor=1.0/S*factor;
    for(unsigned int i=0; i < mx; ++i) {
      unsigned int I=s*mx+i;
      for(unsigned int j=0; j < my; j++) {
        f[I][j]=ffactor*Complex(i,j);
        g[I][j]=gfactor*Complex(2*i,j+1);
      }
    }
  }
}
  
unsigned int outlimit=100;

unsigned int padding(unsigned int m)
{
  unsigned int n=2*m;
  cout << "min padded buffer=" << n << endl;
  unsigned int log2n;
  // Choose next power of 2 for maximal efficiency.
  for(log2n=0; n > ((unsigned int) 1 << log2n); log2n++);
  return 1 << log2n;
}

void add(Complex *f, Complex *F) 
{
  for(unsigned int i=0; i < mx; ++i) {
    unsigned int imy=i*my;
    Complex *fi=f+imy;
    Complex *Fi=F+imy;
    for(unsigned int j=0; j < my; ++j) {
      fi[j] += Fi[j];
    }
  }
}

int main(int argc, char* argv[])
{
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
      case 'h':
      default:
        usage(2);
    }
  }

  if(my == 0) my=mx;

  nx=padding(mx);
  ny=padding(my);

  cout << "nx=" << nx << ", ny=" << ny << endl;
  cout << "mx=" << mx << ", my=" << my << endl;
  
  if(N == 0) {
    N=N0/(nx*ny);
    if(N < 10) N=10;
  }
  cout << "N=" << N << endl;
  
  size_t align=sizeof(Complex);
  int nxp=Explicit ? nx : mx;
  int nyp=Explicit ? ny : my;
  if(Implicit) nxp *= M;
  array2<Complex> f(nxp,nyp,align);
  array2<Complex> g(nxp,nyp,align);

  double *T=new double[N];

  if(Implicit) {
    ImplicitConvolution2 C(mx,my,M);
    unsigned int mxy=mx*my;
    Complex **F=new Complex *[M];
    Complex **G=new Complex *[M];
    for(unsigned int s=0; s < M; ++s) {
      unsigned int smxy=s*mxy;
      F[s]=f+smxy;
      G[s]=g+smxy;
    }
    for(unsigned int i=0; i < N; ++i) {
      init(f,g,M);
      seconds();
      C.convolve(F,G);
//      C.convolve(f,g);
      T[i]=seconds();
    }
    
    timings("Implicit",T,N);

    if(mx*my < outlimit) 
      for(unsigned int i=0; i < mx; i++) {
        for(unsigned int j=0; j < my; j++)
          cout << f[i][j] << "\t";
        cout << endl;
      } else cout << f[0][0] << endl;
    cout << endl;
    
    delete [] G;
    delete [] F;
  }
  
  if(Explicit) {
    ExplicitConvolution2 C(nx,ny,mx,my,f,Pruned);
    for(unsigned int i=0; i < N; ++i) {
      init(f,g);
      seconds();
      C.convolve(f,g);
      T[i]=seconds();
    }

    timings(Pruned ? "Pruned" : "Explicit",T,N);

    if(mx*my < outlimit) 
      for(unsigned int i=0; i < mx; i++) {
        for(unsigned int j=0; j < my; j++)
          cout << f[i][j] << "\t";
        cout << endl;
      } else cout << f[0][0] << endl;
  }

  if(Direct) {
    array2<Complex> h(mx,my,align);
    array2<Complex> f(mx,my,align);
    array2<Complex> g(mx,my,align);
    DirectConvolution2 C(mx,my);
    init(f,g);
    seconds();
    C.convolve(h,f,g);
    T[0]=seconds();
  
    timings("Direct",T,1);

    if(mx*my < outlimit) 
      for(unsigned int i=0; i < mx; i++) {
        for(unsigned int j=0; j < my; j++)
          cout << h[i][j] << "\t";
        cout << endl;
      } else cout << h[0][0] << endl;
  }
  
  delete [] T;
}

