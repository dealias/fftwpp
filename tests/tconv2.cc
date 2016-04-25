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
unsigned int mx=4;
unsigned int my=4;
unsigned int M=1;
unsigned int B=1; // Number of independent outputs

unsigned int nxp;
unsigned int nyp;

bool Direct=false, Implicit=true, Explicit=false, Pruned=false;

unsigned int outlimit=100;

inline void init(Array2<Complex>& e, Array2<Complex>& f, Array2<Complex>& g,
                 unsigned int M=1) 
{
  unsigned int offset=Explicit ? nx/2-mx+1 : 0;
  unsigned int stop=2*mx-1;
  unsigned int stopoffset=stop+(Implicit ? 1 : offset);
  double factor=1.0/cbrt((double) M);
  for(unsigned int s=0; s < M; ++s) {
    double S=sqrt(1.0+s);
    double efactor=1.0/S*factor;
    double ffactor=(1.0+S)*S*factor;
    double gfactor=1.0/(1.0+S)*factor;
#pragma omp parallel for
    for(unsigned int i=0; i < stop; i++) {
      unsigned int I=s*stopoffset+i+offset;
      for(unsigned int j=0; j < my; j++) {
        e[I][j]=efactor*Complex(i,j);
        f[I][j]=ffactor*Complex(i+1,j+2);
        g[I][j]=gfactor*Complex(2*i,j+1);
      }
    }
  }
}

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

  int stats=0; // Type of statistics used in timing test.

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  
  
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"hdeiptA:B:N:m:x:y:n:T:S:");
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
        M=2*atoi(optarg);
        break;
      case 'B':
        B=atoi(optarg);
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
      case 'S':
        stats=atoi(optarg);
        break;
      case 'h':
      default:
        usage(2);
        usageExplicit(2);
        exit(1);
    }
  }

  nx=tpadding(mx);
  ny=tpadding(my);
  
  cout << "nx=" << nx << ", ny=" << ny << endl;
  cout << "mx=" << mx << ", my=" << my << endl;
  
  if(N == 0) {
    N=N0/nx/ny;
    N = max(N, 20);
  }
  cout << "N=" << N << endl;
    
  if(B != 1) {
    cerr << "B=" << B << " is not yet implemented" << endl;
    exit(1);
  }
    
  size_t align=sizeof(Complex);
  array2<Complex> h0;
  if(Direct) h0.Allocate(mx,my,align);
  nxp=Explicit ? nx : (Implicit ? 2*mx : 2*mx-1);
  nyp=Explicit ? ny/2+1 : (Implicit ? my+1 : my);
  unsigned int nxp0=Implicit ? nxp*M : nxp;
  unsigned int offset=Implicit ? -1 : 0;
  Array2<Complex> e(nxp0,nyp,offset,0,align);
  Array2<Complex> f(nxp0,nyp,offset,0,align);
  Array2<Complex> g(nxp0,nyp,offset,0,align);

  double *T=new double[N];

  if(Implicit) {
    ImplicitHTConvolution2 C(mx,my,M);
    cout << "Using " << C.Threads() << " threads."<< endl;
    Complex **E=new Complex *[M];
    Complex **F=new Complex *[M];
    Complex **G=new Complex *[M];
    unsigned int mf=nxp*nyp;
    for(unsigned int s=0; s < M; ++s) {
      unsigned int smf=s*mf;
      E[s]=e+smf;
      F[s]=f+smf;
      G[s]=g+smf;
    }
    for(unsigned int i=0; i < N; ++i) {
      init(e,f,g,M);
      seconds();
      C.convolve(E,F,G);
//      C.convolve(e,f,g);
      T[i]=seconds();
    }
    
    timings("Implicit",mx,T,N,stats);
    
    if(Direct) {
      for(unsigned int i=0; i < mx; i++) 
        for(unsigned int j=0; j < my; j++)
          h0[i][j]=e[i][j];
    }

    if(nxp*my < outlimit)
      for(unsigned int i=0; i < 2*mx-1; i++) {
        for(unsigned int j=0; j < my; j++)
          cout << e[i][j] << "\t";
        cout << endl;
      } else cout << e[1][0] << endl;
    cout << endl;
    
    delete [] G;
    delete [] F;
    delete [] E;
  }
  
  if(Explicit) {
    ExplicitHTConvolution2 C(nx,ny,mx,my,f,Pruned);
    for(unsigned int i=0; i < N; ++i) {
      init(e,f,g);
      seconds();
      C.convolve(e,f,g);
      T[i]=seconds();
    }
    
    timings(Pruned ? "Pruned" : "Explicit",mx,T,N,stats);

    if(Direct) {
      for(unsigned int i=0; i < mx; i++) 
        for(unsigned int j=0; j < my; j++)
          h0[i][j]=e[i][j];
    }

    unsigned int offset=nx/2-mx+1;
    if(2*(mx-1)*my < outlimit) 
      for(unsigned int i=offset; i < offset+2*mx-1; i++) {
        for(unsigned int j=0; j < my; j++)
          cout << e[i][j] << "\t";
        cout << endl;
      } else cout << e[offset][0] << endl;
  }
  
  if(Direct) {
    Explicit=false;
    unsigned int nxp=2*mx-1;
    Array2<Complex> h(nxp,my,0,0,align);
    Array2<Complex> e(nxp,my,0,0,align);
    Array2<Complex> f(nxp,my,0,0,align);
    Array2<Complex> g(nxp,my,0,0,align);
    DirectHTConvolution2 C(mx,my);
    init(e,f,g);
    seconds();
    C.convolve(h,e,f,g);
    T[0]=seconds();
  
    timings("Direct",mx,T,1);

    if(nxp*my < outlimit)
      for(unsigned int i=0; i < nxp; i++) {
        for(unsigned int j=0; j < my; j++)
          cout << h[i][j] << "\t";
        cout << endl;
      } else cout << h[0][0] << endl;

    if(Implicit) { // compare implicit version with direct verion:
      double error=0.0;
      cout << endl;
      double norm=0.0;
      for(unsigned int i=0; i < mx; i++) {
        for(unsigned int j=0; j < my; j++) {
          error += abs2(h0[i][j]-h[i][j]);
          norm += abs2(h[i][j]);
        }
      }
      if(norm > 0) error=sqrt(error/norm);
      cout << "error=" << error << endl;
      if (error > 1e-12) cerr << "Caution! error=" << error << endl;
    }

  }
  
  delete [] T;

  return 0;
}
