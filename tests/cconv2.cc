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

bool Direct=false, Implicit=true, Explicit=false, Pruned=false;

inline void init(Complex **F, unsigned int nxp, unsigned int nyp, unsigned int A) 
{
  if(A%2 == 0) {
    unsigned int M=A/2;
    double factor=1.0/sqrt((double) M);
    for(unsigned int s=0; s < M; ++s) {
      double S=sqrt(1.0+s);
      double ffactor=S*factor;
      double gfactor=1.0/S*factor;
      array2<Complex> f(nxp,nyp,F[s]);
      array2<Complex> g(nxp,nyp,F[M+s]);
#pragma omp parallel for
      for(unsigned int i=0; i < mx; ++i) {
        for(unsigned int j=0; j < my; j++) {
          f[i][j]=ffactor*Complex(i,j);
          g[i][j]=gfactor*Complex(2*i,j+1);
        }
      }
    }
  } else {
    cerr << "Init not implemented for A=" << A << endl;
    exit(1);
  }
}
  
unsigned int outlimit=100;

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
  fftw::maxthreads=get_max_threads();

  unsigned int A=2;
  unsigned int B=1;

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
        A=atoi(optarg);
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

  if(my == 0) my=mx;

  nx=cpadding(mx);
  ny=cpadding(my);

  cout << "nx=" << nx << ", ny=" << ny << endl;
  cout << "mx=" << mx << ", my=" << my << endl;
  
  if(N == 0) {
    N=N0/nx/ny;
    N = max(N, 20);
  }
  cout << "N=" << N << endl;
  
  size_t align=sizeof(Complex);
  array2<Complex> h0;
  if(Direct) h0.Allocate(mx,my,align);
  int nxp=Explicit ? nx : mx;
  int nyp=Explicit ? ny : my;

  if(!Implicit)
    A=2;

  if(B < 1) B=1;
  if(B > A) {
    cerr << "B=" << B << " is not yet implemented for A=" << A << endl;
    exit(1);
  }
  
  // Allocate input/ouput memory and set up pointers
  Complex **F=new Complex *[A];
  for(unsigned int a=0; a < A; ++a)
    F[a]=ComplexAlign(nxp*nyp);

  // For easy access of first element
  array2<Complex> f(mx,my,F[0]);
    
  double *T=new double[N];
    
  if(Implicit) {
    multiplier *mult;
  
    switch(A) {
      case 2: mult=multbinary; break;
      case 4: mult=multbinary2; break;
      case 6: mult=multbinary3; break;
      case 8: mult=multbinary4; break;
      case 16: mult=multbinary8; break;
      default: cout << "Multiplication for A=" << A 
                    << " is not yet implemented" << endl; exit(1);
    }

    ImplicitConvolution2 C(mx,my,A,B);
    cout << "threads=" << C.Threads() << endl << endl;;

    for(unsigned int i=0; i < N; ++i) {
      init(F,mx,my,A);
      seconds();
      C.convolve(F,mult);
//      C.convolve(F[0],F[1]);
      T[i]=seconds();
    }
    
    timings("Implicit",mx,T,N,stats);

    if(Direct) {
      for(unsigned int i=0; i < mx; i++)
        for(unsigned int j=0; j < my; j++)
          h0[i][j]=f[i][j];
    }
    
    if(mx*my < outlimit) {
      for(unsigned int i=0; i < mx; i++) {
        for(unsigned int j=0; j < my; j++)
          cout << f[i][j] << "\t";
        cout << endl;
      }
    } else {
      cout << f[0][0] << endl;
    }
  }
  
  if(Explicit) {
    if(A != 2) {
      cerr << "Explicit convolutions for A=" << A << " are not yet implemented" << endl;
      exit(1);
    }

    ExplicitConvolution2 C(nx,ny,mx,my,F[0],Pruned);
    for(unsigned int i=0; i < N; ++i) {
      init(F,nxp,nyp,A);
      seconds();
      C.convolve(F[0],F[1]);
      T[i]=seconds();
    }

    cout << endl;
    timings(Pruned ? "Pruned" : "Explicit",mx,T,N,stats);

    if(Direct) {
      for(unsigned int i=0; i < mx; i++)
        for(unsigned int j=0; j < my; j++)
          h0[i][j]=F[0][nyp*i+j];
    }
  
    if(mx*my < outlimit) {
      for(unsigned int i=0; i < mx; i++) {
        for(unsigned int j=0; j < my; j++) {
          cout << F[0][nyp*i+j] << "\t";
        }
        cout << endl;
      }
    }
    else
      cout << F[0][0] << endl;
  }

  if(Direct) {
    array2<Complex> h(mx,my,align);
    DirectConvolution2 C(mx,my);
    init(F,mx,my,2);
    seconds();
    C.convolve(h,F[0],F[1]);
    T[0]=seconds();
  
    cout << endl;
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
