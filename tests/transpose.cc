#include "Complex.h"
#include "Array.h"
#include "fftw++.h"
#include "utils.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

// Number of iterations.
unsigned int N0=10000000;
unsigned int N=0;
unsigned int mx=4;
unsigned int my=4;
unsigned int mz=4;

inline void init(array3<Complex>& f, bool transpose=false) 
{
  for(unsigned int i=0; i < mx; ++i) {
    for(unsigned int j=0; j < my; ++j) {
      for(unsigned int k=0; k < mz; ++k) {
        Complex val=Complex(i,j);
        if(!transpose)
          f[i][j][k]=val;
        else
          f[j][i][k]=val;
      }
    }
  }
}
  
unsigned int outlimit=100;

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

  int stats=0; // Type of statistics used in timing test.

  bool inplace = true;
  
#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  
  
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"hN:m:x:y:z:n:T:S:i:d");
    if (c == -1) break;
                
    switch (c) {
      case 0:
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
      case 'i':
        inplace = atoi(optarg);
        break;
      case 'h':
      default:
        usageInplace(2);
        exit(1);
    }
  }

  if(my == 0)
    my=mx;

  cout << "mx=" << mx << ", my=" << my << ", mz=" << mz << endl;
  
  cout << "N=" << N << endl;

  cout << "inplace=" << inplace << endl;
  
  cout << "threads=" << fftw::maxthreads << endl;
  
  size_t align=sizeof(Complex);

  array3<Complex> f(mx,my,mz,align);
  Complex *pg=inplace ? f() : ComplexAlign(mx*my*mz);
  array3<Complex> g(my,mx,mz,pg);

  Transpose transpose(mx,my,mz,f(),g());

  if(N == 0) {
    init(f);

    cout << "Input:" << endl;
    if(mx*my*mz < outlimit) 
      cout << f << endl;
    else 
      cout << f[0][0][0] << endl;

    transpose.transpose(f(),g()); 

    cout << "Output:" << endl;
    if(mx*my*mz < outlimit) 
      cout << g << endl;
    else 
      cout << g[0][0][0] << endl;

    array3<Complex> f0(my,mx,mz,align);
    init(f0,true);
    double errmax = 0.0;
    for(unsigned int i = 0; i < my; ++i) {
      for(unsigned int j = 0; j < mx; ++j) {
        for(unsigned int k = 0; k < mz; ++k) {
          errmax = max(errmax,abs(g[i][j][k]-f0[i][j][k])); 
        }
      }
    }
    cout << "errmax: " << errmax << endl;
    if(errmax > 0.0) {
      cout << "Caution: error too large!" << endl;
      return 1;
    }

  } else {
    double *T=new double[N];
  
    for(unsigned int i=0; i < N; ++i) {
      init(f);
      seconds();
      transpose.transpose(f(),g());
      T[i]=seconds();
    }

    timings("transpose",mx,T,N,stats);
    delete [] T;
  }  

  if(!inplace)
    delete pg;
  
  return 0;
}

