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

bool Direct=false, Implicit=true, Explicit=false, Pruned=false;

inline void init(array2<Complex>& f, unsigned int mx, unsigned int my) 
{
  for(unsigned int i=0; i < mx; ++i)
    for(unsigned int j=0; j < my; j++)
      f[i][j]=Complex(i,j);
}
  
unsigned int outlimit=100;

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

  unsigned int N=0;
  unsigned int mx=4;
  unsigned int my=4;

  int stats=0; // Type of statistics used in timing test.

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  
  
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"hN:m:x:y:n:T:S:");
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
        usageCommon(2);
        exit(0);
    }
  }

  if(my == 0) my=mx;

  cout << "mx=" << mx << ", my=" << my << endl;
  
  if(N == 0) {
    N=N0/mx/my;
    N = max(N, 20);
  }
  cout << "N=" << N << endl;
  
  size_t align=sizeof(Complex);

  array2<Complex> f(mx,my,align);
  array2<Complex> g(mx,my,align);
  
  mfft1d Forward(my,-1,mx,1,my);
  mfft1d Backward(my,1,mx,1,my);

  cout << "\nInput:" << endl;
  init(f,mx,my);
  if(mx*my < outlimit) {
    for(unsigned int i=0; i < mx; i++) {
      for(unsigned int j=0; j < my; j++)
        cout << f[i][j] << "\t";
      cout << endl;
    }
  } else {
    cout << f[0][0] << endl;
  }

  cout << "\nOutput:" << endl;
  Forward.fft(f);
  if(mx*my < outlimit) {
    for(unsigned int i=0; i < mx; i++) {
      for(unsigned int j=0; j < my; j++)
        cout << f[i][j] << "\t";
      cout << endl;
    }
  } else { 
    cout << f[0][0] << endl;
  }

  cout << "\nBack to input:" << endl;
  Backward.fftNormalized(f);
  if(mx*my < outlimit) {
    for(unsigned int i=0; i < mx; i++) {
      for(unsigned int j=0; j < my; j++)
        cout << f[i][j] << "\t";
      cout << endl;
    }
  } else { 
    cout << f[0][0] << endl;
  }

  
  cout << endl;
  double *T=new double[N];
  for(unsigned int i=0; i < N; ++i) {
    init(f,mx,my);
    seconds();
    Forward.fft(f);
    Backward.fft(f);
    T[i]=0.5*seconds();
    Backward.Normalize(f);
  }
  timings("mfft1, in-place",mx,T,N,stats);
  delete [] T;
  
  return 0;
}

