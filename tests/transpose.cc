#include "../Complex.h"
#include "Array.h"
#include "fftw++.h"
#include "utils.h"

using namespace std;
using namespace Array;
using namespace fftwpp;

// Number of iterations.
unsigned int N0=10000000;
unsigned int N=0;
unsigned int mx=4;
unsigned int my=4;


inline void init(array2<Complex>& f) 
{
  for(unsigned int i=0; i < mx; ++i)
    for(unsigned int j=0; j < my; j++)
      f[i][j]=Complex(i, j);
}
  
unsigned int outlimit=100;

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

  unsigned int stats=0; // Type of statistics used in timing test.

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  
  
#ifdef __GNUC__	
  optind=0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"hN:m:x:y:n:T:S:d");
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
      fft_usage(2);
      exit(0);
    }
  }

  if(my == 0) my=mx;

  cout << "mx=" << mx << ", my=" << my << endl;
  
  cout << "N=" << N << endl;
  
  size_t align=sizeof(Complex);

  array2<Complex> f(mx,my,align);
  array2<Complex> g(mx,my,align);

  Transpose transpose(mx, // rows
		      my, // cols
		      1, // length
		      (Complex *)f(), (Complex *)g());

  if(N == 0) {
    init(f);

    cout << "Input:" << endl;
    if(mx*my < outlimit) 
      cout << f << endl;
    else 
      cout << f[0][0] << endl;

    transpose.transpose(f(),g()); 

    cout << "Output:" << endl;
    if(mx*my < outlimit) 
      cout << g << endl;
    else 
      cout << g[0][0] << endl;
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

  return 0;
}

