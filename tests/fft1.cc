#include "Complex.h"
#include "Array.h"
#include "fftw++.h"
#include "utils.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

int main(int argc, char* argv[])
{
  cout << "1D Complex to complex in-place FFT" << endl;

  unsigned int m=11; // Problem size

  int N=1000;
  int stats=MEAN; // Type of statistics used in timing test.

  fftw::maxthreads=get_max_threads();
  int r=-1;

  
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"N:m:x:r:T:S:h");
    if (c == -1) break;
    switch (c) {
      case 0:
        break;
      case 'N':
        N=atoi(optarg);
        break;
      case 'm':
        m=atoi(optarg);
        break;
      case 'r':
        r=atoi(optarg);
        break;
      case 'x':
        m=atoi(optarg);
        break;
      case 'T':
        fftw::maxthreads=max(atoi(optarg),1);
        break;
      case 'S':
        stats=atoi(optarg);
        break;
      case 'h':
      default:
        usageFFT(1);
        exit(0);
    }
  }


  size_t align=sizeof(Complex);
  
  array1<Complex> f(m,align);
  
  array1<Complex> g(m,align);
  
  fft1d Forward(-1,f);
  fft1d Backward(1,f);

  fft1d Forward0(-1,f,g);
  fft1d Backward0(1,g,f);
  
  for(unsigned int i=0; i < m; i++) f[i]=i;

  double *T=new double[N];

  if(r == -1 || r == 0) {
    for(int i=0; i < N; ++i) {
      seconds();
      Forward.fft(f);
      Backward.fft(f);
      T[i]=0.5*seconds();
      Backward.Normalize(f);
    }

    timings("fft1 in-place",m,T,N,stats);
  }

  if(r == -1 || r == 1) {
    for(int i=0; i < N; ++i) {
      seconds();
      Forward0.fft(f,g);
      Backward0.fft(g,f);
      T[i]=0.5*seconds();
      Backward0.Normalize(f);
    }

    timings("fft1 out-of-place",m,T,N,stats);
  }

  //cout << "\nback to input:\n" << f << endl;
}
