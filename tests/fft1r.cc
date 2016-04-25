#include "Complex.h"
#include "Array.h"
#include "fftw++.h"
#include "utils.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

void finit(array1<double> f, unsigned int m)
{
  for(unsigned int i=0; i < m; i++)
    f[i]=i;
}

int main(int argc, char* argv[])
{
  cout << "1D Complex to complex in-place FFT" << endl;

  unsigned int m=11; // Problem size

  int N=1000;
  unsigned int stats=MEAN; // Type of statistics used in timing test.

  fftw::maxthreads=get_max_threads();
  int r=-1;
 
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c=getopt(argc,argv,"N:m:x:r:T:S:h");
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

  unsigned int mp=m/2+1;
  
  array1<double> f(m+2,align);
  array1<Complex> g(mp,align);
  
  double *T=new double[N];

  if(r == -1 || r == 0) {
    rcfft1d Forward(m,f);
    crfft1d Backward(m,f);
    for(int i=0; i < N; ++i) {
      finit(f,m);
      seconds();
      Forward.fft(f);
      Backward.fft(f);
      T[i]=0.5*seconds();
      Backward.Normalize(f);
    }
    timings("fft1 in-place",m,T,N,stats);
  }

  if(r == -1 || r == 1) {
    rcfft1d Forward0(m,f,g);
    crfft1d Backward0(m,g,f);

    for(int i=0; i < N; ++i) {
      finit(f,m);
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
