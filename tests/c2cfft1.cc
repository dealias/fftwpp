#include "Array.h"
#include "fftw++.h"
#include "timing.h"
#include <unistd.h>

// Compile with
// g++ -I .. -fopenmp example1.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace Array;
using namespace fftwpp;

int main(int argc, char* argv[])
{
  cout << "1D Complex to complex in-place FFT" << endl;

  unsigned int m=11; // Problem size

  int N=1000;
  unsigned int stats=MEAN; // Type of statistics used in timing test.

  fftw::maxthreads=get_max_threads();

  
#ifdef __GNUC__	
  optind=0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"N:m:T:S:h");
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
    case 'T':
      fftw::maxthreads=max(atoi(optarg),1);
      break;
    case 'S':
      stats=atoi(optarg);
      break;
    case 'h':
      // FIXME
      break;
    default:
      exit(1);
      //FIXME
    }
  }


  size_t align=sizeof(Complex);
  
  array1<Complex> f(m,align);
  
  fft1d Forward(-1,f);
  fft1d Backward(1,f);
  
  for(unsigned int i=0; i < m; i++) f[i]=i;

  //cout << "\ninput:\n" << f << endl;


  double *T=new double[N];
  for(int i=0; i < N; ++i) {
    seconds();
    Forward.fft(f);
    T[i]=seconds();
    Backward.fftNormalized(f);

  }

  timings("fft1",m,T,N,stats);

  //cout << "\nback to input:\n" << f << endl;
}
