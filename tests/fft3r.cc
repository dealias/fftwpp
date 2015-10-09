#include "../Complex.h"
#include "Array.h"
#include "fftw++.h"
#include "utils.h"

using namespace std;
using namespace Array;
using namespace fftwpp;

void finit(array3<double> f, unsigned int mx, unsigned int my, unsigned int mz)
{
  for(unsigned int i = 0; i < mx; ++i) 
    for(unsigned int j = 0; j < my; ++j)
      for(unsigned int k = 0; k < mz; ++k) 
	f(i, j, k) = i + j + k;
}

int main(int argc, char* argv[])
{
  cout << "3D real-to-complex FFT" << endl;

  unsigned int mx = 11;
  unsigned int my = 4;
  unsigned int mz = 3;
    
  int N = 1000;
  unsigned int stats = MEAN; // Type of statistics used in timing test.

  fftw::maxthreads = get_max_threads();
  int r = -1;
 
#ifdef __GNUC__	
  optind=0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"N:m:x:y:z:r:T:S:h");
    if (c == -1) break;
    switch (c) {
    case 0:
      break;
    case 'N':
      N = atoi(optarg);
      break;
    case 'm':
      mx = my = mz = atoi(optarg);
      break;
    case 'r':
      r = atoi(optarg);
      break;
    case 'x':
      mx = atoi(optarg);
      break;
    case 'y':
      my = atoi(optarg);
      break;
    case 'T':
      fftw::maxthreads = max(atoi(optarg), 1);
      break;
    case 'S':
      stats = atoi(optarg);
      break;
    case 'h':
    default:
      fft_usage(1);
      exit(0);
    }
  }

  size_t align = sizeof(Complex);

  unsigned int mzp = mz / 2 + 1;
  
  array3<Complex> g(mx, my, mzp, align);
  
  double *T = new double[N];

  // if(r == -1 || r == 0) {
  //   rcfft2d Forward(m, f);
  //   //crfft1d Backward(1, f);
  //   for(int i = 0; i < N; ++i) {
  //     finit(f, m);
  //     seconds();
  //     Forward.fft(f);
  //     T[i] = seconds();
  //     //Backward.fftNormalized(f);
  //   }

  //   timings("fft1 in-place", m, T, N, stats);
  // }

  if(r == -1 || r == 1) {
    array3<double> f(mx,my,mz,align);
    rcfft3d Forward0(mx,my,mz,f,g);
    //crfft1d Backward0(1,g,f);

    for(int i = 0; i < N; ++i) {
      finit(f, mx, my, mz);
      seconds();
      Forward0.fft(f, g);
      T[i] = seconds();
      //Backward0.fftNormalized(g, f);
    }

    timings("fft3 out-of-place", mx, T, N, stats);
  }

  //cout << "\nback to input:\n" << f << endl;
}
