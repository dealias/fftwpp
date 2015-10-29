#include "Complex.h"
#include "Array.h"
#include "fftw++.h"
#include "utils.h"

using namespace std;
using namespace utils;
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

  unsigned int outlimit=3000;
  
  int N = 1000;
  unsigned int stats = MEAN; // Type of statistics used in timing test.

  fftw::maxthreads = get_max_threads();
 
#ifdef __GNUC__	
  optind=0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"N:m:x:y:z:T:S:h");
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
    case 'x':
      mx = atoi(optarg);
      break;
    case 'y':
      my = atoi(optarg);
      break;
    case 'z':
      mz = atoi(optarg);
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
  unsigned int mzp = mz/2+1;
  array3<double> f(mx,my,mz,align);
  array3<Complex> g(mx,my,mzp,align);

  rcfft3d Forward0(mx,my,mz,f,g);
  crfft3d Backward0(mx,my,mz,g,f);

  finit(f,mx,my,mz);
  cout << "\nIinput:"  << endl;
  if(mx*my*mz < outlimit) 
    cout << f;
  else 
    cout << f[0][0][0] << endl;

  Forward0.fft(f,g);

  cout << "\nOutput:" << endl;
  if(mx*my*mz < outlimit) 
    cout << g;
  else 
    cout << g[0][0][0] << endl;

  Backward0.fftNormalized(g,f);

  cout << "\nBack to input:\n" << endl;
  if(mx*my*mz < outlimit)
    cout << f;
  else 
    cout << f[0][0][0] << endl;

  if(N > 0) {
    double *T = new double[N];
    for(int i = 0; i < N; ++i) {
      finit(f,mx,my,mz);
      seconds();
      Forward0.fft(f,g);
      T[i] = seconds();
      Backward0.fftNormalized(g, f);
    }
    timings("fft3 out-of-place", mx, T, N, stats);
  }
  
}
