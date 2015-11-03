#include "Complex.h"
#include "Array.h"
#include "fftw++.h"
#include "utils.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

void finit(array2<double> f, unsigned int nx, unsigned int ny,
	   bool inplace=false)
{
  for(unsigned int i = 0; i < nx; ++i) {
    for(unsigned int j = 0; j < ny; ++j) {
      f(i, j) = i + j;
    }
    f(i, ny) = 0;
  }
}

int main(int argc, char* argv[])
{
  cout << "2D real-to-complex FFT" << endl;

  unsigned int nx = 4;
  unsigned int ny = 4;
    
  int N = 1000;
  unsigned int stats = MEAN; // Type of statistics used in timing test.

  bool inplace=false;
  bool shift=false;
  bool quiet=false;
  
  fftw::maxthreads = get_max_threads();
 
#ifdef __GNUC__	
  optind=0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"N:i:m:s:x:y:T:S:hq");
    if (c == -1) break;
    switch (c) {
    case 0:
      break;
    case 'N':
      N = atoi(optarg);
      break;
    case 'm':
      nx = ny = atoi(optarg);
      break;
    case 'i':
      inplace=atoi(optarg);
      break;
    case 'q':
      quiet=true;
      break;
    case 's':
      shift=atoi(optarg);
      break;
    case 'x':
      nx = atoi(optarg);
      break;
    case 'y':
      ny = atoi(optarg);
      break;
    case 'T':
      fftw::maxthreads = max(atoi(optarg), 1);
      break;
    case 'S':
      stats = atoi(optarg);
      break;
    case 'h':
    default:
      usageFFT(1);
      exit(0);
    }
  }

  size_t align=sizeof(Complex);

  unsigned int nyp=ny/2+1;
  
  array2<double> f(nx,ny+2*inplace,align);
  Complex *pg=inplace ? (Complex *) f() : ComplexAlign(nx*nyp);
  array2<Complex> g(nx,nyp,pg);
  
  rcfft2d Forward(nx,ny,f(),g());
  crfft2d Backward(nx,ny,g(),f());

  if(!quiet) {
    finit(f, nx, ny);
    cout << "\ninput:\n" << f << endl;

    if(shift)
      Forward.fft0(f, g);
    else
      Forward.fft(f, g);
    cout << "\noutput:\n" << g << endl;

    if(shift)
      Backward.fft0Normalized(g, f);
    else
      Backward.fftNormalized(g, f);
    cout << "\nback to input:\n" << f << endl;
  }
  
  double *T= new double[N];

  for(int i = 0; i < N; ++i) {
    finit(f, nx, ny);
    if(shift) {
      seconds();
      Forward.fft0(f, g);
      Backward.fft0Normalized(g, f);
      T[i] = seconds();
    } else {
      seconds();
      Forward.fft(f, g);
      Backward.fftNormalized(g, f);
      T[i] = seconds();
    }
  }

  timings("fft2 out-of-place", nx, T, N, stats);
 
}
