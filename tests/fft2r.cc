#include "Complex.h"
#include "Array.h"
#include "fftw++.h"
#include "utils.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

void finit(array2<double> f, unsigned int nx, unsigned int ny)
{
  for(unsigned int i=0; i < nx; ++i) {
    for(unsigned int j=0; j < ny; ++j) {
      f(i,j)=i+j;
    }
  }
}

int main(int argc, char* argv[])
{
  cout << "2D real-to-complex FFT" << endl;

  unsigned int nx=4;
  unsigned int ny=5;
    
  int N=1000;
  unsigned int stats=MEAN; // Type of statistics used in timing test.

  bool inplace=false;
  bool shift=false;
  bool quiet=false;
  
  fftw::maxthreads=get_max_threads();
 
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c=getopt(argc,argv,"N:i:m:s:x:y:T:S:hq");
    if (c == -1) break;
    switch (c) {
      case 0:
        break;
      case 'N':
        N=atoi(optarg);
        break;
      case 'm':
        nx=ny=atoi(optarg);
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
        nx=atoi(optarg);
        break;
      case 'y':
        ny=atoi(optarg);
        break;
      case 'T':
        fftw::maxthreads=max(atoi(optarg),1);
        break;
      case 'S':
        stats=atoi(optarg);
        break;
      case 'h':
      default:
        usageInplace(2);
        exit(0);
    }
  }

  size_t align=sizeof(Complex);

  unsigned int nyp=ny/2+1;
  
  array2<Complex> g(nx,nyp,align);
  array2<double> f;
  
  if(inplace)
    f.Dimension(nx,2*nyp,(double *) g());
  else
    f.Allocate(nx,ny,align);
  
  rcfft2d Forward(nx,ny,f,g);
  crfft2d Backward(nx,ny,g,f);

  if(!quiet) {
    finit(f,nx,ny);
    cout << endl << "Input:" << endl;
    for(unsigned int i=0; i < nx; ++i) {
      for(unsigned int j=0; j < ny; ++j) {
        cout << f(i,j) << " ";
      }
      cout << endl;
    }

    if(shift)
      Forward.fft0(f,g);
    else
      Forward.fft(f,g);
    cout << endl << "output:" << endl << g << endl;

    if(shift)
      Backward.fft0(g,f);
    else
      Backward.fft(g,f);
    Backward.Normalize(f);
    cout << endl << "Back to input:" << endl;
    for(unsigned int i=0; i < nx; ++i) {
      for(unsigned int j=0; j < ny; ++j) {
        cout << f(i,j) << " ";
      }
      cout << endl;
    }
  }
  
  double *T= new double[N];

  for(int i=0; i < N; ++i) {
    finit(f,nx,ny);
    if(shift) {
      seconds();
      Forward.fft0(f,g);
      Backward.fft0(g,f);
      T[i]=0.5*seconds();
      Backward.Normalize(f);
    } else {
      seconds();
      Forward.fft(f,g);
      Backward.fft(g,f);
      T[i]=0.5*seconds();
      Backward.Normalize(f);
    }
  }

  timings("fft2 out-of-place",nx,T,N,stats);
 
}
