#include <getopt.h>

#include "Complex.h"
#include "Array.h"
#include "fftw++.h"
#include "utils.h"
#include "options.h"
#include "timing.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

void finit(array3<double> f, size_t nx, size_t ny, size_t nz)
{
  for(size_t i=0; i < nx; ++i)
    for(size_t j=0; j < ny; ++j)
      for(size_t k=0; k < nz; ++k)
        f(i,j,k)=i+j+k;
}

int main(int argc,char *argv[])
{
  cout << "3D real-to-complex FFT" << endl;

  size_t nx=11;
  size_t ny=4;
  size_t nz=3;

  int N=1000;
  size_t stats=MEAN; // Type of statistics used in timing test.

  bool inplace=false;
  bool shift=false;
  bool quiet=false;

  fftw::maxthreads=parallel::get_max_threads();

#ifdef __GNUC__
  optind=0;
#endif
  for (;;) {
    int c=getopt(argc,argv,"N:i:m:qx:y:z:T:S:h");
    if (c == -1) break;
    switch (c) {
      case 0:
        break;
      case 'N':
        N=atoi(optarg);
        break;
      case 'm':
        nx=ny=nz=atoi(optarg);
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
      case 'z':
        nz=atoi(optarg);
        break;
      case 'T':
        fftw::maxthreads=max(atoi(optarg),1);
        break;
      case 'S':
        stats=atoi(optarg);
        break;
      case 'h':
      default:
        usageInplace(3);
        exit(0);
    }
  }

  size_t align=ALIGNMENT;

  size_t nzp=nz/2+1;

  array3<Complex> g(nx,ny,nzp,align);
  array3<double> f;

  if(inplace)
    f.Dimension(nx,ny,2*nzp,(double *) g());
  else
    f.Allocate(nx,ny,nz,align);

  rcfft3d Forward(nx,ny,nz,f,g);
  crfft3d Backward(nx,ny,nz,g,f);

  if(!quiet) {
    finit(f,nx,ny,nz);
    cout << endl << "Input:" << endl;
    for(size_t i=0; i < nx; ++i) {
      for(size_t j=0; j < ny; ++j) {
        for(size_t k=0; k < nz; ++k) {
          cout << f(i,j,k) << " ";
        }
        cout << endl;
      }
      cout << endl;
    }

    Forward.fft(f,g);

    cout << endl << "Output:" << endl;
    cout << g << endl;

    Backward.fftNormalized(g,f);

    cout << endl << "Back to input:" << endl;
    for(size_t i=0; i < nx; ++i) {
      for(size_t j=0; j < ny; ++j) {
        for(size_t k=0; k < nz; ++k) {
          cout << f(i,j,k) << " ";
        }
        cout << endl;
      }
      cout << endl;
    }
  }

  double *T=new double[N];

  for(int i=0; i < N; ++i) {
    finit(f,nx,ny,nz);
    if(shift) {
      cpuTimer c;
      Forward.fft0(f,g);
      Backward.fft0(g,f);
      T[i]=0.5*c.nanoseconds();
      Backward.Normalize(f);
    } else {
      cpuTimer c;
      Forward.fft(f,g);
      Backward.fft(g,f);
      T[i]=0.5*c.nanoseconds();
      Backward.Normalize(f);
    }
  }
  timings("fft3 out-of-place",nx,T,N,stats);

}
