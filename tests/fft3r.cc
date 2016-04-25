#include "Complex.h"
#include "Array.h"
#include "fftw++.h"
#include "utils.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

void finit(array3<double> f, unsigned int nx, unsigned int ny, unsigned int nz)
{
  for(unsigned int i=0; i < nx; ++i) 
    for(unsigned int j=0; j < ny; ++j)
      for(unsigned int k=0; k < nz; ++k) 
        f(i,j,k)=i+j+k;
}

int main(int argc,char* argv[])
{
  cout << "3D real-to-complex FFT" << endl;

  unsigned int nx=11;
  unsigned int ny=4;
  unsigned int nz=3;

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

  size_t align=sizeof(Complex);
  
  unsigned int nzp=nz/2+1;
  
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
    for(unsigned int i=0; i < nx; ++i) {
      for(unsigned int j=0; j < ny; ++j) {
        for(unsigned int k=0; k < nz; ++k) {
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
    for(unsigned int i=0; i < nx; ++i) {
      for(unsigned int j=0; j < ny; ++j) {
        for(unsigned int k=0; k < nz; ++k) {
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
  timings("fft3 out-of-place",nx,T,N,stats);
  
}
