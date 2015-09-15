#include "mpifftw++.h"
#include "utils.h"
#include "mpiutils.h"
#include <unistd.h>

using namespace std;
using namespace fftwpp;

// Number of iterations.
unsigned int N0=10000000;
unsigned int N=0;
unsigned int mx=4;
unsigned int my=4;
unsigned int mz=4;

inline void init(Complex *f, splitxy d) 
{
  unsigned int c=0;
  for(unsigned int i=0; i < d.x; ++i) {
    unsigned int ii=d.x0+i;
    for(unsigned int j=0; j < d.y; j++) {
      unsigned int jj=d.y0+j;
      for(unsigned int k=0; k < d.Z; k++) {
	unsigned int kk=k;
	f[c++]=Complex(10*kk+ii,jj);
      }
    }
  }
}

unsigned int outlimit=3000;

int main(int argc, char* argv[])
{
#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif
  int retval=0;

  bool quiet=false;
  bool test=false;
  
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"hN:m:x:y:z:n:T:qt");
    if (c == -1) break;
                
    switch (c) {
      case 0:
        break;
      case 'N':
        N=atoi(optarg);
        break;
      case 'm':
        mx=my=mz=atoi(optarg);
        break;
      case 'x':
        mx=atoi(optarg);
        break;
      case 'y':
        my=atoi(optarg);
        break;
      case 'z':
        mz=atoi(optarg);
        break;
      case 'n':
        N0=atoi(optarg);
        break;
      case 'T':
        fftw::maxthreads=atoi(optarg);
        break;
      case 'q':
        quiet=true;
        break;
      case 't':
        test=true;
        break;
      case 'h':
	usage(3);
	exit(0);
	break;
      default:
	cout << "Invalid option." << endl;
        usage(3);
	exit(1);
    }
  }

  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);

  if(my == 0) my=mx;

  if(N == 0) {
    N=N0/mx/my;
    if(N < 10) N=10;
  }
  
  MPIgroup group(MPI_COMM_WORLD,mx,my);

  if(group.size > 1 && provided < MPI_THREAD_FUNNELED)
    fftw::maxthreads=1;

  if(group.rank == 0) {
    cout << "provided: " << provided << endl;
    cout << "fftw::maxthreads: " << fftw::maxthreads << endl;
  }
  
  if(group.rank == 0) {
    cout << "Configuration: " 
         << group.size << " nodes X " << fftw::maxthreads 
         << " threads/node" << endl;
  }
  
  if(group.rank < group.size) {
    bool main=group.rank == 0;
    if(main) {
      cout << "N=" << N << endl;
      cout << "mx=" << mx << ", my=" << my << ", mz=" << mz << endl;
      cout << "size=" << group.size << endl;
    }

    splitxy d(mx,my,mz,group);
    
    Complex *f=ComplexAlign(d.n);

    // Create instance of FFT
    fft3dMPI fft(d,f);
    
    /*
    double *T=new double[N];
    for(unsigned int i=0; i < N; ++i) {
      init(f,d);
      seconds();
      fft.Forwards(f);
      fft.Backwards(f);
      fft.Normalize(f);
      T[i]=seconds();
    }
    if(main) timings("FFT timing:",mx,T,N);
    delete [] T;
    */

    if(mx*my*mz < outlimit) {
      MPI_Barrier(group.active);
      for(int i=0; i < group.size; ++i) {
      	MPI_Barrier(group.active);
      	if(i == group.rank) {
      	  cout << "process " << i << " splity:" << endl;
      	  d.show();
      	  cout << endl;
      	}
	usleep(500); // hack to get around dealing with cout and MPI
      }
      MPI_Barrier(group.active);

      init(f,d);

      if(main) cout << "\ninput:" << endl;
      show(f,d.x,d.y,d.Z,group.active);

      fft.Forwards(f);
      
      std::cout << d.xy.y << std::endl;
      if(main) cout << "\noutput:" << endl;
      show(f,d.X,d.xy.y,d.z,group.active);

      fft.Backwards(f);
      fft.Normalize(f);

      if(main) cout << "\nback to input:" << endl;
      show(f,d.x,d.y,d.Z,group.active);
    }

    deleteAlign(f);
  }
  
  MPI_Finalize();
  
  return retval;
}
  
