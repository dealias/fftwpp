#include "mpifftw++.h"
#include "utils.h"
#include "mpiutils.h"

using namespace std;
using namespace fftwpp;

// Number of iterations.
unsigned int N0=10000000;
unsigned int N=0;
unsigned int mx=4;
unsigned int my=4;

inline void init(Complex *f, splitx d) 
{
  unsigned int c=0;
  for(unsigned int i=0; i < d.x; ++i) {
    unsigned int ii=d.x0+i;
    for(unsigned int j=0; j < d.ny; j++) {
      f[c++]=Complex(ii,j);
    }
  }
}

unsigned int outlimit=100;

int main(int argc, char* argv[])
{
#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif
  int retval=0;
  
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"hN:m:x:y:n:T:");
    if (c == -1) break;
                
    switch (c) {
      case 0:
        break;
      case 'N':
        N=atoi(optarg);
        break;
      case 'm':
        mx=my=atoi(optarg);
        break;
      case 'x':
        mx=atoi(optarg);
        break;
      case 'y':
        my=atoi(optarg);
        break;
      case 'n':
        N0=atoi(optarg);
        break;
      case 'T':
        fftw::maxthreads=atoi(optarg);
        break;
      case 'h':
      default:
        usage(2);
    }
  }

  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);

  if(my == 0) my=mx;

  if(N == 0) {
    N=N0/mx/my;
    if(N < 10) N=10;
  }
  
  MPIgroup group(MPI_COMM_WORLD,mx);

  if(group.size > 1 && provided < MPI_THREAD_FUNNELED) {
    fftw::maxthreads=1;
  } else {
    fftw_init_threads();
    fftw_mpi_init();
  }

  if(group.rank == 0) {
    cout << "provided: " << provided << endl;
    cout << "fftw::maxthreads: " << fftw::maxthreads << endl;
  }
  
  if(group.rank == 0) {
    cout << "Configuration: " 
         << group.size << " nodes X " << fftw::maxthreads 
         << " threads/node" << endl;
  }
  
  if(group.rank < group.size) { // If the process is unused, then do nothing.
    bool main=group.rank == 0;
    if(main) {
      cout << "N=" << N << endl;
      cout << "mx=" << mx << ", my=" << my << endl;
    } 

    splitx d(mx,my,group.active,group.block);
  
    for(int i=0; i < group.size; ++i) {
      MPI_Barrier(group.active);
      if(i == group.rank) {
	cout << "process " << i << " splity:" << endl;
	d.show();
	cout << endl;
      }
    }
    MPI_Barrier(group.active);

    Complex *f=ComplexAlign(d.n);

    // Create instance of FFT
    fft2dMPI fft(d,f);

    MPI_Barrier(group.active);
    if(group.rank == 0)
      cout << "Initialized after " << seconds() << " seconds." << endl;    

    if(mx*my < outlimit) {
      init(f,d);

      if(main) cout << "\ninput:" << endl;
      show(f,d.x,my,group.active);

      fft.Forwards(f);

      if(main) cout << "\noutput:" << endl;
      show(f,mx,d.y,group.active);

      fft.Backwards(f);
      fft.Normalize(f);

      if(main) cout << "\noutput:" << endl;
      show(f,d.x,my,group.active);
    }

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

    deleteAlign(f);
  }
  
  MPI_Finalize();
  
  return retval;
}
