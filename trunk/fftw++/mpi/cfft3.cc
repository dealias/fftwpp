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
unsigned int mz=4;

inline void init(Complex *f, dimensions3 d) 
{
  for(unsigned int i=0; i < d.nx; ++i) {
    unsigned int I=d.y*d.z*i;
    for(unsigned int j=0; j < d.y; j++) {
      unsigned int IJ=I+d.z*j;
      unsigned int jj=d.y0+j;
      for(unsigned int k=0; k < d.z; k++) {
	unsigned int kk=d.z0+k;
	f[IJ+k]=Complex(10*kk+i,jj);
      }
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
    int c = getopt(argc,argv,"hN:m:x:y:z:n:T:");
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
  
  MPIgroup group(MPI_COMM_WORLD,my,mz);

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
  
  if(group.rank < group.size) {
    bool main=group.rank == 0;
    if(main) {
      cout << "N=" << N << endl;
      cout << "mx=" << mx << ", my=" << my << ", mz=" << mz << endl;
      cout << "size=" << group.size << endl;
      cout << "yblock=" << group.yblock << endl;
      cout << "zblock=" << group.zblock << endl;
    }

    dimensions3 d(mx,my,my,mz,group);
    
    Complex *f=ComplexAlign(d.n);

    // Load wisdom
    MPILoadWisdom(group.active);

    // Create instance of FFT
    cfft3dMPI fft(d,f);
    
    bool dofinaltranspose=false; // FIXME: this must always be false for now.
    
    double *T=new double[N];
    for(unsigned int i=0; i < N; ++i) {
      init(f,d);
      seconds();
      fft.Forwards(f,dofinaltranspose);
      fft.Backwards(f,dofinaltranspose);
      fft.Normalize(f);
      T[i]=seconds();
    }
    
    if(main) timings("FFT timing:",mx,T,N);
    delete [] T;

    if(mx*my*mz < outlimit) {
      MPI_Barrier(group.active);
      for(int i=0; i < group.size; ++i) {
	MPI_Barrier(group.active);
	if(i == group.rank) {
	  cout << "process " << i << " dimensions:" << endl;
	  d.show();
	  cout << endl;
	}
      }
      MPI_Barrier(group.active);
      init(f,d);

      MPI_Barrier(group.active);
      if(main) cout << "\ninput:" << endl;
      show(f,d.nx,d.y,d.z,group.active);
      MPI_Barrier(group.active);
      // Uncomment following line to see the order in memory
      //show(f,1,d.nx*d.y*d.z,group.active); 

      
      fft.Forwards(f,dofinaltranspose);
      
      MPI_Barrier(group.active);
      if(main) cout << "\noutput:" << endl;
      show(f,d.x,d.yz.x,d.nz,group.active);
      MPI_Barrier(group.active);
      // Uncomment following line to see the order in memory
      //show(f,1,d.x*d.yz.x*d.nz,group.active);

      fft.Backwards(f,dofinaltranspose);
      fft.Normalize(f);

      if(main) cout << "\nback to input:" << endl;
      show(f,d.nx,d.y,d.z,group.active);
      MPI_Barrier(group.active);
      // Uncomment following line to see the order in memory
      show(f,1,d.nx*d.y*d.z,group.active);
    }

    deleteAlign(f);

    // Load wisdom
    MPISaveWisdom(group.active);
  }
  
  MPI_Finalize();
  
  return retval;
}
  
