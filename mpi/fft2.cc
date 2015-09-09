#include "Array.h"
#include "mpifftw++.h"
#include "utils.h"
#include "mpiutils.h"

using namespace std;
using namespace fftwpp;
using namespace Array;

inline void init(Complex *f, split d) 
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
  int retval = 0; // success!

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  // Number of iterations.
  unsigned int N0=10000000;
  bool Nset = 0;
  unsigned int N=0;
  unsigned int mx=4;
  unsigned int my=4;
  
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"hN:m:X:Y:n:T:");
    if (c == -1) break;
                
    switch (c) {
      case 0:
        break;
      case 'N':
	Nset = true;
        N=atoi(optarg);
        break;
      case 'm':
        mx=my=atoi(optarg);
        break;
      case 'X':
        mx=atoi(optarg);
        break;
      case 'Y':
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

  if(!Nset) {
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

    split d(mx,my,group.active,group.block);
  
    for(int i=0; i < group.size; ++i) {
      if(i == group.rank) {
	cout << "process " << i << " splity:" << endl;
	d.show();
	cout << endl;
      }
    }

    Complex *f=ComplexAlign(d.n);

    // Create instance of FFT
    fft2dMPI fft(d,f);

    if(group.rank == 0)
      cout << "Initialized after " << seconds() << " seconds." << endl;    

    if(N == 0) {
      init(f,d);

      if(mx*my < outlimit) {
	if(main) cout << "\ninput:" << endl;
	show(f,d.x,my,group.active);
      }

      // create local transform for testing purposes
      size_t align=sizeof(Complex);
      array2<Complex> localfin(mx,my,align);
      fft2d localForward2(-1,localfin);
      accumulate_splitx(f, localfin(), d, 1, false, group.active);

      fft.Forwards(f);

      MPI_Barrier(group.active);
      if(mx*my < outlimit) {
      	if(main) cout << "\noutput:" << endl;
      	show(f,mx,d.y,group.active);
      }
      
      MPI_Barrier(group.active);
      if(main) {
	cout << "\nwlocal input:\n" << localfin << endl;
	localForward2.fft(localfin);
	cout << "\nlocal output:\n" << localfin << endl;
      }

      array2<Complex> localfout(mx,my,align);
      accumulate_splitx(f, localfout(), d, 1, true, group.active);
      if(main) {
	cout << "\naccumulated output:\n" << localfout << endl;
	double maxerr = 0.0;
	for(unsigned int i = 0; i < d.nx; ++i) {
	  for(unsigned int j = 0; j < d.ny; ++j) {
	    double diff = abs(localfout[i][j] - localfin[i][j]);
	    if(diff > maxerr)
	      maxerr = diff;
	  }
	}
	cout << "max error: " << maxerr << endl;
	if(maxerr > 1e-10) {
	  cout << "CAUTION: max error is LARGE!" << endl;
	  retval += 1;
	}
      }

      fft.Backwards(f);
      fft.Normalize(f);

      if(mx*my < outlimit) {
      	if(main) cout << "\noutput:" << endl;
      	show(f,d.x,my,group.active);
      }
    } else {
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
    }

    deleteAlign(f);
  }
  
  MPI_Finalize();
  
  return retval;
}
