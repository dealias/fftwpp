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
    for(unsigned int j=0; j < d.Y; j++) {
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
  unsigned int N=0;
  unsigned int mx=4;
  unsigned int my=4;
  bool quiet=false;
  bool test=false;
  
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"hN:m:x:y:n:T:qt");
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
    case 'q':
      quiet=true;
      break;
    case 't':
      test=true;
      break;
    case 'h':
      usage(2);
      exit(0);
    default:
      cout << "Invalid option." << endl;
      usage(2);
      exit(1);
    }
  }

  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);

  if(my == 0) my=mx;

  if(!N == 0) {
    N=N0/mx/my;
    if(N < 10) N=10;
  }
  
  int fftsize=min(mx,my);

  MPIgroup group(MPI_COMM_WORLD,fftsize);

  if(group.size > 1 && provided < MPI_THREAD_FUNNELED)
    fftw::maxthreads=1;

  if(!quiet && group.rank == 0) {
    cout << "provided: " << provided << endl;
    cout << "fftw::maxthreads: " << fftw::maxthreads << endl;
  }
  
  if(!quiet && group.rank == 0) {
    cout << "Configuration: " 
	 << group.size << " nodes X " << fftw::maxthreads 
	 << " threads/node" << endl;
  }

  if(group.rank < group.size) { 
    bool main=group.rank == 0;
    if(!quiet && main) {
      cout << "N=" << N << endl;
      cout << "mx=" << mx << ", my=" << my << endl;
    } 

    split d(mx,my,group.active);
  
    Complex *f=ComplexAlign(d.n);

    // Create instance of FFT
    fft2dMPI fft(d,f);

    if(!quiet && group.rank == 0)
      cout << "Initialized after " << seconds() << " seconds." << endl;    

    if(test) {
      init(f,d);

      if(!quiet && mx*my < outlimit) {
	if(main) cout << "\nDistributed input:" << endl;
	show(f,d.x,my,group.active);
      }

      size_t align=sizeof(Complex);
      array2<Complex> flocal(mx,my,align);
      fft2d localForward(-1,flocal);
      fft2d localBackward(-1,flocal);
      accumulate_split(f, flocal(), d, 1, false, group.active);

      if(!quiet && main) {
	cout << "\nAccumulated input:\n" << flocal << endl;
      }

      fft.Forwards(f);

      if(!quiet && mx*my < outlimit) {
      	if(main) cout << "\nDistributed output:" << endl;
      	show(f,mx,d.y,group.active);
      }
      
      array2<Complex> faccumulated(mx,my,align);
      accumulate_split(f, faccumulated(), d, 1, true, group.active);

      MPI_Barrier(group.active);
      if(main) {
	localForward.fft(flocal);
	if(!quiet) {
	  cout << "\nLocal output:\n" << flocal << endl;
	  cout << "\nAccumulated output:\n" << faccumulated << endl;
	}
	double maxerr = relmaxerror(flocal(),faccumulated(),d.X,d.Y);
	
	cout << "max error: " << maxerr << endl;
	if(maxerr > 1e-10) {
	  cerr << "CAUTION: max error is LARGE!" << endl;
	  retval += 1;
	}
      }

      fft.Backwards(f);
      fft.Normalize(f);

      if(!quiet && mx*my < outlimit) {
      	if(main) cout << "\nDistributed output:" << endl;
      	show(f,d.x,my,group.active);
      }

      accumulate_split(f, faccumulated(), d, 1, true, group.active);
      MPI_Barrier(group.active);
      if(main) {
	localBackward.fftNormalized(flocal);
	if(!quiet) {
	  cout << "\nLocal output:\n" << flocal << endl;
	  cout << "\nAccumulated output:\n" << faccumulated << endl;
	}
	double maxerr = relmaxerror(flocal(),faccumulated(),d.X,d.Y);
	
	cout << "max error: " << maxerr << endl;
	if(maxerr > 1e-10) {
	  cerr << "CAUTION: max error is LARGE!" << endl;
	  retval += 1;
	}
      }

    } else {
      if(N > 0) {
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
    }

    deleteAlign(f);
  }
  
  if(!quiet && group.rank == 0) {
    cout << endl;
    if(retval == 0)
      cout << "pass" << endl;
    else
      cout << "FAIL" << endl;
  }  
  MPI_Finalize();

  return retval;
}
