#include "Array.h"
#include "mpifftw++.h"
#include "utils.h"
#include "mpiutils.h"
#include <unistd.h>
using namespace Array;

using namespace std;
using namespace fftwpp;

// Number of iterations.
unsigned int N0=10000000;
unsigned int N=0;

void init(Complex *f,
	  unsigned int X, unsigned int Y, unsigned int Z,
	  unsigned int x0, unsigned int y0, unsigned int z0,
	  unsigned int x, unsigned int y, unsigned int z)
{
  unsigned int c=0;
  for(unsigned int i=0; i < x; ++i) {
    unsigned int ii=x0+i;
    for(unsigned int j=0; j < y; j++) {
      unsigned int jj=y0+j;
      for(unsigned int k=0; k < Z; k++) {
	unsigned int kk=k;
	f[c++]=Complex(10*kk+ii,jj);
      }
    }
  }
}

void init(Complex *f, splitxy d)
{
  init(f,d.X,d.Y,d.Z,d.x0,d.y0,d.z0,d.x,d.y,d.z);
}

unsigned int outlimit=3000;

int main(int argc, char* argv[])
{
#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif
  int retval=0;

  unsigned int mx=4;
  unsigned int my=0;
  unsigned int mz=0;

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
  if(mz == 0) mz=mx;

  if(N == 0) {
    N=N0/mx/my;
    if(N < 10) N=10;
  }
  
  MPIgroup group(MPI_COMM_WORLD,mz,mx,my);

  if(group.size > 1 && provided < MPI_THREAD_FUNNELED)
    fftw::maxthreads=1;

  if(!quiet) {
    if(group.rank == 0) {
      cout << "provided: " << provided << endl;
      cout << "fftw::maxthreads: " << fftw::maxthreads << endl;
    }
    
    if(group.rank == 0) {
      cout << "Configuration: " 
	   << group.size << " nodes X " << fftw::maxthreads 
         << " threads/node" << endl;
    }
  }
  
  if(group.rank < group.size) {
    bool main=group.rank == 0;
    if(!quiet && main) {
      cout << "N=" << N << endl;
      cout << "mx=" << mx << ", my=" << my << ", mz=" << mz << endl;
      cout << "size=" << group.size << endl;
    }

    splitxy d(mx,my,mz,group);
    
    Complex *f=ComplexAlign(d.n);
    
    fft3dMPI fft(d,f);

    if(test) {
      init(f,d);

      if(!quiet && mx*my < outlimit) {
	if(main) cout << "\ninput:" << endl;
	show(f,d.x,d.y,d.Z,group.active);
      }

      size_t align=sizeof(Complex);
      array3<Complex> fgatherd(mx,my,mz,align);
      fft3d localForward(-1,fgatherd);
      fft3d localBackward(1,fgatherd);
      gatherxy(f, fgatherd(), d, group.active);

      array3<Complex> flocal(mx,my,mz,align);
      init(flocal(),d.X,d.Y,d.Z,0,0,0,d.X,d.Y,d.Z);
      if(main) {
	if(!quiet) {
	  cout << "Gathered input:\n" <<  fgatherd << endl;
	  cout << "Local input:\n" <<  flocal << endl;
	}

	double inputerror = relmaxerror(fgatherd(),flocal(),d.X,d.Y,d.Z);
	if(inputerror > 1e-10) {
	  cout << "Caution!  Inputs differ: " << inputerror << endl;
	  retval += 1;
	} else {
	  cout << "Inputs agree." << endl;
	}
      }
      
      fft.Forwards(f);

      if(main)
	localForward.fft(flocal);
      
      if(!quiet) {
	if(main) cout << "Distributed output:" << endl;
	show(f,d.x,d.xy.y,d.Z,group.active);
      }
      gatherxy(f, fgatherd(), d, group.active); 

      if(!quiet && main) {
	cout << "Gathered output:\n" <<  fgatherd << endl;
	cout << "Local output:\n" <<  flocal << endl;
      }
      
      if(main) {
	double outputerror = relmaxerror(fgatherd(),flocal(),d.X,d.Y,d.Z);
	if(outputerror > 1e-10) {
	  cout << "Caution!  Outputs differ: " << outputerror << endl;
	  retval += 1;
	} else {
	  cout << "Outputs agree." << endl;
	}
      }
      
      fft.Backwards(f);
      fft.Normalize(f);
      if(main)
	localBackward.fftNormalized(flocal);
      if(!quiet) {
	if(main) cout << "Distributed output:" << endl;
	show(f,d.x,d.xy.y,d.Z,group.active);
      }

      gatherxy(f, fgatherd(), d, group.active);
      
      if(!quiet && main) {
	cout << "Gathered output:\n" <<  fgatherd << endl;
	cout << "Local output:\n" <<  flocal << endl;
      }
      
      if(main) {
	double outputerror = relmaxerror(fgatherd(),flocal(),d.X,d.Y,d.Z);
	if(outputerror > 1e-10) {
	  cout << "Caution!  Outputs differ: " << outputerror << endl;
	  retval += 1;
	} else {
	  cout << "Outputs agree." << endl;
	}
      }
      
      if(!quiet) {
	if(main) cout << "\nback to input:" << endl;
	show(f,d.x,d.y,d.Z,group.active);
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
	delete[] T;
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
  
