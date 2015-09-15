#include "Array.h"
#include "mpifftw++.h"
#include "utils.h"
#include "mpiutils.h"

using namespace std;
using namespace fftwpp;
using namespace Array;

inline void init(Complex *f, splitxy d, int transpose=0) 
{
  unsigned int c=0;
  switch(transpose) {
  case 0: 
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
    break;
  case 1:
    // FIXME
    break;
  case 2:
    // FIXME
    break;
  default:
    cerr << "Invalid transposition case" << endl;
    exit(1);
  }
}

unsigned int outlimit=100;

int main(int argc, char* argv[])
{
  int retval=0; // success!

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  unsigned int mx=4;
  unsigned int my=4;
  unsigned int mz=4;
  
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c=getopt(argc,argv,"hm:x:y:z:");
    if (c == -1) break;
                
    switch (c) {
      case 0:
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
      case 'z':
        mz=atoi(optarg);
        break;
      case 'h':
      default:
        cout << "FIXME: usage" << endl;
    }
  }

  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);

  if(my == 0) my=mx;
  
  MPIgroup group(MPI_COMM_WORLD,mx,my);

  // If the process is unused, then do nothing.
  if(group.rank < group.size) {
    
    
    bool main=group.rank == 0;

    splitxy d(mx,my,mz,group);
    
    Complex *f=ComplexAlign(d.n);

    init(f,d);
    
    if(mx * my < outlimit) {
      if(main)
	cout << "\ninput:" << endl;
      cout << "process " << group.rank << endl;
      for(unsigned int i=0; i < d.n; ++i) {
      	cout << f[i] << endl;
      }
    }
      

  }

  if(group.rank == 0) {
    cout << endl;
    if(retval == 0) {
      cout << "Test passed." << endl;
    } else {
      cout << "Test FAILED!!!" << endl;
    }
  }
  
  MPI_Finalize();
  
  return retval;
}
