#include "Array.h"
#include "mpifftw++.h"
#include "utils.h"
#include "mpiutils.h"

using namespace std;
using namespace fftwpp;
using namespace Array;

inline void init(Complex *f, 
		 unsigned int nx,
		 unsigned int ny, 
		 unsigned int x0,
		 unsigned int y0, 
		 unsigned int x,
		 unsigned int y, 
		 bool transposed) 
{
  unsigned int c = 0;
  if(!transposed) {
    for(unsigned int i = 0; i < x; ++i) {
      unsigned int ii = x0 + i;
      for(unsigned int j = 0; j < ny; j++) {
	f[c++] = Complex(ii, j);
      }
    }
  } else {
    for(unsigned int i = 0; i < nx; ++i) {
      for(unsigned int j = 0; j < y; j++) {
	unsigned int jj = y0 + j;
	f[c++] = Complex(i, jj);
      }
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

  unsigned int mx=4;
  unsigned int my=4;
  
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"hm:X:Y:");
    if (c == -1) break;
                
    switch (c) {
      case 0:
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
      case 'h':
      default:
        cout << "FIXME: usage" << endl;
    }
  }

  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);

  if(my == 0) my=mx;
  
  MPIgroup group(MPI_COMM_WORLD,mx);

  // If the process is unused, then do nothing.
  if(group.rank < group.size) { 
    bool main=group.rank == 0;

    splitx d(mx, my, group. active, group.block);
    
    Complex *f = ComplexAlign(d.n);

    init(f, d.nx, d.ny, d.x0, d.y0, d.x, d.y, false);
    
    // if(mx * my < outlimit) {
    //   if(main) cout << "\ninput:" << endl;
    //   show(f,d.x,my,group.active);
    // }

    array2<Complex> localf(mx,my);

    accumulate_splitx(f, localf(), d, false, group.active);
    if(main) {
      //cout << "Local version:\n" << localf << endl;
      array2<Complex> localf0(mx,my);
      init(localf0(), d.nx, d.ny, 0, 0, d.nx, d.ny, false);
      bool same = true;
      for(unsigned int i = 0; i < d.nx; ++i) {
	for(unsigned int j = 0; j < d.ny; ++j) {
	  if(localf(i,j) != localf0(i,j))
	    same = false;
	}
      }
      if(!same) {
	cout << "Error!" << endl;
	retval += 1;
      }
    }

    init(f, d.nx, d.ny, d.x0, d.y0, d.x, d.y, true);
    // if(mx*my < outlimit) {
    //   if(main) cout << "\noutput:" << endl;
    //   show(f,mx,d.y,group.active);
    // }

    localf = 0.0;
    accumulate_splitx(f, localf(), d, true, group.active);
    if(main) {
      //cout << "Local version:\n" << localf << endl;
      array2<Complex> localf0(mx,my);
      init(localf0(), d.nx, d.ny, 0, 0, d.nx, d.ny, false);
      //cout << "Local init:\n" << localf0 << endl;
      bool same = true;
      for(unsigned int i = 0; i < d.nx; ++i) {
	for(unsigned int j = 0; j < d.ny; ++j) {
	  if(localf(i,j) != localf0(i,j))
	    same = false;
	}
      }
      if(!same) {
	cout << "Error!" << endl;
	retval += 1;
      }
    }

      
    MPI_Barrier(group.active);
  }
  
  MPI_Finalize();
  
  return retval;
}
