#include "Array.h"
#include "mpifftw++.h"
#include "utils.h"
#include "mpiutils.h"

using namespace std;
using namespace fftwpp;
using namespace Array;

inline void init(Complex *f,
		 unsigned int X, unsigned int Y, unsigned int Z,
		 unsigned int x0, unsigned int y0, unsigned int z0,
		 unsigned int x, unsigned int y, unsigned int z,
		 int transpose=0) 
{
  unsigned int c=0;
  switch(transpose) {
  case 0: 
    for(unsigned int i=0; i < X; ++i) {
      unsigned int ii=i;
      for(unsigned int j=0; j < y; j++) {
	unsigned int jj=y0+j;
	for(unsigned int k=0; k < z; k++) {
	  unsigned int kk=z0+k;
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

  bool quiet=false;
  unsigned int mx=4;
  unsigned int my=4;
  unsigned int mz=4;
  
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c=getopt(argc,argv,"hm:x:y:z:q");
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
      case 'q':
        quiet=true;
        break;
      case 'h':
      default:
        cout << "FIXME: usage" << endl;
    }
  }

  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);

  if(mx == 0) mx=4;
  if(my == 0) my=mx;
  if(mz == 0) mz=mx;
  
  MPIgroup group(MPI_COMM_WORLD,mx,my);

  // If the process is unused, then do nothing.
  if(group.rank < group.size) {
    
    bool main=group.rank == 0;

    splityz d(mx,my,mz,group);

    if(!quiet) {
      // cout << "process " << group.rank << endl;
      // d.show();
    }

    Complex *pF=ComplexAlign(d.n2);
    array3<Complex> F0(d.X,d.y,d.z,pF);
     
    init(F0(),d.X,d.Y,d.X,d.x0,d.y0,d.z0,d.x,d.y,d.z);
    //F0.Load(group.rank);
    
    if(!quiet){
      if(main) cout << "\ninput:" << endl;
      show(F0(),mx,d.y,d.z,group.active);
    }
    
    if(!quiet && main) cout << "Accumulating... " << endl;
    // Local array for transpose=0
    Complex *pf0=ComplexAlign(mx * my * mz);
    array3<Complex> f0(mx,my,mz,pf0);
    f0.Load(0.0);
    accumulate_splityz(F0(),f0(),
    		       d.X, d.Y, d.Z,
    		       d.x0, d.y0, d.z0,
    		       d.x, d.y, d.z,
    		       0, group.active);
    array3<Complex> g0(d.X,d.Y,d.Z);
    g0.Load(0.0);
    if(!quiet && main) {
      cout << "Local transpose=0:" << endl;
      cout << f0 << endl;
      cout << "local init:" << endl;
      array3<Complex> G0(d.X,d.Y,d.Z);
      init(g0,d.X,d.Y,d.Z,0,0,0,d.X,d.Y,d.Z);
      cout << g0 << endl;
    }
    bool same=true;
    for(unsigned int i = 0; i < d.X; ++i) {
      for(unsigned int j = 0; j < d.Y; ++j) {
	for(unsigned int k = 0; k < d.Z; ++k) {
      if(g0(i,j,k) != f0(i,j,k))
	same=false;
	}
      }
    }
    if(!same)
      retval++;
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
