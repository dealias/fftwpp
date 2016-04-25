#include "Array.h"
#include "mpifftw++.h"
#include "utils.h"

using namespace std;
using namespace utils;
using namespace fftwpp;
using namespace Array;

inline void init(Complex *f,
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
        Complex val=Complex(10*kk+ii,jj);
        f[c++]=val;
      }
    }
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
  
  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if(rank != 0) opterr=0;
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
        if(rank == 0)
          usageGather();
        exit(1);
    }
  }

  if(mx == 0) mx=4;
  if(my == 0) my=mx;
  if(mz == 0) mz=mx;
  
  MPIgroup group(MPI_COMM_WORLD,mx,my);

  // If the process is unused, then do nothing.
  if(group.rank < group.size) {
    
    bool main=group.rank == 0;

    split3 d(mx,my,mz,group);

    unsigned int localsize=d.x*d.y*d.Z;
    Complex *pF=ComplexAlign(localsize);
    Array3<Complex> F0(d.x,d.y,d.Z,pF);
     
    init(F0(),d.X,d.Y,d.Z,d.x0,d.y0,d.z0,d.x,d.y,d.z);
    //F0.Load(group.rank);
    
    if(!quiet){
      if(main) cout << "\ninput:" << endl;
      //show(F0(),d.x,d.y,d.Z,group.active);
    }
    
    if(!quiet) {
      // if(group.rank == 1) {
      //         cout << "process " << group.rank << endl;
      //          d.show();
      // }
      cout << "process " << group.rank << endl;
      cout << F0 << endl;
//      d.show();
    }
    
    if(!quiet && main)
      cout << "Gathering... " << endl;
    // Local array for transpose=0
    //Complex *pf0=ComplexAlign(d.X*d.Y*d.Z);
    Array3<Complex> f0(d.X,d.Y,d.Z);
    f0.Load(0.0);
    gatherxy(F0(),f0(),d,group.active);
    if(main) {
      array3<Complex> g0(d.X,d.Y,d.Z);
      init(g0,d.X,d.Y,d.Z,0,0,0,d.X,d.Y,d.Z);
      if(!quiet) {
        cout << "Local transpose=0:" << endl;
        cout << f0 << endl;
        cout << "local init:" << endl;
        cout << g0 << endl << endl;;
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
  }

  if(group.rank == 0) {
    if(retval == 0) {
      cout << "Test passed." << endl;
    } else {
      cout << "Test FAILED!!!" << endl;
    }
  }

  MPI_Finalize();
  return retval;
}
