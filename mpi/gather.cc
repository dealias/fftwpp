#include "Array.h"
#include "mpifftw++.h"
#include "utils.h"

using namespace std;
using namespace utils;
using namespace fftwpp;
using namespace Array;

inline void init(Complex *f, unsigned int nx, unsigned int ny, unsigned int x0,
                 unsigned int y0, unsigned int x, unsigned int y, 
                 unsigned int nz, bool transposed) 
{
  if(!transposed) {
    for(unsigned int i=0; i < x; ++i) {
      unsigned int ii=x0 + i;
      for(unsigned int j=0; j < ny; j++) {
        for(unsigned int k=0; k < nz; k++) {
          int pos=i*ny*nz+j*nz+k;
          f[pos]=Complex(ii,j+(1.0*k)/nz);
        }
      }
    }
  } else {
    for(unsigned int i=0; i < nx; ++i) {
      for(unsigned int j=0; j < y; j++) {
        unsigned int jj=y0 + j;
        for(unsigned int k=0; k < nz; k++) {
          int pos=i*y*nz+j*nz+k;
          f[pos]=Complex(i,jj+(1.0*k)/nz);
        }
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
  unsigned int mz=1;
  
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

  if(my == 0) my=mx;
  
  MPIgroup group(MPI_COMM_WORLD,mx);

  // If the process is unused, then do nothing.
  if(group.rank < group.size) { 
    bool main=group.rank == 0;

    split d(mx,my,group. active);
    
    Complex *f=ComplexAlign(d.n*mz);

    init(f,d.X,d.Y,d.x0,d.y0,d.x,d.y,mz,false);

    if(!quiet) {
      if(mx * my < outlimit) {
        if(main)
          cout << "\ninput:" << endl;
        cout << "process " << group.rank << endl;
        for(unsigned int i=0; i < d.n*mz; ++i) {
          cout << f[i] << endl;
        }
      }
    }
    
    array3<Complex> localf(mx,my,mz);
    localf=0.0;
    
    gatherx(f,localf(),d,mz,group.active);
    if(main) {
      if(!quiet) {
        cout << "Gathered:" <<endl;
        cout << localf << endl;
      }
      array3<Complex> localf0(mx,my,mz);
      init(localf0(),d.X,d.Y,0,0,d.X,d.Y,mz,false);
      if(!quiet) {
        cout << "Local init:" << endl;
        cout << localf0 << endl;
      }
      bool same=true;
      if(!quiet) {
        for(unsigned int i=0; i < d.X; ++i) {
          for(unsigned int j=0; j < d.Y; ++j) {
            for(unsigned int k=0; k < mz; ++k) {
              cout << localf(i,j,k) << " " << localf0(i,j,k) << endl;
              if(localf(i,j,k) != localf0(i,j,k))
                same=false;
            }
          }
        }
      }
      if(same) {
        cout << "OK." << endl;
      } else {
        cout << "ERROR!" << endl;
        retval += 1;
      }
    }


    if(!quiet)
      cout << "\nTransposed init:" << endl;
    init(f,d.X,d.Y,d.x0,d.y0,d.x,d.y,mz,true);
    if(!quiet) {
      if(mx*my < outlimit) {
        if(main) cout << "\noutput:" << endl;
        show(f,d.X,d.Y,mz,
             d.x0,d.y0,0,
             d.x,d.y,mz,
             group.active);

        cout << "process " << group.rank << endl;
        for(unsigned int i=0; i < d.n * mz; ++i) {
          cout << f[i] << endl;
        }
      }
    }
        
    localf=0.0;
    gathery(f,localf(),d,mz,group.active);
    if(main) {
      if(!quiet) {
        if(mx*my < outlimit) {
          cout << "\nGathered version:" << endl;
          cout << localf << endl;
        }
      }
      array3<Complex> localf0(mx,my,mz);
      init(localf0(),d.X,d.Y,0,0,d.X,d.Y,mz,false);
      if(!quiet) {
        if(mx*my < outlimit) {
          cout << "Local init:" << endl;
          cout << localf0 << endl;
        }
      }
        
      //cout << "Comparison:\n"  << endl;
      bool same=true;
      for(unsigned int i=0; i < d.X; ++i) {
        for(unsigned int j=0; j < d.Y; ++j) {
          for(unsigned int k=0; k < mz; ++k) {
            //cout << localf(i,j,k) << " " << localf0(i,j,k) << endl;
            if(localf(i,j,k) != localf0(i,j,k))
              same=false;
          }
        }
      }
      if(same) {
        cout << "OK." << endl;
      } else {
        cout << "ERROR!" << endl;
        retval += 1;
      }
    }
      
    MPI_Barrier(group.active);
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
