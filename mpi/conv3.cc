#include "mpiconvolution.h"
#include "utils.h"

using namespace std;
using namespace utils;
using namespace fftwpp;
using namespace Array;

unsigned int outlimit=3000;

inline void init(Complex **F, const split3& d, unsigned int A=2,
                 bool xcompact=true, bool ycompact=true, bool zcompact=true)
{
  unsigned int M=A/2;
  double factor=1.0/sqrt((double) M);
  for(unsigned int s=0; s < M; ++s) {
    double S=sqrt(1.0+s);
    array3<Complex> f(d.X,d.y,d.z,F[s]);
    array3<Complex> g(d.X,d.y,d.z,F[M+s]);
    double ffactor=S*factor;
    double gfactor=1.0/S*factor;
    
    if(!xcompact) {
      for(unsigned int j=0; j < d.y; ++j) {
        for(unsigned int k=0; k < d.z; ++k) {
          f[0][j][k]=0.0;
          g[0][j][k]=0.0;
        }
      }
    }
    
    if(!ycompact) {
      for(unsigned int i=0; i < d.X; ++i) {
        for(unsigned int k=0; k < d.z; ++k) {
          f[i][0][k]=0.0;
          g[i][0][k]=0.0;
        }
      }
    }
    
    if(!zcompact && d.z0+d.z == d.Z) { // Last process
      for(unsigned int i=0; i < d.X; ++i) {
        for(unsigned int j=0; j < d.y; ++j) {
          f[i][j][d.z-1]=0.0;
          g[i][j][d.z-1]=0.0;
        }
      }
    }
    
    for(unsigned int i=!xcompact; i < d.X; ++i) {
      unsigned int ii=i-!xcompact;
      unsigned int jstart=!ycompact && d.y0 == 0;
      for(unsigned int j=jstart; j < d.y; ++j) {
        unsigned int jj=d.y0+j-!ycompact;
        unsigned int stop=(d.z0+d.z < d.Z) ? d.z : d.z-!zcompact;
        for(unsigned int k=0; k < stop; ++k) {
          unsigned int kk=d.z0+k;
          f[i][j][k]=ffactor*Complex(ii+kk,jj+kk);
          g[i][j][k]=gfactor*Complex(2*ii+kk,jj+1+kk);
        }
      }
    }
  }
}

int main(int argc, char* argv[])
{
  // Number of iterations.
  unsigned int N0=1000000;
  unsigned int N=0;
  
#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  
  int retval=0;

  unsigned int A=2; // Number of independent inputs
  unsigned int B=1; // Number of outputs
  
  unsigned int mx=4;
  unsigned int my=4;
  unsigned int mz=4;
  
  bool xcompact=true;
  bool ycompact=true;
  bool zcompact=true;
  int divisor=0; // Test for best block divisor
  int alltoall=-1; // Test for best alltoall routine

  bool test=false;
  bool quiet=false;

  int stats=0;
  
  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if(rank != 0) opterr=0;
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"hitqA:B:N:a:m:s:x:y:z:n:T:S:X:Y:Z:");
    if (c == -1) break;
                
    switch (c) {
      case 0:
        break;
      case 'a':
        divisor=atoi(optarg);
        break;
      case 'A':
        A=atoi(optarg);
        break;
      case 'B':
        B=atoi(optarg);
        break;
      case 'N':
        N=atoi(optarg);
        break;
      case 'm':
        mx=my=mz=atoi(optarg);
        break;
      case 's':
        alltoall=atoi(optarg);
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
      case 't':
        test=true;
        break;
      case 'q':
        quiet=true;
        break;
      case 'T':
        fftw::maxthreads=atoi(optarg);
        break;
      case 'S':
        stats=atoi(optarg);
        break;
      case 'X':
        xcompact=atoi(optarg) == 0;
        break;
      case 'Y':
        ycompact=atoi(optarg) == 0;
        break;
      case 'Z':
        zcompact=atoi(optarg) == 0;
        break;
      case 'i':
	// For compatibility reasons with -i option in OpenMP version.
	break;
      case 'h':
      default:
        if(rank == 0) {
          usage(3);
          usageCompact(3);
          usageTranspose();
        }
        exit(1);
    }
  }

  if(my == 0) my=mx;
  if(mz == 0) mz=mx;
  
  if(N == 0) {
    N=N0/mx/my/mz;
    if(N < 10) N=10;
  }

  unsigned int nx=2*mx-xcompact;
  unsigned int ny=2*my-ycompact;
  unsigned int nzp=mz+!zcompact;
    
  int size;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  unsigned int x=ceilquotient(nx,size);
  unsigned int y=ceilquotient(ny,size);
  unsigned int X2=mx+xcompact;
  unsigned int x2=ceilquotient(X2,size);
  bool allowpencil=nx*y == x*ny && X2*y == x2*ny;
  
  MPIgroup group(MPI_COMM_WORLD,ny,nzp,allowpencil);
  
  if(group.size > 1 && provided < MPI_THREAD_FUNNELED)
    fftw::maxthreads=1;
  
  defaultmpithreads=fftw::maxthreads;
    
  if(group.rank < group.size) {
    bool main=group.rank == 0;
    if(!quiet && main) {
      seconds();
      cout << "Configuration: " 
           << group.size << " nodes X " << fftw::maxthreads 
           << " threads/node" << endl;
      cout << "Using MPI VERSION " << MPI_VERSION << endl;
    }

    // Dimensions of the data
    split3 d(nx,ny,nzp,group,true);
    
    // Dimensions used in the MPI convolution
    split3 du(mx+xcompact,ny,my+ycompact,nzp,group,true);
//    du.n=max(du.n,d.n2);
    
    if(B != 1) {
      cerr << "Only B=1 is implemented" << endl;
      exit(1);
    }
    
    Complex **F=new Complex*[A];
    for(unsigned int a=0; a < A; a++)
      F[a]=ComplexAlign(d.n);
    
    realmultiplier *mult;
    
    switch(A) {
      case 2: mult=multbinary; break;
      case 4: mult=multbinary2; break;
      default: if(main) cout << "A=" << A << " is not yet implemented" << endl;
        exit(1);
    }

    if(!quiet && main) {
      if(!test)
        cout << "N=" << N << endl;
      cout << "A=" << A << endl;
      cout << "mx=" << mx << ", my=" << my << ", mz=" << mz << endl;
      cout << "nx=" << nx << ", ny=" << ny << ", nzp=" << nzp << endl;
    }
    
    ImplicitHConvolution3MPI C(mx,my,mz,xcompact,ycompact,zcompact,d,du,F[0],
                               mpiOptions(divisor,alltoall),A,B);
    
    if(test) {
      init(F,d,A,xcompact,ycompact,zcompact);
      
      if(!quiet) {
        if(main) cout << "Distributed input:" << endl;
        for(unsigned int a=0; a < A; a++) {
          if(main) cout << "a: " << a << endl;
          show(F[a],d.X,d.y,d.z,group.active);
        }
      }
      
      if(!quiet && main) cout << "Gathered input:" << endl;
      Complex **F0=new Complex*[A];
      for(unsigned int a=0; a < A; a++) {
        if(main) {
          F0[a]=ComplexAlign(d.X*d.Y*d.Z);
        }
        gatheryz(F[a],F0[a],d,group.active);
        if(!quiet && main) {
          cout << "a: " << a << endl;
          show(F0[a],d.X,d.Y,d.Z,0,0,0,d.X,d.Y,d.Z);
        }
      }

      C.convolve(F,mult);

      if(!quiet && nx*ny*mz < outlimit) {
        if(main) cout << "Distributed output: " << endl;
        show(F[0],d.X,d.y,d.z,group.active);
      }
      
      Complex **F0out=new Complex*[B];
      if(main) {
        for(unsigned int b=0; b < B; b++)
          F0out[b]=ComplexAlign(d.X*d.Y*d.Z);
      }
      for(unsigned int b=0; b < B; b++) {
        gatheryz(F[b],F0out[b],d,group.active);
        if(!quiet && main) {
          cout << "Gathered output:" << endl;
          cout << "b: " << b << endl;
          show(F0out[b],d.X,d.Y,d.Z,0,0,0,d.X,d.Y,d.Z);
        }
      }

      if(main) {
        ImplicitHConvolution3 Clocal(mx,my,mz,xcompact,ycompact,zcompact,A,B);
        Clocal.convolve(F0,mult);
        if(!quiet)
          cout << "Local output:" << endl;
        for(unsigned int b=0; b < B; b++) {
          if(!quiet) {
            cout << "b: " << b << endl;
            show(F0[b],d.X,d.Y,d.Z,0,0,0,d.X,d.Y,d.Z);
          }
          retval += checkerror(F0[b],F0out[b],d.X*d.Y*d.Z);
        }
      }
      
      if(main) {
        for(unsigned int a=0; a < A; a++)
          deleteAlign(F0[a]);
        for(unsigned int b=0; b < B; b++)
          deleteAlign(F0out[b]);
      }
      delete[] F0;
      delete[] F0out;
      
    } else {
      if(!quiet && main)
        cout << "Initialized after " << seconds() << " seconds." << endl;

      MPI_Barrier(group.active);
      
      double *T=new double[N];
      for(unsigned int i=0; i < N; ++i) {
        init(F,d,A,xcompact,ycompact,zcompact);
        if(main) seconds();
        C.convolve(F,mult);
        if(main) T[i]=seconds();
      }
      if(main) 
        timings("Implicit",mx,T,N,stats);
      delete [] T;
      
      if(!quiet && nx*ny*mz < outlimit) {
        if(main) cout << "output: " << endl;
        show(F[0],d.X,d.y,d.z,group.active);
      }
    }
      
    for(unsigned int a=0; a < A; a++)
      deleteAlign(F[a]);
    delete[] F;

    
  }

  MPI_Finalize();
  
  return retval;
}
