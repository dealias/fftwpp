#include "mpiconvolution.h"
#include "utils.h"
#include "mpiutils.h"

using namespace std;
using namespace fftwpp;

// Number of iterations.
unsigned int N0=1000000;
unsigned int N=0;

bool Direct=false, Implicit=true;

unsigned int outlimit=3000;

inline void init(Complex *f, Complex *g, const split3& d, unsigned int M=1,
                 bool xcompact=true, bool ycompact=true, bool zcompact=true)
{
  double factor=1.0/sqrt((double) M);
  for(unsigned int s=0; s < M; ++s) {
    double S=sqrt(1.0+s);
    double ffactor=S*factor;
    double gfactor=1.0/S*factor;
    unsigned int stride=s*d.n;
    
    if(!xcompact) {
      for(unsigned int j=0; j < d.y; ++j) {
        unsigned int IJ=stride+d.z*j;
        for(unsigned int k=0; k < d.z; ++k) {
          f[IJ+k]=0.0;
          g[IJ+k]=0.0;
        }
      }
    }
    
    if(!ycompact) {
      for(unsigned int i=0; i < d.x; ++i) {
        unsigned int IJ=stride+d.y*d.z*i;
        for(unsigned int k=0; k < d.z; ++k) {
          f[IJ+k]=0.0;
          g[IJ+k]=0.0;
        }
      }
    }
    
    if(!zcompact && d.z0+d.z == d.Z) { // Last process
      for(unsigned int i=0; i < d.X; ++i) {
        unsigned int I=stride+d.y*d.z*i;
        for(unsigned int j=0; j < d.y; ++j) {
          unsigned int IJ=I+d.z*j;
          f[IJ+d.z-1]=0.0;
          g[IJ+d.z-1]=0.0;
        }
      }
    }
    
    for(unsigned int i=!xcompact; i < d.X; ++i) {
      unsigned int I=stride+d.y*d.z*i;
      unsigned int ii=i-!xcompact;
      for(unsigned int j=!ycompact && d.y == 0; j < d.y; ++j) {
        unsigned int IJ=I+d.z*j;
        unsigned int jj=d.y0+j-!ycompact;
        unsigned int stop=d.z0+d.z < d.Z ? d.z : d.z-!zcompact;
        for(unsigned int k=0; k < stop; ++k) {
          unsigned int kk=d.z0+k;
          f[IJ+k]=ffactor*Complex(ii+kk,jj+kk);
          g[IJ+k]=gfactor*Complex(2*ii+kk,jj+1+kk);
        }
      }
    }
  }
}

int main(int argc, char* argv[])
{
#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  
  int retval=0;
  bool quiet=false;

  unsigned int mx=4;
  unsigned int my=4;
  unsigned int mz=4;
  unsigned int M=1;
  bool xcompact=true;
  bool ycompact=true;
  bool zcompact=true;
  int divisor=0; // Test for best block divisor
  int alltoall=-1; // Test for best alltoall routine

#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"heiqM:N:a:m:s:x:y:z:n:T:X:Y:Z:");
    if (c == -1) break;
                
    switch (c) {
      case 0:
        break;
      case 'a':
        divisor=atoi(optarg);
        break;
      case 'e':
        Implicit=false;
        break;
      case 'i':
        Implicit=true;
        break;
      case 'M':
        M=atoi(optarg);
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
      case 'q':
        quiet=true;
        break;
      case 'T':
        fftw::maxthreads=atoi(optarg);
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
      case 'h':
      default:
        usage(3,false,false,true);
        exit(1);
    }
  }

  unsigned int A=2*M; // Number of independent inputs
  unsigned int B=1;   // Number of outputs
  
  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);

  if(N == 0) {
    N=N0/mx/my/mz;
    if(N < 10) N=10;
  }
    
  unsigned int nx=2*mx-xcompact;
  unsigned int ny=2*my-ycompact;
  unsigned int nzp=mz+!zcompact;
    
  MPIgroup group(MPI_COMM_WORLD,ny,nzp,nx);
  
  if(group.size > 1 && provided < MPI_THREAD_FUNNELED)
    fftw::maxthreads=1;
  
  if(group.rank < group.size) {
    bool main=group.rank == 0;
    if(!quiet && main) {
      seconds();
      cout << "provided: " << provided << endl;
      cout << "fftw::maxthreads: " << fftw::maxthreads << endl;
    
      cout << "Configuration: " 
           << group.size << " nodes X " << fftw::maxthreads 
           << " threads/node" << endl;
    
      cout << "N=" << N << endl;
      cout << "mx=" << mx << ", my=" << my << ", mz=" << mz << endl;
      cout << "nx=" << nx << ", ny=" << ny << ", nzp=" << nzp << endl;
      cout << "size=" << group.size << endl;
    }

    split3 d(nx,ny,nzp,group,true);
    split3 du(mx+xcompact,ny,my+ycompact,nzp,group,true);
    
    unsigned int Mn=M*d.n;
    Complex *f=ComplexAlign(Mn);
    Complex *g=ComplexAlign(Mn);

    double *T=new double[N];

    realmultiplier *mult;
  
    switch(M) {
      case 1: mult=multbinary; break;
      case 2: mult=multbinary2; break;
      default: cout << "M=" << M << " is not yet implemented" << endl; exit(1);
    }

    if(Implicit) {
      ImplicitHConvolution3MPI C(mx,my,mz,xcompact,ycompact,zcompact,d,du,f,
                                 mpiOptions(divisor,alltoall),A,B);
      Complex **F=new Complex *[A];
      unsigned int stride=d.n;
      for(unsigned int s=0; s < M; ++s) {
        unsigned int sstride=s*stride;
        F[2*s]=f+sstride;
        F[2*s+1]=g+sstride;
      }
      MPI_Barrier(group.active);
      if(!quiet && main)
        cout << "Initialized after " << seconds() << " seconds." << endl;
      for(unsigned int i=0; i < N; ++i) {
        init(f,g,d,M,xcompact,ycompact,zcompact);
        if(main) seconds();
        C.convolve(F,mult);
//      C.convolve(f,g);
        if(main) T[i]=seconds();
      }
    
      if(main) 
        timings("Implicit",mx,T,N);
    
      delete [] F;
      
      if(!quiet && nx*ny*mz < outlimit)
        show(f,d.X,d.y,d.z,
             !xcompact,!ycompact && d.y0 == 0,0,
             d.X,d.y,d.z0+d.z < d.Z ? d.z : d.z-!zcompact,group.active);
    }
  
    deleteAlign(f);
    deleteAlign(g);
  
    delete [] T;
  }

  MPI_Finalize();
  
  return retval;
}
