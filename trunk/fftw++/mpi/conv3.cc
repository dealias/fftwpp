#include "mpiconvolution.h"
#include "utils.h"
#include "mpiutils.h"

using namespace std;
using namespace fftwpp;

// Number of iterations.
unsigned int N0=10000000;
unsigned int N=0;
unsigned int mx=4;
unsigned int my=4;
unsigned int mz=4;
unsigned int M=1;
bool compact=true;

bool Direct=false, Implicit=true;

unsigned int outlimit=3000;

inline void init(Complex *f, Complex *g, const dimensions3& d, unsigned int M=1,
                 bool compact=true)
{
  double factor=1.0/sqrt((double) M);
  for(unsigned int s=0; s < M; ++s) {
    double S=sqrt(1.0+s);
    double ffactor=S*factor;
    double gfactor=1.0/S*factor;
    for(unsigned int i=!compact; i < d.nx; ++i) {
      unsigned int I=s*d.n+d.y*d.z*i;
      unsigned int ii=i-!compact;
      for(unsigned int j=d.y0 == 0 ? !compact : 0; j < d.y; ++j) {
        unsigned int IJ=I+d.z*j;
        unsigned int jj=d.y0-!compact+j;
          for(unsigned int k=0; k < d.z; ++k) {
            unsigned int kk=d.z0+k;
            f[IJ+k]=ffactor*Complex(ii+kk,jj+kk);
            g[IJ+k]=gfactor*Complex(2*ii+kk,jj+1+kk);
        }
      }
    }
  }
}

unsigned int padding(unsigned int m)
{
  unsigned int n=3*m-2;
  cout << "min padded buffer=" << n << endl;
  unsigned int log2n;
  // Choose next power of 2 for maximal efficiency.
  for(log2n=0; n > ((unsigned int) 1 << log2n); log2n++);
  return 1 << log2n;
}

int main(int argc, char* argv[])
{
#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  
  bool dohash=false;
  int retval=0;

#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"heipHc:M:N:m:x:y:z:n:T:");
    if (c == -1) break;
                
    switch (c) {
      case 0:
        break;
      case 'c':
        compact=atoi(optarg) != 0;
        break;
      case 'e':
        Implicit=false;
        break;
      case 'i':
        Implicit=true;
        break;
      case 'p':
        Implicit=false;
        break;
      case 'H':
	dohash=true;
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
      case 'h':
      default:
        usage(3,false,false);
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
    
  unsigned int nx=2*mx-compact;
  unsigned int ny=2*my-compact;
  unsigned int nz=mz+!compact;
    
  MPIgroup group(MPI_COMM_WORLD,nx,ny,nz);
  MPILoadWisdom(group.active);
  
  if(group.size > 1 && provided < MPI_THREAD_FUNNELED) {
    fftw::maxthreads=1;
  } else {
    fftw_init_threads();
    fftw_mpi_init();
  }
  
  if(group.rank == 0) {
    cout << "provided: " << provided << endl;
    cout << "fftw::maxthreads: " << fftw::maxthreads << endl;
    
    cout << "Configuration: " 
         << group.size << " nodes X " << fftw::maxthreads 
         << " threads/node" << endl;
  }
  
  if(group.rank < group.size) {
    bool main=group.rank == 0;
    if(main) {
      seconds();
      cout << "N=" << N << endl;
      cout << "mx=" << mx << ", my=" << my << ", mz=" << mz << endl;
      cout << "nx=" << nx << ", ny=" << ny << ", mz=" << mz << endl;
      cout << "size=" << group.size << endl;
      cout << "yblock=" << group.yblock << endl;
      cout << "zblock=" << group.zblock << endl;
    }

    dimensions3 d(nx,ny,ny,nz,group);
    dimensions3 du(mx+compact,ny,my+compact,nz,group);
    
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
      ImplicitHConvolution3MPI C(mx,my,mz,d,du,f,A,B,compact);
      Complex **F=new Complex *[A];
      unsigned int stride=d.n;
      for(unsigned int s=0; s < M; ++s) {
        unsigned int sstride=s*stride;
        F[2*s]=f+sstride;
        F[2*s+1]=g+sstride;
      }
      MPI_Barrier(group.active);
      if(main)
        cout << "Initialized after " << seconds() << " seconds." << endl;
      for(unsigned int i=0; i < N; ++i) {
        init(f,g,d,M,compact);
        if(main) seconds();
        C.convolve(F,mult);
//      C.convolve(f,g);
        if(main) T[i]=seconds();
      }
    
      if(main) 
        timings("Implicit",mx,T,N);
    
      delete [] F;
      
      if(nx*ny*mz < outlimit)
        show(f,nx,d.y,d.z,
             !compact,!compact && d.y0 == 0,0,
             nx,d.y,compact || d.z0+d.z < nz ? d.z : d.z-1,group.active);
  
      // check if the hash of the rounded output matches a known value
      if(dohash && compact) {
	int hashval=hash(f,nx,d.y,d.z,group.active);
	if(group.rank == 0) cout << hashval << endl;
	if(mx == 4 && my == 4 && mz == 4) {
	  if(hashval != -1156278167) {
	    retval=1;
	    if(group.rank == 0) cout << "error: hash does not match" << endl;
	  } else {
	    if(group.rank == 0) cout << "hash value OK." << endl;
	  }
	}
      }

    }
  
    deleteAlign(f);
    deleteAlign(g);
  
    delete [] T;
  }

  MPISaveWisdom(group.active);
  MPI_Finalize();
  
  return retval;
}
