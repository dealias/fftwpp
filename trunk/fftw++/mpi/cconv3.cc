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

bool Direct=false, Implicit=true, Explicit=false, Pruned=false;

inline void init(Complex *f, Complex *g, const dimensions3& d, unsigned int M=1)
{
  double factor=1.0/sqrt((double) M);
  for(unsigned int s=0; s < M; ++s) {
    double S=sqrt(1.0+s);
    double ffactor=S*factor;
    double gfactor=1.0/S*factor;
    for(unsigned int i=0; i < d.nx; ++i) {
      unsigned int I=s*d.n+d.y*d.z*i;
      for(unsigned int j=0; j < d.y; j++) {
        unsigned int IJ=I+d.z*j;
        unsigned int jj=d.y0+j;
        for(unsigned int k=0; k < d.z; k++) {
          unsigned int kk=d.z0+k;
          f[IJ+k]=ffactor*Complex(i+kk,jj+kk);
          g[IJ+k]=gfactor*Complex(2*i+kk,jj+1+kk);
        }
      }
    }
  }
}

unsigned int outlimit=3000;

inline unsigned int min(unsigned int a, unsigned int b)
{
  return (a < b) ? a : b;
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
    int c = getopt(argc,argv,"heipHM:N:m:x:y:z:n:T:");
    if (c == -1) break;
                
    switch (c) {
      case 0:
        break;
      case 'e':
        Explicit=true;
        Implicit=false;
        Pruned=false;
        break;
      case 'i':
        Implicit=true;
        Explicit=false;
        break;
      case 'p':
        Explicit=true;
        Implicit=false;
        Pruned=true;
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
        usage(3);
    }
  }

  unsigned int A=2*M; // Number of independent inputs
  
  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);

  if(my == 0) my=mx;

  if(N == 0) {
    N=N0/mx/my/mz;
    if(N < 10) N=10;
  }
  
  MPIgroup group(MPI_COMM_WORLD,mx,my,mz);
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
      cout << "N=" << N << endl;
      cout << "mx=" << mx << ", my=" << my << ", mz=" << mz << endl;
      cout << "size=" << group.size << endl;
      cout << "yblock=" << group.yblock << endl;
      cout << "zblock=" << group.zblock << endl;
    }
  
    dimensions3 d(mx,my,my,mz,group);
    
    unsigned int Mn=M*d.n;
    Complex *f=ComplexAlign(Mn);
    Complex *g=ComplexAlign(Mn);

    double *T=new double[N];
  
    multiplier *mult;
  
    switch(M) {
      case 1: mult=multbinary; break;
      case 2: mult=multbinary2; break;
      case 3: mult=multbinary3; break;
      case 4: mult=multbinary4; break;
      case 8: mult=multbinary8; break;
      default: cout << "M=" << M << " is not yet implemented" << endl; exit(1);
    }

    if(Implicit) {
      ImplicitConvolution3MPI C(mx,my,mz,d,A);
      unsigned int stride=d.n;
      Complex **F=new Complex *[A];
      for(unsigned int s=0; s < M; ++s) {
        unsigned int sstride=s*stride;
        F[2*s]=f+sstride;
        F[2*s+1]=g+sstride;
      }
      MPI_Barrier(group.active);
      if(group.rank == 0)
        cout << "Initialized after " << seconds() << " seconds." << endl;
      for(unsigned int i=0; i < N; ++i) {
        init(f,g,d,M);
        if(main) seconds();
        C.convolve(F,mult);
//      C.convolve(f,g);
        if(main) T[i]=seconds();
      }
    
      if(main) 
        timings("Implicit",mx,T,N);
      delete [] F;
    
      if(mx*my*mz < outlimit) 
        show(f,mx,d.y,d.z,group.active);

      // check if the hash of the rounded output matches a known value
      if(dohash) {
	int hashval=hash(f,mx,d.y,d.z,group.active);
	if(group.rank == 0) cout << hashval << endl;
	if(mx == 4 && my == 4 && mz == 4) {
	  if(hashval != 1073202285) {
	    retval=1;
	    if(group.rank == 0) cout << "error: hash does not match" << endl;
	  }  else {
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
