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
unsigned int M=1;

bool Implicit=true, Explicit=false, Pruned=false;

inline void init(Complex *f, Complex *g, dimensions d, unsigned int M=1) 
{
  double factor=1.0/sqrt((double) M);
  for(unsigned int s=0; s < M; ++s) {
    double S=sqrt(1.0+s);
    double ffactor=S*factor;
    double gfactor=1.0/S*factor;
    for(unsigned int i=0; i < d.nx; ++i) {
      unsigned int I=s*d.n+d.y*i;
      for(unsigned int j=0; j < d.y; j++) {
        unsigned int jj=d.y0+j;
        f[I+j]=ffactor*Complex(i,jj);
        g[I+j]=gfactor*Complex(2*i,jj+1);
      }
    }
  }
}

unsigned int outlimit=100;

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
    int c = getopt(argc,argv,"heipHM:N:m:x:y:n:T:");
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
        mx=my=atoi(optarg);
        break;
      case 'x':
        mx=atoi(optarg);
        break;
      case 'y':
        my=atoi(optarg);
        break;
      case 'n':
        N0=atoi(optarg);
        break;
      case 'T':
        fftw::maxthreads=atoi(optarg);
        break;
      case 'h':
      default:
        usage(2);
    }
  }

  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);

  if(provided < MPI_THREAD_FUNNELED) {
    fftw::maxthreads=1;
  } else {
    fftw_init_threads();
    fftw_mpi_init();
  }
  
  if(my == 0) my=mx;

  if(N == 0) {
    N=N0/mx/my;
    if(N < 10) N=10;
  }
  
  MPIgroup group(my);
  
  if(group.rank == 0) {
    cout << "Configuration: " 
         << group.size << " nodes X " << fftw::maxthreads 
         << " threads/node" << endl;
  }
  
  if(group.rank < group.size) {
    bool main=group.rank == 0;
    if(main) {
      cout << "N=" << N << endl;
      cout << "mx=" << mx << ", my=" << my << endl;
    } 

    dimensions d(mx,my,group.active,group.yblock);
  
    unsigned int Mn=M*d.n;
  
    Complex *f=ComplexAlign(Mn);
    Complex *g=ComplexAlign(Mn);
  
    double *T=new double[N];

    if(Implicit) {
      ImplicitConvolution2MPI C(mx,my,d,M);
      Complex **F=new Complex *[M];
      Complex **G=new Complex *[M];
      unsigned int stride=d.n;
      for(unsigned int s=0; s < M; ++s) {
        unsigned int sstride=s*stride;
        F[s]=f+sstride;
        G[s]=g+sstride;
      }

      MPI_Barrier(group.active);
      if(group.rank == 0)
        cout << "Initialized after " << seconds() << " seconds." << endl;
      for(unsigned int i=0; i < N; ++i) {
        init(f,g,d,M);
        seconds();
        C.convolve(F,G);
//      C.convolve(f,g);
        T[i]=seconds();
      }
    
      if(main) 
        timings("Implicit",mx,T,N);
   
      if(mx*my < outlimit)
	show(f,mx,d.y,group);

      // check if the hash of the rounded output matches a known value
      if(dohash) {
	int hashval=hash(f,mx,d.y,group);
	if(group.rank == 0) cout << hashval << endl;
	if(mx == 4 && my == 4) {
	  if(hashval != -268695821) {
	    retval=1;
	    if(group.rank == 0) cout << "error: hash does not match" << endl;
	  }  else {
	    if(group.rank == 0) cout << "hash value OK." << endl;
	  }
	}
      }
      
      delete [] G;
      delete [] F;
    }
  
    deleteAlign(f);
    deleteAlign(g);
  
    delete [] T;
  }
  
  MPI_Finalize();
  
  return retval;
}

