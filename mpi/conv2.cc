#include "mpiconvolution.h"
#include "utils.h"

using namespace std;
using namespace fftwpp;

// Number of iterations.
unsigned int N0=10000000;
unsigned int N=0;
unsigned int nx=0;
unsigned int mx=4;
unsigned int my=4;
unsigned int M=1;

bool Direct=false, Implicit=true, Explicit=false, Pruned=false;

unsigned int outlimit=200;

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
        unsigned int jj=j+d.y0;
        f[I+j]=ffactor*Complex(i,jj);
        g[I+j]=gfactor*Complex(2*i,jj+1);
      }
    }
  }
}

int main(int argc, char* argv[])
{
#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif
  
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"heiptM:N:m:x:y:n:T:");
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
  
  if(N == 0) {
    N=N0/mx/my;
    if(N < 10) N=10;
  }
  
  unsigned int nx=2*mx-1;
  unsigned int mx1=mx+1;
  
  MPIgroup group(my);
  
  if(group.rank < group.size) {
    bool main=group.rank == 0;
    if(main) {
      seconds();
      cout << "N=" << N << endl;
      cout << "mx=" << mx << ", my=" << my << endl;
    }
    
    dimensions d(nx,my,group.active,group.yblock);
    dimensions du(mx1,my,group.active,group.yblock);
  
    unsigned int Mn=M*d.n;
  
    Complex *f=ComplexAlign(Mn);
    Complex *g=ComplexAlign(Mn);
  
    double *T=new double[N];
  
    if(Implicit) {
      ImplicitHConvolution2MPI C(mx,my,d,du,f,M);
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
        if(main) seconds();
        C.convolve(F,G);
        //C.convolve(f,g);
        if(main) T[i]=seconds();
      }
      if(main)
        timings("Implicit",mx,T,N);

      delete [] G;
      delete [] F;
    }
    if(nx*my < outlimit) 
      show(f,nx,d.y,group);
    
    deleteAlign(g);
    deleteAlign(f);
  
    delete [] T;
  }
  
  MPI_Finalize();
  
  return 0;
}
