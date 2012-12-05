#include "mpiconvolution.h"
#include "utils.h"

using namespace std;
using namespace fftwpp;

// Number of iterations.
unsigned int N0=10000000;
unsigned int N=0;
unsigned int mx=4;
unsigned int my=4;
unsigned int mz=4;
unsigned int M=1;

bool Direct=false, Implicit=true;

unsigned int outlimit=3000;

inline void init(Complex *f, Complex *g, const dimensions3& d, unsigned int M=1)
{
  double factor=1.0/sqrt((double) M);
  for(unsigned int s=0; s < M; ++s) {
    double S=sqrt(1.0+s);
    double ffactor=S*factor;
    double gfactor=1.0/S*factor;
    for(unsigned int i=0; i < d.nx; ++i) {
      unsigned int I=s*d.n+d.y*d.z*i;
      for(unsigned int j=0; j < d.y; ++j) {
        unsigned int IJ=I+d.z*j;
        unsigned int jj=d.y0+j;
        for(unsigned int k=0; k < d.z; ++k) {
          unsigned int kk=d.z0+k;
          f[IJ+k]=ffactor*Complex(i+kk,jj+kk);
          g[IJ+k]=gfactor*Complex(2*i+kk,jj+1+kk);
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
  
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"hdeiptM:N:m:x:y:z:n:T:");
    if (c == -1) break;
                
    switch (c) {
      case 0:
        break;
      case 'd':
        Direct=true;
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

  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);

  if(provided < MPI_THREAD_FUNNELED) {
    fftw::maxthreads=1;
  } else {
    fftw_init_threads();
    fftw_mpi_init();
  }
  
  if(N == 0) {
    N=N0/mx/my/mz;
    if(N < 10) N=10;
  }
    
  unsigned int nx=2*mx-1;
  unsigned int ny=2*my-1;
    
  MPIgroup group(ny,mz);
  
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

    dimensions3 d(nx,ny,ny,mz,group);
    dimensions3 du(mx+1,ny,my+1,mz,group);
    
    unsigned int Mn=M*d.n;
    Complex *f=ComplexAlign(Mn);
    Complex *g=ComplexAlign(Mn);

    double *T=new double[N];

    if(Implicit) {
      ImplicitHConvolution3MPI C(mx,my,mz,d,du,f,M);
      Complex **F=new Complex *[M];
      Complex **G=new Complex *[M];
      unsigned int stride=d.n;
      for(unsigned int s=0; s < M; ++s) {
        unsigned int sstride=s*stride;
        F[s]=f+sstride;
        G[s]=g+sstride;
      }
      MPI_Barrier(group.active);
      if(main)
        cout << "Initialized after " << seconds() << " seconds." << endl;
      for(unsigned int i=0; i < N; ++i) {
        init(f,g,d,M);
        seconds();
        C.convolve(F,G);
//      C.convolve(f,g);
        T[i]=seconds();
      }
    
      if(main) 
        timings("Implicit",T,N);
    
      delete [] G;
      delete [] F;
      
      if(nx*ny*mz < outlimit) 
        show(f,nx,d.y,d.z,group);
    }
  
    deleteAlign(f);
    deleteAlign(g);
  
    delete [] T;
  }

  MPI_Finalize();
  
  return 0;
}
