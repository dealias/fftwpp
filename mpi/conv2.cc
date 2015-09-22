#include "mpiconvolution.h"
#include "utils.h"
#include "mpiutils.h"

using namespace std;
using namespace fftwpp;

// Number of iterations.
unsigned int N0=10000000;
unsigned int N=0;
unsigned int nx=0;
unsigned int mx=4;
unsigned int my=4;
unsigned int M=1;
bool xcompact=true;
bool ycompact=true;

bool Direct=false, Implicit=true, Explicit=false, Pruned=false;

unsigned int outlimit=200;

inline void init(Complex *f, Complex *g, split d, unsigned int M=1,
                 bool xcompact=true)
{
  double factor=1.0/sqrt((double) M);
  for(unsigned int s=0; s < M; ++s) {
    double S=sqrt(1.0+s);
    double ffactor=S*factor;
    double gfactor=1.0/S*factor;
    for(unsigned int i=!xcompact; i < d.X; ++i) {
      unsigned int I=s*d.n+d.y*i;
      unsigned int ii=i-!xcompact;
      for(unsigned int j=0; j < d.y; j++) {
        unsigned int jj=j+d.y0;
        f[I+j]=ffactor*Complex(ii,jj);
        g[I+j]=gfactor*Complex(2*ii,jj+1);
      }
    }
  }
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
    int c = getopt(argc,argv,"heipH:M:N:m:x:y:n:T:X:Y:");
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
      case 'X':
        xcompact=atoi(optarg) == 0;
        break;
      case 'Y':
        ycompact=atoi(optarg) == 0;
        break;
      case 'h':
      default:
        usage(2,false,true,true);
    }
  }

  unsigned int A=2*M; // Number of independent inputs
  unsigned int B=1;   // Number of outputs
  
  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);

  if(N == 0) {
    N=N0/mx/my;
    if(N < 10) N=10;
  }
  
  unsigned int nx=2*mx-xcompact;
  unsigned int ny=my+!ycompact;
  
  MPIgroup group(MPI_COMM_WORLD,ny);
  MPILoadWisdom(group.active);
  
  if(group.size > 1 && provided < MPI_THREAD_FUNNELED)
    fftw::maxthreads=1;

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
      cout << "mx=" << mx << ", my=" << my << endl;
      cout << "nx=" << nx << ", ny=" << ny << endl;
    }
    
    split d(nx,ny,group.active);
    split du(mx+xcompact,ny,group.active);
  
    unsigned int Mn=M*d.n;
  
    Complex *f=ComplexAlign(Mn);
    Complex *g=ComplexAlign(Mn);
  
    double *T=new double[N];
  
    if(Implicit) {
      realmultiplier *mult;
  
      switch(M) {
        case 1: mult=multbinary; break;
        case 2: mult=multbinary2; break;
        default: cout << "M=" << M << " is not yet implemented" << endl;
          exit(1);
      }

      convolveOptions options;
      options.xcompact=xcompact;
      options.ycompact=ycompact;
      ImplicitHConvolution2MPI C(mx,my,d,du,f,A,B,options);
      
      Complex **F=new Complex *[A];
      unsigned int stride=d.n;
      for(unsigned int s=0; s < M; ++s) {
        unsigned int sstride=s*stride;
        F[2*s]=f+sstride;
        F[2*s+1]=g+sstride;
      }
      
      MPI_Barrier(group.active);
      if(group.rank == 0)
        cout << "Initialized after " << seconds() << " seconds." << endl;
      for(unsigned int i=0; i < N; ++i) {
        init(f,g,d,M,xcompact);
        if(main) seconds();
        C.convolve(F,mult);
        //C.convolve(f,g);
        if(main) T[i]=seconds();
      }
      
      if(main)
        timings("Implicit",mx,T,N);

      if(nx*my < outlimit) 
        show(f,nx,d.y,
             !xcompact,0,
             nx,ycompact || d.y0+d.y < ny ? d.y : d.y-1,group.active);

      // check if the hash of the rounded output matches a known value
      if(dohash && xcompact && ycompact) {
        int hashval=hash(f,mx,d.y,group.active);
        if(group.rank == 0) cout << hashval << endl;
        if(mx == 4 && my == 4) {
          if(hashval != -268659210) {
            retval=1;
            if(group.rank == 0) cout << "error: hash does not match" << endl;
          } else {
            if(group.rank == 0) cout << "hash value OK." << endl;
          }
        }
      }
      delete [] F;
    }
    deleteAlign(g);
    deleteAlign(f);
  
    delete [] T;
  }
  
  MPISaveWisdom(group.active);
  MPI_Finalize();
  
  return retval;
}
