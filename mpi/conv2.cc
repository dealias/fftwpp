#include "mpiconvolution.h"
#include "utils.h"

using namespace std;
using namespace utils;
using namespace fftwpp;
using namespace Array;


inline void init(Complex **F, split d, unsigned int A=2,
                 bool xcompact=true, bool ycompact=true)
{
  unsigned int M=A/2;
  double factor=1.0/sqrt((double) M);
  for(unsigned int s=0; s < M; ++s) {
    double S=sqrt(1.0+s);
    array2<Complex> f(d.X,d.y,F[s]);
    array2<Complex> g(d.X,d.y,F[M+s]);
    double ffactor=S*factor;
    double gfactor=1.0/S*factor;

    if(!xcompact) {
      for(unsigned int j=0; j < d.y; j++) {
        f[0][j]=0.0;
        g[0][j]=0.0;
      }
    }
      
    if(!ycompact && d.y0+d.y == d.Y) { // Last process
      for(unsigned int i=0; i < d.X; ++i) {
        f[i][d.y-1]=0.0;
        g[i][d.y-1]=0.0;
      }
    }
      
    for(unsigned int i=!xcompact; i < d.X; ++i) {
      unsigned int ii=i-!xcompact;
      unsigned int stop=d.y0+d.y < d.Y ? d.y : d.y-!ycompact;
      for(unsigned int j=0; j < stop; j++) {
        unsigned int jj=d.y0+j;
        f[i][j]=ffactor*Complex(ii,jj);
        g[i][j]=gfactor*Complex(2*ii,jj+1);
      }
    }
  }
}

int main(int argc, char* argv[])
{
  // Number of iterations.
  unsigned int N0=1000000;
  unsigned int N=0;
  unsigned int outlimit=200;
#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif
  int retval=0;
  
  int stats = 0;

  unsigned int A=2; // Number of independent inputs
  unsigned int B=1; // Number of outputs
  
  unsigned int mx=4;
  unsigned int my=4;

  bool xcompact=true;
  bool ycompact=true;
  int divisor=0; // Test for best block divisor
  int alltoall=-1; // Test for best alltoall routine

  bool quiet=false;
  bool test=false;
  
  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if(rank != 0) opterr=0;
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"hqtA:B:iH:N:a:m:n:s:x:y:T:S:X:Y:");
    if (c == -1) break;
                
    switch (c) {
      case 0:
        break;
      case 'A':
        A=atoi(optarg);
        break;
      case 'B':
        B=atoi(optarg);
        break;
      case 'a':
        divisor=atoi(optarg);
        break;
      case 'N':
        N=atoi(optarg);
        break;
      case 'm':
        mx=my=atoi(optarg);
        break;
      case 'q':
        quiet=true;
        break;
      case 's':
        alltoall=atoi(optarg);
        break;
      case 't':
        test=true;
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
      case 'S':
        stats=atoi(optarg);
        break;
      case 'X':
        xcompact=atoi(optarg) == 0;
        break;
      case 'Y':
        ycompact=atoi(optarg) == 0;
        break;
      case 'i':
	// Added for compatibility with the OpenMP version.
        break;
      case 'h':
      default:
        if(rank == 0) {
          usage(2);
          usageCompact(2);
          usageTranspose();
        }
        exit(1);
    }
  }
  
  if(my == 0) my=mx;
  
  if(N == 0) {
    N=N0/mx/my;
    if(N < 10) N=10;
  }
  
  unsigned int nx=2*mx-xcompact;
  unsigned int nyp=my+!ycompact;
  
  MPIgroup group(MPI_COMM_WORLD,nyp);
  
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
    split d(nx,nyp,group.active);
    
    // Dimensions used in the MPI convolution
    split du(mx+xcompact,nyp,group.active);
  
    if(B != 1) {
      cerr << "Only B=1 is implemented" << endl;
      exit(1);
    }
    
    Complex **F=new Complex *[A];
    for(unsigned int i=0; i < A; ++i) {
      F[i]=ComplexAlign(d.n);
    }
  
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
      cout << "mx=" << mx << ", my=" << my << endl;
      cout << "nx=" << nx << ", nyp=" << nyp << endl;
    }
    
    ImplicitHConvolution2MPI C(mx,my,xcompact,ycompact,d,du,F[0],
                               mpiOptions(divisor,alltoall),A,B);
    
    if(test) {
      init(F,d,A,xcompact,ycompact);

      if(!quiet) {
        for(unsigned int a=0; a < A; ++a) {
          if(main) 
            cout << "\nDistributed input " << a  << ":"<< endl;
          show(F[a],mx,d.y,group.active);
        }
      }

      Complex **Flocal=new Complex *[A];
      for(unsigned int i=0; i < A; ++i) {
        Flocal[i]=ComplexAlign(nx*nyp);
        gathery(F[i],Flocal[i],d,1,group.active);
        if(!quiet && main)  {
          cout << "\nGathered input " << i << ":" << endl;
          Array2<Complex> AFlocala(mx,my,Flocal[i]);
          cout << AFlocala << endl;
          // FIXME: add error check
        }
      }

      C.convolve(F,mult);

      Complex *Foutgather=ComplexAlign(nx*nyp);
      gathery(F[0],Foutgather,d,1,group.active);

      if(main) {
        ImplicitHConvolution2 Clocal(mx,my,xcompact,ycompact,A,B);
        Clocal.convolve(Flocal,mult);
        if(!quiet) {
          cout << "Local output:" << endl;
          Array2<Complex> AFlocal0(nx,nyp,Flocal[0]);
          cout << AFlocal0 << endl;
        }
        retval += checkerror(Flocal[0],Foutgather,d.X*d.Y);
      }
    } else {
      if(!quiet && main)
        cout << "Initialized after " << seconds() << " seconds." << endl;

      MPI_Barrier(group.active);
      
      double *T=new double[N];
      for(unsigned int i=0; i < N; ++i) {
        init(F,d,A,xcompact,ycompact);
        if(main) seconds();
        C.convolve(F,mult);
        //C.convolve(f,g);
        if(main) T[i]=seconds();
      }
      if(main)
        timings("Implicit",mx,T,N,stats);
      delete [] T;
    
      if(!quiet && nx*my < outlimit)
        show(F[0],d.X,d.y,group.active);
    }
    
    for(unsigned int i=0; i < A; ++i)
      deleteAlign(F[i]);
    delete [] F;
  }
  
  MPI_Finalize();
  
  return retval;
}
