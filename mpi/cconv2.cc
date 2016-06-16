#include "mpiconvolution.h"
#include "utils.h"

using namespace std;
using namespace utils;
using namespace fftwpp;
using namespace Array;

// Number of iterations.
unsigned int N0=1000000;
unsigned int N=0;
unsigned int mx=4;
unsigned int my=4;
int divisor=0; // Test for best block divisor
int alltoall=-1; // Test for best alltoall routine

inline void init(Complex **F, split d, unsigned int A) 
{
  unsigned int M=A/2;
  double factor=1.0/sqrt((double) M);
  for(unsigned int s=0; s < M; ++s) {
    array2<Complex> f(d.X,d.y,F[s]);
    array2<Complex> g(d.X,d.y,F[M+s]);
    double S=sqrt(1.0+s);
    double ffactor=S*factor;
    double gfactor=1.0/S*factor;
    for(unsigned int i=0; i < d.X; ++i) {
      //unsigned int I=s*d.n+d.y*i;
      for(unsigned int j=0; j < d.y; j++) {
        unsigned int jj=d.y0+j;
        f[i][j]=ffactor*Complex(i,jj);
        g[i][j]=gfactor*Complex(2*i,jj+1);
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
  int retval=0;
  bool test=false;
  bool quiet=false;
  
  unsigned int A=2; // Number of independent inputs
  unsigned int B=1; // Number of outputs

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
    int c = getopt(argc,argv,"hqta:A:B:N:m:s:x:y:n:T:S:i");
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
        mx=my=atoi(optarg);
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
      case 'n':
        N0=atoi(optarg);
        break;
      case 'T':
        fftw::maxthreads=atoi(optarg);
        break;
      case 'S':
        stats=atoi(optarg);
        break;
      case 'i':
	// For compatibility reasons with -i option in OpenMP version.
	break;
      case 't':
        test=true;
        break;
      case 'q':
        quiet=true;
        break;
      case 'h':
      default:
        if(rank == 0) {
          usage(2);
          usageTranspose();
        }
        exit(1);
    }
  }

  if(my == 0) my=mx;

  if(N == 0) {
    N=N0/mx/my;
    if(N < 20) N=20;
  }
  
  MPIgroup group(MPI_COMM_WORLD,my);

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

    split d(mx,my,group.active);
  
    if(B != 1) {
      cerr << "Only B=1 is implemented" << endl;
      exit(1);
    }
    
    Complex **F=new Complex *[A];
    for(unsigned int a=0; a < A; ++a) {
      F[a]=ComplexAlign(d.n);
    }

    multiplier *mult;
  
    switch(A) {
      case 2: mult=multbinary; break;
      case 4: mult=multbinary2; break;
      case 6: mult=multbinary3; break;
      case 8: mult=multbinary4; break;
      case 16: mult=multbinary8; break;
      default: if(main) cout << "A=" << A << " is not yet implemented" << endl;
        exit(1);
    }

    if(!quiet && main) {
      if(!test)
        cout << "N=" << N << endl;
      cout << "A=" << A << endl;
      cout << "mx=" << mx << ", my=" << my << endl;
    }
    
    ImplicitConvolution2MPI C(mx,my,d,mpiOptions(divisor,alltoall),A,B);

    if(test) {
      init(F,d,A);

      if(!quiet) {
        for(unsigned int a=0; a < A; ++a) {
          if(main) 
            cout << "\nDistributed input " << a  << ":"<< endl;
          show(F[a],mx,d.y,group.active);
        }
      }

      Complex **Flocal=new Complex *[A];
      for(unsigned int a=0; a < A; ++a) {
        Flocal[a]=ComplexAlign(mx*my);
        gathery(F[a],Flocal[a],d,1,group.active);
        if(!quiet && main)  {
          cout << "\nGathered input " << a << ":" << endl;
          Array2<Complex> AFlocala(mx,my,Flocal[a]);
          cout << AFlocala << endl;
          // FIXME: add error check
        }
      }
      
      C.convolve(F,mult);

      Complex *Foutgather=ComplexAlign(mx*my);
      gathery(F[0],Foutgather,d,1,group.active);

      if(!quiet) {
        if(main)
          cout << "Distributed output:" << endl;
        show(F[0],mx,d.y,group.active);
      }
      
      if(main) {
        ImplicitConvolution2 Clocal(mx,my,A,1);
        Clocal.convolve(Flocal,mult);
        if(!quiet) {
          cout << "Local output:" << endl;
          Array2<Complex> AFlocal0(mx,my,Flocal[0]);
          cout << AFlocal0 << endl;
        }
        retval += checkerror(Flocal[0],Foutgather,d.X*d.Y);
      }

      deleteAlign(Foutgather);
      for(unsigned int a=0; a < A; ++a)
        deleteAlign(Flocal[a]);
      delete [] Flocal;
      
      MPI_Barrier(group.active);

    } else {
      if(!quiet && main)
        cout << "Initialized after " << seconds() << " seconds." << endl;

      MPI_Barrier(group.active);
      
      double *T=new double[N];
      for(unsigned int i=0; i < N; ++i) {
        init(F,d,A);
        if(main) seconds();
        C.convolve(F,mult);
        //      C.convolve(f,g);
        if(main) T[i]=seconds();
      }
    
      if(main) 
        timings("Implicit",mx,T,N,stats);
      delete [] T;
    }   

    if(!quiet && mx*my < outlimit)
      show(F[0],mx,d.y,group.active);

    for(unsigned int a=0; a < A; ++a)
      deleteAlign(F[a]);
    delete [] F;
  

  
  }

  MPI_Finalize();
  
  return retval;
}
