#include "mpiconvolution.h"
#include "utils.h"

using namespace std;
using namespace utils;
using namespace fftwpp;


void init(Complex **F,
          unsigned int X, unsigned int Y, unsigned int Z,
          unsigned int x0, unsigned int y0, unsigned int z0,
          unsigned int x, unsigned int y, unsigned int z,
          unsigned int A=2)
{
  unsigned int M=A/2;
  double factor=1.0/sqrt((double) M);
  for(unsigned int s=0; s < M; ++s) {
    Complex *Fs=F[s];
    Complex *Gs=F[s+M];
    double S=sqrt(1.0+s);
    double ffactor=S*factor;
    double gfactor=1.0/S*factor;
    for(unsigned int i=0; i < X; ++i) {
      for(unsigned int j=0; j < y; j++) {
        unsigned int jj=y0+j;
        for(unsigned int k=0; k < z; k++) {
          unsigned int kk=z0+k;
          unsigned int pos=i*y*z+j*z+k;
          Fs[pos]=ffactor*Complex(i+kk,jj+kk);
          Gs[pos]=gfactor*Complex(2*i+kk,jj+1+kk);
        }
      }
    }
  }
}

void init(Complex **F,split3 &d, unsigned int A=2)
{
  init(F,
       d.X,d.Y,d.Z,
       d.x0,d.y0,d.z0,
       d.x,d.y,d.z,A);
}

void show(Complex *F, unsigned int X, unsigned int Y, unsigned int Z)
{
  for(unsigned int i=0; i < X; ++i) {
    for(unsigned int j=0; j < Y; ++j) {
      for(unsigned int k=0; k < Z; ++k) {
        unsigned int pos=i*Y*Z+j*Z+k;
        cout << F[pos] << "\t";
      }
      cout << endl;
    }
    cout << endl;
  }
}

unsigned int outlimit=3000;

int main(int argc, char* argv[])
{

  // Number of iterations.
  unsigned int N0=1000000;
  unsigned int N=0;
  int divisor=0; // Test for best block divisor
  int alltoall=-1; // Test for best alltoall routine

  int stats=0;
  
  unsigned int mx=4;
  unsigned int my=0;
  unsigned int mz=0;
  
  unsigned int A=2; // Number of independent inputs
  unsigned int B=1; // Number of outputs
    
#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  
  int retval=0;
  bool test=false;
  bool quiet=false;
  
  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if(rank != 0) opterr=0;
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"ihtqa:A:B:N:T:S:m:n:s:x:y:z:");
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
      case 'i':
	// For compatibility reasons with -i option in OpenMP version.
	break;       
      case 'h':
      default:
        if(rank == 0) {
          usage(3);
          usageTranspose();
        }
        exit(1);
    }
  }
  
  if(my == 0) my=mx;
  if(mz == 0) mz=mx;
  
  if(N == 0) {
    N=N0/mx/my/mz;
    if(N < 20) N=20;
  }

  int size;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  unsigned int x=ceilquotient(mx,size);
  unsigned int y=ceilquotient(my,size);
  bool allowpencil=mx*y == x*my;
  
  MPIgroup group(MPI_COMM_WORLD,my,mz,allowpencil);
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
      cout << "mx=" << mx << ", my=" << my << ", mz=" << mz << endl;
    }
    split3 d(mx,my,mz,group,true);
    
    if(B != 1) {
      cerr << "Only B=1 is implemented" << endl;
      exit(1);
    }
    
    Complex **F=new Complex*[A];
    for(unsigned int a=0; a < A; a++) {
      F[a]=ComplexAlign(d.n);
    }

    ImplicitConvolution3MPI C(mx,my,mz,d,mpiOptions(divisor,alltoall),A,B);

    if(test) {
      init(F,d,A);
      if(!quiet) {
        if(main) cout << "Distributed input:" << endl;
        for(unsigned int a=0; a < A; a++) {
          if(main) cout << "a: " << a << endl;
          show(F[a],mx,d.y,d.z,group.active);
        }
      }

      if(!quiet && main) cout << "Gathered input:" << endl;
      Complex **F0=new Complex*[A];
      for(unsigned int a=0; a < A; a++) {
        if(main) {
          F0[a]=ComplexAlign(mx*my*mz);
        }
        gatheryz(F[a],F0[a],d,group.active);
        if(!quiet && main) {
          cout << "a: " << a << endl;
          show(F0[a],mx,my,mz);
        }
      }

      C.convolve(F,mult);

      if(!quiet) {
        if(main) cout << "Distributed output:" << endl;
        show(F[0],mx,d.y,d.z,group.active);
      }
      
      Complex *FC0=ComplexAlign(mx*my*mz);
      gatheryz(F[0],FC0,d,group.active);
      if(!quiet && main) {
        cout << "Gathered output:" << endl;
        show(FC0,mx,my,mz);
      }

      if(main) {
        unsigned int B=1; // TODO: generalize
        ImplicitConvolution3 C(mx,my,mz,A,B); 
        C.convolve(F0,mult);
        if(!quiet) {
          cout << "Local output:" << endl;
          show(F0[0],mx,my,mz);
        }
      }

      if(main)
        retval += checkerror(F0[0],FC0,d.X*d.Y*d.Z);

      if(main) {
        deleteAlign(FC0);
        for(unsigned int a=0; a < A; a++)
          deleteAlign(F0[a]);
        delete[] F0;
      }
    } else {
      if(!quiet && main)
        cout << "Initialized after " << seconds() << " seconds." << endl;

      MPI_Barrier(group.active);
     
      double *T=new double[N];
      for(unsigned int i=0; i < N; i++) {
        init(F,d,A);
        MPI_Barrier(group.active);
        seconds();
        C.convolve(F,mult);
        // C.convolve(f,g);
        T[i]=seconds();
        MPI_Barrier(group.active);
      }
      if(main) 
        timings("Implicit",mx,T,N,stats);
    
      delete[] T;
      if(!quiet && mx*my*mz < outlimit) 
        show(F[0],mx,d.y,d.z,group.active);
    }
  
    for(unsigned int a=0; a < A; a++)
      deleteAlign(F[a]);
    delete[] F;
  }

  MPI_Finalize();
  
  return retval;
}
