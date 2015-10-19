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

inline void init(Complex **F, const split3& d, unsigned int A=2,
                 bool xcompact=true, bool ycompact=true, bool zcompact=true)
{
  if(A % 2 != 0) {
    cout << "A=" << A << " is not yet implemented" << endl;
    exit(1);
  }

  unsigned int M=A/2;
  double factor=1.0/sqrt((double) M);
  for(unsigned int s=0; s < M; ++s) {
    Complex *f=F[s];
    Complex *g=F[s+M];

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
      for(unsigned int i=0; i < d.X; ++i) {
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
      unsigned int jstart=!ycompact && d.y0 == 0;
      for(unsigned int j=jstart; j < d.y; ++j) {
        unsigned int IJ=I+d.z*j;
        unsigned int jj=d.y0+j-!ycompact;
        unsigned int stop=(d.z0+d.z < d.Z) ? d.z : d.z-!zcompact;
        for(unsigned int k=0; k < stop; ++k) {
          unsigned int kk=d.z0+k;
          f[IJ+k]=ffactor*Complex(ii+kk,jj+kk);
          g[IJ+k]=gfactor*Complex(2*ii+kk,jj+1+kk);
        }
      }
    }
  }
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

int main(int argc, char* argv[])
{
#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  
  int retval=0;
  bool test=false;
  bool quiet=false;

  unsigned int mx=4;
  unsigned int my=4;
  unsigned int mz=4;
  unsigned int A=2;
  unsigned int B=1;
  bool xcompact=true;
  bool ycompact=true;
  bool zcompact=true;
  int divisor=0; // Test for best block divisor
  int alltoall=-1; // Test for best alltoall routine

#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"heitqA:M:N:a:m:s:x:y:z:n:T:X:Y:Z:");
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
      case 'A':
        A=atoi(optarg);
        break;
      case 'M':
        A=2*atoi(optarg);
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

  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);

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

    // Dimensions of the input data
    split3 df(nx,ny,mz+!zcompact,group,true);
    
    Complex **F=new Complex*[A];
    for(unsigned int a=0; a < A; a++)
      F[a]=ComplexAlign(df.n);
    init(F,df,A,xcompact,ycompact,zcompact);
    
    realmultiplier *mult;
    switch(A) {
      case 2: mult=multbinary; break;
      case 4: mult=multbinary2; break;
      default: cout << "A=" << A << " is not yet implemented" << endl; exit(1);
    }

    // Dimensions used in the MPI convolution
    split3 d(nx,ny,nzp,group,true);
    split3 du(mx+xcompact,ny,my+ycompact,nzp,group,true);
    ImplicitHConvolution3MPI C(mx,my,mz,xcompact,ycompact,zcompact,d,du,F[0],
                               mpiOptions(divisor,alltoall),A,B);
    
    MPI_Barrier(group.active);
    if(!quiet && main)
      cout << "Initialized after " << seconds() << " seconds." << endl;

    if(test) {
      if(!quiet) {
        if(main) cout << "Distributed input:" << endl;
        for(unsigned int a=0; a < A; a++) {
          if(main) cout << "a: " << a << endl;
          show(F[a],df.X,df.y,df.z,group.active);
        }
      }
      
      if(!quiet && main) cout << "Gathered input:" << endl;
      Complex **F0=new Complex*[A];
      for(unsigned int a=0; a < A; a++) {
        if(main) {
          F0[a]=ComplexAlign(d.X*d.Y*d.Z);
        }
        gatheryz(F[a],F0[a],d,group.active);
        if(!quiet && main) {
          cout << "a: " << a << endl;
	  show(F0[a],d.X,d.Y,d.Z);
        }
      }


      C.convolve(F,mult);

      if(!quiet && nx*ny*mz < outlimit) {
	if(main) cout << "Distributed output: " << endl;
	show(F[0],df.X,df.y,df.z,group.active);
      }
      
      Complex **F0out=new Complex*[B];
      if(main) {
	for(unsigned int b=0; b < B; b++)
          F0out[b]=ComplexAlign(d.X*d.Y*d.Z);
      }
      for(unsigned int b=0; b < B; b++) {
	gatheryz(F[b],F0out[b],d,group.active);
	if(!quiet && main) {
	  cout << "Gathered output:" << endl;
	  cout << "b: " << b << endl;
	  show(F0out[b],d.X,d.Y,d.Z);
	}
      }

      if(main) {
	ImplicitHConvolution3 Clocal(mx,my,mz,xcompact,ycompact,zcompact,A,B);
	Clocal.convolve(F0,mult);
	if(!quiet && main) {
	  cout << "Local output:" << endl;
	  for(unsigned int b=0; b < B; b++) {
	    cout << "b: " << b << endl;
	    show(F0[b],d.X,d.Y,d.Z);
	    retval += checkerror(F0[b],F0out[b],d.X*d.Y*d.Z);
	  }
	}

      }
      
      if(main) {
	for(unsigned int a=0; a < A; a++)
	  deleteAlign(F0[a]);
	for(unsigned int b=0; b < B; b++)
	  deleteAlign(F0out[b]);
      }
      delete[] F0;
      delete[] F0out;
      
    } else {

      N=N0/mx/my/mz;
      if(N < 10) N=10;

      double *T=new double[N];
      for(unsigned int i=0; i < N; ++i) {
        init(F,d,A,xcompact,ycompact,zcompact);
        seconds();
        C.convolve(F,mult);
        T[i]=seconds();
      }
      if(main) 
        timings("Implicit",mx,T,N);
      delete [] T;
      
      if(!quiet && nx*ny*mz < outlimit) {
	if(main) cout << "output: " << endl;
	show(F[0],df.X,df.y,df.z,group.active);
      }

    }
      
    for(unsigned int a=0; a < A; a++)
      deleteAlign(F[a]);
    delete[] F;

    
  }

  MPI_Finalize();
  
  return retval;
}
