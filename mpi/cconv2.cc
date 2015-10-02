#include "mpiconvolution.h"
#include "utils.h"
#include "mpiutils.h"
#include "Array.h"

using namespace std;
using namespace fftwpp;
using namespace Array;

// Number of iterations.
unsigned int N0=1000000;
unsigned int N=0;
unsigned int mx=4;
unsigned int my=4;

bool Implicit=true, Explicit=false, Pruned=false;

inline void init(Complex **F, split d, unsigned int A) 
{
  if(A%2 == 0) {
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
  } else {
    cerr << "Init not implemented for A=" << A << endl;
    exit(1);
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
  unsigned int A=2;
  
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"heipqHtM:N:m:x:y:n:T:");
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
      case 't':
        test=true;
        break;
      case 'q':
        quiet=true;
        break;
      case 'h':
      default:
        usage(2);
    }
  }

  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);

  if(my == 0) my=mx;

  if(N == 0) {
    N=N0/mx/my;
    if(N < 10) N=10;
  }
  
  MPIgroup group(MPI_COMM_WORLD,my);

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
    } 

    split d(mx,my,group.active);
  
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
      default: cout << "A=" << A << " is not yet implemented" << endl;
        exit(1);
    }

    ImplicitConvolution2MPI C(mx,my,d,A);

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
	gathery(F[a], Flocal[a], d, 1, group.active);
	if(!quiet && main)  {
	  cout << "\nGathered input " << a << ":" << endl;
	  Array2<Complex> AFlocala(mx,my,Flocal[a]);
	  cout << AFlocala << endl;
	}
      }
      
      C.convolve(F,mult);

      Complex *Foutgather=ComplexAlign(mx*my);
      gathery(F[0], Foutgather, d, 1, group.active);

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
	double maxerr = relmaxerror(Flocal[0],Foutgather,d.X,d.Y);
	cout << "maxerr: " << maxerr << endl;
	if(maxerr > 1e-10) {
	  retval += 1;
	  cout << "CAUTION! Large error!" << endl;
	} else {
	  cout << "Error ok." << endl;
	}
      }

      deleteAlign(Foutgather);
      for(unsigned int a=0; a < A; ++a)
	deleteAlign(Flocal[a]);
      delete [] Flocal;
      
      MPI_Barrier(group.active);

    } else {
      if(group.rank == 0)
	cout << "Initialized after " << seconds() << " seconds." << endl;

      double *T=new double[N];

      for(unsigned int i=0; i < N; ++i) {
	init(F,d,A);
	if(main) seconds();
	C.convolve(F,mult);
	//      C.convolve(f,g);
	if(main) T[i]=seconds();
      }
    
      if(main) 
	timings("Implicit",mx,T,N);
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

