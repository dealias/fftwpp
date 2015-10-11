#include "Array.h"
#include "mpifftw++.h"
#include "utils.h"
#include "mpiutils.h"

using namespace std;
using namespace fftwpp;
using namespace Array;

inline void init(double *f, split d) 
{
  unsigned int c=0;
  for(unsigned int i=0; i < d.x; ++i) {
    unsigned int ii=d.x0+i;
    for(unsigned int j=0; j < d.Y; j++) {
      f[c++]=j+ii;
    }
  }
}

unsigned int outlimit=100;

int main(int argc, char* argv[])
{
  int retval = 0; // success!

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  // Number of iterations.
  unsigned int N0=10000000;
  unsigned int N=0;
  unsigned int nx=4;
  unsigned int ny=4;
  int divisor=0; // Test for best block divisor
  int alltoall=-1; // Test for best alltoall routine
  
  bool quiet=false;
  bool test=false;

  bool shift=false;
  
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"hN:a:m:s:x:y:n:T:S:qt");
    if (c == -1) break;
                
    switch (c) {
    case 0:
      break;
    case 'N':
      N=atoi(optarg);
      break;
    case 'a':
      divisor=atoi(optarg);
      break;
    case 'm':
      nx=ny=atoi(optarg);
      break;
    case 's':
      alltoall=atoi(optarg);
      break;
    case 'x':
      nx=atoi(optarg);
      break;
    case 'y':
      ny=atoi(optarg);
      break;
    case 'n':
      N0=atoi(optarg);
      break;
    case 'S':
      shift=atoi(optarg);
      break;
    case 'T':
      fftw::maxthreads=atoi(optarg);
      break;
    case 'q':
      quiet=true;
      break;
    case 't':
      test=true;
      break;
    case 'h':
      usage(2); // TODO: Add shift, compact flags
      usageTranspose();
      exit(1);
    }
  }

  cout << "shift: " << shift << endl;
  
  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);

  if(ny == 0) ny=nx;

  if(!N == 0) {
    N=N0/nx/ny;
    if(N < 10) N=10;
  }

  MPIgroup group(MPI_COMM_WORLD,ny);

  if(group.size > 1 && provided < MPI_THREAD_FUNNELED)
    fftw::maxthreads=1;

  if(!quiet && group.rank == 0) {
    cout << "provided: " << provided << endl;
    cout << "fftw::maxthreads: " << fftw::maxthreads << endl;
  }
  
  if(!quiet && group.rank == 0) {
    cout << "Configuration: " 
	 << group.size << " nodes X " << fftw::maxthreads 
	 << " threads/node" << endl;
  }

  if(group.rank < group.size) { 
    bool main=group.rank == 0;
    if(!quiet && main) {
      cout << "N=" << N << endl;
      cout << "nx=" << nx << ", ny=" << ny << endl;
    } 
    unsigned int nyp=ny/2+1;
    
    split df(nx,ny,group.active);
    split dg(nx,nyp,group.active);
  
    double *f=doubleAlign(df.n);
    Complex *g=ComplexAlign(dg.n);

    // Create instance of FFT
    rcfft2dMPI rcfft(df,dg,f,g,mpiOptions(fftw::maxthreads,divisor,alltoall));

    if(!quiet && group.rank == 0)
      cout << "Initialized after " << seconds() << " seconds." << endl;    

    if(test) {
      init(f,df);

      if(!quiet && nx*ny < outlimit) {
	if(main) cout << "\nDistributed input:" << endl;
	show(f,df.x,ny,group.active);
      }
      
      size_t align=sizeof(Complex);
      array2<double> flocal(nx,ny,align);
      array2<Complex> glocal(nx,nyp,align);
      rcfft2d localForward(nx, ny, flocal(), glocal());
      crfft2d localBackward(nx, ny, glocal(), flocal());

      gatherx(f, flocal(), df, 1, group.active);

      if(!quiet && main) {
	cout << endl << "Gathered input:\n" << flocal << endl;
      }

      if(shift)
	rcfft.Forwards0(f,g);
      else
	rcfft.Forwards(f,g);
      
      if(!quiet && nx*ny < outlimit) {
      	if(main) cout << "\nDistributed output:" << endl;
      	show(g,dg.X,dg.y,group.active);
      }

      array2<Complex> ggather(nx,nyp,align);
      gathery(g, ggather(), dg, 1, group.active);

      MPI_Barrier(group.active);
      if(main) {
	if(shift)
	  localForward.fft0(flocal,glocal);
	else
	  localForward.fft(flocal,glocal);
	if(!quiet) {
	  cout << "\nLocal output:\n" << glocal << endl;
	  cout << "\nGathered output:\n" << ggather << endl;
	}
        retval += checkerror(glocal(),ggather(),dg.X*dg.Y);
      }

      if(shift)
	rcfft.Backwards0Normalized(g,f);
      else
	rcfft.BackwardsNormalized(g,f);

      if(!quiet && nx*ny < outlimit) {
      	if(main) cout << "\nDistributed back to input:" << endl;
      	show(f,df.x,ny,group.active);
      }

      array2<double> flocal0(nx,ny,align);
      gatherx(f, flocal0(), df, 1, group.active);
      MPI_Barrier(group.active);
      if(main) {
	if(shift)
	  localBackward.fft0Normalized(glocal,flocal);
	else
	  localBackward.fftNormalized(glocal,flocal);
      	if(!quiet) {
      	  cout << "\nLocal output:\n" << flocal << endl;
      	  cout << "\nGathered output:\n" << flocal0 << endl;
      	}
	retval += checkerror(flocal0(),flocal(),df.X*df.Y);
      }

      if(!quiet && group.rank == 0) {
        cout << endl;
        if(retval == 0)
          cout << "pass" << endl;
        else
          cout << "FAIL" << endl;
      }  
  
    } else {
       if(N > 0) {
       	double *T=new double[N];
       	for(unsigned int i=0; i < N; ++i) {
       	  init(f,df);
       	  seconds();
       	  rcfft.Forwards(f,g);
       	  rcfft.BackwardsNormalized(g,f);
       	  T[i]=seconds();
       	}    
       	if(main) timings("FFT timing:",nx,T,N);
      	delete [] T;
        
        if(!quiet && nx*ny < outlimit)
          show(f,df.x,ny,group.active);
       }
    }

    deleteAlign(f);
  }
  
  MPI_Finalize();

  return retval;
}
