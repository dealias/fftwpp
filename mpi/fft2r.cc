#include "Array.h"
#include "mpifftw++.h"
#include "utils.h"

using namespace std;
using namespace utils;
using namespace fftwpp;
using namespace Array;

inline void init(array2<double> f, split d)
{
  for(unsigned int i=0; i < d.x; ++i) {
    unsigned int ii=d.x0+i;
    for(unsigned int j=0; j < d.Y; j++) {
      f(i,j)=j+ii;
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

  bool inplace=false;
  
  bool quiet=false;
  bool test=false;
  bool shift=false;
  unsigned int stats=0; // Type of statistics used in timing test.
  
  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if(rank != 0) opterr=0;
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"hN:a:i:m:s:x:y:n:T:S:O:qt");
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
      case 'i':
        inplace=atoi(optarg);
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
      case 'O':
        shift=atoi(optarg);
        break;
      case 'S':
        stats=atoi(optarg);
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
      default:
        if(rank == 0) {
          usageInplace(2);
          usageTranspose();
          usageShift();
        }
        exit(1);
    }
  }

  if(ny == 0) ny=nx;

  if(N == 0) {
    N=N0/nx/ny;
    if(N < 10) N=10;
  }

  MPIgroup group(MPI_COMM_WORLD,ny);

  if(group.size > 1 && provided < MPI_THREAD_FUNNELED)
    fftw::maxthreads=1;

  defaultmpithreads=fftw::maxthreads;
    
  if(group.rank < group.size) { 
    bool main=group.rank == 0;
    if(!quiet && main) {
      cout << "Configuration: " 
           << group.size << " nodes X " << fftw::maxthreads
           << " threads/node" << endl;
      cout << "Using MPI VERSION " << MPI_VERSION << endl;
      cout << "N=" << N << endl;
      cout << "nx=" << nx << ", ny=" << ny << endl;
    } 
    unsigned int nyp=ny/2+1;
    
    split df(nx,ny,group.active);
    split dg(nx,nyp,group.active);
  
    unsigned int dfY=inplace ? 2*dg.Y : df.Y;
      
    array2<Complex> g(dg.x,dg.Y,ComplexAlign(dg.n));
    array2<double> f;
    if(inplace)
      f.Dimension(df.x,2*dg.Y,(double *) g());
    else
      f.Dimension(df.x,df.Y,doubleAlign(df.n));
  
    // Create instance of FFT
    rcfft2dMPI rcfft(df,dg,f,g,mpiOptions(divisor,alltoall));

    if(!quiet && group.rank == 0)
      cout << "Initialized after " << seconds() << " seconds." << endl;    

    if(test) {
      init(f,df);

      if(!quiet && nx*ny < outlimit) {
        if(main) cout << "\nDistributed input:" << endl;
        show(f(),df.x,df.Y,group.active);
      }
      
      split dfgather(nx,dfY,group.active);
      size_t align=sizeof(Complex);
      array2<Complex> ggather(nx,nyp,align);
      array2<Complex> glocal(nx,nyp,align);
      array2<double> fgather(nx,dfY,align);
      array2<double> flocal;
      if(inplace)
        flocal.Dimension(nx,2*nyp,(double *) glocal());
      else
        flocal.Allocate(nx,ny,align);
  
      rcfft2d localForward(nx,ny,flocal,glocal);
      crfft2d localBackward(nx,ny,glocal,flocal);

      gatherx(f(),flocal(),dfgather,1,group.active);
      gatherx(f(),fgather(),dfgather,1,group.active);
      if(!quiet && main)
        cout << endl << "Gathered input:\n" << fgather << endl;
      
      if(shift)
        rcfft.Forward0(f,g);
      else
        rcfft.Forward(f,g);      
      
      if(!quiet && nx*ny < outlimit) {
        if(main) cout << "\nDistributed output:" << endl;
        show(g(),dg.X,dg.y,group.active);
      }

      gathery(g(),ggather(),dg,1,group.active);
      if(main && !quiet)
        cout << "\nGathered output:\n" << ggather << endl;
      
      if(main) {
        if(shift)
          localForward.fft0(flocal,glocal);
        else
          localForward.fft(flocal,glocal);
        if(!quiet)
          cout << "\nLocal output:\n" << glocal << endl;
        retval += checkerror(glocal(),ggather(),dg.X*dg.Y);
      }

      if(shift)
        rcfft.Backward0(g,f);
      else
        rcfft.Backward(g,f);
      rcfft.Normalize(f);

      if(!quiet && nx*ny < outlimit) {
        if(main) cout << "\nDistributed back to input:" << endl;
        show(f(),dfgather.x,dfgather.Y,group.active);
      }

      gatherx(f(),fgather(),dfgather,1,group.active);
      if(!quiet && main)
        cout << endl << "Gathered back to input:\n" << fgather << endl;
      
      if(main) {
        if(shift)
          localBackward.fft0Normalized(glocal,flocal);
        else
          localBackward.fftNormalized(glocal,flocal);
        if(!quiet)
          cout << "\nLocal back to input:\n" << flocal << endl;
        cout << "df.X: " << df.X << endl;
        cout << "df.Y: " << df.Y << endl;
        cout << "dfgather.Y: " << dfgather.Y << endl;
        retval += checkerror(fgather(),flocal(),df.Y,df.X,dfgather.Y);
      }
      
      if(!quiet && group.rank == 0) {
        cout << endl;
        if(retval == 0)
          cout << "pass" << endl;
        else
          cout << "FAIL" << endl;
      }  
  
    } else {
      double *T=new double[N];
      init(f,df);
      for(unsigned int i=0; i < N; ++i) {
        if(shift) {
          init(f,df);
          seconds();
          rcfft.Forward0(f,g);
          rcfft.Backward0(g,f);
          T[i]=0.5*seconds();
          rcfft.Normalize(f);
        } else {
          seconds();
          rcfft.Forward(f,g);
          rcfft.Backward(g,f);
          T[i]=0.5*seconds();
          rcfft.Normalize(f);
        }
      }    
      if(main)
	timings("FFT timing:",nx,T,N,stats);
      delete [] T;
        
      if(!quiet && nx*ny < outlimit)
        show(f(),df.x,dfY,0,0,df.x,df.Y,group.active);
    }
  
    deleteAlign(g());
    if(!inplace)
      deleteAlign(f());
  }
  
  MPI_Finalize();

  return retval;
}
