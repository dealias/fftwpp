#include "Array.h"
#include "mpifftw++.h"
#include "utils.h"

using namespace std;
using namespace utils;
using namespace fftwpp;
using namespace Array;

inline void init(array3<double> f, split3 d)
{
  for(unsigned int i=0; i < d.x; ++i) {
    unsigned int ii=d.x0+i;
    for(unsigned int j=0; j < d.y; j++) {
      unsigned int jj=d.y0+j;
      for(unsigned int k=0; k < d.Z; k++) {
        f(i,j,k)=ii+jj+k +1;
      }
    }
  }
}

unsigned int outlimit=3000;

int main(int argc, char* argv[])
{

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif
  int retval=0;

  // Default number of iterations.
  unsigned int N0=10000000;
  unsigned int N=0;
  int divisor=0; // Test for best block divisor
  int alltoall=-1; // Test for best alltoall routine
  
  unsigned int nx=4;
  unsigned int ny=0;
  unsigned int nz=0;

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
    int c = getopt(argc,argv,"S:hti:N:O:T:a:i:m:n:s:x:y:z:q");
    if (c == -1) break;
                
    switch (c) {
      case 0:
        break;
      case 'a':
        divisor=atoi(optarg);
        break;
      case 'i':
        inplace=atoi(optarg);
        break;
      case 'N':
        N=atoi(optarg);
        break;
      case 'm':
        nx=ny=nz=atoi(optarg);
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
      case 'z':
        nz=atoi(optarg);
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
          usage(3);
          usageTranspose();
          usageShift();
        }
        exit(1);
    }
  }

  if(ny == 0) ny=nx;
  if(nz == 0) nz=nx;

  if(N == 0) {
    N=N0/nx/ny/nz;
    if(N < 10) N=10;
  }
  
  unsigned int nzp=nz/2+1;
  MPIgroup group(MPI_COMM_WORLD,nx,ny);

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
      cout << "nx=" << nx << ", ny=" << ny << ", nz=" << nz << ", nzp=" << nzp
           << endl;
    }

    split3 df(nx,ny,nz,group);
    split3 dg(nx,ny,nzp,group,true);

    unsigned int dfZ=inplace ? 2*dg.Z : df.Z;

    split3 dfgather(nx,ny,dfZ,group);
        
    array3<Complex> g(dg.x,dg.y,dg.Z,ComplexAlign(dg.n));
    array3<double> f;
    if(inplace)
      f.Dimension(df.x,df.y,2*dg.Z,(double *) g());
    else
      f.Dimension(df.x,df.y,df.Z,doubleAlign(df.n));
  
    rcfft3dMPI rcfft(df,dg,f,g,mpiOptions(divisor,alltoall));

    if(!quiet && group.rank == 0)
      cout << "Initialized after " << seconds() << " seconds." << endl;    
    
    if(test) {
      init(f,df);
      
      if(!quiet && nx*ny < outlimit) {
        if(main) cout << "\nDistributed input:" << endl;
        show(f(),dfgather.x,dfgather.y,dfgather.Z,group.active);
      }

      size_t align=sizeof(Complex);

      array3<Complex> ggather(nx,ny,nzp,align);
      array3<Complex> glocal(nx,ny,nzp,align);
      array3<double> fgather(dfgather.X,dfgather.Y,dfgather.Z,align);
      array3<double> flocal;
      if(inplace)
        flocal.Dimension(nx,ny,2*nzp,(double *) glocal());
      else
        flocal.Allocate(nx,ny,nz,align);
      
      rcfft3d localForward(nx,ny,nz,flocal,glocal);
      crfft3d localBackward(nx,ny,nz,glocal,flocal);

      gatherxy(f(), flocal(), dfgather, group.active);
      gatherxy(f(), fgather(), dfgather, group.active);
      if(main && !quiet)
        cout << "Gathered input:\n" <<  fgather << endl;
              
      if(shift)
        rcfft.Forward0(f,g);
      else
        rcfft.Forward(f,g);
      
      if(main) {
        if(shift)
          localForward.fft0(flocal,glocal);
        else
          localForward.fft(flocal,glocal);
        cout << endl;
      }
        
      if(!quiet) {
        if(main)
	  cout << "Distributed output:" << endl;
	if(!quiet && nx*ny < outlimit)
	  show(g(),dg.X,dg.y,dg.z,group.active);
      }

      gatheryz(g(),ggather(),dg,group.active); 
      if(!quiet && main)
        cout << "Gathered output:\n" <<  ggather << endl;

      if(!quiet && main) 
        cout << "Local output:\n" <<  glocal << endl;
      
      if(main)
        retval += checkerror(glocal(),ggather(),dg.X*dg.Y*dg.Z);

      if(shift)
        rcfft.Backward0(g,f);
      else
        rcfft.Backward(g,f);
      rcfft.Normalize(f);

      if(!quiet) {
        if(main)
	  cout << "Distributed back to input:" << endl;
	if(!quiet && nx*ny < outlimit) 
	  show(f(),dfgather.x,dfgather.y,dfgather.Z,group.active);
      }

      gatherxy(f(),fgather(),dfgather,group.active);
      if(!quiet && main)
        cout << "Gathered back to input:\n" <<  fgather << endl;

      if(main) {
        if(shift)
          localBackward.fft0Normalized(glocal,flocal);
        else 
          localBackward.fftNormalized(glocal,flocal);
      }
      
      if(!quiet && main) 
        cout << "Local back to input:\n" <<  flocal << endl;
      
      if(main)
        retval += checkerror(flocal(),fgather(),df.Z,df.X*df.Y,dfgather.Z);
      
      if(!quiet && group.rank == 0) {
        cout << endl;
        if(retval == 0)
          cout << "pass" << endl;
        else
          cout << "FAIL" << endl;
      }

    } else {
      if(main) cout << "N=" << N << endl;
      double *T=new double[N];

      for(unsigned int i=0; i < N; ++i) {
        init(f,df);
        if(shift) {
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
      if(!quiet && nx*ny < outlimit)
        show(f(),df.x,df.y,dfZ,0,0,0,df.x,df.y,df.Z,group.active);
        
      if(main) timings("FFT timing:",nx,T,N,stats);
      delete[] T;
    }
    
    deleteAlign(g());
    if(!inplace) deleteAlign(f());
  }
  
  MPI_Finalize();
  
  return retval;
}
