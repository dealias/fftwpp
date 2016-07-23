#include "Array.h"
#include "mpifftw++.h"
#include "utils.h"

using namespace std;
using namespace utils;
using namespace fftwpp;
using namespace Array;

// Number of iterations.
unsigned int N0=10000000;
unsigned int N=0;
int divisor=0; // Test for best block divisor
int alltoall=-1; // Test for best alltoall routine

void init(Complex *f,
          unsigned int X, unsigned int Y, unsigned int Z,
          unsigned int x0, unsigned int y0, unsigned int z0,
          unsigned int x, unsigned int y, unsigned int z)
{
  unsigned int c=0;
  for(unsigned int i=0; i < x; ++i) {
    unsigned int ii=x0+i;
    for(unsigned int j=0; j < y; j++) {
      unsigned int jj=y0+j;
      for(unsigned int k=0; k < Z; k++) {
        f[c++]=Complex(10*k+ii,jj);
      }
    }
  }
}

void init(Complex *f, split3 d)
{
  init(f,d.X,d.Y,d.Z,d.x0,d.y0,d.z0,d.x,d.y,d.z);
}

unsigned int outlimit=3000;

int main(int argc, char* argv[])
{
#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif
  int retval=0;

  unsigned int nx=4;
  unsigned int ny=0;
  unsigned int nz=0;

  bool inplace=true;
  
  bool quiet=false;
  bool test=false;
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
    int c = getopt(argc,argv,"htN:S:T:a:i:m:n:s:x:y:z:q");
    if (c == -1) break;
                
    switch (c) {
      case 0:
        break;
      case 'a':
        divisor=atoi(optarg);
        break;
      case 'N':
        N=atoi(optarg);
        break;
      case 'i':
        inplace=atoi(optarg);
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
          usageInplace(3);
          usageTranspose();
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
      cout << "N=" << N << endl;
      cout << "nx=" << nx << ", ny=" << ny << ", nz=" << nz << endl;
    }

    split3 d(nx,ny,nz,group);
    
    Complex *f=ComplexAlign(d.n);
    Complex *g=inplace ? f : ComplexAlign(d.n);
    
    fft3dMPI fft(d,f,g,mpiOptions(divisor,alltoall));

    if(test) {
      if(main) std::cout << "Allocated " << d.n << " bytes." << endl;
      init(f,d);

      if(!quiet && nx*ny*nz < outlimit) {
        if(main) cout << "\ninput:" << endl;
        show(f,d.x,d.y,d.Z,group.active);
      }

      size_t align=sizeof(Complex);
      array3<Complex> fgathered(nx,ny,nz,align);
      fft3d localForward(-1,fgathered);
      fft3d localBackward(1,fgathered);
      gatherxy(f,fgathered(),d,group.active);

      array3<Complex> flocal(nx,ny,nz,align);
      init(flocal(),d.X,d.Y,d.Z,0,0,0,d.X,d.Y,d.Z);
      if(main) {
        if(!quiet) {
          cout << "Gathered input:\n" <<  fgathered << endl;
          cout << "Local input:\n" <<  flocal << endl;
        }
        retval += checkerror(flocal(),fgathered(),d.X*d.Y*d.Z);
      }
      
      fft.Forward(f,g);

      if(main)
        localForward.fft(flocal);
      
      if(!quiet) {
        if(main) cout << "Distributed output:" << endl;
        show(f,d.X,d.xy.y,d.z,group.active);
      }
      gatheryz(g,fgathered(),d,group.active); 

      if(!quiet && main) {
        cout << "Gathered output:\n" <<  fgathered << endl;
        cout << "Local output:\n" <<  flocal << endl;
      }
      
      if(main)
        retval += checkerror(flocal(),fgathered(),d.X*d.Y*d.Z);
      
      fft.Backward(g,f);
      fft.Normalize(f);

      if(main)
        localBackward.fftNormalized(flocal);
      if(!quiet) {
        if(main) cout << "Distributed output:" << endl;
        show(f,d.x,d.y,d.Z,group.active);
      }

      gatherxy(f,fgathered(),d,group.active);
      
      if(!quiet && main) {
        cout << "Gathered output:\n" <<  fgathered << endl;
        cout << "Local output:\n" <<  flocal << endl;
      }
      
      if(main)
        retval += checkerror(flocal(),fgathered(),d.X*d.Y*d.Z);
      
      if(!quiet) {
        if(main) cout << "\nback to input:" << endl;
        show(f,d.x,d.y,d.Z,group.active);
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
        unsigned int M=1;
        Complex **F=new Complex *[M];
        Complex **G=inplace ? F : new Complex *[M];
        fft3dMPI **FFT=new fft3dMPI *[M];
        F[0]=ComplexAlign(d.n);
        if(!inplace) G[0]=ComplexAlign(d.n);
        FFT[0]=new fft3dMPI(d,F[0],G[0],mpiOptions(divisor,alltoall),
                            mpiOptions(divisor,alltoall));
        for(unsigned int m=1; m < M; ++m) {
          F[m]=ComplexAlign(d.n);
          if(!inplace) G[m]=ComplexAlign(d.n);
          FFT[m]=new fft3dMPI(d,F[m],G[m],FFT[0]->Txy->Options(),
                              FFT[0]->Tyz ? FFT[0]->Tyz->Options() :
                              defaultmpiOptions);
        }
        if(main) std::cout << "Allocated " << M*d.n << " bytes." << endl;
        
        double *T=new double[N];
        for(unsigned int i=0; i < N; ++i) {
          for(unsigned int m=0; m < M; ++m)
            init(F[m],d);
          seconds();
          for(unsigned int m=0; m < M; ++m)
            FFT[m]->iForward(F[m],G[m]);
          for(unsigned int m=0; m < M; ++m) {
            FFT[m]->ForwardWait(G[m]);
            FFT[m]->iBackward(G[m],F[m]);
          }
          for(unsigned int m=0; m < M; ++m)
            FFT[m]->BackwardWait(F[m]);
          T[i]=0.5*seconds()/M;
          fft.Normalize(F[0]);
        }
        if(!quiet && nx*ny*nz < outlimit) show(G[0],d.x,d.y,d.Z,group.active);
        if(main) timings("FFT timing:",nx,T,N,stats);
        
        for(unsigned int m=0; m < M; ++m) {
          delete FFT[m];
          if(!inplace) deleteAlign(G[m]);
          deleteAlign(F[m]);
        }        
        
        delete[] FFT; 
        if(!inplace) delete[] G;
        delete[] F;
        delete[] T;
      }
    }
  
    deleteAlign(f);
    if(!inplace)
      deleteAlign(g);
  }
  
  MPI_Finalize();
  
  return retval;
}
