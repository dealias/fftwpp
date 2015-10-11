#include "Array.h"

#include "mpifftw++.h"
#include "mpiconvolution.h"
#include "mpiutils.h" // For output of distributed arrays

using namespace std;
using namespace fftwpp;

void init(double *f, const split3 df)
{
  unsigned int c=0;
  for(unsigned int i=0; i < df.x; ++i) {
    unsigned int ii=df.x0+i;
    for(unsigned int j=0; j < df.y; j++) {
      unsigned int jj=df.y0+j;
      for(unsigned int k=0; k < df.Z; k++) {
        unsigned int kk=k;
        f[c++] = ii + jj + kk;
      }
    }
  }
}

int main(int argc, char* argv[])
{
#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  // The real-space problem size:
  unsigned int nx=4;
  unsigned int ny=4;
  unsigned int nz=4;

  // The z-dimension of the array in complex space:
  unsigned int nzp=nz/2+1;

  // Convolution dimensions:
  unsigned int mx=nx/2;
  unsigned int my=ny/2;
  unsigned int mz=nz/2;
  
  int divisor=1;
  int alltoall=1;
  convolveOptions options;
  options.xcompact=false;
  options.ycompact=false;
  options.zcompact=false;
  options.mpi=mpiOptions(fftw::maxthreads,divisor,alltoall);
    
  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);

  MPIgroup group(MPI_COMM_WORLD,nx,ny,nzp);

  if(group.size > 1 && provided < MPI_THREAD_FUNNELED)
    fftw::maxthreads=1;

  if(group.rank == 0) {
    cout << "Configuration: " 
	 << group.size << " nodes X " << fftw::maxthreads
	 << " threads/node" << endl;
  }

  if(group.rank < group.size) { 
    bool main=group.rank == 0;
    if(main) {
      cout << "nx=" << nx
	   << ", ny=" << ny
	   << ", nz=" << nz
	   << ", nzp=" << nzp << endl;
    }

    // Set up per-process dimensions
    split3 df(nx,ny,nz,group);
    split3 dg(nx,ny,nzp,group);
    
    // Allocate complex-aligned memory
    double *f0=doubleAlign(df.n);
    double *f1=doubleAlign(df.n);
    Complex *g0=ComplexAlign(dg.n);
    Complex *g1=ComplexAlign(dg.n);

    // Create instance of FFT
    rcfft3dMPI rcfft(df,dg,f0,g0,options.mpi);

    init(f0,df);
    init(f1,df);

    if(main) cout << "\nDistributed input (split in xy):" << endl;
    if(main) cout << "f0:" << endl;
    show(f0,df.x,df.Y,df.Z,group.active);
    if(main) cout << "f1:" << endl;
    show(f1,df.x,df.Y,df.Z,group.active);
      
    if(main) cout << "\nDistributed output (split in yz:)" << endl;
    if(main) cout << "g0:" << endl;
    rcfft.Forwards0(f0,g0);
    show(g0,dg.X,dg.y,dg.Z,group.active);
    if(main) cout << "g1:" << endl;
    rcfft.Forwards0(f1,g1);
    show(g1,dg.X,dg.y,dg.Z,group.active);
    
    if(main) cout << "\nAfter convolution (split in yz):" << endl;
    // Create instance of convolution
    Complex *G[]={g0,g1};
    split3 du(mx,my,mz,group);
    ImplicitHConvolution3MPI C(mx,my,mz,dg,du,g0,2,1,
                               convolveOptions(options,fftw::maxthreads));
    
    C.convolve(G,multbinary);
    if(main) cout << "g0:" << endl;
    show(g0,dg.X,dg.y,dg.Z,group.active);

    if(main) cout << "\nTransformed back to real-space (split in xy):"
		  << endl;
    if(main) cout << "f0:" << endl;
    rcfft.Backwards0Normalized(g0,f0);
    show(f0,df.x,df.Y,df.Z,group.active);

    deleteAlign(f0);
    deleteAlign(f1);
    deleteAlign(g0);
    deleteAlign(g1);
  }
  
  MPI_Finalize();
}
