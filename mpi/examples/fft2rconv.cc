#include "mpiconvolution.h"

using namespace std;
using namespace utils;
using namespace fftwpp;

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

int main(int argc, char* argv[])
{
#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  // The real-space problem size (must be even in each dimension:
  unsigned int nx=8;
  unsigned int ny=8;

  // The y-dimension of the array in complex space:
  unsigned int nyp=ny/2+1;

  // Convolution dimensions:
  unsigned int mx=(nx+1)/2;
  unsigned int my=(ny+1)/2;
  
  int divisor=0;    // Test for best divisor
  int alltoall=-1;  // Test for best communication routine
  mpiOptions options(divisor,alltoall);
  
  bool xcompact=false;
  bool ycompact=false;
  
  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);

  MPIgroup group(MPI_COMM_WORLD,nyp);

  if(group.size > 1 && provided < MPI_THREAD_FUNNELED)
    fftw::maxthreads=1;
  
  defaultmpithreads=fftw::maxthreads;

  if(group.rank == 0) {
    cout << "Configuration: " 
         << group.size << " nodes X " << fftw::maxthreads
         << " threads/node" << endl;
  }

  if(group.rank < group.size) { 
    bool main=group.rank == 0;
    if(main) {
      cout << "mx=" << mx << ", my=" << my << endl;
      cout << "nx=" << nx << ", ny=" << ny << ", nyp=" << nyp << endl;
    } 

    // Set up per-process dimensions
    split df(nx,ny,group.active);
    split dg(nx,nyp,group.active);
    split du(mx+xcompact,nyp,group.active);

    // Allocate complex-aligned memory
    double *f0=doubleAlign(df.n);
    double *f1=doubleAlign(df.n);
    Complex *g0=ComplexAlign(dg.n);
    Complex *g1=ComplexAlign(dg.n);

    // Create instance of FFT
    rcfft2dMPI rcfft(df,dg,f0,g0,options);

    // Create instance of convolution
    Complex *G[]={g0,g1};
    ImplicitHConvolution2MPI C(mx,my,xcompact,ycompact,dg,du,g0,options);
    
    init(f0,df);
    init(f1,df);

    if(main) cout << "\nDistributed input (split in x direction):" << endl;
    if(main) cout << "f0:" << endl;
    show(f0,df.x,df.Y,group.active);
    if(main) cout << "f1:" << endl;
    show(f1,df.x,df.Y,group.active);
      
    if(main) cout << "\nDistributed output (split in y direction:)" << endl;
    if(main) cout << "g0:" << endl;
    rcfft.Forward0(f0,g0);
    show(g0,dg.X,dg.y,group.active);
    if(main) cout << "g1:" << endl;
    rcfft.Forward0(f1,g1);
    show(g1,dg.X,dg.y,group.active);

    if(main) cout << "\nAfter convolution (split in y direction):" << endl;
    C.convolve(G,multbinary);
    if(main) cout << "g0:" << endl;
    show(g0,dg.X,dg.y,group.active);

    if(main) cout << "\nTransformed back to real-space (split in x direction):"
                  << endl;
    if(main) cout << "f0:" << endl;
    rcfft.Backward0(g0,f0);
    rcfft.Normalize(f0);
    show(f0,df.x,df.Y,group.active);

    deleteAlign(g1);
    deleteAlign(g0);
    deleteAlign(f1);
    deleteAlign(f0);
  }
  
  MPI_Finalize();
}
