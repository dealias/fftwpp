#include "mpifftw++.h"
#include "mpiconvolution.h"
#include "mpiutils.h" // For output of distritubed arrays

using namespace std;
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

  // Must be even for 2/3 padding convolutions.
  unsigned int mx=4;
  unsigned int my=4;
  
  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);

  // MPI group size cannot be larger than minimum dimensions of arrays.
  int fftsize=min(mx,my/2+1);

  MPIgroup group(MPI_COMM_WORLD,fftsize);

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
      cout << "mx=" << mx << ", my=" << my << endl;
    } 
    unsigned int myp=my/2+1;

    // Set up per-process dimensions
    split df(mx,my,group.active);
    split dg(mx,myp,group.active);

    // Allocate complex-aligned memory
    double *f0=doubleAlign(df.n);
    double *f1=doubleAlign(df.n);
    Complex *g0=ComplexAlign(dg.n);
    Complex *g1=ComplexAlign(dg.n);

    // Create instance of FFT
    rcfft2dMPI rcfft(df,dg,f0,g0,mpiOptions(fftw::maxthreads));

    // Create instance of convolution
    Complex **G=new Complex *[2];
    G[0]=g0;
    G[1]=g1;
    ImplicitHConvolution2MPI C(mx/2,my/2,dg,dg,g0,2,1);
    realmultiplier *mult=multbinary;

    if(main) cout << "\nDistributed input (split in x-direction):" << endl;
    if(main) cout << "f0:" << endl;
    init(f0,df);
    show(f0,df.x,my,group.active);
    if(main) cout << "f1:" << endl;
    init(f1,df);
    show(f1,df.x,my,group.active);
      
    if(main) cout << "\nDistributed output (split in y-direction:)" << endl;
    if(main) cout << "g0:" << endl;
    rcfft.Forwards0(f0,g0);
    show(g0,dg.X,dg.y,group.active);
    if(main) cout << "g1:" << endl;
    rcfft.Forwards0(f1,g1);
    show(g1,dg.X,dg.y,group.active);

    if(main) cout << "\nAfter convolution (split in y-direction):" << endl;
    C.convolve(G,mult);
    if(main) cout << "g0:" << endl;
    show(g0,dg.X,dg.y,group.active);

    if(main) cout << "\nTransformed back to real-space (split in x-direction):"
		  << endl;
    if(main) cout << "f0:" << endl;
    rcfft.Backwards0Normalized(g0,f0);
    show(f0,df.x,my,group.active);

    deleteAlign(f0);
    deleteAlign(f1);
    deleteAlign(g0);
    deleteAlign(g1);
    delete[] G;
  }
  
  MPI_Finalize();
}
