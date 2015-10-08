#include "Array.h"
#include "mpiconvolution.h"
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

int main(int argc, char* argv[])
{
#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  unsigned int mx=4;
  unsigned int my=4;
  
  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);

  if(my == 0) my=mx;

  int fftsize=min(mx,my);

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
    
    split df(mx,my,group.active);
    split dg(mx,myp,group.active);
  
    double *f0=doubleAlign(df.n);
    Complex *g0=ComplexAlign(dg.n);
    double *f1=doubleAlign(df.n);
    Complex *g1=ComplexAlign(dg.n);

    // Create instance of FFT
    rcfft2dMPI rcfft(df,dg,f0,g0,mpiOptions(fftw::maxthreads));

    // Create instance of convolution
    Complex **G=new Complex *[2];
    G[0]=g0;
    G[1]=g1;
    ImplicitHConvolution2MPI C(mx,my,dg,dg,g0,2,1);
    realmultiplier *mult=multbinary;

    // Init the real-valued inputs
    init(f0,df);
    init(f1,df);

    if(main) cout << "\nDistributed input:" << endl;
    show(f0,df.x,my,group.active);
    show(f1,df.x,my,group.active);

    // Transform to complex space
    rcfft.Forwards0(f0,g0);
    rcfft.Forwards0(f1,g1);
      
    if(main) cout << "\nDistributed output:" << endl;
    show(g0,dg.X,dg.y,group.active);
    show(g1,dg.X,dg.y,group.active);

    // Convolve in complex space
    C.convolve(G,mult);
    if(main) cout << "\nAfter convolution:" << endl;
    show(g0,dg.X,dg.y,group.active);

    // Transform back to real space.
    rcfft.Backwards0Normalized(g0,f0);

    if(main) cout << "\nTransformed back to real-space:" << endl;
    show(f0,df.x,my,group.active);

    deleteAlign(f0);
    deleteAlign(f1);
    deleteAlign(g0);
    deleteAlign(g1);
    delete[] G;
  }
  
  MPI_Finalize();
}
