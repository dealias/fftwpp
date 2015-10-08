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

int main(int argc, char* argv[])
{
#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  unsigned int mx=4;
  unsigned int my=4;
  bool shift=false;
  
  cout << "shift: " << shift << endl;
  
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
  
    double *f=doubleAlign(df.n);
    Complex *g=ComplexAlign(dg.n);

    // Create instance of FFT
    rcfft2dMPI rcfft(df,dg,f,g,mpiOptions(fftw::maxthreads));
 
    init(f,df);
      
    if(main) cout << "\nDistributed input:" << endl;
    show(f,df.x,my,group.active);
      
    if(shift)
      rcfft.Forwards0(f,g);
    else
      rcfft.Forwards(f,g);
      
    if(main) cout << "\nDistributed output:" << endl;
    show(g,dg.X,dg.y,group.active);

    if(shift)
      rcfft.Backwards0Normalized(g,f);
    else
      rcfft.BackwardsNormalized(g,f);

    if(main) cout << "\nDistributed back to input:" << endl;
    show(f,df.x,my,group.active);

    deleteAlign(f);
  }
  
  MPI_Finalize();
}
