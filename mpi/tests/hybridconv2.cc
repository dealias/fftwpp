#include <getopt.h>

#include "mpiconvolve.h"
#include "utils.h"
#include "timing.h"
#include "options.h"

using namespace std;
using namespace utils;
using namespace fftwpp;
using namespace Array;

size_t A=2; // number of inputs
size_t B=1; // number of outputs

int main(int argc, char* argv[])
{
  int divisor=0; // Test for best block divisor
  int alltoall=-1; // Test for best alltoall routine

  Lx=Ly=4;  // input data length
  Mx=My=8; // minimum padded length

  fftw::maxthreads=parallel::get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  int retval=0;
  bool quiet=false;

  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if(rank != 0) opterr=0;

  optionsHybrid(argc,argv,false,true);

  if(quiet) Output=false;

  MPIgroup group(MPI_COMM_WORLD,Ly);

  if(group.size > 1 && provided < MPI_THREAD_FUNNELED)
    fftw::maxthreads=1;

  defaultmpithreads=fftw::maxthreads;

  if(group.rank < group.size) {
    bool main=group.rank == 0;

    utils::stopWatch *c=NULL;
    if(!quiet && main) {
      c=new utils::stopWatch;
      cout << "Configuration: "
           << group.size << " nodes x " << fftw::maxthreads
           << " threads/node" << endl;
      cout << "Using MPI VERSION " << MPI_VERSION << endl;

      cout << endl;

      cout << "Lx=" << Lx << endl;
      cout << "Ly=" << Ly << endl;
      cout << "Mx=" << Mx << endl;
      cout << "My=" << My << endl;

      cout << "N=" << N << endl;
    }

    Application appx(A,B,multNone,fftw::maxthreads,true,true,mx,Dx,Ix);
    Application appy(A,B,multBinary,appx,my,Dy,Iy);

    split d(Lx,Ly,group.active);

    params P;

    if(main) {
      fftPad fftx(Lx,Mx,appx,d.y);
      P.x.init(&fftx);
      fftPad ffty(Ly,My,appy);
      P.y.init(&ffty);
    }

    MPI_Bcast(&P,sizeof(params),MPI_BYTE,0,group.active);

    fftPad fftx(Lx,Mx,appx,d.y,d.y,P.x.m,P.x.D,P.x.I);
    fftPad ffty(Ly,My,appy,1,1,    P.y.m,P.y.D,P.y.I);

    Convolution2MPI Convolve(&fftx,&ffty,group.active,
                             mpiOptions(divisor,alltoall));

    Complex **f=ComplexAlign(max(A,B),Lx*d.y);

    for(size_t a=0; a < A; ++a) {
      Complex *fa=f[a];
      for(size_t i=0; i < Lx; ++i) {
        for(size_t j=0; j < d.y; ++j) {
          unsigned int J=d.y0+j;
          fa[d.y*i+j]=Output || testError ? Complex(i+a+1,(a+1)*J+3) : 0.0;
        }
      }
    }

    if(Output || testError) {

      if(Output) {
        for(unsigned int a=0; a < A; ++a) {
          if(main)
            cout << "\nDistributed input " << a  << ":"<< endl;
          show(f[a],Lx,d.y,group.active);
        }
      }

      Complex **flocal=new Complex *[A];
      for(unsigned int a=0; a < A; ++a) {
        flocal[a]=ComplexAlign(Lx*Ly);
        gathery(f[a],flocal[a],d,1,group.active);
        if(main && Output)  {
          cout << "\nGathered input " << a << ":" << endl;
          Array2<Complex> flocala(Lx,Ly,flocal[a]);
          cout << flocala << endl;
        }
      }

      Convolve.convolve(f);

      Complex *foutgather=ComplexAlign(Lx*Ly);
      gathery(f[0],foutgather,d,1,group.active);

      if(Output) {
        if(main)
          cout << "Distributed output:" << endl;
        show(f[0],Lx,d.y,group.active);
      }

      if(main) {
        fftPad fftx(Lx,Mx,appx,Ly);
        fftPad ffty(Ly,My,appy);
        Convolution2 Convolve(&fftx,&ffty);
        Convolve.convolve(flocal);

        if(Output) {
          cout << "Local output:" << endl;
          Array2<Complex> flocal0(Lx,Ly,flocal[0]);
          cout << flocal0 << endl;
        }
        retval += checkerror(flocal[0],foutgather,Lx*Ly);
      }

      deleteAlign(foutgather);
      for(unsigned int a=0; a < A; ++a)
        deleteAlign(flocal[a]);
      delete [] flocal;

      MPI_Barrier(group.active);

    } else {
      if(!quiet && main)
        cout << "Initialized after " << c->seconds() << " seconds." << endl;

      MPI_Barrier(group.active);

      vector<double> T;
      for(size_t i=0; i < N; ++i) {
        utils::stopWatch *c;
        if(main) c=new utils::stopWatch;
        Convolve.convolveRaw(f);
        if(main)
          T.push_back(c->nanoseconds());
      }

      if(main) {
        timings("Hybrid",Lx*Ly,T.data(),T.size(),stats);
        T.clear();
      }
    }

    deleteAlign(f[0]); delete [] f;
  }

  MPI_Finalize();

  return retval;
}
