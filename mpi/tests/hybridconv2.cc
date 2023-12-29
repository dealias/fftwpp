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

size_t N=1; // TEMP

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

  optionsHybrid(argc,argv);

  if(Output || testError)
    K=0;
  if(K == 0) minCount=1;

  if(Sx == 0) Sx=Ly;

  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

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
      cout << "K=" << K << endl << endl;
      K *= 1.0e9;

      cout << "Lx=" << Lx << endl;
      cout << "Ly=" << Ly << endl;
      cout << "Mx=" << Mx << endl;
      cout << "My=" << My << endl;
    }

    params P;

    Application appx(A,B,multNone,fftw::maxthreads,false,mx,Dx,Ix);
    Application appy(A,B,multbinary,appx,my,Dy,Iy);

    split d(Lx,Ly,group.active);

    setMPIplanner();

    if(main) {
      fftPad fftx(Lx,Mx,appx,d.y);
      P.x.init(fftx.m,fftx.D,fftx.inplace,fftx.l);
      fftPad ffty(Ly,My,appy);
      P.y.init(ffty.m,ffty.D,ffty.inplace,ffty.l);
    }

    MPI_Bcast(&P,sizeof(params),MPI_BYTE,0,d.communicator);

    split D(P.x.l*P.x.D,Sx,d.communicator);

    D.Activate();

    fftPad fftx(Lx,Mx,appx,d.y,d.y,P.x.m,P.x.D,P.x.I);
    fftPad ffty(Ly,My,appy,1,1,    P.y.m,P.y.D,P.y.I);

    Convolution2MPI Convolve(&fftx,&ffty,D,mpiOptions(divisor,alltoall));

    Complex **f=ComplexAlign(max(A,B),d.n);

    for(size_t a=0; a < A; ++a) {
      Complex *fa=f[a];
      for(size_t i=0; i < d.X; ++i) {
        for(size_t j=0; j < d.y; ++j) {
          unsigned int J=d.y0+j;
          fa[d.y*i+j]=Output || testError ? Complex((1.0+a)*i,J+a) : 0.0;
        }
      }
    }

    if(Output || testError) {

      if(!quiet && Output) {
        for(unsigned int a=0; a < A; ++a) {
          if(main)
            cout << "\nDistributed input " << a  << ":"<< endl;
          show(f[a],d.X,d.y,group.active);
        }
      }

      Complex **flocal=new Complex *[A];
      for(unsigned int a=0; a < A; ++a) {
        flocal[a]=ComplexAlign(d.X*d.Y);
        gathery(f[a],flocal[a],d,1,group.active);
        if(!quiet && main)  {
          cout << "\nGathered input " << a << ":" << endl;
          Array2<Complex> flocala(d.X,d.Y,flocal[a]);
          cout << flocala << endl;
        }
      }

      Convolve.convolve(f);

      Complex *foutgather=ComplexAlign(d.X*d.Y);
      gathery(f[0],foutgather,d,1,group.active);

      if(!quiet && Output) {
        if(main)
          cout << "Distributed output:" << endl;
        show(f[0],d.X,d.y,group.active);
      }

      if(main) {
        fftPad fftx(Lx,Mx,appx,Ly,Sx);
        fftPad ffty(Ly,My,appy);
        Convolution2 Convolve(&fftx,&ffty);
        Convolve.convolve(flocal);

        if(!quiet) {
          cout << "Local output:" << endl;
          Array2<Complex> flocal0(d.X,d.Y,flocal[0]);
          cout << flocal0 << endl;
        }
        retval += checkerror(flocal[0],foutgather,d.X*d.Y);
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
      for(unsigned int i=0; i < N; ++i) {
        utils::stopWatch *c;
        if(main) c=new utils::stopWatch;
        Convolve.convolve(f);
        if(main)
          T.push_back(c->nanoseconds());
      }

      if(main) {
        timings("Hybrid",d.X,T.data(),T.size(),stats);
        T.clear();
      }

      if(!quiet && Output) {
        show(f[0],d.X,d.y,group.active);
      }
    }

    deleteAlign(f[0]); delete f;
  }

  MPI_Finalize();

  return retval;
}
