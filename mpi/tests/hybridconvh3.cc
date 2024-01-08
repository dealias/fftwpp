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

  Lx=Ly=Lz=7;  // input data length
  Mx=My=Mz=10; // minimum padded length

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

  int size;
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  unsigned int Hx=ceilquotient(Lx,2);
  unsigned int Hz=ceilquotient(Lz,2);

  unsigned int x=ceilquotient(Lx,size);
  unsigned int y=ceilquotient(Ly,size);
  unsigned int hx=ceilquotient(Hx,size);
  bool allowpencil=Lx*y == x*Ly && Hx*y == hx*Ly;

  MPIgroup group(MPI_COMM_WORLD,Ly,Hz,allowpencil);

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
      cout << "Lz=" << Lz << endl;
      cout << "Mx=" << Mx << endl;
      cout << "My=" << My << endl;
      cout << "Mz=" << Mz << endl;

      cout << "N=" << N << endl;
    }

    Application appx(A,B,multNone,fftw::maxthreads,true,true,mx,Dx,Ix);
    Application appy(A,B,multNone,appx,my,Dy,Iy);
    Application appz(A,B,realMultBinary,appy,mz,Dz,Iz);

    split3 d(Lx,Ly,Hz,group,true);

    params P;

    if(main) {
      fftPadCentered fftx(Lx,Mx,appx,d.y*d.z);
      P.x.init(&fftx);
      fftPadCentered ffty(Ly,My,appy,d.z);
      P.y.init(&ffty);
      fftPadHermitian fftz(Lz,Mz,appz);
      P.z.init(&fftz);
    }

    MPI_Bcast(&P,sizeof(params),MPI_BYTE,0,d.communicator);

    fftPadCentered fftx(Lx,Mx,appx,d.y*d.z,d.y*d.z,P.x.m,P.x.D,P.x.I);
    fftPadCentered ffty(Ly,My,appy,d.z,d.z,        P.y.m,P.y.D,P.y.I);
    fftPadHermitian fftz(Lz,Mz,appz,1,             P.z.m,P.z.D,P.z.I);

    Convolution3MPI Convolve(&fftx,&ffty,&fftz,group,
                             mpiOptions(divisor,alltoall));

    Complex **f=ComplexAlign(max(A,B),Lx*d.y*d.z);

    for(size_t a=0; a < A; ++a) {
      Complex *fa=f[a];
      for(size_t i=0; i < Lx; ++i) {
        int I=Lx % 2 ? i : i-1;
        for(size_t j=0; j < d.y; ++j) {
          int J=Ly % 2 ? d.y0+j : d.y0+j-1;
          for(size_t k=0; k < d.z; ++k) {
            size_t K=d.z0+k;
            fa[d.y*d.z*i+d.z*j+k]=Output || testError ?
              Complex(I+a*K+1,(a+1)*J+3+K) : 0.0;
          }
        }
      }
      HermitianSymmetrizeXY(d,fa);
    }

    if(Output || testError) {

      if(Output) {
        for(unsigned int a=0; a < A; ++a) {
          if(main)
            cout << "\nDistributed input " << a  << ":"<< endl;
          show(f[a],Lx,d.y,d.z,group.active);
        }
      }

      Complex **flocal=new Complex *[A];
      for(unsigned int a=0; a < A; ++a) {
        flocal[a]=ComplexAlign(Lx*Ly*Hz);
        gatheryz(f[a],flocal[a],d,group.active);
        if(main && Output)  {
          cout << "\nGathered input " << a << ":" << endl;
          Array3<Complex> flocala(Lx,Ly,Hz,flocal[a]);
          cout << flocala << endl;
        }
      }

      Convolve.convolve(f);

      Complex *foutgather=ComplexAlign(Lx*Ly*Hz);
      gatheryz(f[0],foutgather,d,group.active);

      if(Output) {
        if(main)
          cout << "Distributed output:" << endl;
        show(f[0],Lx,d.y,d.z,group.active);
      }

      if(main) {
        fftPadCentered fftx(Lx,Mx,appx,Ly*Hz);
        fftPadCentered ffty(Ly,My,appy,Hz);
        fftPadHermitian fftz(Lz,Mz,appz);
        Convolution3 Convolve(&fftx,&ffty,&fftz);
        Convolve.convolve(flocal);

        if(Output) {
          cout << "Local output:" << endl;
          Array3<Complex> flocal0(Lx,Ly,Hz,flocal[0]);
          cout << flocal0 << endl;
        }
        retval += checkerror(flocal[0],foutgather,Lx*Ly*Hz);
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
        timings("Hybrid",Lx*Ly*Hz,T.data(),T.size(),stats);
        T.clear();
      }
    }

    deleteAlign(f[0]); delete [] f;
  }

  MPI_Finalize();

  return retval;
}
