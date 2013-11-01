#include <mpi.h>
#include <fftw3-mpi.h>
#include "../Complex.h"
#include "../seconds.h"
#include "mpitranspose.h"
#include "utils.h"
#include "mpiutils.h"
#include <unistd.h>

using namespace std;
using namespace fftwpp;

inline unsigned int ceilquotient(unsigned int a, unsigned int b)
{
  return (a+b-1)/b;
}

namespace fftwpp {
void LoadWisdom(const MPI_Comm& active);
void SaveWisdom(const MPI_Comm& active);
}

void init(Complex *data, unsigned int X, unsigned int y, unsigned int Z,
  ptrdiff_t ystart) {
  for(unsigned int i=0; i < X; ++i) { 
    for(unsigned int j=0; j < y; ++j) {
      for(unsigned int k=0; k < Z; ++k) {
        data[(y*i+j)*Z+k].re = i;
        data[(y*i+j)*Z+k].im = ystart+j;
      }
    }
  }
}
  
inline void usage()
{
  std::cerr << "Options: " << std::endl;
  std::cerr << "-h\t\t help" << std::endl;
  std::cerr << "-T\t\t number of threads" << std::endl;
  std::cerr << "-N\t\t number of iterations" << std::endl;
  std::cerr << "-m\t\t size" << std::endl;
  std::cerr << "-X\t\t X size" << std::endl;
  std::cerr << "-Y\t\t Y size" << std::endl;
  std::cerr << "-Z\t\t Z size" << std::endl;
  std::cerr << "-L\t\t local transpose output" << std::endl;
  exit(1);
}

int main(int argc, char **argv)
{

  unsigned int X=8, Y=8, Z=1;
  const unsigned int showlimit=1024;
  int N=1;
  bool outtranspose=false;

#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"hLN:m:T:X:Y:Z:");
    if (c == -1) break;
                
    switch (c) {
      case 0:
        break;
      case 'N':
        N=atoi(optarg);
        break;
      case 'L':
        outtranspose=true;
        break;
      case 'm':
        X=Y=atoi(optarg);
        break;
      case 'X':
        X=atoi(optarg);
        break;
      case 'Y':
        Y=atoi(optarg);
        break;
      case 'Z':
        Z=atoi(optarg);
        break;
      case 'T':
        fftw::maxthreads=atoi(optarg);
        break;
      case 'h':
      default:
        usage();
    }
  }

    
  Complex *data;
  ptrdiff_t x,xstart;
  ptrdiff_t y,ystart;
  
  int provided;
//  MPI_Init(&argc,&argv);
  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);

  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  
  if(rank == 0) {
    cout << "size=" << comm_size << endl;
    cout << "threads=" << fftw::maxthreads << endl;
  }
  
  fftw_mpi_init();
     
  /* get local data size and allocate */
  ptrdiff_t NN[2]={Y,X};
  unsigned int block=ceilquotient(Y,comm_size);
#ifdef OLD  
  ptrdiff_t alloc=
#endif    
    fftw_mpi_local_size_many_transposed(2,NN,Z,block,0,
                                                      MPI_COMM_WORLD,&y,
                                                      &ystart,&x,&xstart);
  if(rank == 0) {
    cout << "x=" << x << endl;
    cout << "y=" << y << endl;
    cout << "X=" << X << endl;
    cout << "Y=" << Y << endl;
    cout << "Z=" << Z << endl;
    cout << "N=" << N << endl;
    cout << endl;
  }
  
#ifndef OLD
  data=ComplexAlign(X*y*Z);
#else  
  data=ComplexAlign(alloc);
#endif  
  
#ifdef OLD
  if(rank == 0) cout << "\nOLD\n" << endl;
  
  fftwpp::LoadWisdom(MPI_COMM_WORLD);
  fftw_plan inplan=fftw_mpi_plan_many_transpose(Y,X,2*Z,block,0,
                                                (double*) data,(double*) data,
                                                MPI_COMM_WORLD,
                                                 FFTW_MPI_TRANSPOSED_IN);
  fftw_plan outplan=fftw_mpi_plan_many_transpose(X,Y,2*Z,0,block,
                                                 (double*) data,(double*) data,
                                                 MPI_COMM_WORLD,
                                                 outtranspose ? 0 : FFTW_MPI_TRANSPOSED_OUT);
  fftwpp::SaveWisdom(MPI_COMM_WORLD);
#else
  mpitranspose T(X,y,x,Y,Z,data,NULL,fftw::maxthreads);
#endif  
  
  init(data,X,y,Z,ystart);
#ifndef OLD  
  // Initialize remaining plans.
  T.transpose(data,false,true);
  T.NmTranspose(data);
  init(data,X,y,Z,ystart);
#endif  

  bool showoutput=X*Y < showlimit && N == 1;
  if(showoutput)
    show(data,X,y*Z);
  
  fftw::statistics Sininit,Sinwait0,Sinwait1,Sin,Soutinit,Soutwait0,Soutwait1,Sout;

  for(int k=0; k < N; ++k) {
    double begin=0.0, Tinit0=0.0, Tinit=0.0, Twait0=0.0, Twait1=0.0;
    if(rank == 0) begin=totalseconds();
#ifndef OLD
    T.inphase0(data);
#else  
    fftw_execute(inplan);
#endif  
    if(rank == 0) Tinit0=totalseconds();
#ifndef OLD
    T.insync0(data);
#endif
    if(rank == 0) Twait0=totalseconds();
#ifndef OLD
    T.inphase1(data);
#endif
    if(rank == 0) Tinit=totalseconds();
#ifndef OLD
    T.insync1(data);
#endif
    if(rank == 0) Twait1=totalseconds();
#ifndef OLD
    T.inpost(data);
#endif
    if(rank == 0) {
      Sin.add(totalseconds()-begin);
      Sininit.add(Tinit0-begin);
      Sinwait0.add(Twait0-Tinit0);
      Sinwait1.add(Twait1-Tinit);
    }

    if(showoutput) {
      if(rank == 0) cout << "\ntranspose:\n" << endl;
      show(data,x,Y*Z);
    }
    
    if(rank == 0) begin=totalseconds();
#ifndef OLD
    T.outphase0(data);
#else  
    fftw_execute(outplan);
#endif  
    if(rank == 0) Tinit0=totalseconds();
#ifndef OLD
    T.outsync0(data);
#endif    
    if(rank == 0) Twait0=totalseconds();
#ifndef OLD
    T.outphase1(data);
#endif
    if(rank == 0) Tinit=totalseconds();
#ifndef OLD
    T.outsync1(data);
#endif    
    if(rank == 0) Twait1=totalseconds();
#ifndef OLD
    if(outtranspose) T.NmTranspose(data);
#endif
    if(rank == 0) {
      Sout.add(totalseconds()-begin);
      Soutinit.add(Tinit0-begin);
      Soutwait0.add(Twait0-Tinit0);
      Soutwait1.add(Twait1-Tinit);
    }
  }
  
  if(showoutput) {
    if(outtranspose) {
      if(rank == 0) cout << "\nout:\n" << endl;
      show(data,y,X*Z);
    } else {
      if(rank == 0) cout << "\noriginal:\n" << endl;
      show(data,X,y*Z);
    }
  }

  if(rank == 0) {
    Sininit.output("Tininit",X);
    Sinwait0.output("Tinwait0",X);
    Sinwait1.output("Tinwait1",X);
    Sin.output("Tin",X);
    cout << endl;
    Soutinit.output("Toutinit",X);
    Soutwait0.output("Toutwait0",X);
    Soutwait1.output("Toutwait1",X);
    Sout.output("Tout",X);
  }
  
#ifdef OLD  
  fftw_destroy_plan(inplan);
  fftw_destroy_plan(outplan);
#endif  
  
  MPI_Finalize();
}
