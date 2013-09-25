#include <mpi.h>
#include <iostream>
#include <fftw3-mpi.h>
#include <cassert>
#include <cstring>
#include "../Complex.h"
#include "../fftw++.h"
#include "../seconds.h"
#include "mpitranspose.h"
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
  exit(1);
}

int main(int argc, char **argv)
{

  unsigned int X=8, Y=8, Z=1;
  const unsigned int showlimit=1024;
  int N=1;

#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"hN:m:X:Y:Z:T:");
    if (c == -1) break;
                
    switch (c) {
      case 0:
        break;
      case 'N':
        N=atoi(optarg);
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
  
//  int provided;
  MPI_Init(&argc,&argv);
//  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);

  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  
  if(rank == 0) cout << "size=" << comm_size << endl;
  
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
    cout << endl;
  }
  
#ifndef OLD
  data=new Complex[X*y*Z];
#else  
  data=new Complex[alloc];
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
                                                 FFTW_MPI_TRANSPOSED_OUT);
  fftwpp::SaveWisdom(MPI_COMM_WORLD);
#else
  transpose T(data,X,y,x,Y,Z);
  init(data,X,y,Z,ystart);
  T.inTransposed(data);
  T.inwait(data);
  T.outTransposed(data);
  T.outwait(data,true);
#endif  
  
  init(data,X,y,Z,ystart);

  bool showoutput=X*Y < showlimit && N == 1;
  if(showoutput)
    show(data,X,y*Z);
  
  double commtime=0;
  double posttime=0;
  double outcommtime=0;
  double outposttime=0;

  for(int k=0; k < N; ++k) {
    if(rank == 0) seconds();
#ifndef OLD
    T.inTransposed(data);
#else  
    fftw_execute(inplan);
#endif  
    if(rank == 0) 
      commtime += seconds();
#ifndef OLD
    T.inwait(data);
#endif
    if(rank == 0) 
      posttime += seconds();

    if(showoutput) {
      if(rank == 0) cout << "\ntranspose:\n" << endl;
      show(data,x,Y*Z);
    }
    
#ifndef OLD
    T.outTransposed(data);
#else  
    fftw_execute(outplan);
#endif  
    if(rank == 0) 
      outcommtime += seconds();
#ifndef OLD
    T.outwait(data,true);
#endif
    if(rank == 0) 
      outposttime += seconds();
  }
  
  if(rank == 0) {
    cout << endl << commtime/N << endl;
    cout << (commtime+posttime)/N << endl;
    cout << endl;
    cout << outcommtime/N << endl;
    cout << (outcommtime+outposttime)/N << endl;
  }
  
  if(showoutput) {
    if(rank == 0) cout << "\noriginal:\n" << endl;
//    show(data,X,y*Z);
    show(data,y,X*Z);
  }
  
#ifdef OLD  
  fftw_destroy_plan(inplan);
  fftw_destroy_plan(outplan);
#endif  
  
  MPI_Finalize();
}
