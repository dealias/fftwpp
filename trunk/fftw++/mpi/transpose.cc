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

void init(Complex *data, unsigned int N0, unsigned int n1, unsigned int N2,
  ptrdiff_t n1start) {
  for(unsigned int i=0; i < N0; ++i) { 
    for(unsigned int j=0; j < n1; ++j) {
      for(unsigned int k=0; k < N2; ++k) {
        data[(n1*i+j)*N2+k].re = i;
        data[(n1*i+j)*N2+k].im = n1start+j;
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
  std::cerr << "-x\t\t x size" << std::endl;
  std::cerr << "-y\t\t y size" << std::endl;
  exit(1);
}

int main(int argc, char **argv)
{

  unsigned int N0=1024, N1=1024;

  const unsigned int showlimit=1024;
  int N=1000;

#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"hN:m:x:y:T:");
    if (c == -1) break;
                
    switch (c) {
      case 0:
        break;
      case 'N':
        N=atoi(optarg);
        break;
      case 'm':
        N0=N1=atoi(optarg);
        break;
      case 'x':
        N0=atoi(optarg);
        break;
      case 'y':
        N1=atoi(optarg);
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
  ptrdiff_t n0,n0start;
  ptrdiff_t n1,n1start;
  const unsigned int N2=1;
  
//  int provided;
  MPI_Init(&argc,&argv);
//  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);

  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  
  if(rank == 0) cout << "size=" << comm_size << endl;
  
  fftw_mpi_init();
     
  /* get local data size and allocate */
  ptrdiff_t NN[2]={N1,N0};
  unsigned int block=ceilquotient(N1,comm_size);
#ifdef OLD  
  ptrdiff_t alloc=
#endif    
    fftw_mpi_local_size_many_transposed(2,NN,N2,block,0,
                                                      MPI_COMM_WORLD,&n1,
                                                      &n1start,&n0,&n0start);
  if(rank == 0) {
    cout << "n=" << n0 << endl;
    cout << "m=" << n1 << endl;
    cout << "N=" << N0 << endl;
    cout << "M=" << N1 << endl;
    cout << endl;
  }
  
#ifndef OLD
  data=new Complex[N0*n1*N2];
#else  
  data=new Complex[alloc];
#endif  
  
#ifdef OLD
  if(rank == 0) cout << "\nOLD\n" << endl;
  
  fftwpp::LoadWisdom(MPI_COMM_WORLD);
  fftw_plan inplan=fftw_mpi_plan_many_transpose(N1,N0,2*N2,block,0,
                                                (double*) data,(double*) data,
                                                MPI_COMM_WORLD,
                                                 FFTW_MPI_TRANSPOSED_IN);
  fftw_plan outplan=fftw_mpi_plan_many_transpose(N0,N1,2*N2,0,block,
                                                 (double*) data,(double*) data,
                                                 MPI_COMM_WORLD,
                                                 FFTW_MPI_TRANSPOSED_OUT);
  fftwpp::SaveWisdom(MPI_COMM_WORLD);
#else
  transpose T(N0,n1,n0,N1,N2);
  init(data,N0,n1,N2,n1start);
  T.inTransposed(data);
  T.inwait(data);
  T.outTransposed(data);
  T.outwait(data);
#endif  
  
  init(data,N0,n1,N2,n1start);

  if(N0*N1 < showlimit)
    show(data,N0,n1*N2);
  
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

    if(N0*N1 < showlimit && N == 1){
      // output tranposed array if the problem size is small and only
      // one iteration is being performed.
      if(rank == 0) cout << "\ntranspose:\n" << endl;
      show(data,n0,N1*N2);
    }
    
#ifndef OLD
    T.outTransposed(data);
#else  
    fftw_execute(outplan);
#endif  
    if(rank == 0) 
      outcommtime += seconds();
#ifndef OLD
    T.outwait(data);
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
  
  if(N0*N1 < showlimit) {  
    if(rank == 0) cout << "\noriginal:\n" << endl;
    show(data,N0,n1*N2);
  }
  
#ifdef OLD  
  fftw_destroy_plan(inplan);
  fftw_destroy_plan(outplan);
#endif  
  
  MPI_Finalize();
}
