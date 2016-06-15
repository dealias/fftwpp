#include <mpi.h>
#include "Complex.h"
#include <fftw3-mpi.h>
#include <iostream>
#include "getopt.h"
#include "seconds.h"
#include "timing.h"
#include "cmult-sse2.h"
#include "exmpiutils.h"
#include "../mpiutils.h"

using namespace std;
using namespace utils;
using namespace fftwpp;

void init(Complex* f, unsigned int N0, unsigned int N1,
	  unsigned int local_0_start, unsigned int local_n0) 
{
  for(unsigned int i=0; i < local_n0; ++i) {
    unsigned int ii=local_0_start+i;
    for(unsigned int j=0; j < N1; j++) {
      f[i*N1+j]=ii + I * j;
    }
  }
}

int main(int argc, char **argv)
{
  int N=4;
  int m=4;
  int nthreads=1; // Number of threads
  int stats=0;
#ifdef __GNUC__	
  optind=0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"N:m:T:S:e");
    if (c == -1) break;
    
    switch (c) {
      case 0:
        break;
      case 'N':
        N=atoi(optarg);
        break;
      case 'm':
        m=atoi(optarg);
        break;
      case 'T':
        nthreads=atoi(optarg);
        break;
      case 'S':
        stats=atoi(optarg);
        break;
      case 'e':
        // For compatibility with timing scripts
        break;
    }
  }

  unsigned int m0 = m;
  unsigned int m1 = m;

  if(N == 0) {
    unsigned int N0=1000000;
    N=N0/m0/m1;
    if(N < 20) N=20;
  }
  
  const unsigned int N0 = m0;
  const unsigned int N1 = m1;
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int threads_ok=provided >= MPI_THREAD_FUNNELED;
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(threads_ok)
    fftw_init_threads();
  fftw_mpi_init();
  
  if(threads_ok)
    fftw_plan_with_nthreads(nthreads);
  else 
    if(nthreads > 1 && rank == 0) cout << "threads not ok!" << endl;
  
  /* get local data size and allocate */
  ptrdiff_t local_n0, local_0_start;
  ptrdiff_t alloc_local=fftw_mpi_local_size_2d(N0,N1,MPI_COMM_WORLD,
					       &local_n0, &local_0_start);
  
  Complex *f=ComplexAlign(alloc_local);
  
  /* create plan for in-place DFT */
  fftw_plan fplan=fftw_mpi_plan_dft_2d(N0,N1,(fftw_complex *) f,
                                       (fftw_complex *) f,MPI_COMM_WORLD,
				       FFTW_FORWARD,
				       FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT);
  fftw_plan iplan=fftw_mpi_plan_dft_2d(N0,N1,(fftw_complex *) f,
                                       (fftw_complex *) f,MPI_COMM_WORLD,
				       FFTW_BACKWARD,
				       FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN);

  unsigned int outlimit = 256;

  if(N0*N1 < outlimit) {
    init(f, N0, N1, local_0_start, local_n0);
    if(rank == 0)
      cout << "input:" << endl;
    show((Complex *)f, N0, N1, 0, 0, local_n0, N1, MPI_COMM_WORLD);

    fftw_mpi_execute_dft(fplan,(fftw_complex *) f,(fftw_complex *) f);

    // determine number of elements per process after tranpose
    ptrdiff_t local_n1, local_1_start;
    fftw_mpi_local_size_2d_transposed(N0,N1,MPI_COMM_WORLD,
				      &local_n0, &local_0_start,
				      &local_n1, &local_1_start);
    if(rank == 0)
      cout << "output:" << endl;
    show(f, N1, N0, 0, 0, local_n1, N0, MPI_COMM_WORLD);

    fftw_mpi_execute_dft(iplan,(fftw_complex *) f,(fftw_complex *) f);
    if(rank == 0)
      cout << "back to input:" << endl;
    show(f, N0, N1, 0, 0, local_n0, N1, MPI_COMM_WORLD);
  }
    
  if(N > 0) {
    double *T=new double[N];
    for(int i=0; i < N; ++i) {
      init(f, N0, N1, local_0_start, local_n0);
      seconds();
      fftw_mpi_execute_dft(fplan,(fftw_complex *) f,(fftw_complex *) f);
      fftw_mpi_execute_dft(iplan,(fftw_complex *) f,(fftw_complex *) f);
      T[i]=0.5*seconds();
    }  
    if(rank == 0)
      timings("FFT",m,T,N,stats);
    delete[] T;
  }

  fftw_destroy_plan(fplan);
  fftw_destroy_plan(iplan);

  MPI_Finalize();
}
