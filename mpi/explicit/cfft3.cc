#include <mpi.h>
#include <complex.h>
#include <fftw3-mpi.h>
#include <iostream>
#include "getopt.h"
#include "seconds.h"
#include "timing.h"
#include "cmult-sse2.h"
#include "exmpiutils.h"

using namespace std;
using namespace utils;
using namespace fftwpp;

int main(int argc, char **argv)
{
  int N=4;
  int m=4;
  int stats=0;
  int nthreads=1; // Number of threads
#ifdef __GNUC__	
  optind=0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"N:m:T:S:");
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
    }
  }

  const unsigned int m0 = m;
  const unsigned int m1 = m;
  const unsigned int m2=m;
  const unsigned int N0 = m0;
  const unsigned int N1 = m1;
  const unsigned int N2 = m2;

  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int threads_ok = provided >= MPI_THREAD_FUNNELED;
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(threads_ok) threads_ok = fftw_init_threads();
  fftw_mpi_init();
  
  if(threads_ok)
    fftw_plan_with_nthreads(nthreads);
  else 
    if(rank ==0) cout << "threads not ok!" << endl;
  
  /* get local data size and allocate */
  ptrdiff_t local_n0;
  ptrdiff_t local_0_start;
  ptrdiff_t alloc_local = fftw_mpi_local_size_3d(N0,N1,N2,MPI_COMM_WORLD,
						 &local_n0, &local_0_start);
  fftw_complex *f=fftw_alloc_complex(alloc_local);
  
  /* create plan for in-place DFT */
  fftw_plan fplan=fftw_mpi_plan_dft_3d(N0,N1,N2,f,f,MPI_COMM_WORLD,
				       FFTW_FORWARD,
				       FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN);
  fftw_plan iplan=fftw_mpi_plan_dft_3d(N0,N1,N2,f,f,MPI_COMM_WORLD,
				       FFTW_BACKWARD,
				       FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT);

  double *T=new double[N];
  for(int i=0; i < N; ++i) {
    initf(f,local_0_start,local_n0,N0,N1,m0,m1);
    seconds();
    fftw_mpi_execute_dft(fplan,f,f);
    fftw_mpi_execute_dft(iplan,f,f);
    T[i]=seconds();
  }  
  if(rank == 0)
    timings("FFT",m,T,N,stats);
  delete[] T;

  fftw_destroy_plan(fplan);
  fftw_destroy_plan(iplan);

  MPI_Finalize();
}
