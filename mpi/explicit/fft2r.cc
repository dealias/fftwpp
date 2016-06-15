#include <mpi.h>
#include <Complex.h>
#include <fftw3-mpi.h>
#include <iostream>
#include "getopt.h"
#include "seconds.h"
#include "timing.h"
#include "cmult-sse2.h"

using namespace std;
using namespace utils;
using namespace fftwpp;


void init2r(double *f,
	    const int local_n0, const int local_n0_start,
	    const int N1)
{
  for (int i = 0; i < local_n0; ++i) {
    for (int j = 0; j < N1; ++j) {
      f[i* (2*(N1/2+1)) + j] = 10 * (i + local_n0_start) + j;
    }
  }
}

void show2r(const double *f, const int local_n0, const int N1)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  for(int r = 0; r < size; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(r == rank) {
      cout << "process " << r << endl;
      for (int i = 0; i < local_n0; ++i) {
	for (int j = 0; j < N1; ++j) {
	  cout << f[i* (2*(N1/2+1)) + j] << " ";
	}
	cout << endl;
      }
    }
  }
}

void show2c(const fftw_complex *F, const int local_n0, const int N1p)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  for(int r = 0; r < size; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(r == rank) {
      cout << "process " << r << endl;
      double *Fd = (double*) F;
      for (int i = 0; i < local_n0; ++i) {
	for (int j = 0; j < N1p; ++j) {
	  int pos = 2 * (i * N1p + j);
	  cout << "(" << Fd[pos] << ","  << Fd[pos + 1] << ") ";
	}
	cout << endl;
      }
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
	// For compabitility with timing tests
        break;
    }
  }

  const unsigned int m0 = m;
  const unsigned int m1 = m;
  const unsigned int N0 = m0;
  const unsigned int N1 = m1;

  const unsigned int N1p = N1 / 2 + 1;

  if(N == 0) {
    unsigned int N0=1000000;
    N=N0/m0/m1;
    if(N < 20) N=20;
  }
  
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int threads_ok=provided >= MPI_THREAD_FUNNELED;
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int mpisize;
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

  if(threads_ok)
    fftw_init_threads();
  fftw_mpi_init();
  
  if(threads_ok)
    fftw_plan_with_nthreads(nthreads);
  else 
    if(nthreads > 1 && rank == 0) cout << "threads not ok!" << endl;
  
  /* get local data size and allocate */
  ptrdiff_t local_n0;
  ptrdiff_t local_n0_start;
  ptrdiff_t alloc_local=fftw_mpi_local_size_2d(N0,N1p,MPI_COMM_WORLD,
					       &local_n0, &local_n0_start);
  
  double* f=fftw_alloc_real(2 * alloc_local);
  fftw_complex* F=fftw_alloc_complex(alloc_local);
  
  fftw_plan rcplan=fftw_mpi_plan_dft_r2c_2d(N0, N1, f, F, MPI_COMM_WORLD,
					    FFTW_MEASURE
					    | FFTW_MPI_TRANSPOSED_OUT);
  
  fftw_plan crplan=fftw_mpi_plan_dft_c2r_2d(N0, N1, F, f, MPI_COMM_WORLD,
					    FFTW_MEASURE
					    | FFTW_MPI_TRANSPOSED_IN);

  unsigned int outlimit=3000;
  
  if(N0*N1 < outlimit) {
    if(rank == 0)
      cout << "input:" << endl;
    init2r(f, local_n0, local_n0_start, N1);
    show2r(f, local_n0, N1);
    if(rank == 0)
      cout << "output:" << endl;
    fftw_mpi_execute_dft_r2c(rcplan,f,F);
    show2c(F, local_n0, N1p);
    if(rank == 0)
      cout << "back to input:" << endl;
    fftw_mpi_execute_dft_c2r(crplan,F,f);
    show2r(f, local_n0, N1);
  }

  if(N > 0) {
    double *T=new double[N];
    for(int i=0; i < N; ++i) {
      init2r(f, local_n0, local_n0_start, N1);
      seconds();
      fftw_mpi_execute_dft_r2c(rcplan,f,F);
      fftw_mpi_execute_dft_c2r(crplan,F,f);
      T[i]=0.5*seconds();
    }  
    if(rank == 0)
      timings("FFT",m,T,N,stats);
    delete[] T;
  }
  
  fftw_destroy_plan(rcplan);
  fftw_destroy_plan(crplan);

  MPI_Finalize();
}
