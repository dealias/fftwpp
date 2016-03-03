#include <mpi.h>
#include <complex.h>
#include <fftw3-mpi.h>
#include <iostream>
#include "getopt.h"
#include "seconds.h"
#include "timing.h"
#include "exmpiutils.h"

using namespace std;
using namespace utils;
using namespace fftwpp;

// compile with
// mpicxx -o conv2 conv2.cc -lfftw3_mpi -lfftw3 -lm


void init2c(fftw_complex *F, const int local_n0, const int local_n0_start,
	    const int N1p,
	    const int m0, const int m1p)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  for(int r = 0; r < size; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(r == rank) {
      double *Fd = (double*) F;
      for (int i = 0; i < local_n0; ++i) {
	int ii = i + local_n0_start;
	for (int j = 0; j < N1p; ++j) {
	  int pos = 2 * (i * N1p + j);
	  if(ii < m0 && j < m1p) {
	    Fd[pos] = ii;
	    Fd[pos + 1] = j;
	  } else {
	    Fd[pos] = 0.0;
	    Fd[pos + 1] = 0.0;
	  }
	}
      }
    }
  }
}

void show2c(const fftw_complex *F, const int local_n0, const int local_n0_start,
	    const int N1p,  const int m0, const int m1p)
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
	int ii = i + local_n0_start;
	for (int j = 0; j < N1p; ++j) {
	  if(ii < m0 && j < m1p) {
	    int pos = 2 * (i * N1p + j);
	    cout << "(" << Fd[pos] << ","  << Fd[pos + 1] << ") ";
	  }
	}
	if(ii < m0)
	  cout << endl;
      }
    }
  }
}

void convolve(fftw_complex *F, fftw_complex *G, 
	      double *f, double *g, double norm,
	      int num, fftw_plan rcplan, fftw_plan crplan) 
{
  // FIXME: these have to be shifted.
    
  fftw_mpi_execute_dft_c2r(crplan,F,f);
  fftw_mpi_execute_dft_c2r(crplan,G,g);

  for (int k = 0; k < num; ++k)
    f[k] *= g[k]*norm;
  
  fftw_mpi_execute_dft_r2c(rcplan,f,F);
}

int main(int argc, char **argv)
{
  int N=0;
  int m=4;
  int stats=0;
#ifdef __GNUC__	
  optind=0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"N:m:S:");
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
      case 'S':
        stats=atoi(optarg);
        break;
    }
  }

  const unsigned int m0 = m;
  const unsigned int m1 = m;
  const unsigned int N0 = m0 * 3 / 2;
  const unsigned int N1 = m1 * 3 / 2;

  const unsigned int m1p = m1 / 2 + 1;
  const unsigned int N1p = N1 / 2 + 1;
  
  if(N == 0) {
    unsigned int N0=1000000;
    N=N0/m0/m1;
    if(N < 20) N=20;
  }
  
  MPI_Init(&argc, &argv);
  fftw_mpi_init();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  cout << "m0: " << m0 << "\tm1: " << m1  << "\tm1p: " << m1p << endl;
  cout << "N0: " << N0 << "\tN1: " << N1  << "\tN1p: " << N1p << endl;
  
  /* get local data size and allocate */
  ptrdiff_t local_n0, local_n0_start;
  ptrdiff_t alloc_local = fftw_mpi_local_size_2d(N0,N1,MPI_COMM_WORLD,
						 &local_n0, &local_n0_start);
   
  double* f=fftw_alloc_real(2 * alloc_local);
  fftw_complex* F=fftw_alloc_complex(alloc_local);

  double* g=fftw_alloc_real(2 * alloc_local);
  fftw_complex* G=fftw_alloc_complex(alloc_local);
  
  /* create plan for in-place DFT */
  fftw_plan rcplan=fftw_mpi_plan_dft_r2c_2d(N0, N1, f, F, MPI_COMM_WORLD,
					    FFTW_MEASURE);
  
  fftw_plan crplan=fftw_mpi_plan_dft_c2r_2d(N0, N1, F, f, MPI_COMM_WORLD,
					    FFTW_MEASURE);

  init2c(F, local_n0, local_n0_start, N1p, m0, m1p);
  init2c(G, local_n0, local_n0_start, N1p, m0, m1p);
  
  cout << "Input F:" << endl;
  show2c(F, local_n0, local_n0_start, N1p, m0, m1p);
  cout << "Input G:" << endl;
  show2c(G, local_n0, local_n0_start, N1p, m0, m1p);

  double *T=new double[N];

  double overN=1.0/((double) (N0*N1));
  for(int i=0; i < N; ++i) {
    init2c(F, local_n0, local_n0_start, N1p, m0, m1p);
    init2c(G, local_n0, local_n0_start, N1p, m0, m1p);
    seconds();
    convolve(F, G, f, g, overN, alloc_local, rcplan, crplan);
    T[i]=seconds();
  }  

  if(rank == 0) {
    timings("Explicit",m,T,N,stats);
  }
    
  if(m0 * m1 < 1000) {
    if(rank == 0) cout << "output:" << endl;
    show2c(F, local_n0, local_n0_start, N1p, m0, m1p);
  }

  fftw_destroy_plan(crplan);
  fftw_destroy_plan(rcplan);

  MPI_Finalize();

  return 0;
}
