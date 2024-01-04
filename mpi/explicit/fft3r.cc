#include <mpi.h>
#include <Complex.h>
#include <fftw3-mpi.h>
#include <iostream>
#include <iostream>
#include "seconds.h"
#include "utils.h"
#include "timing.h"
#include "cmult-sse2.h"
#include "exmpiutils.h"
#include "../mpiutils.h"

using namespace std;
using namespace utils;
using namespace fftwpp;

void init3r(double *f,
	    unsigned  int local_n0, unsigned  int local_n0_start,
	    unsigned  int N1, unsigned  int N2)
{
  for(unsigned int i=0; i < local_n0; ++i) {
    for(unsigned int j=0; j < N1; ++j) {
      for(unsigned int k=0; k < N2; ++k) {
	f[(i*N1+j)*(2*(N2/2+1))+k]=i+local_n0_start+j+k+1;
      }
    }
  }
}

void show3r(const double *f, unsigned int local_n0, unsigned int N1,
            unsigned int N2)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  for(int r=0; r < size; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(r == rank) {
      cout << "process " << r << endl;
      for (unsigned int i=0; i < local_n0; ++i) {
	for (unsigned int j=0; j < N1; ++j) {
	  for (unsigned int k=0; k < N2; ++k) {
	    cout << f[(i*N1+j)*(2*(N2/2+1))+k] << " ";
	  }
	  cout << endl;
	}
	cout << endl;
      }
    }
  }
}

void show3c(const fftw_complex *F, unsigned int N0, unsigned int N1,
            unsigned int N2p)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  for(int r=0; r < size; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(r == rank) {
      cout << "process " << r << endl;
      double *Fd=(double*) F;
      for (unsigned int i=0; i < N0; ++i) {
  	for (unsigned int j=0; j < N1; ++j) {
  	  for (unsigned int k=0; k < N2p; ++k) {
  	    unsigned int pos=2*(i*N1*N2p+j*N2p+k);
	    cout << "(" << Fd[pos] << ","  << Fd[pos+1] << ") ";
	  }
	  cout << endl;
	}
      }
    }
  }
}

int main(int argc, char **argv)
{
  int N=0;
  int m=4;
  int nthreads=1; // Number of threads
  int stats=0;
#ifdef __GNUC__
  optind=0;
#endif
  for (;;) {
    int c=getopt(argc,argv,"N:m:T:S:e");
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

  const unsigned int m0=m;
  const unsigned int m1=m;
  const unsigned int m2=m;
  const unsigned int N0=m0;
  const unsigned int N1=m1;
  const unsigned int N2=m2;

  const unsigned int N2p=N2/2+1;

  if(N == 0) {
    unsigned int N0=1000000;
    N=N0/m0/m1/m2;
    if(N < 20) N=20;
  }

  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
  if(provided < MPI_THREAD_FUNNELED)
    nthreads=1;
  fftw_init_threads();
  fftw_mpi_init();
  fftw_plan_with_nthreads(nthreads);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // local data sizes
  ptrdiff_t local_n0,local_0_start;
  ptrdiff_t local_n1,local_1_start;
  ptrdiff_t alloc_local=
    fftw_mpi_local_size_3d_transposed(N0,N1,N2p,MPI_COMM_WORLD,
                                      &local_n0,&local_0_start,
                                      &local_n1,&local_1_start);

  fftw_complex *F=(fftw_complex *) ComplexAlign(alloc_local);
  double* f=(double *) F;

  fftw_plan rcplan=fftw_mpi_plan_dft_r2c_3d(N0,N1,N2,f,F,MPI_COMM_WORLD,
					    FFTW_MEASURE |
					    FFTW_MPI_TRANSPOSED_OUT);

  fftw_plan crplan=fftw_mpi_plan_dft_c2r_3d(N0,N1,N2,F,f,MPI_COMM_WORLD,
					    FFTW_MEASURE |
					    FFTW_MPI_TRANSPOSED_IN);

  unsigned int outlimit=3000;

  Transpose *TXyZ,*TyXZ;
  TXyZ=new Transpose(N0,local_n1,N2p,F,F,nthreads);
  TyXZ=new Transpose(local_n1,N0,N2p,F,F,nthreads);

  if(N0*N1*N1 < outlimit) {
    if(rank == 0)
      cout << "input:" << endl;
    init3r(f,local_n0,local_0_start,N1,N2);
    show3r(f,local_n0,N1,N2);

    fftw_mpi_execute_dft_r2c(rcplan,f,F);
    TyXZ->transpose(F);

    if(rank == 0)
      cout << "output:" << endl;
    show3c(F,N0,local_n1,N2p);

    TXyZ->transpose(f);
    fftw_mpi_execute_dft_c2r(crplan,F,f);

    double norm=1.0/(N0*N1*N2);
    for(unsigned int i=0; i < local_n0; ++i)
      for(unsigned int j=0; j < N1; ++j)
        for(unsigned int k=0; k < N2; ++k)
          f[(i*N1+j)*(2*(N2/2+1))+k] *= norm;

    if(rank == 0)
      cout << "back to input:" << endl;
    show3r(f,local_n0,N1,N2);
  }

  if(N > 0) {
    double *T=new double[N];
    for(int i=0; i < N; ++i) {
      init3r(f,local_n0,local_0_start,N1,N2);
      cpuTimer c;
      fftw_mpi_execute_dft_r2c(rcplan,f,F);
      TyXZ->transpose(F);
      TXyZ->transpose(F);
      fftw_mpi_execute_dft_c2r(crplan,F,f);
      T[i]=0.5*c.seconds();
    }
    if(rank == 0)
      timings("FFT",m,T,N,stats);
    delete[] T;
  }

  deleteAlign(F);

  fftw_destroy_plan(rcplan);
  fftw_destroy_plan(crplan);

  MPI_Finalize();
}
