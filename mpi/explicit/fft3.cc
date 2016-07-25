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


void init(Complex* f, unsigned int N0, unsigned int N1, unsigned int N2,
	  unsigned int local_0_start, unsigned int local_n0) 
{
  unsigned int c=0;
  for(unsigned int i=0; i < local_n0; ++i) {
    unsigned int ii=local_0_start+i;
    for(unsigned int j=0; j < N1; j++) {
      for(unsigned int k=0; k < N2; k++) {
        f[c++]=Complex(10*k+ii,j);
      }
    }
  }
}

int main(int argc, char **argv)
{
  int N=0;
  int m=4;
  int stats=0;
  int nthreads=1; // Number of threads
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
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  // local data sizes
  ptrdiff_t local_n0,local_0_start;
  ptrdiff_t local_n1,local_1_start;
  ptrdiff_t alloc_local=
    fftw_mpi_local_size_3d_transposed(N0,N1,N2,MPI_COMM_WORLD,
                                      &local_n0,&local_0_start,
                                      &local_n1,&local_1_start);
    
  Complex *f=ComplexAlign(alloc_local);
  
  /* create plan for in-place DFT */
  fftw_plan fplan=fftw_mpi_plan_dft_3d(N0,N1,N2,(fftw_complex *) f,
                                       (fftw_complex *) f,MPI_COMM_WORLD,
				       FFTW_FORWARD,
				       FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT);
  fftw_plan iplan=fftw_mpi_plan_dft_3d(N0,N1,N2,(fftw_complex *) f,
                                       (fftw_complex *) f,MPI_COMM_WORLD,
				       FFTW_BACKWARD,
				       FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN);

  unsigned int outlimit=512;
  
  Transpose *TXyZ,*TyXZ;
  TXyZ=new Transpose(N0,local_n1,N2,f,f,nthreads);
  TyXZ=new Transpose(local_n1,N0,N2,f,f,nthreads);
  
  if(N0*N1*N2 < outlimit) {
    init(f,N0,N1 ,N2,local_0_start,local_n0);
    if(rank == 0)
      cout << "input:" << endl;
    show(f,N0,N1,N2,0,0,0,local_n0,N1,N2,MPI_COMM_WORLD);

    fftw_mpi_execute_dft(fplan,(fftw_complex *) f,(fftw_complex *) f);
    TyXZ->transpose(f);
    
    if(rank == 0)
      cout << "output:" << endl;
    show(f,N0,N1,N2,0,0,0,N0,local_n1,N2,MPI_COMM_WORLD);

    TXyZ->transpose(f);
    fftw_mpi_execute_dft(iplan,(fftw_complex *) f,(fftw_complex *) f);
    
    double norm=1.0/(N0*N1*N2);
    unsigned int c=0;
    for(unsigned int i=0; i < local_n0; ++i)
      for(unsigned int j=0; j < N1; j++)
        for(unsigned int k=0; k < N2; k++)
          f[c++] *= norm;
        
    if(rank == 0)
      cout << "back to input:" << endl;
    show(f,N0,N1,N2,0,0,0,local_n0,N1,N2,MPI_COMM_WORLD);
  }

  if(N > 0) {
    double *T=new double[N];
    for(int i=0; i < N; ++i) {
      init(f,N0,N1,N2,local_0_start,local_n0);
      seconds();
      fftw_mpi_execute_dft(fplan,(fftw_complex *) f,(fftw_complex *) f);
      TyXZ->transpose(f);
      TXyZ->transpose(f);
      fftw_mpi_execute_dft(iplan,(fftw_complex *) f,(fftw_complex *) f);
      T[i]=0.5*seconds();
    }  
    if(rank == 0)
      timings("FFT",m,T,N,stats);
    delete[] T;
  }
  
  deleteAlign(f);
  
  fftw_destroy_plan(fplan);
  fftw_destroy_plan(iplan);

  MPI_Finalize();
}
