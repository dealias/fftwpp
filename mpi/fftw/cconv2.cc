#include<complex.h>
#include <fftw3-mpi.h>
#include <iostream>
#include "getopt.h"
#include "seconds.h"
#include "timing.h"
#include <stdlib.h>
#include "Complex.h"
#include "cmult-sse2.h"

using namespace std;
using namespace fftwpp;

namespace fftwpp {
#ifdef __SSE2__
  const union uvec sse2_pm = {
    { 0x00000000,0x00000000,0x00000000,0x80000000 }
  };
  
  const union uvec sse2_mm = {
    { 0x00000000,0x80000000,0x00000000,0x80000000 }
  };
#endif
}

// compile with
// mpicxx -o cconv2 cconv2.cc -lfftw3_mpi -lfftw3 -lm

void show(fftw_complex *f, int local_0_start, int local_n0, 
	  int N1, int m0, int m1, int A)
{
  int stop=local_0_start+local_n0;
  for (int i = local_0_start; i < stop; ++i) {
    if(i < m0) {
      int ii=i-local_0_start;
      cout << A << i << ": ";
      for (int j = 0; j < m1; ++j) {
	cout << "("<<creal(f[ii*N1 + j])<<","<<cimag(f[ii*N1 + j]) << ")  ";
      }
      cout << endl;
    }
  }
} 

void init(fftw_complex *f, fftw_complex *g, int local_0_start, int local_n0, 
	  int N0, int N1, int m0, int m1)
{
  int stop=local_0_start+local_n0;
  for (int i = local_0_start; i < stop; ++i) {
    if(i < m0) {
      int ii=i-local_0_start;
      for (int j = 0; j < m1; ++j) {
	f[ii*N1+j]=i +j*I;
	g[ii*N1+j]=2*i +(j+1)*I;

	// f[ii*N1+j]=i*I;
	// g[ii*N1+j]=(i == 0 && j == 0) ? 1.0 : 0.0; //i*I;
      }
      for (int j = m1; j < N1; ++j) {
	f[ii*N1+j]=0.0;
	g[ii*N1+j]=0.0;
      }
    }
  }
  
  for (int i = 0; i < local_n0; ++i) {
    if(i+local_0_start >= m0) {
      for (int j = 0; j < N1; ++j) {
	f[i*N1+j]=0.0;
	g[i*N1+j]=0.0;
      }
    }
  }

}

void convolve(fftw_complex *f, fftw_complex *g, double norm,
	      int num, fftw_plan fplan, fftw_plan iplan) 
{
  fftw_mpi_execute_dft(fplan,f,f);
  fftw_mpi_execute_dft(fplan,g,g);
  Complex *F = (Complex *) f;
  Complex *G = (Complex *) g;
#ifdef __SSE2__
  Vec Ninv=LOAD(norm);
  for (int k = 0; k < num; ++k)
    STORE(F+k,Ninv*ZMULT(LOAD(F+k),LOAD(G+k)));
#else
  for (int k = 0; k < num; ++k)
    f[k] *= g[k]*norm;
#endif
  fftw_mpi_execute_dft(iplan,f,f);
}

int main(int argc, char **argv)
{

  int N=4, m=4;
#ifdef __GNUC__	
  optind=0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"N:m:");
    if (c == -1) break;
    
    switch (c) {
      case 0:
        break;
      case 'n':
        N=atoi(optarg);
        break;
      case 'm':
        m=atoi(optarg);
        break;
    }
  }

  const unsigned int m0 = m, m1 = m;
  const unsigned int N0 = 2*m0, N1 = 2*m1;
  fftw_plan fplan, iplan;
  fftw_complex *f, *g;
  ptrdiff_t alloc_local, local_n0, local_0_start;
  
  MPI_Init(&argc, &argv);
  fftw_mpi_init();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  /* get local data size and allocate */
  alloc_local = fftw_mpi_local_size_2d(N0,N1,MPI_COMM_WORLD,
				       &local_n0, &local_0_start);
  f=fftw_alloc_complex(alloc_local);
  g=fftw_alloc_complex(alloc_local);
  
  /* create plan for in-place DFT */
  fplan=fftw_mpi_plan_dft_2d(N0,N1,f,f,MPI_COMM_WORLD,FFTW_FORWARD,
			     FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN);
  iplan=fftw_mpi_plan_dft_2d(N0,N1,f,f,MPI_COMM_WORLD,FFTW_BACKWARD,
			     FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT);

  
  init(f,g,local_0_start,local_n0,N0,N1,m0,m1);

  // show(f,local_0_start,local_n0,N1,m0,m1,0);
  // show(f,local_0_start,local_n0,N1,m0,m1,0);
  //show(g,local_0_start,local_n0,N1,N0,N1,1);

  // determine number of elements per process after tranpose
  ptrdiff_t local_n1, local_1_start;
  unsigned int transize=
    fftw_mpi_local_size_2d_transposed(N0,N1,MPI_COMM_WORLD,
   				      &local_n0, &local_0_start,
   				      &local_n1, &local_1_start);
  
  double *T=new double[N];

  for(int i=0; i < N; ++i) {
    init(f,g,local_0_start,local_n0,N0,N1,m0,m1);
    double overN=1.0/((double) (N0*N1));
    seconds();
    convolve(f,g,overN,transize,fplan,iplan);
    T[i]=seconds();
  }  

  if(rank == 0) timings("Explicit",m,T,N);
  
  if(m0*m1<100) {
    if(rank == 0) cout << "output:" << endl;
    show(f,local_0_start,local_n0,N1,m0,m1,2);
  }
  fftw_destroy_plan(fplan);
  fftw_destroy_plan(iplan);

  MPI_Finalize();

  return 0;
}
