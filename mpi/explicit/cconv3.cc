#include <mpi.h>
#include "Complex.h"
#include <fftw3-mpi.h>
#include <iostream>
#include "getopt.h"
#include "seconds.h"
#include "timing.h"
#include "exmpiutils.h"
#include "cmult-sse2.h"
#include "../mpiutils.h"

using namespace std;
using namespace utils;
using namespace fftwpp;

// compile with
// mpicxx -o cconv3 cconv3.cc -lfftw3_mpi -lfftw3 -lm

void init(Complex *f,
	  unsigned int N0, unsigned int N1, unsigned int N2,
	  unsigned int m0, unsigned int m1, unsigned int m2,
	  unsigned int local_0_start, unsigned int local_n0) 
{
  // Load everything with zeros.
  for(unsigned int i=0; i < local_n0; ++i) {
    for(unsigned int j=0; j < N1; j++) {
      for(unsigned int k=0; k < N2; k++) {
	f[i * N1 * N2 + j * N2 + k] = 0.0;
      }
    }
  }

  unsigned int local_0_stop = local_0_start + local_n0;
  for(unsigned int ii=local_0_start; ii < local_0_stop; ++ii) {
    if(ii < m0) {
      // The logical index:
      unsigned int i = ii - local_0_start; 
      for(unsigned int j=0; j < m1; j++) {
	for(unsigned int k=0; k < m2; k++) {
	  f[i * N1 * N2 + j * N2 + k] = ii + k*k + I*j ;
	}
      }
    }
  }
}

void unpad_local(const Complex *f, Complex *f_nopad,
		 unsigned int N1, unsigned int N2,
		 unsigned int local_m0, unsigned int m1,unsigned int m2)
{
  for(unsigned int i=0; i < local_m0; ++i) {
    for(unsigned int j=0; j < m1; j++) {
      for(unsigned int k=0; k < m2; k++) {
	f_nopad[i * m1 * m2  + j * m2 + k] = f[i * N1 * N2 + j * N2 + k];
      }
    }
  }
}

void convolve(Complex *f, Complex *g, double norm,
	      int num, fftw_plan fplan, fftw_plan iplan) 
{
  fftw_mpi_execute_dft(fplan,(fftw_complex *) f,(fftw_complex *) f);
  fftw_mpi_execute_dft(fplan,(fftw_complex *) g,(fftw_complex *) g);
  Complex *F=f;
  Complex *G=g;
#ifdef __SSE2__
  Vec Ninv=LOAD(norm);
  for (int k = 0; k < num; ++k)
    STORE(F+k,Ninv*ZMULT(LOAD(F+k),LOAD(G+k)));
#else
  for (int k = 0; k < num; ++k)
    f[k] *= g[k]*norm;
#endif
  fftw_mpi_execute_dft(iplan,(fftw_complex *) f,(fftw_complex *) f);
}

int threads_ok;

int main(int argc, char **argv)
{
  int threads=1;
  int N=0;
  int m=4;
  int stats=0;

#ifdef __GNUC__	
  optind=0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"N:m:S:T:e");
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
      case 'T':
        threads=atoi(optarg);
        break;
      case 'e':
	// For compatibility reasons with -e option in OpenMP version.
	break;
    }
  }

  const unsigned int m0 = m;
  const unsigned int m1 = m;
  const unsigned int m2=m;
  const unsigned int N0 = 2*m0;
  const unsigned int N1 = 2*m1;
  const unsigned int N2 = 2*m2;

  if(N == 0) {
    unsigned int N0=1000000;
    N=N0/m0/m1/m2;
    if(N < 20) N=20;
  }
  
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  
  threads_ok=provided >= MPI_THREAD_FUNNELED;
    
  if(threads_ok)
    fftw_init_threads();

  fftw_mpi_init();

  if(threads_ok) 
    fftw_plan_with_nthreads(threads);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(threads_ok && rank == 0) {
    cout << "Threads ok!" << endl;
  }
  
  
  /* get local data size and allocate */
  ptrdiff_t local_n0;
  ptrdiff_t local_0_start;
  ptrdiff_t alloc_local = fftw_mpi_local_size_3d(N0,N1,N2,MPI_COMM_WORLD,
						 &local_n0, &local_0_start);
  
  ptrdiff_t local_n1, local_1_start;
  int transize=fftw_mpi_local_size_3d_transposed(N0,N1,N2,MPI_COMM_WORLD,
						 &local_n0,&local_0_start,
						 &local_n1,&local_1_start);
  Complex *f=ComplexAlign(alloc_local);
  Complex *g=ComplexAlign(alloc_local);
  
  /* create plan for in-place DFT */
  fftw_plan fplan=fftw_mpi_plan_dft_3d(N0,N1,N2,(fftw_complex *) f,
                                       (fftw_complex *) f,MPI_COMM_WORLD,
				       FFTW_FORWARD,
				       FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT);
  fftw_plan iplan=fftw_mpi_plan_dft_3d(N0,N1,N2,(fftw_complex *) f,
                                       (fftw_complex *) f,MPI_COMM_WORLD,
				       FFTW_BACKWARD,
				       FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN);
  double overN=1.0/((double) (N0*N1*N2));
  unsigned int outlimit = 1000;

  if(N0*N1*N2 < outlimit) {
    init(f,N0,N1,N2, m0,m1,m2,local_0_start,local_n0);
    init(g,N0,N1,N2,m0,m1,m2,local_0_start,local_n0);

    int local_m0 =
      (local_0_start < m0) ? m0 - local_0_start : 0;
    if(local_m0 > local_n0)
      local_m0 = local_n0;
    
    unsigned int n_nopad = local_m0 * m1 * m2;
    Complex *f_nopad = n_nopad > 0 ? ComplexAlign(n_nopad) : NULL;
    Complex *g_nopad = n_nopad > 0 ? ComplexAlign(n_nopad) : NULL;
    
    if(rank == 0)
      cout << "input f:" << endl;
    unpad_local(f, f_nopad, N1, N2, local_m0, m1, m2);
    show(f_nopad, local_m0, m1, m2, MPI_COMM_WORLD);

    if(rank == 0)
      cout << "input g:" << endl;
    unpad_local(g, g_nopad, N1, N2, local_m0, m1, m2);
    show(g_nopad, local_m0, m1, m2, MPI_COMM_WORLD);

    convolve(f,g,overN,transize,fplan,iplan);

    if(rank == 0)
      cout << "output f:" << endl;
    unpad_local(f, f_nopad, N1, N2, local_m0, m1, m2);
    show(f_nopad, local_m0, m1, m2, MPI_COMM_WORLD);

    if(n_nopad > 0) {
      delete[] f_nopad;
      delete[] g_nopad;
    }
      
  }
  
  if(N > 0) {
    double *T=new double[N];
    for(int i=0; i < N; ++i) {
      init(f,N0,N1,N2, m0,m1,m2,local_0_start,local_n0);
      init(g,N0,N1,N2,m0,m1,m2,local_0_start,local_n0);
      seconds();
      convolve(f,g,overN,transize,fplan,iplan);
      T[i]=seconds();
    }  
    if(rank == 0)
      timings("Explicit",m,T,N,stats);
  }

  fftw_destroy_plan(fplan);
  fftw_destroy_plan(iplan);

  MPI_Finalize();

  return 0;
}
