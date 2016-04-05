#include <mpi.h>
#include <complex.h>
#include <fftw3-mpi.h>
#include <iostream>
#include "getopt.h"
#include "seconds.h"
#include "timing.h"
#include "cmult-sse2.h"
#include "../mpiutils.h"

#ifdef __SSE2__
namespace fftwpp {
  const union uvec sse2_pm = {
    { 0x00000000,0x00000000,0x00000000,0x80000000 }
  };
  const union uvec sse2_mm = {
    { 0x00000000,0x80000000,0x00000000,0x80000000 }
  };
}
#endif


using namespace std;
using namespace utils;
using namespace fftwpp;

// compile with
// mpicxx -o cconv2 cconv2.cc -lfftw3_mpi -lfftw3 -lm

void init(fftw_complex* f, unsigned int N0, unsigned int N1,
	  unsigned int m0, unsigned int m1,
	  unsigned int local_0_start, unsigned int local_n0) 
{
  // Load everything with zeros.
  for(unsigned int i=0; i < local_n0; ++i) {
    for(unsigned int j=0; j < N1; j++) {
      f[i * N1 + j] = 0.0;
    }
  }

  unsigned int local_0_stop = local_0_start + local_n0;
  for(unsigned int ii=local_0_start; ii < local_0_stop; ++ii) {
    if(ii < m0) {
      // The logical index:
      unsigned int i = ii - local_0_start; 
      for(unsigned int j=0; j < m1; j++) {
	f[i*N1+j]=ii + I * j;
      }
    }
  }
}

void unpad_local(const fftw_complex* f, fftw_complex* f_nopad,
		unsigned int N0, unsigned int N1,
		unsigned int m0, unsigned int m1,
		unsigned int local_0_start, unsigned int local_n0)
{
  unsigned int local_0_stop = local_0_start + local_n0;
  for(unsigned int ii=local_0_start; ii < local_0_stop; ++ii) {
    if(ii < m0) {
      unsigned int i = ii - local_0_start; 
      for(unsigned int j=0; j < m1; j++) {
	f_nopad[i * m1 + j] = f[i*N1+j];
      }
    }
  }
}

void convolve(fftw_complex *f, fftw_complex *g, double norm,
	      int num, fftw_plan fplan, fftw_plan iplan) 
{
  fftw_mpi_execute_dft(fplan,f,f);
  fftw_mpi_execute_dft(fplan,g,g);

#ifdef __SSE2__
  Complex *F = (Complex *) f;
  Complex *G = (Complex *) g;
  Vec Ninv=LOAD(norm);
  for (int k = 0; k < num; ++k)
    STORE(F+k,Ninv*ZMULT(LOAD(F+k),LOAD(G+k)));
#else
  for (int k = 0; k < num; ++k)
    f[k] *= g[k]*norm;
#endif
  
  fftw_mpi_execute_dft(iplan,f,f);
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
  const unsigned int N0 = 2*m0;
  const unsigned int N1 = 2*m1;
  
  if(N == 0) {
    unsigned int N0=1000000;
    N=N0/m0/m1;
    if(N < 20) N=20;
  }

  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  
  threads_ok = provided >= MPI_THREAD_FUNNELED;
    
  if(threads_ok)
    threads_ok = fftw_init_threads();

  fftw_mpi_init();

  if(threads_ok) 
    fftw_plan_with_nthreads(threads);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(threads_ok && rank == 0) {
    cout << "Threads ok!" << endl;
  }
  
  /* get local data size and allocate */
  ptrdiff_t local_n0, local_0_start;
  ptrdiff_t alloc_local = fftw_mpi_local_size_2d(N0,N1,MPI_COMM_WORLD,
						 &local_n0, &local_0_start);
  fftw_complex *f=fftw_alloc_complex(alloc_local);
  fftw_complex *g=fftw_alloc_complex(alloc_local);
  
  /* create plan for in-place DFT */
  fftw_plan fplan=fftw_mpi_plan_dft_2d(N0,N1,f,f,MPI_COMM_WORLD,
				       FFTW_FORWARD,
				       FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT);
  fftw_plan iplan=fftw_mpi_plan_dft_2d(N0,N1,f,f,MPI_COMM_WORLD,
				       FFTW_BACKWARD,
				       FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN);

  // determine number of elements per process after tranpose
  ptrdiff_t local_n1, local_1_start;
  unsigned int transize=
    fftw_mpi_local_size_2d_transposed(N0,N1,MPI_COMM_WORLD,
   				      &local_n0, &local_0_start,
   				      &local_n1, &local_1_start);

  unsigned int outlimit = 256;

  double overN=1.0/((double) (N0*N1));
  
  if(m0 * m1 < outlimit) {
    init(f, N0, N1, m0, m1, local_0_start, local_n0);
    init(g, N0, N1, m0, m1, local_0_start, local_n0);
    Complex* f_nopad = new Complex[m0 * m1];
    Complex* g_nopad = new Complex[m0 * m1];
    unpad_local(f, (fftw_complex*) f_nopad,
		N0, N1, m0, m1, local_0_start, local_n0);
    unpad_local(g, (fftw_complex*) g_nopad,
		N0, N1, m0, m1, local_0_start, local_n0);

    unsigned int local_0_end = local_0_start + local_n0;
    unsigned int local_m0 = local_0_start <= m0 ? local_0_end - m0 : 0;
    
    if(rank == 0)
      cout << "input f:" << endl;
    show(f_nopad, m0, m1, 0, 0, local_m0, m1, MPI_COMM_WORLD);
    if(rank == 0)
      cout << "input g:" << endl;
    show(g_nopad, m0, m1, 0, 0, local_m0, m1, MPI_COMM_WORLD);

    convolve(f,g,overN,transize,fplan,iplan);
    
    unpad_local(f, (fftw_complex*) f_nopad,
		N0, N1, m0, m1, local_0_start, local_n0);
    if(rank == 0)
      cout << "output:" << endl;
    show(f_nopad, m0, m1, 0, 0, local_m0, m1, MPI_COMM_WORLD);
    delete[] g_nopad;
    delete[] f_nopad;
  }

  if(N > 0) {
    double *T=new double[N];
  
    for(int i=0; i < N; ++i) {
      init(f, N0, N1, m0, m1, local_0_start, local_n0);
      init(g, N0, N1, m0, m1, local_0_start, local_n0);
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
