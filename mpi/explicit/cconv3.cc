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

// compile with
// mpicxx -o cconv3 cconv3.cc -lfftw3_mpi -lfftw3 -lm

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
  const unsigned int m2=m;
  const unsigned int N0 = 2*m0;
  const unsigned int N1 = 2*m1;
  const unsigned int N2 = 2*m2;

  
  if(N == 0) {
    unsigned int N0=1000000;
    N=N0/m0/m1/m2;
    if(N < 20) N=20;
  }
  
  
  MPI_Init(&argc, &argv);
  fftw_mpi_init();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  /* get local data size and allocate */
  ptrdiff_t local_n0;
  ptrdiff_t local_0_start;
  ptrdiff_t alloc_local = fftw_mpi_local_size_3d(N0,N1,N2,MPI_COMM_WORLD,
						 &local_n0, &local_0_start);
  
  ptrdiff_t local_n1, local_1_start;
  int transize=fftw_mpi_local_size_3d_transposed(N0,N1,N2,MPI_COMM_WORLD,
						 &local_n0,&local_0_start,
						 &local_n1,&local_1_start);
  fftw_complex *f=fftw_alloc_complex(alloc_local);
  fftw_complex *g=fftw_alloc_complex(alloc_local);
  
  /* create plan for in-place DFT */
  fftw_plan fplan=fftw_mpi_plan_dft_3d(N0,N1,N2,f,f,MPI_COMM_WORLD,
				       FFTW_FORWARD,
				       FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN);
  fftw_plan iplan=fftw_mpi_plan_dft_3d(N0,N1,N2,f,f,MPI_COMM_WORLD,
				       FFTW_BACKWARD,
				       FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT);

  initf(f,local_0_start,local_n0,N0,N1,N2,m0,m1,m2);
  initg(g,local_0_start,local_n0,N0,N1,N2,m0,m1,m2);


  //show(f,local_0_start,local_n0,N1,N2,m0,m1,m2,0);
  //show(f,local_0_start,local_n0,N1,N2,N0,N1,N2,1);
  //show(g,local_0_start,local_n0,N1,N2,m0,m1,m2,1);

  double *T=new double[N];
  for(int i=0; i < N; ++i) {
    initf(f,local_0_start,local_n0,N0,N1,N2,m0,m1,m2);
    initg(g,local_0_start,local_n0,N0,N1,N2,m0,m1,m2);
    double overN=1.0/((double) (N0*N1*N2));
    seconds();
    convolve(f,g,overN,transize,fplan,iplan);
    T[i]=seconds();
  }  

  if(rank == 0) timings("Explicit",m,T,N,stats);
  
  if(m0*m1*m2<100) {
    if(rank == 0) cout << "output:" << endl;
    show(f,local_0_start,local_n0,N1,N2,m0,m1,m2,2);
  }

  fftw_destroy_plan(fplan);
  fftw_destroy_plan(iplan);

  MPI_Finalize();

  return 0;
}
