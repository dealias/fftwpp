#include <mpi.h>
#include <complex.h>
#include <fftw3-mpi.h>
#include <iostream>
#include "getopt.h"
#include "seconds.h"
#include "timing.h"
#include <stdlib.h>
#include "Complex.h"
#include "cmult-sse2.h"
#include "exmpiutils.h"

using namespace std;
#ifdef __SSE2__
using namespace fftwpp;

namespace fftwpp {
  const union uvec sse2_pm = {
    { 0x00000000,0x00000000,0x00000000,0x80000000 }
  };
  const union uvec sse2_mm = {
    { 0x00000000,0x80000000,0x00000000,0x80000000 }
  };
}
#endif

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
      case 'N':
        N=atoi(optarg);
        break;
      case 'm':
        m=atoi(optarg);
        break;
    }
  }

  const unsigned int m0 = m, m1 = m, m2=m;
  const unsigned int N0 = 2*m0, N1 = 2*m1, N2 = 2*m2;
  fftw_plan fplan, iplan;
  fftw_complex *f, *g;
  ptrdiff_t alloc_local, local_n0, local_0_start;
  
  MPI_Init(&argc, &argv);
  fftw_mpi_init();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  /* get local data size and allocate */
  alloc_local = fftw_mpi_local_size_3d(N0,N1,N2,MPI_COMM_WORLD,
				       &local_n0, &local_0_start);
  
  ptrdiff_t local_n1, local_1_start;
  int transize=fftw_mpi_local_size_3d_transposed(N0,N1,N2,MPI_COMM_WORLD,
						 &local_n0,&local_0_start,
						 &local_n1,&local_1_start);
  f=fftw_alloc_complex(alloc_local);
  g=fftw_alloc_complex(alloc_local);
  
  /* create plan for in-place DFT */
  fplan=fftw_mpi_plan_dft_3d(N0,N1,N2,f,f,MPI_COMM_WORLD,FFTW_FORWARD,
			     FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN);
  iplan=fftw_mpi_plan_dft_3d(N0,N1,N2,f,f,MPI_COMM_WORLD,FFTW_BACKWARD,
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

  if(rank == 0) timings("Explicit",m,T,N);
  
  if(m0*m1*m2<100) {
    if(rank == 0) cout << "output:" << endl;
    show(f,local_0_start,local_n0,N1,N2,m0,m1,m2,2);
  }

  fftw_destroy_plan(fplan);
  fftw_destroy_plan(iplan);

  MPI_Finalize();

  return 0;
}
