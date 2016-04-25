#include <mpi.h>
#include <Complex.h>
#include <fftw3-mpi.h>
#include <iostream>
#include <stdlib.h>
#include "Complex.h"
#include "exmpiutils.h"
#include "cmult-sse2.h"

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

void show(Complex *f, int local_0_start, int local_n0, 
	  int N1, int m0, int m1, int A)
{
  int stop=local_0_start+local_n0;
  for (int i = local_0_start; i < stop; ++i) {
    if(i < m0) {
      int ii=i-local_0_start;
      std::cout << A << i << ": ";
      for (int j = 0; j < m1; ++j)
	std::cout << f[ii*N1 + j] << " "; 
      std::cout << std::endl;
    }
  }
}

void initf(Complex *f, int local_0_start, int local_n0, 
	  int N0, int N1, int m0, int m1)
{
  int stop=local_0_start+local_n0;
  for (int i = local_0_start; i < stop; ++i) {
    if(i < m0) {
      int ii=i-local_0_start;
      for (int j = 0; j < m1; ++j) {
	f[ii*N1+j]=i +j*I;

	// f[ii*N1+j]=i*I;
      }
      for (int j = m1; j < N1; ++j) {
	f[ii*N1+j]=0.0;
      }
    }
  }
  
  for (int i = 0; i < local_n0; ++i) {
    if(i+local_0_start >= m0) {
      for (int j = 0; j < N1; ++j) {
	f[i*N1+j]=0.0;
      }
    }
  }

}

void initg(Complex *g, int local_0_start, int local_n0, 
	  int N0, int N1, int m0, int m1)
{
  int stop=local_0_start+local_n0;
  for (int i = local_0_start; i < stop; ++i) {
    if(i < m0) {
      int ii=i-local_0_start;
      for (int j = 0; j < m1; ++j) {
	g[ii*N1+j]=2*i +(j+1)*I;

	// g[ii*N1+j]=(i == 0 && j == 0) ? 1.0 : 0.0; //i*I;
      }
      for (int j = m1; j < N1; ++j) {
	g[ii*N1+j]=0.0;
      }
    }
  }
  
  for (int i = 0; i < local_n0; ++i) {
    if(i+local_0_start >= m0) {
      for (int j = 0; j < N1; ++j) {
	g[i*N1+j]=0.0;
      }
    }
  }

}
// 3D
void show(Complex *f, int local_0_start, int local_n0, 
	  int N1, int N2, int m0, int m1, int m2, int A)
{
  int stop=local_0_start+local_n0;
  for (int i = local_0_start; i < stop; ++i) {
    if(i < m0) {
      int ii=i-local_0_start;
      for (int j = 0; j < m1; ++j) {
	std::cout << A << "-" << i << ": ";
	for (int k = 0; k < m2; ++k) {
	  int index=ii*N1*N2 + j*N1 +k;
	  std::cout << f[index] << "  ";
	}
	std::cout << std::endl;
      }
      std::cout << std::endl;
    }
  }
} 


