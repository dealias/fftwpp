#include <mpi.h>
#include <complex.h>
#include <fftw3-mpi.h>
#include <iostream>
#include <stdlib.h>
#include "Complex.h"

void show(fftw_complex *f, int local_0_start, int local_n0, 
	  int N1, int m0, int m1, int A)
{
  int stop=local_0_start+local_n0;
  for (int i = local_0_start; i < stop; ++i) {
    if(i < m0) {
      int ii=i-local_0_start;
      std::cout << A << i << ": ";
      for (int j = 0; j < m1; ++j)
	std::cout << "(" << creal(f[ii*N1 + j]) 
		  << "," << cimag(f[ii*N1 + j]) 
		  << ")  ";
      std::cout << std::endl;
    }
  }
}

void initf(fftw_complex *f, int local_0_start, int local_n0, 
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

void initg(fftw_complex *g, int local_0_start, int local_n0, 
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
void show(fftw_complex *f, int local_0_start, int local_n0, 
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
	  std::cout << "("<<creal(f[index])<<","<<cimag(f[index]) << ")  ";
	}
	std::cout << std::endl;
      }
      std::cout << std::endl;
    }
  }
} 

void initf(fftw_complex *f, int local_0_start, int local_n0, 
	  int N0, int N1, int N2, int m0, int m1, int m2)
{
  int stop=local_0_start+local_n0;
  for (int i = local_0_start; i < stop; ++i) {
    if(i < m0) {
      int ii=i-local_0_start;
      for (int j = 0; j < m1; ++j) {
	for (int k = 0; k < m2; ++k) {
	  int index=ii*N1*N2 + j*N1 +k;
	  f[index]=i+k +(j+k)*I;
	  	  
	  // f[ii*N1+j]=i*I;
	}
	for (int k = m2; k < N2; ++k) {
	  int index=ii*N1*N2 + j*N1 +k;
	  f[index]=0.0;
	}
	
      }
      for (int j = m1; j < N1; ++j) {
	for (int k = 0; k < N2; ++k) {
	  int index=ii*N1*N2 + j*N1 +k;
	  f[index]=0.0;
	}
      }
    }
  }
  
  for (int i = 0; i < local_n0; ++i) {
    if(i+local_0_start >= m0) {
      for (int j = 0; j < N1; ++j) {
	for (int k = 0; k < N2; ++k) {
	  int index=i*N1*N2 + j*N1 +k;
	  f[index]=0.0;
	}
      }
    }
  }

}

void initg(fftw_complex *g, int local_0_start, int local_n0, 
	  int N0, int N1, int N2, int m0, int m1, int m2)
{
  int stop=local_0_start+local_n0;
  for (int i = local_0_start; i < stop; ++i) {
    if(i < m0) {
      int ii=i-local_0_start;
      for (int j = 0; j < m1; ++j) {
	for (int k = 0; k < m2; ++k) {
	  int index=ii*N1*N2 + j*N1 +k;
	  g[index]=2*i+k +(j+1+k)*I;
	  
	  // g[ii*N1+j]=(i == 0 && j == 0) ? 1.0 : 0.0; //i*I;
	}
	for (int k = m2; k < N2; ++k) {
	  int index=ii*N1*N2 + j*N1 +k;
	  g[index]=0.0;
	}
	
      }
      for (int j = m1; j < N1; ++j) {
	for (int k = 0; k < N2; ++k) {
	  int index=ii*N1*N2 + j*N1 +k;
	  g[index]=0.0;
	}
      }
    }
  }
  
  for (int i = 0; i < local_n0; ++i) {
    if(i+local_0_start >= m0) {
      for (int j = 0; j < N1; ++j) {
	for (int k = 0; k < N2; ++k) {
	  int index=i*N1*N2 + j*N1 +k;
	  g[index]=0.0;
	}
      }
    }
  }

}
