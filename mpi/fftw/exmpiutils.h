#ifndef __exmpiutils_h__
#define __exmpiutils_h__ 1

#include <Complex.h>

// 2D
void show(fftw_complex *f, int local_0_start, int local_n0, 
	  int N1, int m0, int m1, int A);
void initf(fftw_complex *f, int local_0_start, int local_n0, 
	  int N0, int N1, int m0, int m1);
void initg(fftw_complex *g, int local_0_start, int local_n0, 
	  int N0, int N1, int m0, int m1);

// 3D
void show(fftw_complex *f, int local_0_start, int local_n0, 
	  int N1, int N2, int m0, int m1, int m2, int A);
void initf(fftw_complex *f, int local_0_start, int local_n0, 
	  int N0, int N1, int N2, int m0, int m1, int m2);
void initg(fftw_complex *g, int local_0_start, int local_n0, 
	  int N0, int N1, int N2, int m0, int m1, int m2);

static const Complex I(0,1);

#endif
