#ifndef __exmpiutils_h__
#define __exmpiutils_h__ 1

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
void init(fftw_complex *f, fftw_complex *g, int local_0_start, int local_n0, 
	  int N0, int N1, int N2, int m0, int m1, int m2);

#endif
