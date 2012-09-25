/* cfftw++.h - C callable FFTW++ wrapper
 *
 * Not all of the FFTW++ routines are wrapped.
 *
 * Authors: 
 * Matthew Emmett <memmett@unc.edu> and 
 * Malcolm Roberts <malcolm.i.w.roberts@gmail.com>
 */

#ifndef CFFTWPP_H
#define CFFTWPP_H

/* #include<complex.h> */

typedef struct Complex Complex;

typedef struct ImplicitConvolution ImplicitConvolution;
typedef struct ImplicitHConvolution ImplicitHConvolution;

/* FIXME: */
/*  expected ‘double *’ but argument is of type ‘struct Complex *’ */
/* Complex *create_ComplexAlign(unsigned int n); */

void init(double *f, unsigned int n); /* temp */
void show(double *f, unsigned int n); /* temp */
void init2(double *f, double *g, unsigned int m); /* temp */

ImplicitConvolution *fftwpp_create_conv1d(unsigned int m);
void fftwpp_conv1d_convolve(ImplicitConvolution *conv, double *a, double *b);

ImplicitHConvolution *fftwpp_create_hconv1d(unsigned int m);
void fftwpp_hconv1d_convolve(ImplicitHConvolution *conv, double *a, double *b);

#endif
