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

typedef struct ImplicitConvolution ImplicitConvolution;
typedef struct ImplicitHConvolution ImplicitHConvolution;

#ifdef  __cplusplus

#include "fftw++.h"

ImplicitConvolution *fftwpp_create_conv1d(unsigned int m);
void fftwpp_conv1d_convolve(ImplicitConvolution *conv, double *a, double *b);

ImplicitHConvolution *fftwpp_create_hconv1d(unsigned int m);
void fftwpp_hconv1d_convolve(ImplicitHConvolution *conv, double *a, double *b);


#endif


#endif
