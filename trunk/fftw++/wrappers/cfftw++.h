/* cfftw++.h - C callable FFTW++ wrapper
 *
 * Not all of the FFTW++ routines are wrapped.
 *
 * Author: Matthew Emmett <memmett@unc.edu>
 */


#ifndef CFFTWPP_H
#define CFFTWPP_H

//#ifdef  __cplusplus
#include "fftw++.h"

typedef struct ImplicitHConvolution ImplicitHConvolution;

ImplicitHConvolution *fftwpp_create_hconv1d(unsigned int m);
void fftwpp_hconv1d_convolve(ImplicitHConvolution *conv, double *a, double *b);



//#endif //  __cplusplus

#endif // CFFTWPP_H
