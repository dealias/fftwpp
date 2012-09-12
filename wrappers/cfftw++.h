/* cfftw++.h - C callable FFTW++ wrapper
 *
 * Not all of the FFTW++ routines are wrapped.
 *
 * Author: Matthew Emmett <memmett@unc.edu>
 */

typedef struct ImplicitHConvolution hconv1d_t;

hconv1d_t *fftwpp_create_hconv1d(unsigned int m);
void fftwpp_hconv1d_convolve(hconv1d_t *conv, double *a, double *b);


