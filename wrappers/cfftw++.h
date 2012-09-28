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


double __complex__  *create_complexAlign(unsigned int n);

typedef struct ImplicitConvolution ImplicitConvolution;
ImplicitConvolution *fftwpp_create_conv1d(unsigned int m);
void fftwpp_conv1d_convolve(ImplicitConvolution *conv, 
			    double __complex__ *a, double __complex__  *b);
void fftwpp_conv1d_delete(ImplicitConvolution *conv);

typedef struct ImplicitHConvolution ImplicitHConvolution;
ImplicitHConvolution *fftwpp_create_hconv1d(unsigned int m);
void fftwpp_hconv1d_convolve(ImplicitHConvolution *conv, 
			     double __complex__ *a, double __complex__  *b);
void fftwpp_hconv1d_delete(ImplicitHConvolution *conv);


#endif
