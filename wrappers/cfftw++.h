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

// wrappers for allocating aligned memory arrays
double *create_doublealign(unsigned int n);
void delete_complexAlign(double __complex__ * p);
double __complex__  *create_complexAlign(unsigned int n);
void delete_complexAlign(double __complex__ * p);

// 1d complex non-centered convolution
typedef struct ImplicitConvolution ImplicitConvolution;
ImplicitConvolution *fftwpp_create_conv1d(unsigned int m);
void fftwpp_conv1d_convolve(ImplicitConvolution *conv, 
			    double __complex__ *a, double __complex__  *b);
void fftwpp_conv1d_delete(ImplicitConvolution *conv);

// 1d Hermitian-symmetric entered convolution
typedef struct ImplicitHConvolution ImplicitHConvolution;
ImplicitHConvolution *fftwpp_create_hconv1d(unsigned int m);
void fftwpp_hconv1d_convolve(ImplicitHConvolution *conv, 
			     double __complex__ *a, double __complex__  *b);
void fftwpp_hconv1d_delete(ImplicitHConvolution *conv);

// 2d complex non-centered convolution
typedef struct ImplicitConvolution2 ImplicitConvolution2;
ImplicitConvolution2 *fftwpp_create_conv2d(unsigned int mx, unsigned int my);
void fftwpp_conv2d_convolve(ImplicitConvolution2 *conv, 
			    double __complex__ *a, double __complex__  *b);
void fftwpp_conv2d_delete(ImplicitConvolution2 *conv);

// 2d Hermitian-symmetric centered convolution
typedef struct ImplicitHConvolution2 ImplicitHConvolution2;
ImplicitHConvolution2 *fftwpp_create_hconv2d(unsigned int mx, unsigned int my);
void fftwpp_hconv2d_convolve(ImplicitHConvolution2 *conv, 
			    double __complex__ *a, double __complex__  *b);
void fftwpp_hconv2d_delete(ImplicitHConvolution2 *conv);

// 3d complex non-centered convolution
typedef struct ImplicitConvolution3 ImplicitConvolution3;
ImplicitConvolution3 *fftwpp_create_conv3d(unsigned int mx, unsigned int my, 
					   unsigned int mz);
void fftwpp_conv3d_convolve(ImplicitConvolution3 *conv, 
			    double __complex__ *a, double __complex__  *b);
void fftwpp_conv3d_delete(ImplicitConvolution3 *conv);

// 3d Hermitian-symmetric centered convolution
typedef struct ImplicitHConvolution3 ImplicitHConvolution3;
ImplicitHConvolution3 *fftwpp_create_hconv3d(unsigned int mx, unsigned int my, 
					   unsigned int mz);
void fftwpp_hconv3d_convolve(ImplicitHConvolution3 *conv, 
			    double __complex__ *a, double __complex__  *b);
void fftwpp_hconv3d_delete(ImplicitHConvolution3 *conv);

#endif
