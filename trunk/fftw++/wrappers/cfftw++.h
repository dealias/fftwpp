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

// wrappers for multiple threads
unsigned int get_fftwpp_maxthreads();
void set_fftwpp_maxthreads(unsigned int nthreads);

// 1d complex non-centered convolution
typedef struct ImplicitConvolution ImplicitConvolution;
ImplicitConvolution *fftwpp_create_conv1d(unsigned int m);
ImplicitConvolution *fftwpp_create_conv1d_work(unsigned int m,
					       double __complex__ *u, 
					       double __complex__ *v);
void fftwpp_conv1d_convolve(ImplicitConvolution *conv, 
			    double __complex__ *a, double __complex__  *b);
void fftwpp_conv1d_delete(ImplicitConvolution *conv);

// 1d Hermitian-symmetric entered convolution
typedef struct ImplicitHConvolution ImplicitHConvolution;
ImplicitHConvolution *fftwpp_create_hconv1d(unsigned int m);
ImplicitHConvolution *fftwpp_create_hconv1d_work(unsigned int m,
						 double __complex__ *u,
						 double __complex__ *v,
						 double __complex__ *w);
void fftwpp_hconv1d_convolve(ImplicitHConvolution *conv, 
			     double __complex__ *a, double __complex__  *b);
void fftwpp_hconv1d_delete(ImplicitHConvolution *conv);

// 2d complex non-centered convolution
typedef struct ImplicitConvolution2 ImplicitConvolution2;
ImplicitConvolution2 *fftwpp_create_conv2d(unsigned int mx, unsigned int my);
ImplicitConvolution2 *fftwpp_create_conv2d_work(unsigned int mx, 
						unsigned int my,
						double __complex__ *u1, 
						double __complex__ *v1,
						double __complex__ *u2, 
						double __complex__ *v2);

void fftwpp_conv2d_convolve(ImplicitConvolution2 *conv, 
			    double __complex__ *a, double __complex__  *b);
void fftwpp_conv2d_delete(ImplicitConvolution2 *conv);

// 2d Hermitian-symmetric centered convolution
typedef struct ImplicitHConvolution2 ImplicitHConvolution2;
ImplicitHConvolution2 *fftwpp_create_hconv2d(unsigned int mx, unsigned int my);
ImplicitHConvolution2 *fftwpp_create_hconv2d_work(unsigned int mx, 
						  unsigned int my,
						  double __complex__ *u1, 
						  double __complex__ *v1,
						  double __complex__ *w1,
						  double __complex__ *u2, 
						  double __complex__ *v2);
void fftwpp_hconv2d_convolve(ImplicitHConvolution2 *conv, 
			    double __complex__ *a, double __complex__  *b);
void fftwpp_hconv2d_delete(ImplicitHConvolution2 *conv);

// 3d complex non-centered convolution
typedef struct ImplicitConvolution3 ImplicitConvolution3;
ImplicitConvolution3 *fftwpp_create_conv3d(unsigned int mx, unsigned int my, 
					   unsigned int mz);
ImplicitConvolution3 *fftwpp_create_conv3d_work(unsigned int mx,
						unsigned int my,
						unsigned int mz,
						double __complex__ *u1,
						double __complex__ *v1,
						double __complex__ *u2,
						double __complex__ *v2,
						double __complex__ *u3,
						double __complex__ *v3);
void fftwpp_conv3d_convolve(ImplicitConvolution3 *conv, 
			    double __complex__ *a, double __complex__ *b);
void fftwpp_conv3d_delete(ImplicitConvolution3 *conv);

// 3d Hermitian-symmetric centered convolution
typedef struct ImplicitHConvolution3 ImplicitHConvolution3;
ImplicitHConvolution3 *fftwpp_create_hconv3d(unsigned int mx, unsigned int my, 
					   unsigned int mz);
ImplicitHConvolution3 *fftwpp_create_hconv3d_work(unsigned int mx, 
						  unsigned int my, 
						  unsigned int mz,
						  double __complex__ *u1, 
						  double __complex__ *v1, 
						  double __complex__ *w1,
						  double __complex__ *u2, 
						  double __complex__ *v2,
						  double __complex__ *u3, 
						  double __complex__ *v3);
void fftwpp_hconv3d_convolve(ImplicitHConvolution3 *conv, 
			    double __complex__ *a, double __complex__ *b);
void fftwpp_hconv3d_delete(ImplicitHConvolution3 *conv);

// 1d Hermitian-symmetric ternary convolution
typedef struct ImplicitHTConvolution ImplicitHTConvolution;
ImplicitHTConvolution *fftwpp_create_htconv1d(unsigned int m);
ImplicitHTConvolution *fftwpp_create_htconv1d_work(unsigned int m,
						   double __complex__ *u, 
						   double __complex__ *v,
						   double __complex__ *w);
void fftwpp_htconv1d_convolve(ImplicitHTConvolution *conv, 
			      double __complex__ *a, double __complex__ *b,
			      double __complex__ *c);
void fftwpp_htconv1d_delete(ImplicitHTConvolution *conv);

// 2d Hermitian-symmetric ternary convolution
typedef struct ImplicitHTConvolution2 ImplicitHTConvolution2;
ImplicitHTConvolution2 *fftwpp_create_htconv2d(unsigned int mx,unsigned int my);
ImplicitHTConvolution2 *fftwpp_create_htconv2d_work(unsigned int mx,
						    unsigned int my,
						    double __complex__ *u1, 
						    double __complex__ *v1, 
						    double __complex__ *w1,
						    double __complex__ *u2,
						    double __complex__ *v2, 
						    double __complex__ *w2);
void fftwpp_htconv2d_convolve(ImplicitHTConvolution2 *conv, 
			      double __complex__ *a, double __complex__ *b,
			      double __complex__ *c);
void fftwpp_htconv2d_delete(ImplicitHTConvolution2 *conv);

#endif
