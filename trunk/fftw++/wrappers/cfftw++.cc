/* cfftw++.cc - C callable FFTW++ wrapper.
 *
 * These C callable wrappers make the Python wrapper fairly easy.  Not
 * all of the FFTW++ routines are wrapped.
 *
 * Authors: 
 * Matthew Emmett <memmett@unc.edu> and 
 * Malcolm Roberts <malcolm.i.w.roberts@gmail.com>
 */

#ifdef  __cplusplus
#include "Complex.h"
#include "cfftw++.h"
#include "convolution.h"
#include<complex.h>

namespace fftwpp {

  // prototypes
  extern "C" {

    // wrappers for allocating aligned memory arrays
    double *create_doubleAlign(unsigned int n) {
      return (double  * ) fftwpp::doubleAlign(n); 
    }
    void delete_doubleAlign(double * p) {
      deleteAlign(p);
    }
    double __complex__  *create_complexAlign(unsigned int n) {
      return (double __complex__ * ) fftwpp::ComplexAlign(n); 
    }
    void delete_complexAlign(double __complex__ * p) {
      deleteAlign(p);
    }

    // wrappers for multiple threads
    unsigned int get_fftwpp_maxthreads() {
      return fftw::maxthreads;
    }
    void set_fftwpp_maxthreads(unsigned int nthreads) {
      fftw::maxthreads=nthreads;
    }
    
    // 1d complex non-centered convolution
    ImplicitConvolution *fftwpp_create_conv1d(unsigned int m);
    ImplicitConvolution *fftwpp_create_conv1d_work(unsigned int m,
						   double __complex__ *u, 
						   double __complex__ *v);
    void fftwpp_conv1d_delete(ImplicitConvolution *conv);
    void fftwpp_conv1d_convolve(ImplicitConvolution *conv, 
				double __complex__ *a, double __complex__ *b);

    // 1d Hermitian-symmetric entered convolution
    ImplicitHConvolution *fftwpp_create_hconv1d(unsigned int m);
    ImplicitHConvolution *fftwpp_create_hconv1d_work(unsigned int m,
						     double __complex__ *u, 
						     double __complex__ *v, 
						     double __complex__ *w);
    void fftwpp_hconv1d_delete(ImplicitHConvolution *conv);
    void fftwpp_hconv1d_convolve(ImplicitHConvolution *conv, 
				 double __complex__*a, double __complex__ *b);

    // 2d complex wrappers
    ImplicitConvolution2 *fftwpp_create_conv2d(unsigned int mx, 
					       unsigned int my);
    void fftwpp_conv2d_delete(ImplicitConvolution2 *conv);
    void fftwpp_conv2d_convolve(ImplicitConvolution2 *conv, 
				double *a, double *b);

    // 2d Hermitian symmetric  wrappers
    ImplicitHConvolution2 *fftwpp_create_hconv2d(unsigned int mx, 
						 unsigned int my);
    void fftwpp_hconv2d_delete(ImplicitHConvolution2 *conv);
    void fftwpp_hconv2d_convolve(ImplicitHConvolution2 *conv, 
				 double *a, double *b);

    // 3d complex wrappers
    ImplicitConvolution3 *fftwpp_create_conv3d(unsigned int mx, 
					       unsigned int my, 
					       unsigned int mz);
    void fftwpp_conv3d_delete(ImplicitConvolution3 *conv);
    void fftwpp_conv3d_convolve(ImplicitConvolution3 *conv, 
				double *a, double *b);

    // 3d Hermitian symmetric  wrappers
    ImplicitHConvolution3 *fftwpp_create_hconv3d(unsigned int mx, 
						 unsigned int my, 
						 unsigned int mz);
    void fftwpp_hconv3d_delete(ImplicitHConvolution3 *conv);
    void fftwpp_hconv3d_convolve(ImplicitHConvolution3 *conv, 
				 double *a, double *b);

    // 1d centered Hermitian-symmetric ternary  convolution
    ImplicitHTConvolution *fftwpp_create_htconv1d(unsigned int mx);
    void fftwpp_htconv1d_convolve(ImplicitHTConvolution *conv, 
				double *a, double *b, double *c);
    void fftwpp_htconv1d_delete(ImplicitHTConvolution *conv);

    // 2d centered Hermitian-symmetric ternary  convolution
    ImplicitHTConvolution2 *fftwpp_create_htconv2d(unsigned int mx, 
						   unsigned int my);
    void fftwpp_htconv2d_convolve(ImplicitHTConvolution2 *conv, 
				double *a, double *b, double *c);
    void fftwpp_htconv2d_delete(ImplicitHTConvolution2 *conv);

  } // extern 'C'


  // 1d complex wrappers
  ImplicitConvolution *fftwpp_create_conv1d(unsigned int m) {
    return new ImplicitConvolution(m);
  }

  ImplicitConvolution *fftwpp_create_conv1d_work(unsigned int m,
						 double __complex__ *u, 
						 double __complex__ *v) {
    return new ImplicitConvolution(m,(Complex *) u,(Complex *) v);
  }


  void fftwpp_conv1d_convolve(ImplicitConvolution *conv, 
			      double __complex__ *a, double __complex__  *b){
    conv->convolve((Complex *) a, (Complex *) b);
  }

  void fftwpp_conv1d_delete(ImplicitConvolution *conv) {
    delete conv;
  }

  // 1d Hermitian symmetric wrappers
  ImplicitHConvolution *fftwpp_create_hconv1d(unsigned int m) {
    return new ImplicitHConvolution(m);
  }

  ImplicitHConvolution *fftwpp_create_hconv1d_work(unsigned int m,
						   double __complex__ *u, 
						   double __complex__ *v, 
						   double __complex__ *w) {
    return new ImplicitHConvolution(m, (Complex *) u, (Complex *) v, 
				    (Complex *) w);
  }
  
  void fftwpp_hconv1d_convolve(ImplicitHConvolution *conv, 
			       double __complex__  *a, double __complex__  *b) {
    conv->convolve((Complex *) a, (Complex *) b);
  }

  void fftwpp_hconv1d_delete(ImplicitHConvolution *conv) {
    delete conv;
  }

  // 2d non-centered complex convolution
  ImplicitConvolution2 *fftwpp_create_conv2d(unsigned int mx, unsigned int my) {
    return new ImplicitConvolution2(mx, my);
  }

  void fftwpp_conv2d_convolve(ImplicitConvolution2 *conv, 
			      double *a, double *b) {
    conv->convolve((Complex *) a, (Complex *) b);
  }

  void fftwpp_conv2d_delete(ImplicitConvolution2 *conv) {
    delete conv;
  }

  // 2d centered Hermitian-symmetric convolution
  ImplicitHConvolution2 *fftwpp_create_hconv2d(unsigned int mx, 
					       unsigned int my) {
    return new ImplicitHConvolution2(mx, my);
  }

  void fftwpp_hconv2d_convolve(ImplicitHConvolution2 *conv, 
			       double *a, double *b) {
    conv->convolve((Complex *) a, (Complex *) b);
  }

  void fftwpp_hconv2d_delete(ImplicitHConvolution2 *conv) {
    delete conv;
  }

  // 3d non-centered complex convolution
  ImplicitConvolution3 *fftwpp_create_conv3d(unsigned int mx, 
					     unsigned int my, 
					     unsigned int mz) {
    return new ImplicitConvolution3(mx, my, mz);
  }

  void fftwpp_conv3d_convolve(ImplicitConvolution3 *conv, 
			      double *a, double *b) {
    conv->convolve((Complex *) a, (Complex *) b);
  }

  void fftwpp_conv3d_delete(ImplicitConvolution3 *conv) {
    delete conv;
  }

  // 3d centered Hermitian-symmetric convolution
  ImplicitHConvolution3 *fftwpp_create_hconv3d(unsigned int mx, 
					       unsigned int my, 
					       unsigned int mz) {
    return new ImplicitHConvolution3(mx, my, mz);
  }
  void fftwpp_hconv3d_convolve(ImplicitHConvolution3 *conv, 
			       double *a, double *b) {
    conv->convolve((Complex *) a, (Complex *) b);
  }
  void fftwpp_hconv3d_delete(ImplicitHConvolution3 *conv) {
    delete conv;
  }

  // 1d centered Hermitian-symmetric ternary  convolution
  ImplicitHTConvolution *fftwpp_create_htconv1d(unsigned int mx) {
    return new ImplicitHTConvolution(mx);
  }
  void fftwpp_htconv1d_convolve(ImplicitHTConvolution *conv, 
				double *a, double *b, double *c) {
    conv->convolve((Complex *) a, (Complex *) b, (Complex *) c);
  }
  void fftwpp_htconv1d_delete(ImplicitHTConvolution *conv) {
    delete conv;
  }

  // 2d centered Hermitian-symmetric ternary  convolution
  ImplicitHTConvolution2 *fftwpp_create_htconv2d(unsigned int mx, 
						 unsigned int my) {
    return new ImplicitHTConvolution2(mx,my);
  }
  void fftwpp_htconv2d_convolve(ImplicitHTConvolution2 *conv, 
				double *a, double *b, double *c) {
    conv->convolve((Complex *) a, (Complex *) b, (Complex *) c);
  }
  void fftwpp_htconv2d_delete(ImplicitHTConvolution2 *conv) {
    delete conv;
  }

  
}

#endif //__cplusplus
