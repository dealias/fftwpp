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
    double __complex__  *create_complexAlign(unsigned int n) {
      return (double __complex__ * ) fftwpp::ComplexAlign(n); 
    }

    // 1d complex wrappers
    ImplicitConvolution *fftwpp_create_conv1d(unsigned int m);
    void fftwpp_conv1d_delete(ImplicitConvolution *conv);
    void fftwpp_conv1d_convolve(ImplicitConvolution *conv, 
				double __complex__ *a, double __complex__ *b);

    // 1d Hermitian wrappers
    ImplicitHConvolution *fftwpp_create_hconv1d(unsigned int m);
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

    /*
    // 3d
    ImplicitHConvolution3 *fftwpp_create_hconv3d(unsigned int mx, 
						 unsigned int my, 
						 unsigned int mz);
    void fftwpp_hconv3d_delete(ImplicitHConvolution3 *conv);
    void fftwpp_hconv3d_convolve(ImplicitHConvolution3 *conv, 
				 double *a, double *b);

    ImplicitConvolution3 *fftwpp_create_conv3d(unsigned int mx, 
					       unsigned int my, 
					       unsigned int mz);
    void fftwpp_conv3d_delete(ImplicitConvolution3 *conv);
    void fftwpp_conv3d_convolve(ImplicitConvolution3 *conv, 
				double *a, double *b);
    */
  } // extern 'C'


  // 1d complex wrappers
  ImplicitConvolution *fftwpp_create_conv1d(unsigned int m) {
    return new ImplicitConvolution(m);
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
    std::cout << "ImplicitHConvolution2" << std::endl;
    return new ImplicitHConvolution2(mx, my);
  }

  void fftwpp_hconv2d_convolve(ImplicitHConvolution2 *conv, 
			       double *a, double *b) {
    conv->convolve((Complex *) a, (Complex *) b);
  }

  void fftwpp_hconv2d_delete(ImplicitHConvolution2 *conv) {
    delete conv;
  }

  /*
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
  */

  /*
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

  */
}

#endif //__cplusplus
