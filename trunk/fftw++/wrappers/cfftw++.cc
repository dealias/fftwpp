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
#include "fftw++.h"
#include "cfftw++.h"
#include "convolution.h"
#include <iostream> // temp

namespace array {
  extern "C" {
    // Complex arrays
    Complex *create_ComplexAlign(unsigned int n) {
      return fftwpp::ComplexAlign(n);
    }

    // FIXME: add delete option too!

    void init(Complex * f, unsigned int n) {
      for(unsigned int i=0; i < n; ++i) {
	f[i]=2.0*i;
      }
    }

    void show(Complex * f, unsigned int n) {
      for(unsigned int i=0; i < n; ++i) {
	std::cout << f[i] << std::endl;
      }
    }
    
    void init2(Complex *f, Complex *g, unsigned int m) 
    {
      for(unsigned int i=0; i < m; i += m) {
	Complex *fi=f+i;
	Complex *gi=g+i;
	for(unsigned int k=0; k < m; k++) fi[k]=Complex(k,k+1);
	for(unsigned int k=0; k < m; k++) gi[k]=Complex(k,2*k+1);
      }
    }




  }
}

namespace fftwpp {

  // prototypes
  extern "C" {
    // FIXME: why prototype and impliment in same file?

    // 1d complex wrappers
    ImplicitConvolution *fftwpp_create_conv1d(unsigned int m);
    void fftwpp_conv1d_delete(ImplicitConvolution *conv);
    void fftwpp_conv1d_convolve(ImplicitConvolution *conv, 
				double *a, double *b);

    // 1d Hermitian wrappers
    ImplicitHConvolution *fftwpp_create_hconv1d(unsigned int m);
    void fftwpp_hconv1d_delete(ImplicitHConvolution *conv);
    void fftwpp_hconv1d_convolve(ImplicitHConvolution *conv, 
				 double *a, double *b);

    /*
    // 2d
    ImplicitHConvolution2 *fftwpp_create_hconv2d(unsigned int mx, 
						 unsigned int my);
    void fftwpp_hconv2d_delete(ImplicitHConvolution2 *conv);
    void fftwpp_hconv2d_convolve(ImplicitHConvolution2 *conv, 
				 double *a, double *b);

    ImplicitConvolution2 *fftwpp_create_conv2d(unsigned int mx, 
					       unsigned int my);
    void fftwpp_conv2d_delete(ImplicitConvolution2 *conv);
    void fftwpp_conv2d_convolve(ImplicitConvolution2 *conv, 
				double *a, double *b);

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


  // 1d wrappers complex wrappers
  ImplicitConvolution *fftwpp_create_conv1d(unsigned int m) {
    return new ImplicitConvolution(m);
  }

  void fftwpp_conv1d_convolve(ImplicitConvolution *conv, double *a, double *b){
    conv->convolve((Complex *) a, (Complex *) b);
  }

  void fftwpp_conv1d_delete(ImplicitConvolution *conv) {
    delete conv;
  }

  // 1d wrappers Hermitian symmetric
  ImplicitHConvolution *fftwpp_create_hconv1d(unsigned int m) {
    return new ImplicitHConvolution(m);
  }

  void fftwpp_hconv1d_convolve(ImplicitHConvolution *conv, 
			       double *a, double *b) {
    conv->convolve((Complex *) a, (Complex *) b);
  }

  void fftwpp_hconv1d_delete(ImplicitHConvolution *conv) {
    delete conv;
  }

  /*
  // 2d wrappers
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

  // 3d wrappers
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
}

#endif //__cplusplus
