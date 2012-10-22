/* cfftw++.cc - C callable FFTW++ wrapper.
 *
 * These C callable wrappers make the Python wrapper fairly easy.  Not
 * all of the FFTW++ routines are wrapped.
 *
 * Authors: 
 * Matthew Emmett <memmett@gmail.com> and 
 * Malcolm Roberts <malcolm.i.w.roberts@gmail.com>
 */

#include "Complex.h"
#include "cfftw++.h"
#include "convolution.h"
#include <complex.h>

extern "C" {

  namespace fftwpp {

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
    
    // 1d complex wrappers
    ImplicitConvolution *fftwpp_create_conv1d(unsigned int m) {
      return new ImplicitConvolution(m);
    }

    ImplicitConvolution *fftwpp_create_conv1d_dot(unsigned int m,
						  unsigned int M) {
      return new ImplicitConvolution(m,M);
    }

    ImplicitConvolution *fftwpp_create_conv1d_work(unsigned int m,
						   double __complex__ *u, 
						   double __complex__ *v) {
      return new ImplicitConvolution(m,(Complex *) u,(Complex *) v);
    }
    ImplicitConvolution *fftwpp_create_conv1d_work_dot(unsigned int m,
						       double __complex__ *u,
						       double __complex__ *v,
						       unsigned int M) {
      return new ImplicitConvolution(m,(Complex *) u,(Complex *) v, M);
    }

    void fftwpp_conv1d_convolve(ImplicitConvolution *conv, 
				double __complex__ *a, double __complex__ *b) {
      conv->convolve((Complex *) a, (Complex *) b);
    }
    void fftwpp_conv1d_convolve_dot(ImplicitConvolution *conv, 
				  double __complex__ **a, 
				  double __complex__ **b) {
      conv->convolve((Complex **) a, (Complex **) b);
    }

    void fftwpp_conv1d_convolve_dotf(ImplicitConvolution *conv, 
				     double __complex__ *a, 
				     double __complex__ *b) {
      unsigned int M=conv->getM();
      unsigned int m=conv->getm();
      Complex **A=new Complex *[M];
      Complex **B=new Complex *[M];
      for(unsigned int s=0; s < M; ++s) {
	unsigned int sm=s*m;
	A[s]=(Complex *) a+sm;
	B[s]=(Complex *) b+sm;
      }
      conv->convolve((Complex **) A, (Complex **) B);
      delete[] B;      
      delete[] A;
    }

    void fftwpp_conv1d_delete(ImplicitConvolution *conv) {
      delete conv;
    }

    // 1d Hermitian symmetric wrappers
    ImplicitHConvolution *fftwpp_create_hconv1d(unsigned int m) {
      return new ImplicitHConvolution(m);
    }
    ImplicitHConvolution *fftwpp_create_hconv1d_dot(unsigned int m,
						    unsigned int M) {
      return new ImplicitHConvolution(m,M);
    }

    ImplicitHConvolution *fftwpp_create_hconv1d_work(unsigned int m,
						     double __complex__ *u, 
						     double __complex__ *v, 
						     double __complex__ *w) {
      return new ImplicitHConvolution(m, (Complex *) u, (Complex *) v, 
				      (Complex *) w);
    }
    ImplicitHConvolution *fftwpp_create_hconv1d_work_dot(unsigned int m,
							 double __complex__ *u, 
							 double __complex__ *v, 
							 double __complex__ *w,
							 unsigned int M) {
      return new ImplicitHConvolution(m, (Complex *) u, (Complex *) v, 
				      (Complex *) w, M);
    }
  
    void fftwpp_hconv1d_convolve(ImplicitHConvolution *conv, 
				 double __complex__ *a, double __complex__ *b) {
      conv->convolve((Complex *) a, (Complex *) b);
    }

    void fftwpp_hconv1d_convolve_dot(ImplicitHConvolution *conv, 
				     double __complex__ **a, 
				     double __complex__ **b) {
      conv->convolve((Complex **) a, (Complex **) b);
    }

    void fftwpp_hconv1d_convolve_dotf(ImplicitHConvolution *conv, 
				      double __complex__ *a, 
				      double __complex__ *b) {
      unsigned int M=conv->getM();
      unsigned int m=conv->getm();
      Complex **A=new Complex *[M];
      Complex **B=new Complex *[M];
      for(unsigned int s=0; s < M; ++s) {
	unsigned int sm=s*m;
	A[s]=(Complex *) a+sm;
	B[s]=(Complex *) b+sm;
      }
      conv->convolve((Complex **) A, (Complex **) B);
      delete[] B;      
      delete[] A;
    }

    void fftwpp_hconv1d_delete(ImplicitHConvolution *conv) {
      delete conv;
    }

    // 2d non-centered complex convolution
    ImplicitConvolution2 *fftwpp_create_conv2d(unsigned int mx, 
					       unsigned int my) {
      return new ImplicitConvolution2(mx, my);
    }
    ImplicitConvolution2 *fftwpp_create_conv2d_dot(unsigned int mx, 
						   unsigned int my,
						   unsigned int M) {
      return new ImplicitConvolution2(mx, my, M);
    }
    ImplicitConvolution2 *fftwpp_create_conv2d_work(unsigned int mx, 
						    unsigned int my,
						    double __complex__ *u1, 
						    double __complex__ *v1,
						    double __complex__ *u2, 
						    double __complex__ *v2) {
      return new ImplicitConvolution2(mx, my,
				      (Complex *) u1, (Complex *) v1, 
				      (Complex *) u2, (Complex *) v2);
    }
    ImplicitConvolution2 *fftwpp_create_conv2d_work_dot(unsigned int mx, 
							unsigned int my,
							double __complex__ *u1, 
							double __complex__ *v1,
							double __complex__ *u2, 
							double __complex__ *v2,
							unsigned int M) {
      return new ImplicitConvolution2(mx, my,
				      (Complex *) u1, (Complex *) v1, 
				      (Complex *) u2, (Complex *) v2,
				      M);
    }
    void fftwpp_conv2d_convolve(ImplicitConvolution2 *conv, 
				double __complex__ *a, double __complex__ *b) {
      conv->convolve((Complex *) a, (Complex *) b);
    }
    
    void fftwpp_conv2d_convolve_dot(ImplicitConvolution2 *conv, 
     				    double __complex__ **a, 
				    double __complex__ **b) {
      conv->convolve((Complex **) a, (Complex **) b);
    }
    void fftwpp_conv2d_convolve_dotf(ImplicitConvolution2 *conv, 
				     double __complex__ *a, 
				     double __complex__ *b) {
      unsigned int M=conv->getM();
      unsigned int m=conv->getmx()*conv->getmy();
      Complex **A=new Complex *[M];
      Complex **B=new Complex *[M];
      for(unsigned int s=0; s < M; ++s) {
	unsigned int sm=s*m;
	A[s]=(Complex *) a+sm;
	B[s]=(Complex *) b+sm;
      }
      conv->convolve((Complex **) A, (Complex **) B);
      delete[] A;      
      delete[] B;
    }


    void fftwpp_conv2d_delete(ImplicitConvolution2 *conv) {
      delete conv;
    }

    // 2d centered Hermitian-symmetric convolution
    ImplicitHConvolution2 *fftwpp_create_hconv2d(unsigned int mx, 
						 unsigned int my) {
      return new ImplicitHConvolution2(mx, my);
    }
    ImplicitHConvolution2 *fftwpp_create_hconv2d_dot(unsigned int mx, 
						     unsigned int my,
						     unsigned int M) {
      return new ImplicitHConvolution2(mx, my, M);
    }

    ImplicitHConvolution2 *fftwpp_create_hconv2d_work(unsigned int mx, 
						      unsigned int my,
						      double __complex__ *u1, 
						      double __complex__ *v1,
						      double __complex__ *w1,
						      double __complex__ *u2, 
						      double __complex__ *v2) {
      return new ImplicitHConvolution2(mx, my,(Complex *) u1,(Complex *)  v1,
				       (Complex *) w1,(Complex *)  u2,
				       (Complex *) v2);
    }
    ImplicitHConvolution2 *fftwpp_create_hconv2d_work_dot(unsigned int mx, 
							  unsigned int my,
							  double __complex__*u1,
							  double __complex__*v1,
							  double __complex__*w1,
							  double __complex__*u2,
							  double __complex__*v2,
							  unsigned int M
							  ) {
      return new ImplicitHConvolution2(mx, my,(Complex *) u1,(Complex *)  v1,
				       (Complex *) w1,(Complex *)  u2,
				       (Complex *) v2, M);
    }
    void fftwpp_hconv2d_convolve(ImplicitHConvolution2 *conv, 
				 double __complex__ *a, double __complex__ *b) {
      conv->convolve((Complex *) a, (Complex *) b);
    }
    void fftwpp_hconv2d_convolve_dot(ImplicitHConvolution2 *conv, 
				     double __complex__ **a, 
				     double __complex__ **b) {
      conv->convolve((Complex **) a, (Complex **) b);
    }
    void fftwpp_hconv2d_convolve_dotf(ImplicitHConvolution2 *conv, 
				      double __complex__ *a, 
				      double __complex__ *b) {
      unsigned int M=conv->getM();
      unsigned int mx=conv->getmx();
      unsigned int my=conv->getmy();
      unsigned int m=(2*mx-1)*my;
      Complex **A=new Complex *[M];
      Complex **B=new Complex *[M];
      for(unsigned int s=0; s < M; ++s) {
	unsigned int sm=s*m;
	A[s]=(Complex *) a+sm;
	B[s]=(Complex *) b+sm;
      }
      conv->convolve((Complex **) A, (Complex **) B);
      delete[] A;      
      delete[] B;
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
    ImplicitConvolution3 *fftwpp_create_conv3d_dot(unsigned int mx, 
						   unsigned int my, 
						   unsigned int mz,
						   unsigned int M) {
      return new ImplicitConvolution3(mx, my, mz, M);
    }

    ImplicitConvolution3 *fftwpp_create_conv3d_work(unsigned int mx, 
						    unsigned int my, 
						    unsigned int mz,
						    double __complex__ *u1, 
						    double __complex__ *v1, 
						    double __complex__ *u2, 
						    double __complex__ *v2, 
						    double __complex__ *u3, 
						    double __complex__ *v3) {
      return new ImplicitConvolution3(mx, my, mz,
				      (Complex *) u1,  (Complex *) v1,
				      (Complex *) u2,  (Complex *) v2,
				      (Complex *) u3,  (Complex *) v3);
    }
    ImplicitConvolution3 *fftwpp_create_conv3d_work_dot(unsigned int mx, 
							unsigned int my, 
							unsigned int mz,
							double __complex__ *u1, 
							double __complex__ *v1, 
							double __complex__ *u2, 
							double __complex__ *v2, 
							double __complex__ *u3, 
							double __complex__ *v3,
							unsigned int M) {
      return new ImplicitConvolution3(mx, my, mz,
				      (Complex *) u1,  (Complex *) v1,
				      (Complex *) u2,  (Complex *) v2,
				      (Complex *) u3,  (Complex *) v3,
				      M);
    }
  
    void fftwpp_conv3d_convolve(ImplicitConvolution3 *conv, 
				double __complex__ *a, double __complex__ *b) {
      conv->convolve((Complex *) a, (Complex *) b);
    }
    void fftwpp_conv3d_convolve_dot(ImplicitConvolution3 *conv, 
				    double __complex__ **a, 
				    double __complex__ **b) {
      conv->convolve((Complex **) a, (Complex **) b);
    }
    void fftwpp_conv3d_convolve_dotf(ImplicitConvolution3 *conv, 
				     double __complex__ *a, 
				     double __complex__ *b) {
      unsigned int M=conv->getM();
      unsigned int m=conv->getmx()*conv->getmy()*conv->getmz();
      Complex **A=new Complex *[M];
      Complex **B=new Complex *[M];
      for(unsigned int s=0; s < M; ++s) {
	unsigned int sm=s*m;
	A[s]=(Complex *) a+sm;
	B[s]=(Complex *) b+sm;
      }
      conv->convolve((Complex **) A, (Complex **) B);
      delete[] A;
      delete[] B;
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
    ImplicitHConvolution3 *fftwpp_create_hconv3d_dot(unsigned int mx, 
						     unsigned int my, 
						     unsigned int mz,
						     unsigned int M) {
      return new ImplicitHConvolution3(mx, my, mz, M);
    }

    ImplicitHConvolution3 *fftwpp_create_hconv3d_work(unsigned int mx, 
						      unsigned int my, 
						      unsigned int mz,
						      double __complex__ *u1, 
						      double __complex__ *v1, 
						      double __complex__ *w1,
						      double __complex__ *u2, 
						      double __complex__ *v2,
						      double __complex__ *u3, 
						      double __complex__ *v3) {
      return new ImplicitHConvolution3(mx, my,  mz,
				       (Complex *) u1, (Complex *) v1,
				       (Complex *) w1,
				       (Complex *) u2, (Complex *) v2,
				       (Complex *) u3, (Complex *) v3);

    }
    ImplicitHConvolution3 *fftwpp_create_hconv3d_work_dot(unsigned int mx, 
						      unsigned int my, 
						      unsigned int mz,
						      double __complex__ *u1, 
						      double __complex__ *v1, 
						      double __complex__ *w1,
						      double __complex__ *u2, 
						      double __complex__ *v2,
						      double __complex__ *u3, 
						      double __complex__ *v3,
						      unsigned int M) {
      return new ImplicitHConvolution3(mx, my,  mz,
				       (Complex *) u1, (Complex *) v1,
				       (Complex *) w1,
				       (Complex *) u2, (Complex *) v2,
				       (Complex *) u3, (Complex *) v3,
				       M);

    }
    void fftwpp_hconv3d_convolve(ImplicitHConvolution3 *conv, 
				 double __complex__ *a, double __complex__ *b) {
      conv->convolve((Complex *) a, (Complex *) b);
    }
    void fftwpp_hconv3d_convolve_dot(ImplicitHConvolution3 *conv, 
				     double __complex__ **a, 
				     double __complex__ **b) {
      conv->convolve((Complex **) a, (Complex **) b);
    }
    void fftwpp_hconv3d_convolve_dotf(ImplicitHConvolution3 *conv, 
				      double __complex__ *a, 
				      double __complex__ *b) {
      unsigned int mx=conv->getmx();
      unsigned int my=conv->getmy();
      unsigned int mz=conv->getmz();
      unsigned int m=(2*mx-1)*(2*my-1)*mz;
      unsigned int M=conv->getM();
      Complex **A=new Complex *[M];
      Complex **B=new Complex *[M];
      for(unsigned int s=0; s < M; ++s) {
	unsigned int sm=s*m;
	A[s]=(Complex *) a+sm;
	B[s]=(Complex *) b+sm;
      }
      conv->convolve((Complex **) A, (Complex **) B);
      delete[] A;      
      delete[] B;
    }
    void fftwpp_hconv3d_delete(ImplicitHConvolution3 *conv) {
      delete conv;
    }

    // 1d centered Hermitian-symmetric ternary  convolution
    ImplicitHTConvolution *fftwpp_create_htconv1d(unsigned int mx) {
      return new ImplicitHTConvolution(mx);
    }
    ImplicitHTConvolution *fftwpp_create_htconv1d_dot(unsigned int mx,
						      unsigned int M) {
      return new ImplicitHTConvolution(mx,M);
    }
    ImplicitHTConvolution *fftwpp_create_htconv1d_work(unsigned int m,
						       double __complex__ *u, 
						       double __complex__ *v,
						       double __complex__ *w) {
      return new ImplicitHTConvolution(m, (Complex *) u, (Complex *) v,
				       (Complex *) w);
    }
    ImplicitHTConvolution *fftwpp_create_htconv1d_work_dot(unsigned int m,
							   double __complex__*u,
							   double __complex__*v,
							   double __complex__*w,
							   unsigned int M) {
      return new ImplicitHTConvolution(m, (Complex *) u, (Complex *) v,
				       (Complex *) w, M);
    }

    void fftwpp_htconv1d_convolve(ImplicitHTConvolution *conv, 
				  double __complex__ *a, 
				  double __complex__ *b, 
				  double __complex__ *c) {
      conv->convolve((Complex *) a, (Complex *) b, (Complex *) c);
    }
    void fftwpp_htconv1d_convolve_dot(ImplicitHTConvolution *conv, 
				      double __complex__ **a, 
				      double __complex__ **b, 
				      double __complex__ **c) {
      conv->convolve((Complex **) a, (Complex **) b, (Complex **) c);
    }
    void fftwpp_htconv1d_delete(ImplicitHTConvolution *conv) {
      delete conv;
    }

    // 2d centered Hermitian-symmetric ternary  convolution
    ImplicitHTConvolution2 *fftwpp_create_htconv2d(unsigned int mx, 
						   unsigned int my) {
      return new ImplicitHTConvolution2(mx,my);
    }
    ImplicitHTConvolution2 *fftwpp_create_htconv2d_dot(unsigned int mx, 
						       unsigned int my,
						       unsigned int M) {
      return new ImplicitHTConvolution2(mx,my,M);
    }
    ImplicitHTConvolution2 *fftwpp_create_htconv2d_work(unsigned int mx,
							unsigned int my,
							double __complex__ *u1, 
							double __complex__ *v1, 
							double __complex__ *w1,
							double __complex__ *u2,
							double __complex__ *v2, 
							double __complex__ *w2){
      return new ImplicitHTConvolution2(mx,my, 
					(Complex *) u1,  (Complex *) v1,  
					(Complex *) w1,
					(Complex *) u2,  (Complex *) v2,  
					(Complex *) w2);
    }
    ImplicitHTConvolution2 *fftwpp_create_htconv2d_work_dot(unsigned int mx,
							    unsigned int my,
					    double __complex__ *u1, 
					    double __complex__ *v1, 
					    double __complex__ *w1,
					    double __complex__ *u2,
					    double __complex__ *v2, 
					    double __complex__ *w2,
					    unsigned int M){
      return new ImplicitHTConvolution2(mx,my, 
					(Complex *) u1,  (Complex *) v1,  
					(Complex *) w1,
					(Complex *) u2,  (Complex *) v2,  
					(Complex *) w2,
					M);
    }
    void fftwpp_htconv2d_convolve(ImplicitHTConvolution2 *conv, 
				  double __complex__ *a, 
				  double __complex__ *b,
				  double __complex__ *c) {
      conv->convolve((Complex *) a, (Complex *) b, (Complex *) c);
    }
    void fftwpp_htconv2d_convolve_dot(ImplicitHTConvolution2 *conv, 
				      double __complex__ **a, 
				      double __complex__ **b,
				      double __complex__ **c) {
      conv->convolve((Complex **) a, (Complex **) b, (Complex **) c);
    }
    void fftwpp_htconv2d_delete(ImplicitHTConvolution2 *conv) {
      delete conv;
    }
  
  }
}
