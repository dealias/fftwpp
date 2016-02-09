/* cfftw++.h - C callable FFTW++ wrapper
 *
 * Not all of the FFTW++ routines are wrapped.
 *
 * Authors: 
 * Matthew Emmett <memmett@gmail.com> and 
 * Malcolm Roberts <malcolm.i.w.roberts@gmail.com>
 */

#ifndef _CFFTWPP_H_
#define _CFFTWPP_H_

#ifdef  __cplusplus
namespace fftwpp { extern "C" {
#endif

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
  ImplicitConvolution *fftwpp_create_conv1d(unsigned int nx);
  ImplicitConvolution *fftwpp_create_conv1d_dot(unsigned int m, 
                                                unsigned int M);
  ImplicitConvolution *fftwpp_create_conv1d_work(unsigned int m,
                                                 double __complex__ *u, 
                                                 double __complex__ *v);
  ImplicitConvolution *fftwpp_create_conv1d_work_dot(unsigned int m,
                                                     double __complex__ *u, 
                                                     double __complex__ *v,
                                                     unsigned int M);
  void fftwpp_conv1d_convolve(ImplicitConvolution *conv, 
                              double __complex__ *a, double __complex__  *b);
  void fftwpp_conv1d_correlate(ImplicitConvolution *conv, 
                               double __complex__ *a, double __complex__  *b);
  void fftwpp_conv1d_autocorrelate(ImplicitConvolution *conv, 
                                   double __complex__ *a);
  void fftwpp_conv1d_convolve_dot(ImplicitConvolution *conv, 
                                  double __complex__ **a, 
                                  double __complex__ **b);
  void fftwpp_conv1d_convolve_dotf(ImplicitConvolution *conv, 
                                   double __complex__ *a, 
                                   double __complex__ *b);
  void fftwpp_conv1d_delete(ImplicitConvolution *conv);

// 1d Hermitian-symmetric entered convolution
  typedef struct ImplicitHConvolution ImplicitHConvolution;
  ImplicitHConvolution *fftwpp_create_hconv1d(unsigned int m);
  ImplicitHConvolution *fftwpp_create_hconv1d_dot(unsigned int m,
                                                  unsigned int M);
  ImplicitHConvolution *fftwpp_create_hconv1d_work(unsigned int m,
                                                   double __complex__ *u,
                                                   double __complex__ *v,
                                                   double __complex__ *w);
  ImplicitHConvolution *fftwpp_create_hconv1d_work_dot(unsigned int m,
                                                       double __complex__ *u,
                                                       double __complex__ *v,
                                                       double __complex__ *w,
                                                       unsigned int M);
  void fftwpp_hconv1d_convolve(ImplicitHConvolution *conv, 
                               double __complex__ *a, double __complex__  *b);
  void fftwpp_hconv1d_convolve_dot(ImplicitHConvolution *conv, 
                                   double __complex__ **a, 
                                   double __complex__ **b);
  void fftwpp_hconv1d_convolve_dotf(ImplicitHConvolution *conv, 
                                    double __complex__ *a, 
                                    double __complex__ *b);
  void fftwpp_hconv1d_delete(ImplicitHConvolution *conv);

// 2d complex non-centered convolution
  typedef struct ImplicitConvolution2 ImplicitConvolution2;
  ImplicitConvolution2 *fftwpp_create_conv2d(unsigned int mx, unsigned int my);
  ImplicitConvolution2 *fftwpp_create_conv2d_dot(unsigned int mx, unsigned int my,
                                                 unsigned int M);
  ImplicitConvolution2 *fftwpp_create_conv2d_work(unsigned int mx, 
                                                  unsigned int my,
                                                  double __complex__ *u1, 
                                                  double __complex__ *v1,
                                                  double __complex__ *u2, 
                                                  double __complex__ *v2);
  ImplicitConvolution2 *fftwpp_create_conv2d_work_dot(unsigned int mx, 
                                                      unsigned int my,
                                                      double __complex__ *u1, 
                                                      double __complex__ *v1,
                                                      double __complex__ *u2, 
                                                      double __complex__ *v2,
                                                      unsigned int M);

  void fftwpp_conv2d_convolve(ImplicitConvolution2 *conv, 
                              double __complex__ *a, double __complex__ *b);
  void fftwpp_conv2d_convolve_dot(ImplicitConvolution2 *conv, 
                                  double __complex__ **a, double __complex__ **b);
  void fftwpp_conv2d_convolve_dotf(ImplicitConvolution2 *conv, 
                                   double __complex__ *a, double __complex__ *b);
  void fftwpp_conv2d_delete(ImplicitConvolution2 *conv);

// 2d Hermitian-symmetric centered convolution
  typedef struct ImplicitHConvolution2 ImplicitHConvolution2;
  ImplicitHConvolution2 *fftwpp_create_hconv2d(unsigned int mx, unsigned int my);
  ImplicitHConvolution2 *fftwpp_create_hconv2d_dot(unsigned int mx, 
                                                   unsigned int my,
                                                   unsigned int M);
  ImplicitHConvolution2 *fftwpp_create_hconv2d_work(unsigned int mx, 
                                                    unsigned int my,
                                                    double __complex__ *u1, 
                                                    double __complex__ *v1,
                                                    double __complex__ *w1,
                                                    double __complex__ *u2, 
                                                    double __complex__ *v2);
  ImplicitHConvolution2 *fftwpp_create_hconv2d_work_dot(unsigned int mx, 
                                                        unsigned int my,
                                                        double __complex__ *u1, 
                                                        double __complex__ *v1,
                                                        double __complex__ *w1,
                                                        double __complex__ *u2, 
                                                        double __complex__ *v2,
                                                        unsigned int M);
  void fftwpp_hconv2d_convolve(ImplicitHConvolution2 *conv, 
                               double __complex__ *a, double __complex__ *b);
  void fftwpp_hconv2d_convolve_dot(ImplicitHConvolution2 *conv, 
                                   double __complex__ **a, 
                                   double __complex__ **b);
  void fftwpp_hconv2d_convolve_dotf(ImplicitHConvolution2 *conv, 
                                    double __complex__ *a, 
                                    double __complex__ *b);
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
  ImplicitConvolution3 *fftwpp_create_conv3d_dot(unsigned int mx, 
                                                 unsigned int my, 
                                                 unsigned int mz,
                                                 unsigned int M);
  void fftwpp_conv3d_convolve(ImplicitConvolution3 *conv, 
                              double __complex__ *a, double __complex__ *b);
  void fftwpp_conv3d_convolve_dot(ImplicitConvolution3 *conv, 
                                  double __complex__ **a, double __complex__ **b);
  void fftwpp_conv3d_convolve_dotf(ImplicitConvolution3 *conv, 
                                   double __complex__ *a, double __complex__ *b);
  void fftwpp_conv3d_delete(ImplicitConvolution3 *conv);

// 3d Hermitian-symmetric centered convolution
  typedef struct ImplicitHConvolution3 ImplicitHConvolution3;
  ImplicitHConvolution3 *fftwpp_create_hconv3d(unsigned int mx, unsigned int my, 
                                               unsigned int mz);
  ImplicitHConvolution3 *fftwpp_create_hconv3d_dot(unsigned int mx, 
                                                   unsigned int my, 
                                                   unsigned int mz, 
                                                   unsigned int M);
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
                                                        unsigned int M);
  void fftwpp_hconv3d_convolve(ImplicitHConvolution3 *conv, 
                               double __complex__ *a, double __complex__ *b);
  void fftwpp_hconv3d_convolve_dot(ImplicitHConvolution3 *conv, 
                                   double __complex__ **a, 
                                   double __complex__ **b);
  void fftwpp_hconv3d_delete(ImplicitHConvolution3 *conv);

// 1d Hermitian-symmetric ternary convolution
  typedef struct ImplicitHTConvolution ImplicitHTConvolution;
  ImplicitHTConvolution *fftwpp_create_htconv1d(unsigned int m);
  ImplicitHTConvolution *fftwpp_create_htconv1d_dot(unsigned int m,
                                                    unsigned int M);
  ImplicitHTConvolution *fftwpp_create_htconv1d_work(unsigned int m,
                                                     double __complex__ *u, 
                                                     double __complex__ *v,
                                                     double __complex__ *w);
  ImplicitHTConvolution *fftwpp_create_htconv1d_work_dot(unsigned int m,
                                                         double __complex__ *u, 
                                                         double __complex__ *v,
                                                         double __complex__ *w,
                                                         unsigned int M);
  void fftwpp_htconv1d_convolve(ImplicitHTConvolution *conv, 
                                double __complex__ *a, double __complex__ *b,
                                double __complex__ *c);
  void fftwpp_htconv1d_convolve_dot(ImplicitHTConvolution *conv, 
                                    double __complex__ **a, 
                                    double __complex__ **b,
                                    double __complex__ **c);
  void fftwpp_htconv1d_delete(ImplicitHTConvolution *conv);

// 2d Hermitian-symmetric ternary convolution
  typedef struct ImplicitHTConvolution2 ImplicitHTConvolution2;
  ImplicitHTConvolution2 *fftwpp_create_htconv2d(unsigned int mx,unsigned int my);
  ImplicitHTConvolution2 *fftwpp_create_htconv2d_dot(unsigned int mx,
                                                     unsigned int my,
                                                     unsigned int M);
  ImplicitHTConvolution2 *fftwpp_create_htconv2d_work(unsigned int mx,
                                                      unsigned int my,
                                                      double __complex__ *u1, 
                                                      double __complex__ *v1, 
                                                      double __complex__ *w1,
                                                      double __complex__ *u2,
                                                      double __complex__ *v2, 
                                                      double __complex__ *w2);
  ImplicitHTConvolution2 *fftwpp_create_htconv2d_work_dot(unsigned int mx,
                                                          unsigned int my,
                                                          double __complex__ *u1, 
                                                          double __complex__ *v1, 
                                                          double __complex__ *w1,
                                                          double __complex__ *u2,
                                                          double __complex__ *v2, 
                                                          double __complex__ *w2,
                                                          unsigned int M);
  void fftwpp_htconv2d_convolve(ImplicitHTConvolution2 *conv, 
                                double __complex__ *a, double __complex__ *b,
                                double __complex__ *c);
  void fftwpp_htconv2d_convolve_dot(ImplicitHTConvolution2 *conv, 
                                    double __complex__ **a, 
                                    double __complex__ **b,
                                    double __complex__ **c);
  void fftwpp_htconv2d_delete(ImplicitHTConvolution2 *conv);

#ifdef  __cplusplus
} }
#endif

#endif
