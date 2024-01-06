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
  double *create_doublealign(size_t n);
  void delete_complexAlign(double __complex__ * p);
  double __complex__  *create_complexAlign(size_t n);
  void delete_complexAlign(double __complex__ * p);

// wrappers for multiple threads
  size_t get_fftwpp_maxthreads();
  void set_fftwpp_maxthreads(size_t nthreads);

// 1d complex non-centered convolution
  typedef struct HybridConvolution HybridConvolution;
  HybridConvolution *fftwpp_create_conv1d(size_t nx);
  HybridConvolution *fftwpp_create_conv1d_dot(size_t m,
                                              size_t M);
  HybridConvolution *fftwpp_create_conv1d_work(size_t m,
                                               double __complex__ *u,
                                               double __complex__ *v);
  HybridConvolution *fftwpp_create_conv1d_work_dot(size_t m,
                                                   double __complex__ *u,
                                                   double __complex__ *v,
                                                   size_t M);
  void fftwpp_conv1d_convolve(HybridConvolution *conv,
                              double __complex__ *a, double __complex__  *b);

  void fftwpp_conv1d_convolve_dot(HybridConvolution *conv,
                                  double __complex__ **a,
                                  double __complex__ **b);
  void fftwpp_conv1d_convolve_dotf(HybridConvolution *conv,
                                   double __complex__ *a,
                                   double __complex__ *b);
  void fftwpp_conv1d_delete(HybridConvolution *conv);

// 1d Hermitian-symmetric entered convolution/
  typedef struct HybridConvolutionHermitian HybridConvolutionHermitian;
  HybridConvolutionHermitian *fftwpp_create_hconv1d(size_t m);
  HybridConvolutionHermitian *fftwpp_create_hconv1d_dot(size_t m,
                                                        size_t M);
  HybridConvolutionHermitian *fftwpp_create_hconv1d_work(size_t m,
                                                         double __complex__ *u,
                                                         double __complex__ *v,
                                                         double __complex__ *w);
  HybridConvolutionHermitian *fftwpp_create_hconv1d_work_dot(size_t m,
                                                             double __complex__ *u,
                                                             double __complex__ *v,
                                                             double __complex__ *w,
                                                             size_t M);
  void fftwpp_hconv1d_convolve(HybridConvolutionHermitian *conv,
                               double __complex__ *a, double __complex__  *b);
  void fftwpp_hconv1d_convolve_dot(HybridConvolutionHermitian *conv,
                                   double __complex__ **a,
                                   double __complex__ **b);
  void fftwpp_hconv1d_convolve_dotf(HybridConvolutionHermitian *conv,
                                    double __complex__ *a,
                                    double __complex__ *b);
  void fftwpp_hconv1d_delete(HybridConvolutionHermitian *conv);

// 2d complex non-centered convolution
  typedef struct HybridConvolution2 HybridConvolution2;
  HybridConvolution2 *fftwpp_create_conv2d(size_t mx, size_t my);
  HybridConvolution2 *fftwpp_create_conv2d_dot(size_t mx, size_t my,
                                               size_t M);
  HybridConvolution2 *fftwpp_create_conv2d_work(size_t mx,
                                                size_t my,
                                                double __complex__ *u1,
                                                double __complex__ *v1,
                                                double __complex__ *u2,
                                                double __complex__ *v2);
  HybridConvolution2 *fftwpp_create_conv2d_work_dot(size_t mx,
                                                    size_t my,
                                                    double __complex__ *u1,
                                                    double __complex__ *v1,
                                                    double __complex__ *u2,
                                                    double __complex__ *v2,
                                                    size_t M);

  void fftwpp_conv2d_convolve(HybridConvolution2 *conv,
                              double __complex__ *a, double __complex__ *b);
  void fftwpp_conv2d_convolve_dot(HybridConvolution2 *conv,
                                  double __complex__ **a, double __complex__ **b);
  void fftwpp_conv2d_convolve_dotf(HybridConvolution2 *conv,
                                   double __complex__ *a, double __complex__ *b);
  void fftwpp_conv2d_delete(HybridConvolution2   *conv);

  // 2d Hermitian-symmetric centered convolution
  typedef struct HybridConvolutionHermitian2 HybridConvolutionHermitian2;
  HybridConvolutionHermitian2 *fftwpp_create_hconv2d(size_t mx, size_t my);
  HybridConvolutionHermitian2 *fftwpp_create_hconv2d_dot(size_t mx,
                                                         size_t my,
                                                         size_t M);
  HybridConvolutionHermitian2 *fftwpp_create_hconv2d_work(size_t mx,
                                                          size_t my,
                                                          double __complex__ *u1,
                                                          double __complex__ *v1,
                                                          double __complex__ *w1,
                                                          double __complex__ *u2,
                                                          double __complex__ *v2);
  HybridConvolutionHermitian2 *fftwpp_create_hconv2d_work_dot(size_t mx,
                                                              size_t my,
                                                              double __complex__ *u1,
                                                              double __complex__ *v1,
                                                              double __complex__ *w1,
                                                              double __complex__ *u2,
                                                              double __complex__ *v2,
                                                              size_t M);
  void fftwpp_hconv2d_convolve(HybridConvolutionHermitian2 *conv,
                               double __complex__ *a, double __complex__ *b);
  void fftwpp_hconv2d_convolve_dot(HybridConvolutionHermitian2 *conv,
                                   double __complex__ **a,
                                   double __complex__ **b);
  void fftwpp_hconv2d_convolve_dotf(HybridConvolutionHermitian2 *conv,
                                    double __complex__ *a,
                                    double __complex__ *b);
  void fftwpp_hconv2d_delete(HybridConvolutionHermitian2 *conv);

// 3d complex non-centered convolution
  typedef struct HybridConvolution3 HybridConvolution3;
  HybridConvolution3 *fftwpp_create_conv3d(size_t mx, size_t my,
                                           size_t mz);
  HybridConvolution3 *fftwpp_create_conv3d_work(size_t mx,
                                                size_t my,
                                                size_t mz,
                                                double __complex__ *u1,
                                                double __complex__ *v1,
                                                double __complex__ *u2,
                                                double __complex__ *v2,
                                                double __complex__ *u3,
                                                double __complex__ *v3);
  HybridConvolution3 *fftwpp_create_conv3d_dot(size_t mx,
                                               size_t my,
                                               size_t mz,
                                               size_t M);
  void fftwpp_conv3d_convolve(HybridConvolution3 *conv,
                              double __complex__ *a, double __complex__ *b);
  void fftwpp_conv3d_convolve_dot(HybridConvolution3 *conv,
                                  double __complex__ **a, double __complex__ **b);
  void fftwpp_conv3d_convolve_dotf(HybridConvolution3 *conv,
                                   double __complex__ *a, double __complex__ *b);
  void fftwpp_conv3d_delete(HybridConvolution3 *conv);

  // 3d Hermitian-symmetric centered convolution
  typedef struct HybridConvolutionHermitian3 HybridConvolutionHermitian3;
  HybridConvolutionHermitian3 *fftwpp_create_hconv3d(size_t mx, size_t my,
                                                     size_t mz);
  HybridConvolutionHermitian3 *fftwpp_create_hconv3d_dot(size_t mx,
                                                         size_t my,
                                                         size_t mz,
                                                         size_t M);
  HybridConvolutionHermitian3 *fftwpp_create_hconv3d_work(size_t mx,
                                                          size_t my,
                                                          size_t mz,
                                                          double __complex__ *u1,
                                                          double __complex__ *v1,
                                                          double __complex__ *w1,
                                                          double __complex__ *u2,
                                                          double __complex__ *v2,
                                                          double __complex__ *u3,
                                                          double __complex__ *v3);
  HybridConvolutionHermitian3 *fftwpp_create_hconv3d_work_dot(size_t mx,
                                                              size_t my,
                                                              size_t mz,
                                                              double __complex__ *u1,
                                                              double __complex__ *v1,
                                                              double __complex__ *w1,
                                                              double __complex__ *u2,
                                                              double __complex__ *v2,
                                                              double __complex__ *u3,
                                                              double __complex__ *v3,
                                                              size_t M);
  void fftwpp_hconv3d_convolve(HybridConvolutionHermitian3 *conv,
                               double __complex__ *a, double __complex__ *b);
  void fftwpp_hconv3d_convolve_dot(HybridConvolutionHermitian3 *conv,
                                   double __complex__ **a,
                                   double __complex__ **b);
  void fftwpp_hconv3d_delete(HybridConvolutionHermitian3 *conv);

  void fftwpp_HermitianSymmetrize(double __complex__ *f);
  void fftwpp_HermitianSymmetrizeX(size_t Hx, size_t Hy,
                                   size_t x0, double __complex__ *f);
  void fftwpp_HermitianSymmetrizeXY(size_t Hx, size_t Hy,
                                    size_t Hz,
                                    size_t x0, size_t y0,
                                    double __complex__ *f);


#ifdef  __cplusplus
} }
#endif

#endif
