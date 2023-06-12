/* cfftw++.cc - C callable FFTW++ wrapper.
 *
 * These C callable wrappers make the Python wrapper fairly easy.  Not
 * all of the FFTW++ routines are wrapped.
 *
 * Authors:
 * Matthew Emmett <memmett@gmail.com> and
 * Malcolm Roberts <malcolm.i.w.roberts@gmail.com>
 * Robert Joseph George <rjoseph1@ualberta.ca>
 * John C. Bowman <bowman@ualberta.ca>
 * Noel Murasko <murasko@ualbert.ca>
 */

#include "Complex.h"
#include "cfftw++.h"
#include "convolve.h"
#include <complex.h>
#include "HybridConvolution.h"

using namespace fftwpp;
using namespace parallel;

extern "C" {

  namespace fftwpp {

  // Wrappers for allocating aligned memory arrays
  double *create_doubleAlign(unsigned int n) {
    return (double  *) utils::doubleAlign(n);
  }

  void delete_doubleAlign(double * p) {
    utils::deleteAlign(p);
  }

  double __complex__  *create_complexAlign(unsigned int n) {
    return (double __complex__ * ) utils::ComplexAlign(n);
  }

  void delete_complexAlign(double __complex__ * p) {
    utils::deleteAlign(p);
  }

  // Wrappers for multiple threads
  unsigned int get_fftwpp_maxthreads() {
    return fftw::maxthreads;
  }

  void set_fftwpp_maxthreads(unsigned int nthreads) {
    fftw::maxthreads = nthreads;
  }

  // 1d complex wrappers
  HybridConvolution *fftwpp_create_conv1d(unsigned int L) {
    return new HybridConvolution(L);
  }

  HybridConvolution *fftwpp_create_conv1dAB(unsigned int L, unsigned int M,
                                      unsigned int A,
                                      unsigned int B) {
    return new HybridConvolution(L,multbinary,M,A,B);
  }

  HybridConvolution *fftwpp_create_correlate1d(unsigned int L) {
    return new HybridConvolution(L,multcorrelation);
  }

  void fftwpp_conv1d_delete(HybridConvolution *conv) {
    delete conv;
  }

  void fftwpp_conv1d_convolve(HybridConvolution *conv,
                              double __complex__ *a, double __complex__ *b) {
    Complex *F[]={(Complex *) a,(Complex *) b};
    conv->convolve->convolve(F);
  }

  
  void fftwpp_conv1d_correlate(HybridConvolution *conv,
                               double __complex__ *a, double __complex__ *b) {
    Complex *F[]={(Complex *) a,(Complex *) b};
    conv->convolve->convolve(F);
    
  }

  void fftwpp_conv1d_autoconvolve(HybridConvolution *conv,
                                  double __complex__ *a) {
    Complex *F[]={(Complex *) a,(Complex *) a};                     
    conv->convolve->convolve(F);
  }

  void fftwpp_conv1d_autocorrelate(HybridConvolution *conv,
                                   double __complex__ *a) {
    Complex *F[]={(Complex *) a,(Complex *) a};                     
    conv->convolve->convolve(F);
  }
  
  // 1d Hermitian symmetric wrappers
  HybridConvolutionHermitian *fftwpp_create_hconv1d(unsigned int Lx) {
    return new HybridConvolutionHermitian(Lx);
  }

  void fftwpp_hconv1d_delete(HybridConvolutionHermitian *conv) {
    delete conv;
  }

  void fftwpp_hconv1d_convolve(HybridConvolutionHermitian *hconv,
                               double __complex__ *a, double __complex__ *b) {
    Complex *F[]={(Complex *) a,(Complex *) b};
    hconv->convolve->convolve(F);
  }

  // 2d non-centered complex convolution
  HybridConvolution2 *fftwpp_create_conv2d(unsigned int Lx,
                                             unsigned int Ly) {
    return new HybridConvolution2(Lx, Ly);
  }

  HybridConvolution2 *fftwpp_create_correlate2d(unsigned int Lx, unsigned int Ly) {
    return new HybridConvolution2(Lx,Ly,multcorrelation);
  }

  void fftwpp_conv2d_delete(HybridConvolution2 *pconv) {
    delete pconv;
  }

  void fftwpp_conv2d_convolve(HybridConvolution2 *conv,
                              double __complex__ *a, double __complex__ *b) 
  {
    Complex *F[]={(Complex *) a,(Complex *) b};
    conv->convolve2->convolve(F);
  }


  void fftwpp_conv2d_correlate(HybridConvolution2 *conv,
                               double __complex__ *a, double __complex__ *b) {
    Complex *F[]={(Complex *) a,(Complex *) b}; // assuming reversed b
    conv->convolve2->convolve(F);
  }

  void fftwpp_conv2d_autoconvolve(HybridConvolution2 *conv,
                                  double __complex__ *a) {
    Complex *F[]={(Complex *) a,(Complex *) a};
    conv->convolve2->convolve(F);
  }

  void fftwpp_conv2d_autocorrelate(HybridConvolution2 *conv,
                                   double __complex__ *a) {
    Complex *F[]={(Complex *) a,(Complex *) a}; // assuming reversed a
    conv->convolve2->convolve(F);
  }
  // 2d centered Hermitian-symmetric convolution
  HybridConvolutionHermitian2 *fftwpp_create_hconv2d(unsigned int nx,
                                               unsigned int ny) {
    return new HybridConvolutionHermitian2(nx, ny);
  }

  void fftwpp_hconv2d_delete(HybridConvolutionHermitian2 *hconv2) {
    delete hconv2;
  }

  void fftwpp_hconv2d_convolve(HybridConvolutionHermitian2 *hconv2,
                               double __complex__ *a, double __complex__ *b) {
    Complex *F[]={(Complex *) a,(Complex *) b}; 
    hconv2->convolve2->convolve(F);
  }

  // 3d non-centered complex convolution
  HybridConvolution3 *fftwpp_create_conv3d(unsigned int Lx,
                                             unsigned int Ly,
                                             unsigned int Lz) {
    return new HybridConvolution3(Lx, Ly, Lz);
  }

  HybridConvolution3 *fftwpp_create_correlate3d(unsigned int Lx, unsigned int Ly, unsigned int Lz) {
    return new HybridConvolution3(Lx,Ly,Lz,multcorrelation);
  }

  void fftwpp_conv3d_delete(HybridConvolution3 *pconv) {
    delete pconv;
  }

  void fftwpp_conv3d_convolve(HybridConvolution3 *pconv,
                              double __complex__ *a, double __complex__ *b) {
    Complex *F[]={(Complex *) a,(Complex *) b};
    pconv->convolve3->convolve(F);
  }

  void fftwpp_conv3d_correlate(HybridConvolution3 *conv,
                               double __complex__ *a, double __complex__ *b) {
    Complex *F[]={(Complex *) a,(Complex *) b}; // assuming reversed b
    conv->convolve3->convolve(F);
  }

  void fftwpp_conv3d_autoconvolve(HybridConvolution3 *pconv,
                                  double __complex__ *a) {
    Complex *F[]={(Complex *) a,(Complex *) a};
    pconv->convolve3->convolve(F);
  }

  void fftwpp_conv3d_autocorrelate(HybridConvolution3 *pconv,
                                   double __complex__ *a) {
    Complex *F[]={(Complex *) a,(Complex *) a}; // assuming reversed a
    pconv->convolve3->convolve(F);
  }

  // 3d non-centered complex convolution
  HybridConvolutionHermitian3 *fftwpp_create_hconv3d(unsigned int nx,
                                               unsigned int ny,
                                               unsigned int nz) {
    return new HybridConvolutionHermitian3(nx, ny, nz);
  }

  void fftwpp_hconv3d_delete(HybridConvolutionHermitian3 *pconv) {
    delete pconv;
  }

  void fftwpp_hconv3d_convolve(HybridConvolutionHermitian3 *pconv,
                               double __complex__ *a, double __complex__ *b) {
    Complex *F[]={(Complex *) a,(Complex *) b};
    pconv->convolve3->convolve(F);
  }

  /*
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
    */
  }
}
