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
  double *create_doubleAlign(size_t n) {
    return (double  *) utils::doubleAlign(n);
  }

  void delete_doubleAlign(double * p) {
    utils::deleteAlign(p);
  }

  double __complex__  *create_complexAlign(size_t n) {
    return (double __complex__ * ) utils::ComplexAlign(n);
  }

  void delete_complexAlign(double __complex__ * p) {
    utils::deleteAlign(p);
  }

  // Wrappers for multiple threads
  size_t get_fftwpp_maxthreads() {
    return fftw::maxthreads;
  }

  void set_fftwpp_maxthreads(size_t nthreads) {
    fftw::maxthreads = nthreads;
  }

  // 1d complex wrappers
  HybridConvolution *fftwpp_create_conv1d(size_t L) {
    return new HybridConvolution(L);
  }

  void fftwpp_conv1d_delete(HybridConvolution *conv) {
    delete conv;
  }

  void fftwpp_conv1d_convolve(HybridConvolution *conv,
                              double __complex__ *a, double __complex__ *b) {
    Complex *F[]={(Complex *) a,(Complex *) b};
    conv->convolve->convolve(F);
  }

  // 1d Hermitian symmetric wrappers
  HybridConvolutionHermitian *fftwpp_create_hconv1d(size_t L) {
    return new HybridConvolutionHermitian(L);
  }

  void fftwpp_hconv1d_delete(HybridConvolutionHermitian *conv) {
    delete conv;
  }

  void fftwpp_HermitianSymmetrize(double __complex__ *f) {
    HermitianSymmetrize((Complex *) f);
  }

  void fftwpp_hconv1d_convolve(HybridConvolutionHermitian *hconv,
                               double __complex__ *a, double __complex__ *b) {
    Complex *F[]={(Complex *) a,(Complex *) b};
    hconv->convolve->convolve(F);
  }

  // 2d non-centered complex convolution
  HybridConvolution2 *fftwpp_create_conv2d(size_t Lx,
                                           size_t Ly) {
    return new HybridConvolution2(Lx, Ly);
  }

  void fftwpp_conv2d_delete(HybridConvolution2 *pconv) {
    delete pconv;
  }

  void fftwpp_HermitianSymmetrizeX(size_t Hx, size_t Hy,
                                   size_t x0, double __complex__ *f) {
    HermitianSymmetrizeX(Hx,Hy,x0,(Complex *) f);
  }

  void fftwpp_conv2d_convolve(HybridConvolution2 *conv,
                              double __complex__ *a, double __complex__ *b)
  {
    Complex *F[]={(Complex *) a,(Complex *) b};
    conv->convolve2->convolve(F);
  }

  // 2d centered Hermitian-symmetric convolution
  HybridConvolutionHermitian2 *fftwpp_create_hconv2d(size_t Lx,
                                                     size_t Ly) {
    return new HybridConvolutionHermitian2(Lx, Ly);
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
  HybridConvolution3 *fftwpp_create_conv3d(size_t Lx,
                                           size_t Ly,
                                           size_t Lz) {
    return new HybridConvolution3(Lx, Ly, Lz);
  }

  void fftwpp_conv3d_delete(HybridConvolution3 *pconv) {
    delete pconv;
  }

  void fftwpp_HermitianSymmetrizeXY(size_t Hx, size_t Hy,
                                    size_t Hz,
                                    size_t x0, size_t y0,
                                    double __complex__ *f) {
    HermitianSymmetrizeXY(Hx,Hy,Hz,x0,y0,(Complex *) f);
  }

  void fftwpp_conv3d_convolve(HybridConvolution3 *pconv,
                              double __complex__ *a, double __complex__ *b) {
    Complex *F[]={(Complex *) a,(Complex *) b};
    pconv->convolve3->convolve(F);
  }

  // 3d non-centered complex convolution
  HybridConvolutionHermitian3 *fftwpp_create_hconv3d(size_t nx,
                                                     size_t ny,
                                                     size_t nz) {
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

  }
}
