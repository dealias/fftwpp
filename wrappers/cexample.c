#include<stdio.h>
#include<complex.h>
#include "cfftw++.h"

void init(double complex *f, double complex *g, size_t m)
{
  for(size_t i=0; i < m; i++) {
    f[i]=i+1+I*(i+3);
    g[i]=i+2+I*(2*i+3);
  }
}

void show(double complex *f, size_t m)
{
  for(size_t i=0; i < m; i++)
    printf("(%.2f,%.2f)\n", creal(f[i]), cimag(f[i]));
}

void init2(double complex* f, double complex* g,
	   size_t mx, size_t my)
{
  for(size_t i=0; i < mx; ++i) {
    for(size_t j=0; j < my; j++) {
      f[i*my+j]=i+1+I*(j+3);
      g[i*my+j]=(i+2)+I*(2*j+3);
    }
  }
}

void show2(double complex* f,
	   size_t mx, size_t my)
{
  size_t pos=0;
  for(size_t i=0; i < mx; i++) {
    for(size_t j=0; j < my; j++) {
      printf("(%.1f,%.1f) ", creal(f[pos]), cimag(f[pos]));
      pos++;
    }
    printf("\n");
  }
}

void init3(double complex *f, double complex *g,
	   size_t mx, size_t my, size_t mz)
{
  size_t pos=0;
  for(size_t i=0; i < mx; ++i) {
    for(size_t j=0; j < my; j++) {
      for(size_t k=0; k < mz; k++) {
	f[pos]=i+1+I*(j+3+k);
	g[pos]=i+k+1+I*(2*j+3+k);
	pos++;
      }
    }
  }
}

void show3(double complex *f,
	   size_t mx, size_t my, size_t mz)
{
  size_t pos=0;
  for(size_t i=0; i < mx; ++i) {
    for(size_t j=0; j < my; j++) {
      for(size_t k=0; k < mz; k++) {
	printf("(%.0f,%.0f) ", creal(f[pos]), cimag(f[pos]));
	pos++;
      }
      printf("\n");
    }
    printf("\n");
  }
}

int main()
{
  printf("Example of calling fftw++ convolutions from C:\n");

  size_t nthreads=2;

  size_t M=2; /* dimension of dot product */
  double overM=1.0/(double) M;

  double complex *pf[M];
  double complex *pg[M];

  int returnflag=0;

  set_fftwpp_maxthreads(nthreads);

  {
    printf("Complex, non-centered 1D example:\n");
    size_t L = 8;

    HybridConvolution *cconv = fftwpp_create_conv1d(L);

    double complex *f = create_complexAlign(L);
    double complex *g = create_complexAlign(L);

    init(f, g, L); /* set the input data */
    printf("Input f:\n");
    show(f, L);
    printf("Input g:\n");
    show(g, L);

    fftwpp_conv1d_convolve(cconv, f, g);

    printf("Output f:\n");
    show(f, L);

    delete_complexAlign(g);
    delete_complexAlign(f);

    fftwpp_conv1d_delete(cconv);
    printf("\n");
  }

  {
    printf("Complex, Hermitian-symmetric 1D example:\n");
    size_t H = 4;

    size_t L = 2*H-1;
    HybridConvolutionHermitian *hconv = fftwpp_create_hconv1d(L);

    double complex *f = create_complexAlign(H);
    double complex *g = create_complexAlign(H);

    init(f, g, H); /* set the input data */
    fftwpp_HermitianSymmetrize(f);

    printf("Input f:\n");
    show(f, H);
    printf("Input g:\n");
    show(g, H);

    fftwpp_hconv1d_convolve(hconv, f, g);

    printf("Output f:\n");
    show(f, H);

    delete_complexAlign(g);
    delete_complexAlign(f);

    fftwpp_hconv1d_delete(hconv);
    printf("\n");
  }

  {
    printf("Complex, non-centered 2D example:\n");
    size_t Lx = 4;
    size_t Ly = 4;

    HybridConvolution2 *cconv2 = fftwpp_create_conv2d(Lx, Ly );

    double complex *f = create_complexAlign(Lx * Ly);
    double complex *g = create_complexAlign(Lx * Ly);

    init2(f, g, Lx, Ly); /* set the input data */
    printf("Input f:\n");
    show2(f, Lx, Ly);
    printf("Input g:\n");
    show2(g, Lx, Ly);

    fftwpp_conv2d_convolve(cconv2, f, g);

    printf("Output f:\n");
    show2(f, Lx, Ly);

    delete_complexAlign(g);
    delete_complexAlign(f);

    fftwpp_conv2d_delete(cconv2);
    printf("\n");
  }

  {
    printf("Complex, Hermitian-symmertic, centered 2D example:\n");
    size_t Hx = 4;
    size_t Hy = 4;

    size_t Lx = 2 * Hx - 1;
    size_t Ly = 2 * Hy - 1;
    HybridConvolutionHermitian2 *hconv2 = fftwpp_create_hconv2d(Lx, Ly);

    double complex *f = create_complexAlign(Lx * Hy);
    double complex *g = create_complexAlign(Lx * Hy);

    init2(f, g, Lx, Hy); /* set the input data */
    size_t x0=Lx/2;
    fftwpp_HermitianSymmetrizeX(Hx,Hy,x0,f);
    fftwpp_HermitianSymmetrizeX(Hx,Hy,x0,g);

    printf("Input f:\n");
    show2(f, Lx, Hy);
    printf("Input g:\n");
    show2(g, Lx, Hy);

    fftwpp_hconv2d_convolve(hconv2, f, g);

    printf("Output f:\n");
    show2(f, Lx, Hy);

    delete_complexAlign(g);
    delete_complexAlign(f);

    fftwpp_hconv2d_delete(hconv2);
    printf("\n");
  }

  {
    printf("Complex, non-centered 3D example:\n");
    size_t Lx = 4;
    size_t Ly = 4;
    size_t Lz = 4;

    HybridConvolution3 *cconv3 = fftwpp_create_conv3d(Lx, Ly, Lz);

    double complex *f = create_complexAlign(Lx * Ly * Lz);
    double complex *g = create_complexAlign(Lx * Ly * Lz);

    init3(f, g, Lx, Ly, Lz); /* set the input data */
    printf("Input f:\n");
    show3(f, Lx, Ly, Lz);
    printf("Input g:\n");
    show3(g, Lx, Ly, Lz);

    fftwpp_conv3d_convolve(cconv3, f, g);

    printf("Output f:\n");
    show3(f, Lx, Ly, Lz);

    delete_complexAlign(g);
    delete_complexAlign(f);

    fftwpp_conv3d_delete(cconv3);
    printf("\n");
  }

  {
    printf("Complex, Hermitian-symmetric, centered 3D example:\n");
    size_t Hx = 4;
    size_t Hy = 4;
    size_t Hz = 4;

    size_t Lx = 2 * Hx - 1;
    size_t Ly = 2 * Hy - 1;
    size_t Lz = 2 * Hz - 1;

    HybridConvolutionHermitian3 *hconv3 = fftwpp_create_hconv3d(Lx, Ly, Lz);

    double complex *f = create_complexAlign(Lx * Ly * Hz);
    double complex *g = create_complexAlign(Lx * Ly * Hz);

    init3(f, g, Lx, Ly, Hz); /* set the input data */
    size_t x0=Lx/2;
    size_t y0=Ly/2;
    fftwpp_HermitianSymmetrizeXY(Hx,Hy,Hz,x0,y0,f);
    fftwpp_HermitianSymmetrizeXY(Hx,Hy,Hz,x0,y0,g);

    printf("Input f:\n");
    show3(f, Lx, Ly, Hz);
    printf("Input g:\n");
    show3(g, Lx, Ly, Hz);

    fftwpp_hconv3d_convolve(hconv3, f, g);

    printf("Output f:\n");
    show3(f, Lx, Ly, Hz);

    delete_complexAlign(g);
    delete_complexAlign(f);

    fftwpp_hconv3d_delete(hconv3);
    printf("\n");
  }

}
