#include<stdio.h>
#include<complex.h>
#include "cfftw++.h"

void init(double complex *f, double complex *g, unsigned int m) 
{
  for(unsigned int k=0; k < m; k++) {
    f[k] = k + (k + 1) * I;
    g[k] = k + (2 * k + 1) * I;
  }
}

void show(double complex *f, unsigned int m) 
{
  for(unsigned int k=0; k < m; k++) 
    printf("(%.2f,%.2f)\n", creal(f[k]), cimag(f[k]));
}

void init2(double complex* f, double complex* g,
	   unsigned int mx, unsigned int my)
{
  for(unsigned int i=0; i < mx; ++i) {
    for(unsigned int j=0; j < my; j++) {
      f[i*my+j]=i+j*I;
      g[i*my+j]=2*i+(j+1)*I;
    }
  }
}

void initM2(double complex* f, double complex* g,
	    unsigned int mx, unsigned int my,
	    unsigned int M)
{
  unsigned int stride=mx*my;
  for(unsigned int s=0; s < M; ++s)  
    init2(f+s*stride,g+s*stride,mx,my);
}


void show2(double complex* f, 
	   unsigned int mx, unsigned int my)
{
  int i,j,pos=0;
  for(i=0; i < mx; i++) {
    for(j=0; j < my; j++) {
      printf("(%.1f,%.1f) ", creal(f[pos]), cimag(f[pos]));
      pos++;
    }
    printf("\n");
  }
}

void init3(double complex *f, double complex *g, 
	   unsigned int mx, unsigned int my, unsigned int mz)
{
  int pos=0;
  for(unsigned int i=0; i < mx; ++i) {
    for(unsigned int j=0; j < my; j++) {
      for(unsigned int k=0; k < mz; k++) {
	f[pos]=(i+k) +I*(j+k);
	g[pos]=(2*i+k)+I*(j+1+k);
	pos++;
      }
    }
  }
}

void show3(double complex *f, 
	   unsigned int mx, unsigned int my, unsigned int mz)
{
  int pos=0;
  for(unsigned int i=0; i < mx; ++i) {
    for(unsigned int j=0; j < my; j++) {
      for(unsigned int k=0; k < mz; k++) {
	printf("(%.0f,%.0f) ", creal(f[pos]), cimag(f[pos]));
	pos++;
      }
      printf("\n");
    }
    printf("\n");
  }
}

void initM3(double complex* f, double complex* g,
	    unsigned int mx, unsigned int my, unsigned int mz,
	    unsigned int M)
{
  unsigned int stride=mx*my*mz;
  for(unsigned int s=0; s < M; ++s)  
    init3(f+s*stride,g+s*stride,mx,my,mz);
}

void initMpointers(double complex *f, double complex *F[], 
	   unsigned int M, unsigned int stride)
{
  for(unsigned int s=0; s < M; ++s)
      F[s]=(double complex*) f+s*stride;
}

void normalize(double complex *f, unsigned int N, double overM)
{
  for(unsigned int i=0; i < N; ++i)
    f[i] *= overM;
}


int main()
{
  printf("Example of calling fftw++ convolutions from C:\n");
  
  unsigned int nthreads=2;

  unsigned int M=2; /* dimension of dot product */
  double overM=1.0/(double) M;

  double complex *pf[M];
  double complex *pg[M];
  
  int returnflag=0;

  set_fftwpp_maxthreads(nthreads);

  { 
    printf("Complex, non-centered 1D example:\n");
    unsigned int nx = 8;

    /* ImplicitConvolution *cconv=fftwpp_create_conv1d(m); */
    ImplicitConvolution *cconv = fftwpp_create_conv1d(nx);

    double complex *f = create_complexAlign(nx);
    double complex *g = create_complexAlign(nx);
    
    init(f, g, nx); /* set the input data */
    printf("Input f:\n");
    show(f, nx);
    printf("Input g:\n");
    show(g, nx);
    
    fftwpp_conv1d_convolve(cconv, f, g);
    //fftwpp_conv1d_correlate(cconv, f, g);
    
    printf("Output f:\n");
    show(f, nx);
    
    delete_complexAlign(g);
    delete_complexAlign(f);

    fftwpp_conv1d_delete(cconv);
    printf("\n");
  }
  
  { 
    printf("Complex, Hermitian-symmetric, centered 1D example:\n");
    unsigned int nx = 8;

    ImplicitHConvolution *hconv = fftwpp_create_hconv1d(nx);

    double complex *f = create_complexAlign(nx);
    double complex *g = create_complexAlign(nx);
    
    init(f, g, nx); /* set the input data */
    printf("Input f:\n");
    show(f, nx);
    printf("Input g:\n");
    show(g, nx);
    
    fftwpp_hconv1d_convolve(hconv, f, g);
    
    printf("Output f:\n");
    show(f, nx);
    
    delete_complexAlign(g);
    delete_complexAlign(f);

    fftwpp_hconv1d_delete(hconv);
    printf("\n");
  }

  { 
    printf("Complex, non-centered 2D example:\n");
    unsigned int nx = 4;
    unsigned int ny = 4;

    ImplicitConvolution2 *cconv2 = fftwpp_create_conv2d(nx, ny );

    double complex *f = create_complexAlign(nx * ny);
    double complex *g = create_complexAlign(nx * ny);
    
    init2(f, g, nx, ny); /* set the input data */
    printf("Input f:\n");
    show2(f, nx, ny);
    printf("Input g:\n");
    show2(g, nx, ny);
    
    fftwpp_conv2d_convolve(cconv2, f, g);
    
    printf("Output f:\n");
    show2(f, nx, ny);
    
    delete_complexAlign(g);
    delete_complexAlign(f);

    fftwpp_conv2d_delete(cconv2);
    printf("\n");
  }
    
  { 
    printf("Complex, Hermitian-symmertic, centered 2D example:\n");
    unsigned int nx = 4;
    unsigned int ny = 4;

    ImplicitHConvolution2 *hconv2 = fftwpp_create_hconv2d(nx, ny );

    unsigned int nxp = 2 * nx - 1;
    double complex *f = create_complexAlign(nxp * ny);
    double complex *g = create_complexAlign(nxp * ny);
    
    init2(f, g, nxp, ny); /* set the input data */
    printf("Input f:\n");
    show2(f, nxp, ny);
    printf("Input g:\n");
    show2(g, nxp, ny);
    
    fftwpp_hconv2d_convolve(hconv2, f, g);
    
    printf("Output f:\n");
    show2(f, nxp, ny);
    
    delete_complexAlign(g);
    delete_complexAlign(f);

    fftwpp_hconv2d_delete(hconv2);
    printf("\n");
  }

  { 
    printf("Complex, non-centered 3D example:\n");
    unsigned int nx = 4;
    unsigned int ny = 4;
    unsigned int nz = 4;

    ImplicitConvolution3 *cconv3 = fftwpp_create_conv3d(nx, ny, nz);

    double complex *f = create_complexAlign(nx * ny * nz);
    double complex *g = create_complexAlign(nx * ny * nz);
    
    init3(f, g, nx, ny, nz); /* set the input data */
    printf("Input f:\n");
    show3(f, nx, ny, nz);
    printf("Input g:\n");
    show3(g, nx, ny, nz);
    
    fftwpp_conv3d_convolve(cconv3, f, g);
    
    printf("Output f:\n");
    show3(f, nx, ny, nz);
    
    delete_complexAlign(g);
    delete_complexAlign(f);

    fftwpp_conv3d_delete(cconv3);
    printf("\n");
  }


  { 
    printf("Complex, non-centered 3D example:\n");
    unsigned int nx = 4;
    unsigned int ny = 4;
    unsigned int nz = 4;

    ImplicitConvolution3 *cconv3 = fftwpp_create_conv3d(nx, ny, nz);

    double complex *f = create_complexAlign(nx * ny * nz);
    double complex *g = create_complexAlign(nx * ny * nz);
    
    init3(f, g, nx, ny, nz); /* set the input data */
    printf("Input f:\n");
    show3(f, nx, ny, nz);
    printf("Input g:\n");
    show3(g, nx, ny, nz);
    
    fftwpp_conv3d_convolve(cconv3, f, g);
    
    printf("Output f:\n");
    show3(f, nx, ny, nz);
    
    delete_complexAlign(g);
    delete_complexAlign(f);

    fftwpp_conv3d_delete(cconv3);
    printf("\n");
  }
  
  { 
    printf("Complex, Hermitian-symmetric, centered 3D example:\n");
    unsigned int nx = 4;
    unsigned int ny = 4;
    unsigned int nz = 4;
    
    ImplicitHConvolution3 *hconv3 = fftwpp_create_hconv3d(nx, ny, nz);

    unsigned int nxp = 2 * nx - 1;
    unsigned int nyp = 2 * ny - 1;
    
    double complex *f = create_complexAlign(nxp * nyp * nz);
    double complex *g = create_complexAlign(nxp * nyp * nz);
    
    init3(f, g, nxp, nyp, nz); /* set the input data */
    printf("Input f:\n");
    show3(f, nxp, nyp, nz);
    printf("Input g:\n");
    show3(g, nxp, nyp, nz);
    
    fftwpp_hconv3d_convolve(hconv3, f, g);
    
    printf("Output f:\n");
    show3(f, nxp, nyp, nz);
    
    delete_complexAlign(g);
    delete_complexAlign(f);

    fftwpp_hconv3d_delete(hconv3);
    printf("\n");
  }
}


