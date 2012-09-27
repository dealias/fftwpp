#include<stdio.h>
#include "cfftw++.h"
#include<complex.h>


void init(double complex *f, double complex *g, unsigned int m) 
{
  for(unsigned int k=0; k < m; k++) {
    f[k]=k + (k+1) * I;
    g[k]=k + (2*k +1) * I;
  }
}

void show(double complex *f, unsigned int m) 
{
  for(unsigned int k=0; k < m; k++) 
    printf("(%.2f,%.2f)\n", creal(f[k]), cimag(f[k]));
}


int main()
{
  printf("Example of calling fftw++ convolutions from C:\n");
  unsigned int m=8; /* problem size */
  
  /* input arrays must be aligned */
  double complex *f=create_complexAlign(m);
  double complex *g=create_complexAlign(m);
 
  init(f,g,m); /* set the input data */

  printf("\ninput f:\n");
  show(f,m);
  printf("\ninput g:\n");
  show(g,m);
  
  printf("\n1d non-centered complex convolution:\n");
  ImplicitConvolution *cconv=fftwpp_create_conv1d(m);
  fftwpp_conv1d_convolve(cconv,f,g);
  fftwpp_conv1d_delete(cconv);
  show(f,m);

  init(f,g,m); /* reset the inputs */

  printf("\n1d centered Hermitian-symmetric complex convolution:\n");
  ImplicitHConvolution *conv=fftwpp_create_hconv1d(m);
  fftwpp_hconv1d_convolve(conv,f,g);
  fftwpp_hconv1d_delete(conv);
  show(f,m);

  return 0;
}


