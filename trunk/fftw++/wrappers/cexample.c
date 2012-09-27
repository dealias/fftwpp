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
  unsigned int m=8;
  
  double complex z1 = 1.0 + 3.0 * I;
  printf("Starting values: Z1 = %.2f + %.2fi\n", creal(z1), cimag(z1));

  double complex *f=create_complexAlign(m);
  double complex *g=create_complexAlign(m);
 
  init(f,g,m);
  printf("input f:\n");
  show(f,m);
  printf("input g:\n");
  show(g,m);
  
  printf("problem size=%u\n",m);

  ImplicitConvolution *conv=fftwpp_create_conv1d(m);
  fftwpp_conv1d_convolve(conv,f,g);
  printf("1d non-centered complex convolution:\n");
  show(f,m);

  return 0;
}


