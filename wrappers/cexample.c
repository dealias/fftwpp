#include<stdio.h>
#include "cfftw++.h"

int main()
{
  printf("Example of calling fftw++ convolutions from C:\n");
  unsigned int m=8;
  
  Complex *f=create_ComplexAlign(m);
  Complex *g=create_ComplexAlign(m);

  printf("problem size=%u\n",m);

  init2(f,g,m);
  printf("input f:\n");
  show(f,m);
  printf("input g:\n");
  show(g,m);

  /* FIXME: warning: initialization makes pointer from integer without a cast*/
  ImplicitConvolution *conv=fftwpp_create_conv1d(m);
  
  fftwpp_conv1d_convolve(conv,f,g);
  printf("1d non-centered complex convolution:\n");
  show(f,m);

  return 0;
}


