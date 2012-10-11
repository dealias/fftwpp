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

void init2(double complex** F, double complex** G,
	   unsigned int mx, unsigned int my)
{
  for(unsigned int i=0; i < mx; ++i) {
    for(unsigned int j=0; j < my; j++) {
      F[i][j]=i+j*I;
      G[i][j]=2*i+(j+1)*I;
    }
  }
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

int hash(double complex *a, unsigned int m)
{
  int hash=0;
  unsigned int i;
  for(i=0; i<m; ++i) {
    hash= (hash+(324723947+(int)(creal(a[i])+0.5)))^93485734985;
    hash= (hash+(324723947+(int)(cimag(a[i])+0.5)))^93485734985;
  }
  return hash;
}

int hasht2(double complex *a, unsigned int mx, unsigned int my)
{
  int hash=0;
  unsigned int i;
  
    int j,pos;
    unsigned int stop=2*mx-1;
    for(i=0; i < stop; i++) {
      int ii=i+1;
      for(j=0; j < my; j++) {
	pos=ii*(my+1)+j;
	hash= (hash+(324723947+(int)(creal(a[pos])+0.5)))^93485734985;
	hash= (hash+(324723947+(int)(cimag(a[pos])+0.5)))^93485734985;
      }
    }
  return hash;
}
int main()
{
  printf("Example of calling fftw++ convolutions from C:\n");
  
  unsigned int nthreads=2;

  int returnflag=0;

  set_fftwpp_maxthreads(nthreads);
  
  /* 1D examples */
  { 
    unsigned int m=8; /* problem size */

    /* input arrays must be aligned */
    double complex *f=create_complexAlign(m);
    double complex *g=create_complexAlign(m);

    /* optional work arrays */
    double complex *u=create_complexAlign(m);
    double complex *v=create_complexAlign(m);
    double complex *w=create_complexAlign(3);
    
    init(f,g,m); /* set the input data */
    
    /*
    printf("\ninput f:\n");
    show(f,m);
    printf("\ninput g:\n");
    show(g,m);
    */    

    printf("\n1d non-centered complex convolution:\n");
    /* ImplicitConvolution *cconv=fftwpp_create_conv1d(m); */
    ImplicitConvolution *cconv=fftwpp_create_conv1d_work(m,u,v);
    fftwpp_conv1d_convolve(cconv,f,g);
    fftwpp_conv1d_delete(cconv);
    show(f,m);
    if(m == 8) {
      if(hash(f,m) != -1208058208) {
	printf("ImplicitConvolution output incorecct.\n");
	returnflag += 1;
      }
    }

    init(f,g,m);
    
    printf("\n1d centered Hermitian-symmetric complex convolution:\n");
    /* ImplicitHConvolution *conv=fftwpp_create_hconv1d(m); */
    ImplicitHConvolution *conv=fftwpp_create_hconv1d_work(m,u,v,w);
    fftwpp_hconv1d_convolve(conv,f,g);
    fftwpp_hconv1d_delete(conv);
    show(f,m);

    if(m == 8) {
      if(hash(f,m) != -1208087538) {
	printf("ImplicitHConvolution output incorecct.\n");
	returnflag += 2;
      }
    }

    /* free memory */
    delete_complexAlign(g);
    delete_complexAlign(f);

    delete_complexAlign(u);
    delete_complexAlign(v);
    delete_complexAlign(w);
  }

   /* 2D examples */
  { 
    printf("\n2d non-centered complex convolution:\n");
    unsigned int mx=4, my=4;  /* problem size */
    double complex *f=create_complexAlign(mx*my);
    double complex *g=create_complexAlign(mx*my);

    /* optional work arrays */
    double complex *u1=create_complexAlign(my*nthreads);
    double complex *v1=create_complexAlign(my*nthreads);
    double complex *u2=create_complexAlign(mx*my);
    double complex *v2=create_complexAlign(mx*my);

    /* 2D arrays for convenience */
    double complex* (F)[mx];
    double complex* (G)[mx];
    int i;
    for(i=0; i < mx; ++i)  {
      F[i]=&f[i*my];
      G[i]=&g[i*my];
    }

    init2(F,G,mx,my);

    /*
    printf("\ninput f:\n");
    show2(f,mx,my);
    printf("\ninput g:\n");
    show2(g,mx,my);
    */    

    /* ImplicitConvolution2 *cconv=fftwpp_create_conv2d(mx,my); */
    ImplicitConvolution2 *cconv=fftwpp_create_conv2d_work(mx,my,u1,v1,u2,v2);
    fftwpp_conv2d_convolve(cconv,f,g);
    fftwpp_conv2d_delete(cconv);
    printf("\noutput:\n");
    show2(f,mx,my);

    if(mx==4 && my==4) {
      if(hash(f,mx*my) != -268695633) {
	printf("ImplicitConvolution2 output incorecct.\n");
	returnflag += 4;
      }
    }

    delete_complexAlign(g);
    delete_complexAlign(f);

    delete_complexAlign(v1);
    delete_complexAlign(u1);
    delete_complexAlign(v2);
    delete_complexAlign(u2);
  }

  {
    printf("\n2d centered Hermitian-symmetric convolution:\n");
    unsigned int mx=4, my=4;  /* problem size */
    
    unsigned int Mx=2*mx-1;
    double complex *f=create_complexAlign(Mx*my);
    double complex *g=create_complexAlign(Mx*my);

    /* optional work arrays */
    double complex *u1=create_complexAlign((my/2+1)*nthreads);
    double complex *v1=create_complexAlign((my/2+1)*nthreads);
    double complex *w1=create_complexAlign(3*nthreads);
    double complex *u2=create_complexAlign((mx+1)*my);
    double complex *v2=create_complexAlign((mx+1)*my);

    
    /* 2D arrays for convenience */
    double complex* (F)[Mx];
    double complex* (G)[Mx];
    int i;
    for(i=0; i < Mx; ++i)  {
      F[i]=&f[i*my];
      G[i]=&g[i*my];
    }

    init2(F,G,Mx,my);
    
    /*
    printf("\ninput f:\n");
    show2(f,Mx,my);
    printf("\ninput g:\n");
    show2(g,Mx,my);
    */    

    /* ImplicitHConvolution2 *conv=fftwpp_create_hconv2d(mx,my); */
    ImplicitHConvolution2 *conv=fftwpp_create_hconv2d_work(mx,my,
							   u1,v1,w1,u2,v2);
    fftwpp_hconv2d_convolve(conv,f,g);
    fftwpp_hconv2d_delete(conv);

    printf("\noutput:\n");
    show2(f,Mx,my);
    if(mx==4 && my==4) {
      if(hash(f,Mx*my) != -947771835) {
	printf("ImplicitHConvolution2 output incorecct.\n");
	returnflag += 8;
      }
    }
 
    

    delete_complexAlign(g);
    delete_complexAlign(f);

    delete_complexAlign(w1);
    delete_complexAlign(v1);
    delete_complexAlign(u1);
    delete_complexAlign(v2);
    delete_complexAlign(u2);
  }

   /* 3D examples */
  {
    printf("\n3d non-centered complex convolution:\n");
    
    unsigned int mx=4, my=4, mz=4;  /* problem size */
    unsigned int mxyz=mx*my*mz;
    double complex *f=create_complexAlign(mxyz);
    double complex *g=create_complexAlign(mxyz);

    /* optional work arrays */
    double complex *u1=create_complexAlign(mz*nthreads);
    double complex *v1=create_complexAlign(mz*nthreads);
    double complex *u2=create_complexAlign(my*my*nthreads);
    double complex *v2=create_complexAlign(mz*my*nthreads);
    double complex *u3=create_complexAlign(mx*my*mz);
    double complex *v3=create_complexAlign(mx*my*mz);
    
    init3(f,g,mx,my,mz);

    /*
    printf("\ninput f:\n");
    show3(f,mx,my,mz);
    printf("\ninput g:\n");
    show3(g,mx,my,mz);
    */    

    ImplicitConvolution3 *cconv=fftwpp_create_conv3d(mx,my,mz);
    fftwpp_conv3d_convolve(cconv,f,g); 
    fftwpp_conv3d_delete(cconv);
    
    printf("\noutput:\n");
    show3(f,mx,my,mz);

    if(mx==4 && my==4 && mz==4) {
      if(hash(f,mx*my*mz) != 1073436205) {
	printf("ImplicitConvolution3 output incorecct.\n");
	returnflag += 16;
      }
    }
    

    delete_complexAlign(g);
    delete_complexAlign(f);

    delete_complexAlign(v1);
    delete_complexAlign(u1);
    delete_complexAlign(v2);
    delete_complexAlign(u2);
    delete_complexAlign(v3);
    delete_complexAlign(u3);

  }

  {
    printf("\n3d non-centered complex convolution:\n");
    
    unsigned int mx=4, my=4, mz=4;  /* problem size */
    unsigned int Mx=2*mx-1;
    unsigned int My=2*my-1;

    unsigned int mxyz=Mx*My*mz;
    double complex *f=create_complexAlign(mxyz);
    double complex *g=create_complexAlign(mxyz);

    /* optional work arrays */
    double complex *u1=create_complexAlign((mz/2+1)*nthreads);
    double complex *v1=create_complexAlign((mz/2+1)*nthreads);
    double complex *w1=create_complexAlign(3*nthreads);
    double complex *u2=create_complexAlign((my+1)*mz*nthreads);
    double complex *v2=create_complexAlign((my+1)*mz*nthreads);
    double complex *u3=create_complexAlign((mx+1)*(2*my-1)*mz);
    double complex *v3=create_complexAlign((mx+1)*(2*my-1)*mz);

    init3(f,g,Mx,My,mz);

    /*
    printf("\ninput f:\n");
    show3(f,Mx,My,mz);
    printf("\ninput g:\n");
    show3(g,Mx,My,mz);
    */    

    /* ImplicitHConvolution3 *conv=fftwpp_create_hconv3d(mx,my,mz); */
    ImplicitHConvolution3 *conv=fftwpp_create_hconv3d_work(mx,my,mz,
							   u1,v1,w1,
							   u2,v2,u3,v3);
    fftwpp_hconv3d_convolve(conv,f,g); 
    fftwpp_hconv3d_delete(conv);
    
    if(mx==4 && my==4 && mz==4) {
      if(hash(f,mxyz) != -472674783) {
	printf("ImplicitHConvolution3 output incorecct.\n");
	returnflag += 32;
      }
    }
    
    printf("\noutput:\n");
    show3(f,Mx,My,mz);

    delete_complexAlign(g);
    delete_complexAlign(f);

    delete_complexAlign(w1);
    delete_complexAlign(v1);
    delete_complexAlign(u1);
    delete_complexAlign(v2);
    delete_complexAlign(u2);
    delete_complexAlign(v3);
    delete_complexAlign(u3);
  }

  /* Ternary convolutions */
  {
    printf("\n1d centered Hermitian-symmetric ternary convolution:\n");
    unsigned int m=12; /* problem size */
    unsigned int m1=m+1;
    double complex *e=create_complexAlign(m1);
    double complex *f=create_complexAlign(m1);
    double complex *g=create_complexAlign(m1);

    /* optional work arrays */
    double complex *u=create_complexAlign(m1);
    double complex *v=create_complexAlign(m1);
    double complex *w=create_complexAlign(m1);

    e[0]=1.0;
    f[0]=1.0;
    g[0]=2.0;
    for(unsigned int k=1; k < m; k++) {
      e[k]=k+I*(k+1);
      f[k]=k+I*(k+1);
      g[k]=k+I*(2*k+1);
    }

    /*    
    printf("\ninput e:\n");
    show(e,m);
    printf("\ninput f:\n");
    show(f,m);
    printf("\ninput g:\n");
    show(g,m);
    */

    /* ImplicitHTConvolution *conv=fftwpp_create_htconv1d(m); */
    ImplicitHTConvolution *conv=fftwpp_create_htconv1d_work(m,u,v,w);
    fftwpp_htconv1d_convolve(conv,e,f,g);
    fftwpp_htconv1d_delete(conv);
    printf("\noutput:\n");
    show(e,m);

    if(m==12) {
      if(hash(e,m) != -778218684) {
	printf("ImplicitHTConvolution output incorecct.\n");
	returnflag += 64;
      }
    }

    delete_complexAlign(g);
    delete_complexAlign(f);
    delete_complexAlign(e);

    delete_complexAlign(w);
    delete_complexAlign(v);
    delete_complexAlign(u);
  }

  {
    printf("\n2d centered Hermitian-symmetric ternary convolution:\n");
    
    unsigned int mx=4, my=4;  /* problem size */
    unsigned int Mx=2*mx, my1=my+1;
    unsigned int Mxy1=(Mx+1)*my1;

    double complex *e=create_complexAlign(Mxy1);
    double complex *f=create_complexAlign(Mxy1);
    double complex *g=create_complexAlign(Mxy1);

    double complex *u1=create_complexAlign(my1*nthreads);
    double complex *v1=create_complexAlign(my1*nthreads);
    double complex *w1=create_complexAlign(my1*nthreads);
    double complex *u2=create_complexAlign(Mxy1);
    double complex *v2=create_complexAlign(Mxy1);
    double complex *w2=create_complexAlign(Mxy1);

    int i;
    for(i=0; i < Mxy1; i++) {
      e[i]=0.0;
      f[i]=0.0;
      g[i]=0.0;
    }

    int j,pos;
    unsigned int stop=2*mx-1;
    for(i=0; i < stop; i++) {
      int ii=i+1;
      for(j=0; j < my; j++) {
	pos=ii*(my+1)+j;
	e[pos]=i+I*j;
	f[pos]=2.0*((i+1.0)+I*(j+2.0));
	g[pos]=0.5*((2.0*i)+I*(j+1.0));
      }
    }

    /*
    printf("\ninput e:\n");
    show2(e,2*mx,my1);
    printf("\ninput f:\n");
    show2(f,2*mx,my1);
    printf("\ninput g:\n");
    show2(g,2*mx,my1);
    */

    /* ImplicitHTConvolution2 *conv=fftwpp_create_htconv2d(mx,my); */
    ImplicitHTConvolution2 *conv=fftwpp_create_htconv2d_work(mx,my,
							     u1,v1,w1,
							     u2,v2,w2);
    fftwpp_htconv2d_convolve(conv,e,f,g);
    fftwpp_htconv2d_delete(conv);

    /* set unused array elements to zero for presentation's sake */
    for(i=0; i < my1; i++) e[i]=0.0;
    for(i=0; i < 2*mx; i++) e[i*my1+mx]=0.0;

    printf("\noutput:\n");
    show2(e,2*mx,my1);

    if(mx==4 && my==4) {
      if(hasht2(e,mx,my) != 1432369516) {
	printf("ImplicitHTConvolution2 output incorecct.\n");
	returnflag += 128;
      }
    }

    delete_complexAlign(e);
    delete_complexAlign(f);
    delete_complexAlign(g);

    delete_complexAlign(u1);
    delete_complexAlign(v1);
    delete_complexAlign(w1);
    delete_complexAlign(u2);
    delete_complexAlign(v2);
    delete_complexAlign(w2);
  }

  return returnflag;
}


