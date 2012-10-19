#include<stdio.h>
#include "cfftw++.h"
#include "chash.h"
#include<complex.h>


void init(double complex *f, double complex *g, unsigned int m,
	  unsigned int M) 
{
  for(unsigned int k=0; k < m; k++) {
    for(unsigned int s=0; s < M; s++) {
      f[k+s*m]=k + (k+1) * I;
      g[k+s*m]=k + (2*k +1) * I;
    }
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
  
  /* 1D examples */
  { 
    unsigned int m=8; /* problem size */

    /* input arrays must be aligned */
    double complex *f=create_complexAlign(m*M);
    double complex *g=create_complexAlign(m*M);

    /* optional work arrays */
    double complex *u=create_complexAlign(m);
    double complex *v=create_complexAlign(m);
    
    init(f,g,m,M); /* set the input data */
    
    printf("\ninput f:\n");
    show(f,m);
    printf("\ninput g:\n");
    show(g,m);

    /* for M > 1 */
    initMpointers(f,pf,M,m);
    initMpointers(g,pg,M,m);

    printf("\n1d non-centered complex convolution:\n");
    /* ImplicitConvolution *cconv=fftwpp_create_conv1d(m); */
    ImplicitConvolution *cconv=fftwpp_create_conv1d_dot(m,M);
    /* fftwpp_conv1d_convolve(cconv,f,g); */
    fftwpp_conv1d_convolve_dot(cconv,pf,pg);
    fftwpp_conv1d_delete(cconv);

    normalize(f,m,overM);
    show(f,m);

    /* compare hash of output for unit test: */
    if(m == 8) {
      if(hash(f,m) != -1208058208) {
	printf("ImplicitConvolution output incorect.\n");
	returnflag += 1;
      }
    }

    init(f,g,m,M);
    
    /* optional work arrays (one more for the Hermitian convolution) */
    double complex *w=create_complexAlign(3);

    printf("\n1d centered Hermitian-symmetric complex convolution:\n");
    /* ImplicitHConvolution *conv=fftwpp_create_hconv1d(m); */
    /* ImplicitHConvolution *conv=fftwpp_create_hconv1d_work(m,u,v,w); */
    ImplicitHConvolution *conv=fftwpp_create_hconv1d_dot(m,M);
    /* fftwpp_hconv1d_convolve(conv,f,g); */
    fftwpp_hconv1d_convolve_dot(conv,pf,pg);
    fftwpp_hconv1d_delete(conv);

    normalize(f,m,overM);
    show(f,m);

    /* compare hash of output for unit test: */
    if(m == 8) {
      if(hash(f,m) != -1208087538) {
	printf("ImplicitHConvolution output incorect.\n");
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
    double complex *f=create_complexAlign(M*mx*my);
    double complex *g=create_complexAlign(M*mx*my);

    initMpointers(f,pf,M,mx*my);
    initMpointers(g,pg,M,mx*my);

    /* optional work arrays */
    double complex *u1=create_complexAlign(my*nthreads);
    double complex *v1=create_complexAlign(my*nthreads);
    double complex *u2=create_complexAlign(mx*my);
    double complex *v2=create_complexAlign(mx*my);

    /* 2D arrays for convenience */
    initM2(f,g,mx,my,M);
    

    /*
    printf("\ninput f:\n");
    show2(f,mx,my);
    printf("\ninput g:\n");
    show2(g,mx,my);
    */    

    /* ImplicitConvolution2 *cconv=fftwpp_create_conv2d(mx,my); */
 /*ImplicitConvolution2 *cconv=fftwpp_create_conv2d_work(mx,my,u1,v1,u2,v2);*/
    ImplicitConvolution2 *cconv=fftwpp_create_conv2d_dot(mx,my,M);
    /* fftwpp_conv2d_convolve(cconv,f,g); */
    fftwpp_conv2d_convolve_dot(cconv,pf,pg);
    fftwpp_conv2d_delete(cconv);


    printf("\noutput:\n");
    normalize(f,mx*my,overM);
    show2(f,mx,my);

    /* compare hash of output for unit test: */
    if(mx==4 && my==4) {
      if(hash(f,mx*my) != -268695633) {
	printf("ImplicitConvolution2 output incorect.\n");
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
    double complex *f=create_complexAlign(M*Mx*my);
    double complex *g=create_complexAlign(M*Mx*my);

    initMpointers(f,pf,M,Mx*my);
    initMpointers(g,pg,M,Mx*my);

    /* optional work arrays */
    double complex *u1=create_complexAlign((my/2+1)*nthreads);
    double complex *v1=create_complexAlign((my/2+1)*nthreads);
    double complex *w1=create_complexAlign(3*nthreads);
    double complex *u2=create_complexAlign((mx+1)*my);
    double complex *v2=create_complexAlign((mx+1)*my);

    initM2(f,g,Mx,my,M);
    
    /*
    printf("\ninput f:\n");
    show2(f,Mx,my);
    printf("\ninput g:\n");
    show2(g,Mx,my);
    */    

    /* ImplicitHConvolution2 *conv=fftwpp_create_hconv2d(mx,my); */
    /* ImplicitHConvolution2 *conv=fftwpp_create_hconv2d_work(mx,my, */
    /* 							   u1,v1,w1,u2,v2); */
    ImplicitHConvolution2 *conv=fftwpp_create_hconv2d_dot(mx,my,M); 
    /* fftwpp_hconv2d_convolve(conv,f,g); */
    fftwpp_hconv2d_convolve_dot(conv,pf,pg);
    fftwpp_hconv2d_delete(conv);

    printf("\noutput:\n");
    normalize(f,Mx*my,overM);
    show2(f,Mx,my);

    /* compare hash of output for unit test: */
    if(mx==4 && my==4) {
      if(hash(f,Mx*my) != -947771835) {
	printf("ImplicitHConvolution2 output incorect.\n");
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
    double complex *f=create_complexAlign(M*mxyz);
    double complex *g=create_complexAlign(M*mxyz);

    initMpointers(f,pf,M,mxyz);
    initMpointers(g,pg,M,mxyz);

    /* optional work arrays */
    double complex *u1=create_complexAlign(mz*nthreads);
    double complex *v1=create_complexAlign(mz*nthreads);
    double complex *u2=create_complexAlign(my*my*nthreads);
    double complex *v2=create_complexAlign(mz*my*nthreads);
    double complex *u3=create_complexAlign(mx*my*mz);
    double complex *v3=create_complexAlign(mx*my*mz);
    
    initM3(f,g,mx,my,mz,M);

    /*
    printf("\ninput f:\n");
    show3(f,mx,my,mz);
    printf("\ninput g:\n");
    show3(g,mx,my,mz);
    */    

    /* ImplicitConvolution3 *cconv=fftwpp_create_conv3d(mx,my,mz); */
    ImplicitConvolution3 *cconv=fftwpp_create_conv3d_dot(mx,my,mz,M);
    /* fftwpp_conv3d_convolve(cconv,f,g);  */
    fftwpp_conv3d_convolve_dot(cconv,pf,pg); 
    fftwpp_conv3d_delete(cconv);
    
    printf("\noutput:\n");
    normalize(f,mx*my*mz,overM);
    show3(f,mx,my,mz);

    /* compare hash of output for unit test: */
    if(mx==4 && my==4 && mz==4) {
      if(hash(f,mx*my*mz) != 1073436205) {
	printf("ImplicitConvolution3 output incorect.\n");
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
    printf("\n3d centered Hermitian convolution:\n");
    
    unsigned int mx=4, my=4, mz=4;  /* problem size */
    unsigned int Mx=2*mx-1;
    unsigned int My=2*my-1;

    unsigned int mxyz=Mx*My*mz;
    double complex *f=create_complexAlign(M*mxyz);
    double complex *g=create_complexAlign(M*mxyz);

    initMpointers(f,pf,M,mxyz);
    initMpointers(g,pg,M,mxyz);

    /* optional work arrays */
    double complex *u1=create_complexAlign((mz/2+1)*nthreads);
    double complex *v1=create_complexAlign((mz/2+1)*nthreads);
    double complex *w1=create_complexAlign(3*nthreads);
    double complex *u2=create_complexAlign((my+1)*mz*nthreads);
    double complex *v2=create_complexAlign((my+1)*mz*nthreads);
    double complex *u3=create_complexAlign((mx+1)*(2*my-1)*mz);
    double complex *v3=create_complexAlign((mx+1)*(2*my-1)*mz);

    for(unsigned int s=0; s < M; ++s) 
      init3(pf[s],pg[s],Mx,My,mz);

    /*
    printf("\ninput f:\n");
    show3(f,Mx,My,mz);
    printf("\ninput g:\n");
    show3(g,Mx,My,mz);
    */    

    /* ImplicitHConvolution3 *conv=fftwpp_create_hconv3d(mx,my,mz); */
    /* ImplicitHConvolution3 *conv=fftwpp_create_hconv3d_work(mx,my,mz, */
    /* 							   u1,v1,w1, */
    /* 							   u2,v2,u3,v3); */
    ImplicitHConvolution3 *conv=fftwpp_create_hconv3d_dot(mx,my,mz,M);
    /* fftwpp_hconv3d_convolve(conv,f,g);  */
    fftwpp_hconv3d_convolve_dot(conv,pf,pg); 
    fftwpp_hconv3d_delete(conv);

    printf("\noutput:\n");
    normalize(f,Mx*My*mz,overM);
    show3(f,Mx,My,mz);

    /* compare hash of output for unit test: */    
    if(mx==4 && my==4 && mz==4) {
      if(hash(f,mxyz) != -472674783) {
	printf("ImplicitHConvolution3 output incorect.\n");
	returnflag += 32;
      }
    }

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
  double complex *pe[M];
  {
    printf("\n1d centered Hermitian-symmetric ternary convolution:\n");
    unsigned int m=12; /* problem size */
    unsigned int m1=m+1;
    double complex *e=create_complexAlign(M*m1);
    double complex *f=create_complexAlign(M*m1);
    double complex *g=create_complexAlign(M*m1);

    initMpointers(e,pe,M,m1);
    initMpointers(f,pf,M,m1);
    initMpointers(g,pg,M,m1);

    /* optional work arrays */
    double complex *u=create_complexAlign(m1);
    double complex *v=create_complexAlign(m1);
    double complex *w=create_complexAlign(m1);

    for(unsigned int s=0; s < M; ++s) {
      double complex *ei=e+s*m1;
      double complex *fi=f+s*m1;
      double complex *gi=g+s*m1;
      ei[0]=1.0;
      fi[0]=1.0;
      gi[0]=2.0;
      for(unsigned int k=1; k < m; k++) {
	ei[k]=k+I*(k+1);
	fi[k]=k+I*(k+1);
	gi[k]=k+I*(2*k+1);
      }
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
    /* ImplicitHTConvolution *conv=fftwpp_create_htconv1d_work(m,u,v,w); */
    ImplicitHTConvolution *conv=fftwpp_create_htconv1d_dot(m,M);
    /* fftwpp_htconv1d_convolve(conv,e,f,g); */
    fftwpp_htconv1d_convolve_dot(conv,pe,pf,pg);
    fftwpp_htconv1d_delete(conv);

    normalize(e,m,overM);

    printf("\noutput:\n");
    show(e,m);

    /* compare hash of output for unit test: */
    if(m==12) {
      if(hash(e,m) != -778218684) {
	printf("ImplicitHTConvolution output incorect.\n");
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

    double complex *e=create_complexAlign(M*Mxy1);
    double complex *f=create_complexAlign(M*Mxy1);
    double complex *g=create_complexAlign(M*Mxy1);

    initMpointers(e,pe,M,Mxy1);
    initMpointers(f,pf,M,Mxy1);
    initMpointers(g,pg,M,Mxy1);

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

    for(unsigned int s=0; s < M; ++s) {
      unsigned int sMxy1=s*Mxy1;
      double complex *ei=e+sMxy1;
      double complex *fi=f+sMxy1;
      double complex *gi=g+sMxy1;
      int j,pos;
      unsigned int stop=2*mx-1;
      for(i=0; i < stop; i++) {
	int ii=i+1;
	for(j=0; j < my; j++) {
	  pos=ii*(my+1)+j;
	  ei[pos]=i+I*j;
	  fi[pos]=2.0*((i+1.0)+I*(j+2.0));
	  gi[pos]=0.5*((2.0*i)+I*(j+1.0));
	}
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
    /* ImplicitHTConvolution2 *conv=fftwpp_create_htconv2d_work(mx,my, */
    /* 							     u1,v1,w1, */
    /* 							     u2,v2,w2); */
    ImplicitHTConvolution2 *conv=fftwpp_create_htconv2d_dot(mx,my,M);
    /* fftwpp_htconv2d_convolve(conv,e,f,g); */
    fftwpp_htconv2d_convolve_dot(conv,pe,pf,pg);
    fftwpp_htconv2d_delete(conv);

    /* set unused array elements to zero for presentation's sake */
    for(i=0; i < my1; i++) e[i]=0.0;
    for(i=0; i < 2*mx; i++) e[i*my1+mx]=0.0;

    printf("\noutput:\n");
    normalize(e,2*mx*my1,overM);
    show2(e,2*mx,my1);

    /* compare hash of output for unit test: */
    if(mx==4 && my==4) {
      if(hasht2(e,mx,my) != 1432369516) {
	printf("ImplicitHTConvolution2 output incorect.\n");
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


