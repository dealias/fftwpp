#include "Complex.h"
#include "convolution.h"

namespace fftwpp {

#ifdef __SSE2__
const union uvec sse2_pm = {
  { 0x00000000,0x00000000,0x00000000,0x80000000 }
};

const union uvec sse2_mm = {
  { 0x00000000,0x80000000,0x00000000,0x80000000 }
};
#endif

#ifndef FFTWPP_SINGLE_THREAD
#define PARALLEL(code)                                  \
  if(threads > 1) {                                     \
    _Pragma("omp parallel for num_threads(threads)")    \
      code                                              \
      } else {                                          \
    code                                                \
      }
#else
#define PARALLEL(code)                          \
  {                                             \
    code                                        \
      }
#endif

inline unsigned int min(unsigned int a, unsigned int b)
{
  return (a < b) ? a : b;
}

inline unsigned int max(unsigned int a, unsigned int b)
{
  return (a > b) ? a : b;
}

const double sqrt3=sqrt(3.0);
const double hsqrt3=0.5*sqrt3;
const Complex zeta3(-0.5,hsqrt3);

unsigned int BuildZeta(unsigned int n, unsigned int m,
                       Complex *&ZetaH, Complex *&ZetaL, unsigned int threads)
{
  unsigned int s=(int) sqrt((double) m);
  unsigned int t=m/s;
  if(s*t < m) ++t;
  static const double twopi=2.0*M_PI;
  double arg=twopi/n;
  ZetaH=ComplexAlign(t);
  
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif    
  for(unsigned int a=0; a < t; ++a) {
    double theta=s*a*arg;
    ZetaH[a]=Complex(cos(theta),sin(theta));
  }
  ZetaL=ComplexAlign(s);
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif    
  for(unsigned int b=0; b < s; ++b) {
    double theta=b*arg;
    ZetaL[b]=Complex(cos(theta),sin(theta));
  }
  return s;
}

// a[0][k]=sum_i a[i][k]*b[i][k]
void ImplicitConvolution::mult(Complex *a, Complex **B, unsigned int offset) {
  if(M == 1) { // a[k]=a[k]*b[k]
    Complex *B0=B[0]+offset;
    PARALLEL(
      for(unsigned int k=0; k < m; ++k) {
        Complex *p=a+k;
#ifdef __SSE2__      
        STORE(p,ZMULT(LOAD(p),LOAD(B0+k)));
#else
        Complex ak=*p;
        Complex bk=*(B0+k);
        p->re=ak.re*bk.re-ak.im*bk.im;
        p->im=ak.re*bk.im+ak.im*bk.re;
#endif      
      }
      )
      } else if(M == 2) {
    Complex *a1=a+m;
    Complex *B0=B[0]+offset;
    Complex *B1=B[1]+offset;
    PARALLEL(
      for(unsigned int k=0; k < m; ++k) {
        Complex *p=a+k;
#ifdef __SSE2__
        STORE(p,ZMULT(LOAD(p),LOAD(B0+k))+ZMULT(LOAD(a1+k),LOAD(B1+k)));
#else
        Complex ak=*p;
        Complex bk=B0[k];
        double re=ak.re*bk.re-ak.im*bk.im;
        double im=ak.re*bk.im+ak.im*bk.re;
        ak=a1[k];
        bk=B1[k];
        re += ak.re*bk.re-ak.im*bk.im;
        im += ak.re*bk.im+ak.im*bk.re; 
        p->re=re;
        p->im=im;
#endif      
      }
      )
      } else if(M == 3) {
    Complex *a1=a+m;
    Complex *a2=a+2*m;
    Complex *B0=B[0]+offset;
    Complex *B1=B[1]+offset;
    Complex *B2=B[2]+offset;
    PARALLEL(
      for(unsigned int k=0; k < m; ++k) {
        Complex *p=a+k;
#ifdef __SSE2__
        STORE(p,ZMULT(LOAD(p),LOAD(B0+k))+ZMULT(LOAD(a1+k),LOAD(B1+k))
              +ZMULT(LOAD(a2+k),LOAD(B2+k)));
#else
        Complex ak=*p;
        Complex bk=B0[k];
        double re=ak.re*bk.re-ak.im*bk.im;
        double im=ak.re*bk.im+ak.im*bk.re;
        ak=a1[k];
        bk=B1[k];
        re += ak.re*bk.re-ak.im*bk.im;
        im += ak.re*bk.im+ak.im*bk.re; 
        ak=a2[k];
        bk=B2[k];
        re += ak.re*bk.re-ak.im*bk.im;
        im += ak.re*bk.im+ak.im*bk.re; 
        p->re=re;
        p->im=im;
#endif      
      }
      )
      } else {
    Complex *A=a-offset;
    Complex *B0=B[0];
    unsigned int stop=offset+m;
    PARALLEL(
      for(unsigned int k=offset; k < stop; ++k) {
        Complex *p=A+k;
#ifdef __SSE2__      
        Vec sum=ZMULT(LOAD(p),LOAD(B0+k));
        for(unsigned int i=1; i < M; ++i)
          sum += ZMULT(LOAD(p+m*i),LOAD(B[i]+k));
        STORE(p,sum);
#else
        Complex ak=*p;
        Complex bk=B0[k];
        double re=ak.re*bk.re-ak.im*bk.im;
        double im=ak.re*bk.im+ak.im*bk.re;
        for(unsigned int i=1; i < M; ++i) {
          Complex ak=p[m*i];
          Complex bk=B[i][k];
          re += ak.re*bk.re-ak.im*bk.im;
          im += ak.re*bk.im+ak.im*bk.re; 
        }
        p->re=re;
        p->im=im;
#endif      
      }
      )
      }
}

void ImplicitConvolution::convolve(Complex **F, Complex **G,
                                   Complex *u, Complex **V,
                                   unsigned int offset)
{
  // all 6M FFTs are out-of-place
  Complex *v=V[0];

  for(unsigned int i=0; i < M; ++i) {
    unsigned int im=i*m;
    Backwards->fft(F[i]+offset,u+im);
    Backwards->fft(G[i]+offset,v+im);
  }
  
  mult(u,V);

  for(unsigned int i=0; i < M; ++i) {
    Complex *f=F[i]+offset;
    Complex *g=G[i]+offset;
    PARALLEL(
      for(unsigned int K=0; K < m; K += s) {
        Complex *ZetaL0=ZetaL-K;
        unsigned int stop=min(K+s,m);
#ifdef __SSE2__      
        Vec Zeta=LOAD(ZetaH+K/s);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(CONJ(Zeta),Zeta);
        for(unsigned int k=K; k < stop; ++k) {
          Vec Zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
          Complex *fki=f+k;
          Complex *gki=g+k;
          Vec Fki=LOAD(fki);
          Vec Gki=LOAD(gki);
          STORE(fki,ZMULT(Zetak,Fki));
          STORE(gki,ZMULT(Zetak,Gki));
        }
#else
        Complex *p=ZetaH+K/s;
        double Hre=p->re;
        double Him=p->im;
        for(unsigned int k=K; k < stop; ++k) {
          Complex *P=f+k;
          Complex *Q=g+k;
          Complex fk=*P;
          Complex gk=*Q;
          Complex L=*(ZetaL0+k);
          double Re=Hre*L.re-Him*L.im;
          double Im=Hre*L.im+Him*L.re;
          P->re=Re*fk.re-Im*fk.im;
          P->im=Im*fk.re+Re*fk.im;
          Q->re=Re*gk.re-Im*gk.im;
          Q->im=Im*gk.re+Re*gk.im;
        }
#endif      
      }
      )
      Backwards->fft(f,v+i*m);
    Backwards->fft(g,f);
  }
    
  mult(v,F,offset);

  Complex *f=F[0]+offset;
  Forwards->fft(u,f);
  Forwards->fft(v,u);

  double ninv=0.5/m;
#ifdef __SSE2__      
  Vec Ninv=LOAD(ninv);
#endif    
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      unsigned int stop=min(K+s,m);
      Complex *ZetaL0=ZetaL-K;
#ifdef __SSE2__
      Vec Zeta=Ninv*LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(CONJ(Zeta),Zeta);
      for(unsigned int k=K; k < stop; ++k) {
        Vec Zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
        Complex *fk=f+k;
        STORE(fk,ZMULTC(Zetak,LOAD(u+k))+Ninv*LOAD(fk));
      }
#else      
      Complex *p=ZetaH+K/s;
      double Hre=ninv*p->re;
      double Him=ninv*p->im;
      for(unsigned int k=K; k < stop; ++k) {
        Complex *p=f+k;
        Complex fk=*p;
        Complex fkm=*(u+k);
        Complex L=*(ZetaL0+k);
        double Re=Hre*L.re-Him*L.im;
        double Im=Him*L.re+Hre*L.im;
        p->re=ninv*fk.re+Re*fkm.re+Im*fkm.im;
        p->im=ninv*fk.im-Im*fkm.re+Re*fkm.im;
      }
#endif
    }
    )
    }

// Reverse and conjugate an array of length m.
void ImplicitHConvolution::conjreverse(Complex *f, unsigned int m)
{
  unsigned int c=m/2;
  unsigned int m1=m-1;
  PARALLEL(
    for(unsigned int k=0; k < c; ++k) {
      Complex *p=f+k;
      Complex *q=f+m1-k;
      double re=p->re;
      double im=p->im;
      p->re=q->re;
      p->im=-q->im;
      q->re=re;
      q->im=-im;
    }
    )
    if(2*c < m) f[c]=conj(f[c]);
}

// Compute a[0][k]=sum_i a[i][k]*b[i][k]
void ImplicitHConvolution::mult(double *a, double **B, unsigned int offset)
{
  unsigned int m1=m-1;
  if(M == 1) { // a[k]=a[k]*b[k]
    double *B0=B[0]+offset;
    PARALLEL(
#ifdef __SSE2__        
      for(unsigned int k=0; k < m1; k += 2) {
        double *p=a+k;
        STORE(p,LOAD(p)*LOAD(B0+k));
      }
      if(odd)
        a[m1] *= B0[m1];
#else       
      for(unsigned int k=0; k < m; ++k)
        a[k] *= B0[k];
#endif
      )
      } else if(M == 2) {
    double *a1=a+stride;
    double *B0=B[0]+offset;
    double *B1=B[1]+offset;
    PARALLEL(
#ifdef __SSE2__        
      for(unsigned int k=0; k < m1; k += 2) {
        double *ak=a+k;
        STORE(ak,LOAD(ak)*LOAD(B0+k)+LOAD(a1+k)*LOAD(B1+k));
      }
      if(odd)
        a[m1]=a[m1]*B0[m1]+a1[m1]*B1[m1];
#else        
      for(unsigned int k=0; k < m; ++k)
        a[k]=a[k]*B0[k]+a1[k]*B1[k];
#endif        
      )
      } else if(M == 3) {
    double *a1=a+stride;
    double *a2=a1+stride;
    double *B0=B[0]+offset;
    double *B1=B[1]+offset;
    double *B2=B[2]+offset;
    PARALLEL(
#ifdef __SSE2__        
      for(unsigned int k=0; k < m1; k += 2) {
        double *ak=a+k;
        STORE(ak,LOAD(ak)*LOAD(B0+k)+LOAD(a1+k)*LOAD(B1+k)+
              LOAD(a2+k)*LOAD(B2+k));
      }
      if(odd)
        a[m1]=a[m1]*B0[m1]+a1[m1]*B1[m1]+a2[m1]*B2[m1];
#else        
      for(unsigned int k=0; k < m; ++k)
        a[k]=a[k]*B0[k]+a1[k]*B1[k]+a2[k]*B2[k];
#endif        
      )
      } else {
    double *A=a-offset;
    double *B0=B[0];
    unsigned int stop=m1+offset;
    PARALLEL(
#ifdef __SSE2__        
      for(unsigned int k=offset; k < stop; k += 2) {
        double *p=A+k;
        Vec sum=LOAD(p)*LOAD(B0+k);
        for(unsigned int i=1; i < M; ++i)
          sum += LOAD(p+i*stride)*LOAD(B[i]+k);
        STORE(p,sum);
      }
      if(odd) {
        double *p=A+stop;
        double sum=(*p)*B0[stop];
        for(unsigned int i=1; i < M; ++i)
          sum += p[i*stride]*B[i][stop];
        *p=sum;
      }
#else        
      for(unsigned int k=offset; k <= stop; ++k) {
        double *p=A+k;
        double sum=(*p)*B0[k];
        for(unsigned int i=1; i < M; ++i)
          sum += p[i*stride]*B[i][k];
        *p=sum;
      }
#endif        
      )
      }
}

void ImplicitHConvolution::convolve(Complex **F, Complex **G, 
                                    Complex **U, Complex *v, Complex *w,
                                    unsigned int offset)
{
  Complex *u=U[0];
  
  if(m == 1) {
    double sum=0.0;
    for(unsigned int i=0; i < M; ++i)
      sum += (F[i]+offset)->re*(G[i]+offset)->re;
    *(F[0]+offset)=sum;
    return;
  }

  bool even=2*c == m;
  unsigned int cp1=c+1;
  
  // 9M-2 of 9M FFTs are out-of-place 
  
  for(unsigned int i=0; i < M; ++i) {
    Complex *fi=F[i]+offset;
    Complex *gi=G[i]+offset;
    unsigned int icp1=i*cp1;
    Complex *ui=u+icp1;
    Complex *vi=v+icp1;
    if(i+1 < M) {
      ui += cp1;
      vi += cp1;
    }
    
    double f0=fi->re;
    double g0=gi->re;

    ui[0]=f0;
    vi[0]=g0;
    Complex fc=fi[c];
    unsigned int m1=m-1;
    Complex fmk=fi[m1];
    fi[m1]=f0;
    Complex gc=gi[c];
    Complex gmk=gi[m1];
    gi[m1]=g0;
    unsigned int stop=s;
    Complex *ZetaL0=ZetaL;
    
#ifdef __SSE2__      
    Vec Zetak;
    Vec Fmk=LOAD(&fmk);
    Vec Gmk=LOAD(&gmk);
    Vec Mhalf=LOAD(-0.5);
    Vec HSqrt3=LOAD(hsqrt3);
#else
    double fmkre=fmk.re;
    double fmkim=fmk.im;
    double gmkre=gmk.re;
    double gmkim=gmk.im;
#endif
    
    unsigned int k=1;
    for(unsigned int a=0; k < c; ++a) {
#ifdef __SSE2__      
      Vec Zeta=LOAD(ZetaH+a);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(CONJ(Zeta),Zeta);
      for(; k < stop; ++k) {
        Zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
        Complex *p=fi+k;
        Complex *q=gi+k;
        Vec A=LOAD(p);
        Vec B=LOAD(q);
        Vec C=Fmk*Mhalf+CONJ(A);
        Vec D=Gmk*Mhalf+CONJ(B);
        STORE(p,A+CONJ(Fmk));
        STORE(q,B+CONJ(Gmk));
        Fmk *= HSqrt3;
        Gmk *= HSqrt3;
        A=ZMULTC(Zetak,UNPACKL(C,Fmk));
        B=ZMULTIC(Zetak,UNPACKH(C,Fmk));
        C=ZMULTC(Zetak,UNPACKL(D,Gmk));
        D=ZMULTIC(Zetak,UNPACKH(D,Gmk));
        STORE(ui+k,A-B);
        STORE(vi+k,C-D);
        p=fi+m1-k;
        Fmk=LOAD(p);
        STORE(p,A+B);
        q=gi+m1-k;
        Gmk=LOAD(q);
        STORE(q,C+D);
      }
#else
      Complex *p=ZetaH+a;
      double Hre=p->re;
      double Him=-p->im;
      for(; k < stop; ++k) {
        Complex *p=fi+k;
        Complex *q=gi+k;
        Complex L=*(ZetaL0+k);
        double Re=Hre*L.re+Him*L.im;
        double Im=Him*L.re-Hre*L.im;
        double re=-0.5*fmkre+p->re;
        double im=hsqrt3*fmkre;
        double Are=Re*re-Im*im;
        double Aim=Re*im+Im*re;
        re=-0.5*fmkim-p->im;
        im=hsqrt3*fmkim;
        p->re += fmkre;
        p->im -= fmkim;
        double Bre=-Re*im-Im*re;
        double Bim=Re*re-Im*im;
        p=ui+k;
        p->re=Are-Bre;
        p->im=Aim-Bim;
        p=fi+m1-k;
        fmkre=p->re;
        fmkim=p->im;
        p->re=Are+Bre;
        p->im=Aim+Bim;

        re=-0.5*gmkre+q->re;
        im=hsqrt3*gmkre;
        Are=Re*re-Im*im;
        Aim=Re*im+Im*re;
        re=-0.5*gmkim-q->im;
        im=hsqrt3*gmkim;
        q->re += gmkre;
        q->im -= gmkim;
        Bre=-Re*im-Im*re;
        Bim=Re*re-Im*im;
        q=vi+k;
        q->re=Are-Bre;
        q->im=Aim-Bim;
        q=gi+m1-k;
        gmkre=q->re;
        gmkim=q->im;
        q->re=Are+Bre;
        q->im=Aim+Bim;
      }
#endif
      ZetaL0=ZetaL-stop;
      stop=min(stop+s,c);
    }
    
    Complex *wi=w+3*i;
    if(even) {
      double A=fc.re;
      double B=sqrt3*fc.im;
      ui[c]=A+B;
      wi[0].re=A-B;
      Complex *fic=fi+c;
      wi[1]=*fic;
      *fic=2.0*A;
    
      A=gc.re;
      B=sqrt3*gc.im;
      vi[c]=A+B;
      wi[0].im=A-B;
      Complex *gic=gi+c;
      wi[2]=*gic;
      *gic=2.0*A;
    } else {
      unsigned int a=c/s;
      Complex Zetak=conj(ZetaH[a]*ZetaL[c-s*a]);
      Complex *fic=fi+c;
#ifdef __SSE2__      
      STORE(&fmk,Fmk);
      STORE(&gmk,Gmk);
#else
      fmk=Complex(fmkre,fmkim);
      gmk=Complex(gmkre,gmkim);
#endif      
      *fic=fc+conj(fmk);
      Complex A=Zetak*(fc.re+zeta3*fmk.re);
      Complex B=Zetak*(fc.im-zeta3*fmk.im);
      ui[c]=Complex(A.re-B.im,A.im+B.re);
      wi[0]=Complex(A.re+B.im,A.im-B.re);
      
      Complex *gic=gi+c;
      *gic=gc+conj(gmk);
      A=Zetak*(gc.re+zeta3*gmk.re);
      B=Zetak*(gc.im-zeta3*gmk.im);
      vi[c]=Complex(A.re-B.im,A.im+B.re);
      wi[1]=Complex(A.re+B.im,A.im-B.re);
    }
    
    // r=-1:
    if(i+1 < M) {
      cro->fft(ui,(double *) (ui-cp1));
      cro->fft(vi,(double *) (vi-cp1));
    } else {
      cr->fft(ui);
      cr->fft(vi);
    }
  }
    
  mult((double *) v,(double **) U);
  rco->fft((double *) v,u); // v is now free

  // r=0:
  for(unsigned int i=0; i < M; ++i) {
    Complex *fi=F[i]+offset;
    unsigned int icp1=i*cp1;
    cro->fft(fi,(double *) (v+icp1));
    cro->fft(G[i]+offset,(double *) fi);
  }
  mult((double *) v,(double **) F,2*offset);
  Complex *f=F[0]+offset;
  rco->fft((double *) v,f);
  
  unsigned int start=m-c-1;
  Complex *fstart=f+start;
  Complex S=*fstart;
  double T=(f+c)->re;

  // r=1:
  unsigned int offsetstart=offset+start;
  unsigned int offsetcm1=offset+c-1;
  for(unsigned int i=0; i < M; ++i) {
    Complex *wi=w+3*i;
    Complex *f1=F[i]+offsetstart;
    Complex *g1=G[i]+offsetstart;
    if(even) {
      f1[0]=wi[0].re;
      f1[1]=wi[1];
      g1[0]=wi[0].im;
      g1[1]=wi[2];
    } else {
      f1[0]=wi[0];
      g1[0]=wi[1];
      conjreverse(f1,cp1);
      conjreverse(g1,cp1);
    }
    
    cro->fft(g1,(double *) (v+i*cp1));
    cro->fft(f1,(double *) (G[i]+offsetcm1));
  }
  mult((double *) v,(double **) G,2*offsetcm1);
  Complex *gcm1=G[0]+offsetcm1;
  rco->fft((double *) v,gcm1);

  double ninv=1.0/(3.0*m);
  f[0]=(f[0].re+gcm1[0].re+u[0].re)*ninv;
  Complex *fm=f+m;
#ifdef __SSE2__      
  Vec Ninv=LOAD(ninv);
  Vec Mhalf=LOAD(-0.5);
  Vec HSqrt3=LOAD(hsqrt3);
#endif
  
  PARALLEL(
    for(unsigned int K=0; K < start; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,start);
#ifdef __SSE2__      
      Vec Zeta=Ninv*LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(CONJ(Zeta),Zeta);
      for(unsigned int k=max(1,K); k < stop; ++k) {
        Complex *p=f+k;
        Complex *s=fm-k;
        Vec Zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
        Vec F0=LOAD(p)*Ninv;
        Vec F1=ZMULTC(Zetak,LOAD(gcm1+k));
        Vec F2=ZMULT(Zetak,LOAD(u+k));
        Vec S=F1+F2;
        STORE(p,F0+S);
        STORE(s,CONJ(F0+Mhalf*S)-HSqrt3*FLIP(F1-F2));
      }
#else
      Complex *p=ZetaH+K/s;
      double Hre=ninv*p->re;
      double Him=ninv*p->im;
      for(unsigned int k=max(1,K); k < stop; ++k) {
        Complex *p=f+k;
        Complex *s=fm-k;
        Complex q=gcm1[k];
        Complex r=u[k];
        Complex L=*(ZetaL0+k);
        double Re=Hre*L.re-Him*L.im;
        double Im=Him*L.re+Hre*L.im;
        double f0re=p->re*ninv;
        double f0im=p->im*ninv;
        double f1re=Re*q.re+Im*q.im;
        double f2re=Re*r.re-Im*r.im;
        double sre=f1re+f2re;
        double f1im=Re*q.im-Im*q.re;
        double f2im=Re*r.im+Im*r.re;
        double sim=f1im+f2im;
        p->re=f0re+sre;
        p->im=f0im+sim;
        s->re=f0re-0.5*sre-hsqrt3*(f1im-f2im);
        s->im=-f0im+0.5*sim-hsqrt3*(f1re-f2re);
      }
#endif    
    }
    )
    
    unsigned int a=start/s;
  Complex Zetak0=ninv*ZetaH[a]*ZetaL[start-s*a];
  S *= ninv;
  Complex f1k=conj(Zetak0)*gcm1[start];
  Complex f2k=Zetak0*u[start];
  f[start]=S+f1k+f2k;
  if(c > 1 || !even) f[c+1]=conj(S+zeta3*f1k)+zeta3*conj(f2k);
  
  if(even)
    f[c]=(T-gcm1[c].re*zeta3-u[c].re*conj(zeta3))*ninv;
}

void fftpad::backwards(Complex *f, Complex *u)
{
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
#ifdef __SSE2__      
      Vec H=LOAD(ZetaH+K/s);
      for(unsigned int k=K; k < stop; ++k) {
        Vec Zetak=ZMULT(H,LOAD(ZetaL0+k));
        Vec X=UNPACKL(Zetak,Zetak);
        Vec Y=UNPACKH(CONJ(Zetak),Zetak);
        unsigned int kstride=k*stride;
        Complex *fk=f+kstride;
        Complex *uk=u+kstride;
        for(unsigned int i=0; i < M; ++i)
          STORE(uk+i,ZMULT(X,Y,LOAD(fk+i)));
      }
#else
      Complex H=ZetaH[K/s];
      for(unsigned int k=K; k < stop; ++k) {
        Complex L=ZetaL0[k];
        double Re=H.re*L.re-H.im*L.im;
        double Im=H.re*L.im+H.im*L.re;
        unsigned int kstride=k*stride;
        Complex *fk=f+kstride;
        Complex *uk=u+kstride;
        for(unsigned int i=0; i < M; ++i) {
          Complex *p=uk+i;
          Complex fki=*(fk+i);
          p->re=Re*fki.re-Im*fki.im;
          p->im=Im*fki.re+Re*fki.im;
        }
      }
#endif      
    }
    )
  
    Backwards->fft(f);
  Backwards->fft(u);
}
  
void fftpad::forwards(Complex *f, Complex *u)
{
  Forwards->fft(f);
  Forwards->fft(u);

  double ninv=0.5/m;
#ifdef __SSE2__
  Vec Ninv=LOAD(ninv);
#endif    
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
#ifdef __SSE2__      
      Vec H=Ninv*LOAD(ZetaH+K/s);
      for(unsigned int k=K; k < stop; ++k) {
        Vec Zetak=ZMULT(H,LOAD(ZetaL0+k));
        Vec X=UNPACKL(Zetak,Zetak);
        Vec Y=UNPACKH(Zetak,CONJ(Zetak));
        unsigned int kstride=k*stride;
        Complex *uk=u+kstride;
        Complex *fk=f+kstride;
        for(unsigned int i=0; i < M; ++i) {
          Complex *p=fk+i;
          STORE(p,LOAD(p)*Ninv+ZMULT(X,Y,LOAD(uk+i)));
        }
      }
#else
      Complex H=ninv*ZetaH[K/s];
      for(unsigned int k=K; k < stop; ++k) {
        Complex L=ZetaL0[k];
        double Re=H.re*L.re-H.im*L.im;
        double Im=H.re*L.im+H.im*L.re;
        unsigned int kstride=k*stride;
        Complex *uk=u+kstride;
        Complex *fk=f+kstride;
        for(unsigned int i=0; i < M; ++i) {
          Complex *p=fk+i;
          Complex fki=*p;
          Complex fkm=*(uk+i);
          p->re=ninv*fki.re+Re*fkm.re+Im*fkm.im;
          p->im=ninv*fki.im-Im*fkm.re+Re*fkm.im;
        }
      }
#endif     
    }
    )
    }

void fft0pad::backwards(Complex *f, Complex *u)
{
  unsigned int m1=m-1;
  unsigned int m1stride=m1*stride;
  Complex *fm1stride=f+m1stride;
  for(unsigned int i=0; i < M; ++i)
    u[i]=fm1stride[i];
    
#ifdef __SSE2__      
  Vec Mhalf=LOAD(-0.5);
  Vec Mhsqrt3=LOAD(-hsqrt3);
#endif    
  unsigned int stop=s;
  
  for(unsigned int K=0; K < m; K += s) {
    Complex *ZetaL0=ZetaL-K;
#ifdef __SSE2__
    Vec H=LOAD(ZetaH+K/s);
    for(unsigned int k=max(1,K); k < stop; ++k) {
      Vec Zetak=ZMULT(H,LOAD(ZetaL0+k));
      Vec X=UNPACKL(Zetak,Zetak);
      Vec Y=UNPACKH(CONJ(Zetak),Zetak);
      unsigned int kstride=k*stride;
      Complex *uk=u+kstride;
      Complex *fk=f+kstride;
      Complex *fmk=fm1stride+kstride;
      for(unsigned int i=0; i < M; ++i) {
        Complex *p=fmk+i;
        Complex *q=f+i;
        Complex *r=fk+i;
        Vec A=LOAD(p);
        Vec B=LOAD(q);
        Vec Z=B*Mhalf+A;
        STORE(q,LOAD(r));
        STORE(r,B+A);
        B *= Mhsqrt3;
        A=ZMULT(X,Y,UNPACKL(Z,B));
        B=ZMULTI(X,Y,UNPACKH(Z,B));
        STORE(p,A+B);
        STORE(uk+i,CONJ(A-B));
      }
    }
#else        
    Complex H=ZetaH[K/s];
    for(unsigned int k=max(1,K); k < stop; ++k) {
      Complex L=ZetaL0[k];
      double Re=H.re*L.re-H.im*L.im;
      double Im=H.re*L.im+H.im*L.re;
      unsigned int kstride=k*stride;
      Complex *uk=u+kstride;
      Complex *fk=f+kstride;
      Complex *fmk=fm1stride+kstride;
      for(unsigned int i=0; i < M; ++i) {
        Complex *p=fmk+i;
        Complex *q=f+i;
        double fkre=q->re;
        double fkim=q->im;
        double fmkre=p->re;
        double fmkim=p->im;
        double re=-0.5*fkre+fmkre;
        double im=-hsqrt3*fkre;
        double Are=Re*re-Im*im;
        double Aim=Re*im+Im*re;
        re=fmkim-0.5*fkim;
        im=-hsqrt3*fkim;
        double Bre=-Re*im-Im*re;
        double Bim=Re*re-Im*im;
        p->re=Are+Bre;
        p->im=Aim+Bim;
        p=uk+i;
        p->re=Are-Bre;
        p->im=Bim-Aim;
        p=fk+i;
        q->re=p->re;
        q->im=p->im;
        p->re=fkre+fmkre;
        p->im=fkim+fmkim;
      }
    }
#endif
    stop=min(stop+s,m);
  }
  
  Backwards->fft(f);
  Complex *umstride=u+m*stride;
  for(unsigned int i=0; i < M; ++i) {
    umstride[i]=fm1stride[i]; // Store extra value here.
    fm1stride[i]=u[i];
  }
    
  Backwards->fft(fm1stride);
  Backwards->fft(u);
}
  

void fft0pad::forwards(Complex *f, Complex *u)
{
  unsigned int m1stride=(m-1)*stride;
  Complex *fm1stride=f+m1stride;
  Forwards->fft(fm1stride);
  Complex *umstride=u+m*stride;
  for(unsigned int i=0; i < M; ++i) {
    Complex temp=umstride[i];
    umstride[i]=fm1stride[i];
    fm1stride[i]=temp;
  }
    
  Forwards->fft(f);
  Forwards->fft(u);

  double ninv=1.0/(3.0*m);
  for(unsigned int i=0; i < M; ++i)
    umstride[i]=(umstride[i]+f[i]+u[i])*ninv;
#ifdef __SSE2__      
  Vec Ninv=LOAD(ninv);
  Vec Mhalf=LOAD(-0.5);
  Vec HSqrt3=LOAD(hsqrt3);
#endif    
  unsigned int stop=s;
  
  for(unsigned int K=0; K < m; K += s) {
    Complex *ZetaL0=ZetaL-K;
#ifdef __SSE2__      
    Vec H=LOAD(ZetaH+K/s)*Ninv;
#else
    Complex H=ninv*ZetaH[K/s];
#endif    
    for(unsigned int k=max(1,K); k < stop; ++k) {
#ifdef __SSE2__      
      Vec Zetak=ZMULT(H,LOAD(ZetaL0+k));
      Vec X=UNPACKL(Zetak,Zetak);
      Vec Y=UNPACKH(CONJ(Zetak),Zetak);
      unsigned int kstride=k*stride;
      Complex *fk=f+kstride;
      Complex *fm1k=fm1stride+kstride;
      Complex *uk=u+kstride;
      for(unsigned int i=0; i < M; ++i) {
        Complex *p=fk+i;
        Complex *q=fm1k+i;
        Vec F0=LOAD(p)*Ninv;
        Vec F1=ZMULT(X,-Y,LOAD(q));
        Vec F2=ZMULT(X,Y,LOAD(uk+i));
        Vec S=F1+F2;
        STORE(p-stride,F0+Mhalf*S+HSqrt3*ZMULTI(F1-F2));
        STORE(q,F0+S);
      }
#else
      Complex L=ZetaL0[k];
      double Re=H.re*L.re-H.im*L.im;
      double Im=H.re*L.im+H.im*L.re;
      unsigned int kstride=k*stride;
      Complex *fk=f+kstride;
      Complex *fm1k=fm1stride+kstride;
      Complex *uk=u+kstride;
      for(unsigned int i=0; i < M; ++i) {
        Complex *p=fk+i;
        Complex *q=fm1k+i;
        Complex z=*q;
        Complex r=uk[i];
        double f0re=p->re*ninv;
        double f0im=p->im*ninv;
        double f1re=Re*z.re+Im*z.im;
        double f1im=Re*z.im-Im*z.re;
        double f2re=Re*r.re-Im*r.im;
        double f2im=Re*r.im+Im*r.re;
        double sre=f1re+f2re;
        double sim=f1im+f2im;
        p -= stride;
        p->re=f0re-0.5*sre-hsqrt3*(f1im-f2im);
        p->im=f0im-0.5*sim+hsqrt3*(f1re-f2re);
        q->re=f0re+sre;
        q->im=f0im+sim;
      }
#endif      
    }
    stop=min(stop+s,m);
  }
    
  for(unsigned int i=0; i < M; ++i)
    fm1stride[i]=umstride[i];
}


// a[0][k]=sum_i a[i][k]*b[i][k]*c[i][k]
void ImplicitHTConvolution::mult(double *a, double *b, double **C,
                                 unsigned int offset)
{
  unsigned int twom=2*m;
  if(M == 1) { // a[k]=a[k]*b[k]*c[k]
    double *C0=C[0]+offset;
    PARALLEL(
#ifdef __SSE2__      
      for(unsigned int k=0; k < twom; k += 2) {
        double *ak=a+k;
        STORE(ak,LOAD(ak)*LOAD(b+k)*LOAD(C0+k));
      }
#else  
      for(unsigned int k=0; k < twom; ++k)
        a[k] *= b[k]*C0[k];
#endif        
      )
      } else if(M == 2) {
    double *a1=a+stride;
    double *b1=b+stride;
    double *C0=C[0]+offset;
    double *C1=C[1]+offset;
    PARALLEL(
#ifdef __SSE2__        
      for(unsigned int k=0; k < twom; k += 2) {
        double *ak=a+k;
        STORE(ak,LOAD(ak)*LOAD(b+k)*LOAD(C0+k)+
              LOAD(a1+k)*LOAD(b1+k)*LOAD(C1+k));
      }
#else        
      for(unsigned int k=0; k < twom; ++k)
        a[k]=a[k]*b[k]*C0[k]+a1[k]*b1[k]*C1[k];
#endif        
      )
      } else if(M == 3) {
    double *a1=a+stride;
    double *a2=a1+stride;
    double *b1=b+stride;
    double *b2=b1+stride;
    double *C0=C[0]+offset;
    double *C1=C[1]+offset;
    double *C2=C[2]+offset;
    PARALLEL(
#ifdef __SSE2__        
      for(unsigned int k=0; k < twom; k += 2) {
        double *ak=a+k;
        STORE(ak,LOAD(ak)*LOAD(b+k)*LOAD(C0+k)+
              LOAD(a1+k)*LOAD(b1+k)*LOAD(C1+k)+
              LOAD(a2+k)*LOAD(b2+k)*LOAD(C2+k));
      }
#else        
      for(unsigned int k=0; k < twom; ++k)
        a[k]=a[k]*b[k]*C0[k]+a1[k]*b1[k]*C1[k]+a2[k]*b2[k]*C2[k];
#endif        
      )
      } else {
    double *A=a-offset;
    double *B=b-offset;
    double *C0=C[0];
    unsigned int stop=twom+offset;
    PARALLEL(
#ifdef __SSE2__        
      for(unsigned int k=offset; k < stop; k += 2) {
        double *p=A+k;
        double *q=B+k;
        Vec sum=LOAD(p)*LOAD(q)*LOAD(C0+k);
        for(unsigned int i=1; i < M; ++i) {
          unsigned int istride=i*stride;
          sum += LOAD(p+istride)*LOAD(q+istride)*LOAD(C[i]+k);
        }
        STORE(p,sum);
      }
#else        
      for(unsigned int k=offset; k < stop; ++k) {
        double *p=A+k;
        double *q=B+k;
        double sum=(*p)*(*q)*C0[k];
        for(unsigned int i=1; i < M; ++i) {
          unsigned int istride=i*stride;
          sum += p[istride]*q[istride]*C[i][k];
        }
        *p=sum;
      }
#endif        
      )
      }
}

void ImplicitHTConvolution::convolve(Complex **F, Complex **G, Complex **H,
                                     Complex *u, Complex *v, Complex **W,
                                     unsigned int offset)
{
  // 8M-3 of 8M FFTs are out-of-place
  Complex *w=W[0];
    
  unsigned int m1=m+1;
  for(unsigned int i=0; i < M; ++i) {
    Complex *fi=F[i]+offset;
    Complex *gi=G[i]+offset;
    Complex *hi=H[i]+offset;
    unsigned int im1=i*m1;
    Complex *ui=u+im1;
    Complex *vi=v+im1;
    Complex *wi=w+im1;
    if(i+1 < M) {
      ui += m1;
      vi += m1;
      wi += m1;
    }
    PARALLEL(
      for(unsigned int K=0; K < m; K += s) {
        Complex *ZetaL0=ZetaL-K;
        unsigned int stop=min(K+s,m);
#ifdef __SSE2__      
        Vec Zeta=LOAD(ZetaH+K/s);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(CONJ(Zeta),Zeta);
        for(unsigned int k=K; k < stop; ++k) {
          Vec Zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
          Vec Fk=LOAD(fi+k);
          Vec Gk=LOAD(gi+k);
          Vec Hk=LOAD(hi+k);
          STORE(ui+k,ZMULT(Zetak,Fk));
          STORE(vi+k,ZMULT(Zetak,Gk));
          STORE(wi+k,ZMULT(Zetak,Hk));
        }
#else
        Complex *p=ZetaH+K/s;
        double Hre=p->re;
        double Him=p->im;
        for(unsigned int k=K; k < stop; ++k) {
          Complex *P=ui+k;
          Complex *Q=vi+k;
          Complex *R=wi+k;
          Complex fk=*(fi+k);
          Complex gk=*(gi+k);
          Complex hk=*(hi+k);
          Complex L=*(ZetaL0+k);
          double Re=Hre*L.re-Him*L.im;
          double Im=Hre*L.im+Him*L.re;
          P->re=Re*fk.re-Im*fk.im;
          P->im=Im*fk.re+Re*fk.im;
          Q->re=Re*gk.re-Im*gk.im;
          Q->im=Im*gk.re+Re*gk.im;
          R->re=Re*hk.re-Im*hk.im;
          R->im=Im*hk.re+Re*hk.im;
        }
#endif      
      }  
      )
  
      ui[m]=0.0;
    vi[m]=0.0;
    wi[m]=0.0;
    
    if(i+1 < M) {
      cro->fft(ui,(double *) (ui-m1));
      cro->fft(vi,(double *) (vi-m1));
      cro->fft(wi,(double *) (wi-m1));
    } else {
      cr->fft(ui);
      cr->fft(vi);
      cr->fft(wi);
    }
  }   
    
  mult((double *) v,(double *) u,(double **) W);
  rco->fft((double *) v,u); // v and w are now free

  for(unsigned int i=0; i < M; ++i) {
    Complex *fi=F[i]+offset;
    Complex *gi=G[i]+offset;
    Complex *hi=H[i]+offset;
    unsigned int im1=i*m1;
    fi[m]=0.0;
    cro->fft(fi,(double *) (v+im1));
    gi[m]=0.0;
    cro->fft(gi,(double *) (w+im1));
    hi[m]=0.0;
    cro->fft(hi,(double *) gi);
  }
  
  mult((double *) v,(double *) w,(double **) G,2*offset);
  Complex *f=F[0]+offset;
  rco->fft((double *) v,f);
    
  double ninv=0.25/m;
#ifdef __SSE2__      
  Vec Ninv=LOAD(ninv);
#endif    
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
#ifdef __SSE2__
      Vec Zeta=Ninv*LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(CONJ(Zeta),Zeta);
      for(unsigned int k=K; k < stop; ++k) {
        Vec Zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
        Complex *fk=f+k;
        STORE(fk,ZMULTC(Zetak,LOAD(u+k))+Ninv*LOAD(fk));
      }
#else      
      Complex *p=ZetaH+K/s;
      double Hre=ninv*p->re;
      double Him=ninv*p->im;
      for(unsigned int k=K; k < stop; ++k) {
        Complex *p=f+k;
        Complex fk=*p;
        Complex fkm=*(u+k);
        Complex L=*(ZetaL0+k);
        double Re=Hre*L.re-Him*L.im;
        double Im=Him*L.re+Hre*L.im;
        p->re=ninv*fk.re+Re*fkm.re+Im*fkm.im;
        p->im=ninv*fk.im-Im*fkm.re+Re*fkm.im;
      }
#endif      
    }
    )
    }

// a[k]=a[k]*b[k]*b[k]
void ImplicitHFGGConvolution::mult(double *a, double *b)
{
  unsigned int twom=2*m;
  PARALLEL(
#ifdef __SSE2__      
    for(unsigned int k=0; k < twom; k += 2) {
      double *ak=a+k;
      STORE(ak,LOAD(ak)*LOAD(ak)*LOAD(b+k));
    }
#else  
    for(unsigned int k=0; k < twom; ++k) {
      double ak=a[k];
      a[k]=ak*ak*b[k];
    }
#endif        
    )
    }

void ImplicitHFGGConvolution::convolve(Complex *f, Complex *g,
                                       Complex *u, Complex *v)
{
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
#ifdef __SSE2__      
      Vec Zeta=LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(CONJ(Zeta),Zeta);
      for(unsigned int k=K; k < stop; ++k) {
        Vec Zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
        Vec Fk=LOAD(f+k);
        Vec Gk=LOAD(g+k);
        STORE(u+k,ZMULT(Zetak,Fk));
        STORE(v+k,ZMULT(Zetak,Gk));
      }
#else
      Complex *p=ZetaH+K/s;
      double Hre=p->re;
      double Him=p->im;
      for(unsigned int k=K; k < stop; ++k) {
        Complex *P=u+k;
        Complex *Q=v+k;
        Complex fk=*(f+k);
        Complex gk=*(g+k);
        Complex L=*(ZetaL0+k);
        double Re=Hre*L.re-Him*L.im;
        double Im=Hre*L.im+Him*L.re;
        P->re=Re*fk.re-Im*fk.im;
        P->im=Im*fk.re+Re*fk.im;
        Q->re=Re*gk.re-Im*gk.im;
        Q->im=Im*gk.re+Re*gk.im;
      }
#endif      
    }  
    )
  
    u[m]=0.0;
  v[m]=0.0;
    
  cr->fft(u);
  cr->fft(v);
    
  mult((double *) v,(double *) u);
  rco->fft((double *) v,u); // v is now free

  g[m]=0.0;
  cro->fft(g,(double *) v);
  
  f[m]=0.0;
  cro->fft(f,(double *) g);
  
  mult((double *) v,(double *) g);
  rco->fft((double *) v,f);
    
  double ninv=0.25/m;
#ifdef __SSE2__      
  Vec Ninv=LOAD(ninv);
#endif    
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
#ifdef __SSE2__
      Vec Zeta=Ninv*LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(CONJ(Zeta),Zeta);
      for(unsigned int k=K; k < stop; ++k) {
        Vec Zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
        Complex *fk=f+k;
        STORE(fk,ZMULTC(Zetak,LOAD(u+k))+Ninv*LOAD(fk));
      }
#else      
      Complex *p=ZetaH+K/s;
      double Hre=ninv*p->re;
      double Him=ninv*p->im;
      for(unsigned int k=K; k < stop; ++k) {
        Complex *p=f+k;
        Complex fk=*p;
        Complex fkm=*(u+k);
        Complex L=*(ZetaL0+k);
        double Re=Hre*L.re-Him*L.im;
        double Im=Him*L.re+Hre*L.im;
        p->re=ninv*fk.re+Re*fkm.re+Im*fkm.im;
        p->im=ninv*fk.im-Im*fkm.re+Re*fkm.im;
      }
#endif      
    }
    )
    }

// a[k]=a[k]^3
void ImplicitHFFFConvolution::mult(double *a)
{
  unsigned int twom=2*m;
  PARALLEL(
#ifdef __SSE2__      
    for(unsigned int k=0; k < twom; k += 2) {
      double *p=a+k;
      Vec ak=LOAD(p);
      STORE(p,ak*ak*ak);
    }
#else  
    for(unsigned int k=0; k < twom; ++k) {
      double ak=a[k];
      a[k]=ak*ak*ak;
    }
#endif        
    )
    }
  
void ImplicitHFFFConvolution::convolve(Complex *f, Complex *u)
{
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
#ifdef __SSE2__      
      Vec Zeta=LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(CONJ(Zeta),Zeta);
      for(unsigned int k=K; k < stop; ++k)
        STORE(u+k,ZMULT(ZMULT(X,Y,LOAD(ZetaL0+k)),LOAD(f+k)));
#else
      Complex *p=ZetaH+K/s;
      double Hre=p->re;
      double Him=p->im;
      for(unsigned int k=K; k < stop; ++k) {
        Complex *P=u+k;
        Complex fk=*(f+k);
        Complex L=*(ZetaL0+k);
        double Re=Hre*L.re-Him*L.im;
        double Im=Hre*L.im+Him*L.re;
        P->re=Re*fk.re-Im*fk.im;
        P->im=Im*fk.re+Re*fk.im;
      }
#endif      
    }  
    )
  
    u[m]=0.0;
  cr->fft(u);
  mult((double *) u);
  rc->fft(u);

  f[m]=0.0;
  cr->fft(f);
  mult((double *) f);
  rc->fft(f);
  double ninv=0.25/m;
#ifdef __SSE2__      
  Vec Ninv=LOAD(ninv);
#endif    
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
#ifdef __SSE2__
      Vec Zeta=Ninv*LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(CONJ(Zeta),Zeta);
      for(unsigned int k=K; k < stop; ++k) {
        Complex *p=f+k;
        STORE(p,ZMULTC(ZMULT(X,Y,LOAD(ZetaL0+k)),LOAD(u+k))+Ninv*LOAD(p));
      }
#else      
      Complex *p=ZetaH+K/s;
      double Hre=ninv*p->re;
      double Him=ninv*p->im;
      for(unsigned int k=K; k < stop; ++k) {
        Complex *p=f+k;
        Complex fk=*p;
        Complex fkm=*(u+k);
        Complex L=*(ZetaL0+k);
        double Re=Hre*L.re-Him*L.im;
        double Im=Him*L.re+Hre*L.im;
        p->re=ninv*fk.re+Re*fkm.re+Im*fkm.im;
        p->im=ninv*fk.im-Im*fkm.re+Re*fkm.im;
      }
#endif      
    }
    )
    }


void fft0bipad::backwards(Complex *f, Complex *u)
{
  for(unsigned int i=0; i < M; ++i)
    f[i]=0.0;
  for(unsigned int i=0; i < M; ++i)
    u[i]=0.0;
    
  unsigned int twom=2*m;
  PARALLEL(
    for(unsigned int K=0; K < twom; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,twom);
#ifdef __SSE2__      
      Vec H=-LOAD(ZetaH+K/s);
      for(unsigned int k=max(1,K); k < stop; ++k) {
        Vec Zetak=ZMULT(H,LOAD(ZetaL0+k));
        Vec X=UNPACKL(Zetak,Zetak);
        Vec Y=UNPACKH(CONJ(Zetak),Zetak);
        unsigned int kstride=k*stride;
        Complex *fk=f+kstride;
        Complex *uk=u+kstride;
        for(unsigned int i=0; i < M; ++i)
          STORE(uk+i,ZMULTI(X,Y,LOAD(fk+i)));
      }
#else
      Complex H=ZetaH[K/s];
      for(unsigned int k=max(1,K); k < stop; ++k) {
        Complex L=ZetaL0[k];
        double Re=H.im*L.re+H.re*L.im;
        double Im=H.im*L.im-H.re*L.re;
        unsigned int kstride=k*stride;
        Complex *fk=f+kstride;
        Complex *uk=u+kstride;
        for(unsigned int i=0; i < M; ++i) {
          Complex *p=uk+i;
          Complex fki=*(fk+i);
          p->re=Re*fki.re-Im*fki.im;
          p->im=Im*fki.re+Re*fki.im;
        }
      }
#endif      
    }
    )
    
    Backwards->fft(f);
  Backwards->fft(u);
}

void fft0bipad::forwards(Complex *f, Complex *u)
{
  Forwards->fft(f);
  Forwards->fft(u);

  double ninv=0.25/m;
#ifdef __SSE2__
  Vec Ninv=LOAD(ninv);
#endif    
  unsigned int twom=2*m;
  PARALLEL(
    for(unsigned int K=0; K < twom; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,twom);
#ifdef __SSE2__      
      Vec H=Ninv*LOAD(ZetaH+K/s);
      for(unsigned int k=max(1,K); k < stop; ++k) {
        Vec Zetak=ZMULT(H,LOAD(ZetaL0+k));
        Vec X=UNPACKL(Zetak,Zetak);
        Vec Y=UNPACKH(Zetak,CONJ(Zetak));
        unsigned int kstride=k*stride;
        Complex *uk=u+kstride;
        Complex *fk=f+kstride;
        for(unsigned int i=0; i < M; ++i) {
          Complex *p=fk+i;
          STORE(p,LOAD(p)*Ninv+ZMULTI(X,Y,LOAD(uk+i)));
        }
      }
#else        
      Complex H=ninv*ZetaH[K/s];
      for(unsigned int k=max(1,K); k < stop; ++k) {
        Complex L=ZetaL0[k];
        double Re=H.im*L.re+H.re*L.im;
        double Im=H.re*L.re-H.im*L.im;
        unsigned int kstride=k*stride;
        Complex *uk=u+kstride;
        Complex *fk=f+kstride;
        for(unsigned int i=0; i < M; ++i) {
          Complex *p=fk+i;
          Complex fki=*p;
          Complex fkm=*(uk+i);
          p->re=ninv*fki.re+Re*fkm.re-Im*fkm.im;
          p->im=ninv*fki.im+Im*fkm.re+Re*fkm.im;
        }
      }
#endif     
    }
    )
    }

}
