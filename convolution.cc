#include "Complex.h"
#include "convolution.h"

using namespace fftwpp;

namespace fftwpp {

#ifdef __SSE2__
const union uvec sse2_pm = {
  { 0x00000000,0x00000000,0x00000000,0x80000000 }
};

const union uvec sse2_mm = {
  { 0x00000000,0x80000000,0x00000000,0x80000000 }
};
#endif

inline unsigned int min(unsigned int a, unsigned int b)
{
  return (a < b) ? a : b;
}

const double sqrt3=sqrt(3.0);
const double hsqrt3=0.5*sqrt3;

const Complex hSqrt3(hsqrt3,hsqrt3);
const Complex mhsqrt3(-hsqrt3,-hsqrt3);
const Complex mhalf(-0.5,-0.5);
const Complex zeta3(-0.5,hsqrt3);

unsigned int BuildZeta(unsigned int n, unsigned int m,
                       Complex *&ZetaH, Complex *&ZetaL)
{
  unsigned int s=(int) sqrt((double) m);
  unsigned int t=m/s;
  if(s*t < m) ++t;
  static const double twopi=2.0*M_PI;
  double arg=twopi/n;
  ZetaH=ComplexAlign(t);
  for(unsigned int a=0; a < t; ++a) {
    double theta=s*a*arg;
    ZetaH[a]=Complex(cos(theta),sin(theta));
  }
  ZetaL=ComplexAlign(s);
  for(unsigned int b=0; b < s; ++b) {
    double theta=b*arg;
    ZetaL[b]=Complex(cos(theta),sin(theta));
  }
  return s;
}

void ExplicitConvolution::pad(Complex *f)
{
  for(unsigned int k=m; k < n; ++k) f[k]=0.0;
}
  
void ExplicitConvolution::backwards(Complex *f)
{
  Backwards->fft(f);
}
  
void ExplicitConvolution::forwards(Complex *f)
{
  Forwards->fft(f);
}
  
void ExplicitConvolution::convolve(Complex *f, Complex *g)
{
  pad(f);
  backwards(f);
  
  pad(g);
  backwards(g);
      
  double ninv=1.0/n;
#ifdef __SSE2__      
  const Complex ninv2(ninv,ninv);
  Vec Ninv=LOAD(&ninv2);
  for(unsigned int k=0; k < n; ++k)
    STORE(f+k,Ninv*ZMULT(LOAD(f+k),LOAD(g+k)));
#else    
  for(unsigned int k=0; k < n; ++k)
    f[k] *= g[k]*ninv;
#endif    
	
  forwards(f);
}

void ImplicitConvolution::convolve(Complex **F, Complex **G,
                                   unsigned int offset)
{
  // all 6M FFTs are out-of-place
    
  for(unsigned int i=0; i < M; ++i) {
    unsigned int im=i*m;
    Backwards->fft(F[i]+offset,u+im);
    Backwards->fft(G[i]+offset,v+im);
  }

  mult(u,V);

  for(unsigned int i=0; i < M; ++i) {
    Complex *f=F[i]+offset;
    Complex *g=G[i]+offset;
    for(unsigned int a=0, k=0; k < m; ++a) {
      unsigned int stop=min(k+s,m);
      Complex *ZetaL0=ZetaL-k;
#ifdef __SSE2__      
      Vec Zeta=LOAD(ZetaH+a);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(CONJ(Zeta),Zeta);
      for(; k < stop; ++k) {
        Vec Zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
        Complex *fki=f+k;
        Complex *gki=g+k;
        Vec Fki=LOAD(fki);
        Vec Gki=LOAD(gki);
        STORE(fki,ZMULT(Zetak,Fki));
        STORE(gki,ZMULT(Zetak,Gki));
      }
#else
      Complex *p=ZetaH+a;
      double Hre=p->re;
      double Him=p->im;
      for(; k < stop; ++k) {
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
    Backwards->fft(f,v+i*m);
    Backwards->fft(g,f);
  }
    
  mult(v,F,offset);

  Complex *f=F[0]+offset;
  Forwards->fft(u,f);
  Forwards->fft(v,u);

  double ninv=0.5/m;
#ifdef __SSE2__      
  const Complex ninv2(ninv,ninv);
  Vec Ninv=LOAD(&ninv2);
#endif    
  for(unsigned int a=0, k=0; k < m; ++a) {
    unsigned int stop=min(k+s,m);
    Complex *ZetaL0=ZetaL-k;
#ifdef __SSE2__
    Vec Zeta=Ninv*LOAD(ZetaH+a);
    Vec X=UNPACKL(Zeta,Zeta);
    Vec Y=UNPACKH(CONJ(Zeta),Zeta);
    for(; k < stop; ++k) {
      Vec Zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
      Complex *fk=f+k;
      STORE(fk,ZMULTC(Zetak,LOAD(u+k))+Ninv*LOAD(fk));
    }
#else      
    Complex *p=ZetaH+a;
    double Hre=ninv*p->re;
    double Him=ninv*p->im;
    for(; k < stop; ++k) {
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
}

void DirectConvolution::convolve(Complex *h, Complex *f, Complex *g)
{
  for(unsigned int i=0; i < m; ++i) {
    Complex sum=0.0;
    for(unsigned int j=0; j <= i; ++j) sum += f[j]*g[i-j];
    h[i]=sum;
  }
}	

void ExplicitHConvolution::pad(Complex *f)
{
  unsigned int n2=n/2;
  for(unsigned int i=m; i <= n2; ++i) f[i]=0.0;
}
  
void ExplicitHConvolution::backwards(Complex *f)
{
  cr->fft(f);
}
  
void ExplicitHConvolution::forwards(Complex *f)
{
  rc->fft(f);
}

void ExplicitHConvolution::convolve(Complex *f, Complex *g)
{
  pad(f);
  backwards(f);
  
  pad(g);
  backwards(g);
      
  double *F=(double *) f;
  double *G=(double *) g;
    
  double ninv=1.0/n;
  for(unsigned int k=0; k < n; ++k)
    F[k] *= G[k]*ninv;
  
  forwards(f);
}

// Reverse and conjugate an array of length m.
inline void conjreverse(Complex *f, unsigned int m)
{
  unsigned int c=m/2;
  unsigned int m1=m-1;
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
  if(2*c < m) f[c]=conj(f[c]);
}

void ImplicitHConvolution::convolve(Complex **F, Complex **G, 
                                    unsigned int offset)
{
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
    Vec Mhalf=LOAD(&mhalf);
    Vec HSqrt3=LOAD(&hSqrt3);
    for(unsigned int a=0, k=1; k < c; ++a) {
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
      stop=min(k+s,c);
      ZetaL0=ZetaL-k;
    }
#else
    double fmkre=fmk.re;
    double fmkim=fmk.im;
    double gmkre=gmk.re;
    double gmkim=gmk.im;
    for(unsigned int a=0, k=1; k < c; ++a) {
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
      stop=min(k+s,c);
      ZetaL0=ZetaL-k;
    }
#endif
    
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
  unsigned int stop=s;
  Complex *ZetaL0=ZetaL;
#ifdef __SSE2__      
  const Complex ninv2(ninv,ninv);
  Vec Ninv=LOAD(&ninv2);
  Vec Mhalf=LOAD(&mhalf);
  Vec HSqrt3=LOAD(&hSqrt3);
  for(unsigned int a=0, k=1; k < start; ++a) {
    Vec Zeta=Ninv*LOAD(ZetaH+a);
    Vec X=UNPACKL(Zeta,Zeta);
    Vec Y=UNPACKH(CONJ(Zeta),Zeta);
    for(; k < stop; ++k) {
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
    stop=min(k+s,start);
    ZetaL0=ZetaL-k;
  }
#else
  for(unsigned int a=0, k=1; k < start; ++a) {
    Complex *p=ZetaH+a;
    double Hre=ninv*p->re;
    double Him=ninv*p->im;
    for(; k < stop; ++k) {
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
    stop=min(k+s,start);
    ZetaL0=ZetaL-k;
  }
#endif    
    
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

void DirectHConvolution::convolve(Complex *h, Complex *f, Complex *g)
{
  for(unsigned int i=0; i < m; ++i) {
    Complex sum=0.0;
    for(unsigned int j=0; j <= i; ++j) sum += f[j]*g[i-j];
    for(unsigned int j=i+1; j < m; ++j) sum += f[j]*conj(g[j-i]);
    for(unsigned int j=1; j < m-i; ++j) sum += conj(f[j])*g[i+j];
    h[i]=sum;
  }
}	

void fftpad::backwards(Complex *f, Complex *u)
{
  for(unsigned int a=0, k=0; k < m; ++a) {
    unsigned int stop=min(k+s,m);
    Complex *ZetaL0=ZetaL-k;
#ifdef __SSE2__      
    Vec H=LOAD(ZetaH+a);
    for(; k < stop; ++k) {
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
    Complex H=ZetaH[a];
    for(; k < stop; ++k) {
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
    
  Backwards->fft(f);
  Backwards->fft(u);
}
  
void fftpad::forwards(Complex *f, Complex *u)
{
  Forwards->fft(f);
  Forwards->fft(u);

  double ninv=0.5/m;
#ifdef __SSE2__
  const Complex ninv2(ninv,ninv);
  Vec Ninv=LOAD(&ninv2);
#endif    
  for(unsigned int a=0, k=0; k < m; ++a) {
    unsigned int stop=min(k+s,m);
    Complex *ZetaL0=ZetaL-k;
#ifdef __SSE2__      
    Vec H=Ninv*LOAD(ZetaH+a);
    for(; k < stop; ++k) {
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
    Complex H=ninv*ZetaH[a];
    for(; k < stop; ++k) {
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
}

void fft0pad::backwards(Complex *f, Complex *u)
{
  unsigned int m1=m-1;
  unsigned int m1stride=m1*stride;
  Complex *fm1stride=f+m1stride;
  for(unsigned int i=0; i < M; ++i)
    u[i]=fm1stride[i];
    
#ifdef __SSE2__      
  Vec Mhalf=LOAD(&mhalf);
  Vec Mhsqrt3=LOAD(&mhsqrt3);
#endif    
  unsigned int stop=s;
  Complex *ZetaL0=ZetaL;
  for(unsigned int a=0, k=1; k < m; ++a) {
#ifdef __SSE2__
    Vec H=LOAD(ZetaH+a);
    for(; k < stop; ++k) {
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
    Complex H=ZetaH[a];
    for(; k < stop; ++k) {
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
    stop=min(k+s,m);
    ZetaL0=ZetaL-k;
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
  const Complex ninv2(ninv,ninv);
  Vec Ninv=LOAD(&ninv2);
  Vec Mhalf=LOAD(&mhalf);
  Vec HSqrt3=LOAD(&hSqrt3);
#endif    
  unsigned int stop=s;
  Complex *ZetaL0=ZetaL;
  for(unsigned int a=0, k=1; k < m; ++a) {
#ifdef __SSE2__      
    Vec H=LOAD(ZetaH+a)*Ninv;
    for(; k < stop; ++k) {
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
    }
#else
    Complex H=ninv*ZetaH[a];
    for(; k < stop; ++k) {
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
    }
#endif      
    stop=min(k+s,m);
    ZetaL0=ZetaL-k;
  }
  for(unsigned int i=0; i < M; ++i)
    fm1stride[i]=umstride[i];
}

void ExplicitConvolution2::pad(Complex *f)
{
  for(unsigned int i=0; i < mx; ++i) {
    unsigned int nyi=ny*i;
    unsigned int stop=nyi+ny;
    for(unsigned int j=nyi+my; j < stop; ++j)
      f[j]=0.0;
  }
    
  for(unsigned int i=mx; i < nx; ++i) {
    unsigned int nyi=ny*i;
    unsigned int stop=nyi+ny;
    for(unsigned int j=nyi; j < stop; ++j)
      f[j]=0.0;
  }
}

void ExplicitConvolution2::backwards(Complex *f)
{
  if(prune) {
    xBackwards->fft(f);
    yBackwards->fft(f);
  } else
    Backwards->fft(f);
}
  
void ExplicitConvolution2::forwards(Complex *f)
{
  if(prune) {
    yForwards->fft(f);
    xForwards->fft(f);
  } else
    Forwards->fft(f);
}
  
void ExplicitConvolution2::convolve(Complex *f, Complex *g)
{
  pad(f);
  backwards(f);
  
  pad(g);
  backwards(g);
      
  unsigned int n=nx*ny;
  double ninv=1.0/n;
#ifdef __SSE2__      
  const Complex ninv2(ninv,ninv);
  Vec Ninv=LOAD(&ninv2);
  for(unsigned int k=0; k < n; ++k)
    STORE(f+k,Ninv*ZMULT(LOAD(f+k),LOAD(g+k)));
#else    
  for(unsigned int k=0; k < n; ++k)
    f[k] *= g[k]*ninv;
#endif    
	
  forwards(f);
}

void DirectConvolution2::convolve(Complex *h, Complex *f, Complex *g)
{
  for(unsigned int i=0; i < mx; ++i) {
    for(unsigned int j=0; j < my; ++j) {
      Complex sum=0.0;
      for(unsigned int k=0; k <= i; ++k)
        for(unsigned int p=0; p <= j; ++p)
          sum += f[k*my+p]*g[(i-k)*my+j-p];
      h[i*my+j]=sum;
    }
  }
}	

void ExplicitHConvolution2::pad(Complex *f)
{
  unsigned int nyp=ny/2+1;
  unsigned int nx2=nx/2;

  // zero-pad left block
  unsigned int stop=(nx2-mx+1)*nyp;
  for(unsigned int i=0; i < stop; ++i) 
    f[i]=0.0;

  // zero-pad top-middle block
  unsigned int stop2=stop+2*mx*nyp;
  unsigned int diff=nyp-my;
  for(unsigned int i=stop+nyp; i < stop2; i += nyp) {
    for(unsigned int j=i-diff; j < i; ++j)
      f[j]=0.0;
  }

  // zero-pad right block
  stop=nx*nyp;
  for(unsigned int i=(nx2+mx)*nyp; i < stop; ++i) 
    f[i]=0.0;
}

void oddShift(unsigned int nx, unsigned int ny, Complex *f, int sign,
              unsigned int s, Complex *ZetaH, Complex *ZetaL)
{
  unsigned int nyp=ny/2+1;
  int Sign=-1;
  sign=-sign;
  unsigned int stop=s;
  Complex *ZetaL0=ZetaL;
  for(unsigned int a=0, k=1; k < nx; ++a) {
    Complex H=ZetaH[a];
    for(; k < stop; ++k) {
      Complex zeta=Sign*H*ZetaL0[k];
      zeta.im *= sign;
      unsigned int j=nyp*k;
      unsigned int stop=j+nyp;
      for(; j < stop; ++j)
        f[j] *= zeta;
      Sign=-Sign;
    }
    stop=min(k+s,nx);
    ZetaL0=ZetaL-k;
  }
}

void ExplicitHConvolution2::backwards(Complex *f, bool shift)
{
  if(prune) {
    xBackwards->fft(f);
    if(nx % 2 == 0) {
      if(shift) fftw::Shift(f,nx,ny);
    } else oddShift(nx,ny,f,-1,s,ZetaH,ZetaL);
    yBackwards->fft(f);
  } else {
    if(shift)
      Backwards->fft0(f);
    else
      Backwards->fft(f);
  }
}

void ExplicitHConvolution2::forwards(Complex *f)
{
  if(prune) {
    yForwards->fft(f);
    if(nx % 2 == 0) {
      fftw::Shift(f,nx,ny);
    } else oddShift(nx,ny,f,1,s,ZetaH,ZetaL);
    xForwards->fft(f);
  } else
    Forwards->fft0(f);
}

void ExplicitHConvolution2::convolve(Complex *f, Complex *g, bool symmetrize)
{
  unsigned int xorigin=nx/2;
  unsigned int nyp=ny/2+1;
    
  if(symmetrize) HermitianSymmetrizeX(mx,nyp,xorigin,f);
  pad(f);
  backwards(f,false);
  
  if(symmetrize) HermitianSymmetrizeX(mx,nyp,xorigin,g);
  pad(g);
  backwards(g,false);
    
  double *F=(double *) f;
  double *G=(double *) g;
    
  double ninv=1.0/(nx*ny);
  unsigned int nyp2=2*nyp;

  for(unsigned int i=0; i < nx; ++i) {
    unsigned int nyp2i=nyp2*i;
    unsigned int stop=nyp2i+ny;
    for(unsigned int j=nyp2i; j < stop; ++j)
      F[j] *= G[j]*ninv;
  }
	
  forwards(f);
}

void DirectHConvolution2::convolve(Complex *h, Complex *f, Complex *g,
                                   bool symmetrize)
{
  unsigned int xorigin=mx-1;
    
  if(symmetrize) {
    HermitianSymmetrizeX(mx,my,xorigin,f);
    HermitianSymmetrizeX(mx,my,xorigin,g);
  }
    
  int xstart=-(int)xorigin;
  int ystart=1-(int) my;
  int xstop=mx;
  int ystop=my;
  for(int kx=xstart; kx < xstop; ++kx) {
    for(int ky=0; ky < ystop; ++ky) {
      Complex sum=0.0;
      for(int px=xstart; px < xstop; ++px) {
        for(int py=ystart; py < ystop; ++py) {
          int qx=kx-px;
          if(qx >= xstart && qx < xstop) {
            int qy=ky-py;
            if(qy >= ystart && qy < ystop) {
              sum += ((py >= 0) ? f[(xorigin+px)*my+py] : 
                      conj(f[(xorigin-px)*my-py])) *
                ((qy >= 0) ? g[(xorigin+qx)*my+qy] : 
                 conj(g[(xorigin-qx)*my-qy]));
            }
          }
        }
        h[(xorigin+kx)*my+ky]=sum;
      }
    }
  }	
}

void ExplicitConvolution3::pad(Complex *f)
{
  for(unsigned int i=0; i < mx; ++i) {
    unsigned int nyi=ny*i;
    for(unsigned int j=0; j < my; ++j) {
      unsigned int nyzij=nz*(nyi+j);
      unsigned int stop=nyzij+nz;
      for(unsigned int k=nyzij+mz; k < stop; ++k)
        f[k]=0.0;
    }
  }
    
  unsigned int nyz=ny*nz;
  for(unsigned int i=mx; i < nx; ++i) {
    unsigned int nyzi=nyz*i;
    for(unsigned int j=0; j < ny; ++j) {
      unsigned int nyzij=nyzi+nz*j;
      unsigned int stop=nyzij+nz;
      for(unsigned int k=nyzij; k < stop; ++k)
        f[k]=0.0;
    }
  }
    
  for(unsigned int i=0; i < nx; ++i) {
    unsigned int nyzi=nyz*i;
    for(unsigned int j=mx; j < ny; ++j) {
      unsigned int nyzij=nyzi+nz*j;
      unsigned int stop=nyzij+nz;
      for(unsigned int k=nyzij; k < stop; ++k)
        f[k]=0.0;
    }
  }
}

void ExplicitConvolution3::backwards(Complex *f)
{
  unsigned int nyz=ny*nz;
  if(prune) {
    for(unsigned int i=0; i < mx; ++i)
      yBackwards->fft(f+i*nyz);
    xBackwards->fft(f);
    zBackwards->fft(f);
  } else
    Backwards->fft(f);
}
  
void ExplicitConvolution3::forwards(Complex *f)
{
  if(prune) {
    zForwards->fft(f);
    xForwards->fft(f);
    unsigned int nyz=ny*nz;
    for(unsigned int i=0; i < mx; ++i)
      yForwards->fft(f+i*nyz);
  } else
    Forwards->fft(f);
}

void ExplicitConvolution3::convolve(Complex *f, Complex *g)
{
  pad(f);
  backwards(f);
  
  pad(g);
  backwards(g);
    
  unsigned int n=nx*ny*nz;
  double ninv=1.0/n;
#ifdef __SSE2__      
  const Complex ninv2(ninv,ninv);
  Vec Ninv=LOAD(&ninv2);
  for(unsigned int k=0; k < n; ++k)
    STORE(f+k,Ninv*ZMULT(LOAD(f+k),LOAD(g+k)));
#else    
  for(unsigned int k=0; k < n; ++k)
    f[k] *= g[k]*ninv;
#endif    
	
  forwards(f);
}

void DirectConvolution3::convolve(Complex *h, Complex *f, Complex *g)
{
  for(unsigned int i=0; i < mx; ++i) {
    for(unsigned int j=0; j < my; ++j) {
      for(unsigned int k=0; k < mz; ++k) {
        Complex sum=0.0;
        for(unsigned int r=0; r <= i; ++r)
          for(unsigned int p=0; p <= j; ++p)
            for(unsigned int q=0; q <= k; ++q)
              sum += f[r*myz+p*mz+q]*g[(i-r)*myz+(j-p)*mz+(k-q)];
        h[i*myz+j*mz+k]=sum;
      }
    }
  }
}	

void DirectHConvolution3::convolve(Complex *h, Complex *f, Complex *g, 
                                   bool symmetrize)
{
  unsigned int xorigin=mx-1;
  unsigned int yorigin=my-1;
  unsigned int ny=2*my-1;
  
  if(symmetrize) {
    HermitianSymmetrizeXY(mx,my,mz,ny,xorigin,yorigin,f);
    HermitianSymmetrizeXY(mx,my,mz,ny,xorigin,yorigin,g);
  }
    
  int xstart=-(int) xorigin;
  int ystart=-(int) yorigin;
  int zstart=1-(int) mz;
  int xstop=mx;
  int ystop=my;
  int zstop=mz;
  for(int kx=xstart; kx < xstop; ++kx) {
    for(int ky=ystart; ky < ystop; ++ky) {
      for(int kz=0; kz < zstop; ++kz) {
        Complex sum=0.0;
        for(int px=xstart; px < xstop; ++px) {
          for(int py=ystart; py < ystop; ++py) {
            for(int pz=zstart; pz < zstop; ++pz) {
              int qx=kx-px;
              if(qx >= xstart && qx < xstop) {
                int qy=ky-py;
                if(qy >= ystart && qy < ystop) {
                  int qz=kz-pz;
                  if(qz >= zstart && qz < zstop) {
                    sum += ((pz >= 0) ? 
                            f[((xorigin+px)*ny+yorigin+py)*mz+pz] : 
                            conj(f[((xorigin-px)*ny+yorigin-py)*mz-pz])) *
                      ((qz >= 0) ? g[((xorigin+qx)*ny+yorigin+qy)*mz+qz] :    
                       conj(g[((xorigin-qx)*ny+yorigin-qy)*mz-qz]));
                  }
                }
              }
            }
          }
        }
        h[((xorigin+kx)*ny+yorigin+ky)*mz+kz]=sum;
      }
    }
  }	
}

void ExplicitHTConvolution::pad(Complex *f)
{
  unsigned int n2=n/2;
  for(unsigned int i=m; i <= n2; ++i) f[i]=0.0;
}

void ExplicitHTConvolution::backwards(Complex *f)
{
  cr->fft(f);
}

void ExplicitHTConvolution::forwards(Complex *f)
{
  rc->fft(f);
}

void ExplicitHTConvolution::convolve(Complex *f, Complex *g, Complex *h)
{
  pad(f);
  backwards(f);
  
  pad(g);
  backwards(g);
      
  pad(h);
  backwards(h);
	
  double *F=(double *) f;
  double *G=(double *) g;
  double *H=(double *) h;
    
  double ninv=1.0/n;
  for(unsigned int k=0; k < n; ++k)
    F[k] *= G[k]*H[k]*ninv;

  forwards(f);
}

void ImplicitHTConvolution::convolve(Complex **F, Complex **G, Complex **H,
                                      unsigned int offset)
{
  // 8M-3 of 8M FFTs are out-of-place
    
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
    for(unsigned int a=0, k=0; k < m; ++a) {
      unsigned int stop=min(k+s,m);
      Complex *ZetaL0=ZetaL-k;
#ifdef __SSE2__      
      Vec Zeta=LOAD(ZetaH+a);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(CONJ(Zeta),Zeta);
      for(; k < stop; ++k) {
        Vec Zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
        Vec Fk=LOAD(fi+k);
        Vec Gk=LOAD(gi+k);
        Vec Hk=LOAD(hi+k);
        STORE(ui+k,ZMULT(Zetak,Fk));
        STORE(vi+k,ZMULT(Zetak,Gk));
        STORE(wi+k,ZMULT(Zetak,Hk));
      }
#else
      Complex *p=ZetaH+a;
      double Hre=p->re;
      double Him=p->im;
      for(; k < stop; ++k) {
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
  const Complex ninv2(ninv,ninv);
  Vec Ninv=LOAD(&ninv2);
#endif    
  for(unsigned int a=0, k=0; k < m; ++a) {
    unsigned int stop=min(k+s,m);
    Complex *ZetaL0=ZetaL-k;
#ifdef __SSE2__
    Vec Zeta=Ninv*LOAD(ZetaH+a);
    Vec X=UNPACKL(Zeta,Zeta);
    Vec Y=UNPACKH(CONJ(Zeta),Zeta);
    for(; k < stop; ++k) {
      Vec Zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
      STORE(f+k,ZMULTC(Zetak,LOAD(u+k))+Ninv*LOAD(f+k));
    }
#else      
    Complex *p=ZetaH+a;
    double Hre=ninv*p->re;
    double Him=ninv*p->im;
    for(; k < stop; ++k) {
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
}

void DirectHTConvolution::convolve(Complex *h, Complex *e, Complex *f,
                                    Complex *g)
{
  int stop=m;
  int start=1-m;
  for(int k=0; k < stop; ++k) {
    Complex sum=0.0;
    for(int p=start; p < stop; ++p) {
      Complex E=(p >= 0) ? e[p] : conj(e[-p]);
      for(int q=start; q < stop; ++q) {
        int r=k-p-q;
        if(r >= start && r < stop)
          sum += E*
            ((q >= 0) ? f[q] : conj(f[-q]))*
            ((r >= 0) ? g[r] : conj(g[-r]));
      }
    }
    h[k]=sum;
  }
}

void fft0bipad::backwards(Complex *f, Complex *u)
{
  for(unsigned int i=0; i < M; ++i)
    f[i]=0.0;
  for(unsigned int i=0; i < M; ++i)
    u[i]=0.0;
    
  unsigned int twom=2*m;
  unsigned int stop=s;
  Complex *ZetaL0=ZetaL;
  for(unsigned int a=0, k=1; k < twom; ++a) {
#ifdef __SSE2__      
    Vec H=-LOAD(ZetaH+a);
    for(; k < stop; ++k) {
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
    Complex H=ZetaH[a];
    for(; k < stop; ++k) {
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
    stop=min(k+s,twom);
    ZetaL0=ZetaL-k;
  }
    
  Backwards->fft(f);
  Backwards->fft(u);
}

void fft0bipad::forwards(Complex *f, Complex *u)
{
  Forwards->fft(f);
  Forwards->fft(u);

  double ninv=0.25/m;
#ifdef __SSE2__
  const Complex ninv2(ninv,ninv);
  Vec Ninv=LOAD(&ninv2);
#endif    
  unsigned int twom=2*m;
  unsigned int stop=s;
  Complex *ZetaL0=ZetaL;
  for(unsigned int a=0, k=1; k < twom; ++a) {
#ifdef __SSE2__      
    Vec H=Ninv*LOAD(ZetaH+a);
    for(; k < stop; ++k) {
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
    Complex H=ninv*ZetaH[a];
    for(; k < stop; ++k) {
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
    stop=min(k+s,twom);
    ZetaL0=ZetaL-k;
  }
}

void ExplicitHTConvolution2::pad(Complex *f)
{
  unsigned int nyp=ny/2+1;
  unsigned int nx2=nx/2;
  unsigned int end=nx2-mx;
  for(unsigned int i=0; i <= end; ++i) {
    unsigned int nypi=nyp*i;
    unsigned int stop=nypi+nyp;
    for(unsigned int j=nypi; j < stop; ++j)
      f[j]=0.0;
  }
    
  for(unsigned int i=nx2+mx; i < nx; ++i) {
    unsigned int nypi=nyp*i;
    unsigned int stop=nypi+nyp;
    for(unsigned int j=nypi; j < stop; ++j)
      f[j]=0.0;
  }
  for(unsigned int i=0; i < nx; ++i) {
    unsigned int nypi=nyp*i;
    unsigned int stop=nypi+nyp;
    for(unsigned int j=nypi+my; j < stop; ++j)
      f[j]=0.0;
  }
}

void ExplicitHTConvolution2::backwards(Complex *f, bool shift)
{
  if(prune) {
    xBackwards->fft(f);
    if(nx % 2 == 0) {
      if(shift) fftw::Shift(f,nx,ny);
    } else oddShift(nx,ny,f,-1,s,ZetaH,ZetaL);
    yBackwards->fft(f);
  } else
    return Backwards->fft(f);
}

void ExplicitHTConvolution2::forwards(Complex *f, bool shift)
{
  if(prune) {
    yForwards->fft(f);
    if(nx % 2 == 0) {
      if(shift) fftw::Shift(f,nx,ny);
    } else oddShift(nx,ny,f,1,s,ZetaH,ZetaL);
    xForwards->fft(f);
  } else
    Forwards->fft(f);
}

void ExplicitHTConvolution2::convolve(Complex *f, Complex *g, Complex *h,
                                       bool symmetrize)
{
  unsigned int xorigin=nx/2;
  unsigned int nyp=ny/2+1;
    
  if(symmetrize) HermitianSymmetrizeX(mx,nyp,xorigin,f);
  pad(f);
  backwards(f,false);
  
  if(symmetrize) HermitianSymmetrizeX(mx,nyp,xorigin,g);
  pad(g);
  backwards(g,false);
      
  if(symmetrize) HermitianSymmetrizeX(mx,nyp,xorigin,h);
  pad(h);
  backwards(h,false);
	
  double *F=(double *) f;
  double *G=(double *) g;
  double *H=(double *) h;
    
  double ninv=1.0/(nx*ny);
  unsigned int nyp2=2*nyp;

  for(unsigned int i=0; i < nx; ++i) {
    unsigned int nyp2i=nyp2*i;
    unsigned int stop=nyp2i+ny;
    for(unsigned int j=nyp2i; j < stop; ++j)
      F[j] *= G[j]*H[j]*ninv;
  }
	
  forwards(f,false);
}

void DirectHTConvolution2::convolve(Complex *h, Complex *e, Complex *f,
                                     Complex *g, bool symmetrize)
{
  if(symmetrize) {
    HermitianSymmetrizeX(mx,my,mx-1,e);
    HermitianSymmetrizeX(mx,my,mx-1,f);
    HermitianSymmetrizeX(mx,my,mx-1,g);
  }
    
  unsigned int xorigin=mx-1;
  int xstart=-(int) xorigin;
  int xstop=mx;
  int ystop=my;
  int ystart=1-(int) my;
  for(int kx=xstart; kx < xstop; ++kx) {
    for(int ky=0; ky < ystop; ++ky) {
      Complex sum=0.0;
      for(int px=xstart; px < xstop; ++px) {
        for(int py=ystart; py < ystop; ++py) {
          Complex E=(py >= 0) ? e[(xorigin+px)*my+py] : 
            conj(e[(xorigin-px)*my-py]);
          for(int qx=xstart; qx < xstop; ++qx) {
            for(int qy=ystart; qy < ystop; ++qy) {
              int rx=kx-px-qx;
              if(rx >= xstart && rx < xstop) {
                int ry=ky-py-qy;
                if(ry >= ystart && ry < ystop) {
                  sum += E *
                    ((qy >= 0) ? f[(xorigin+qx)*my+qy] : 
                     conj(f[(xorigin-qx)*my-qy])) *
                    ((ry >= 0) ? g[(xorigin+rx)*my+ry] : 
                     conj(g[(xorigin-rx)*my-ry]));
                }
              }
            }
          }
        }
        h[(xorigin+kx)*my+ky]=sum;
      }
    }
  }	
}

}

