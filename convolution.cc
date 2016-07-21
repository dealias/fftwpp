#include "convolution.h"

using namespace std;
using namespace utils;

namespace fftwpp {

#ifdef __SSE2__
const union uvec sse2_pm = {
  { 0x00000000,0x00000000,0x00000000,0x80000000 }
};

const union uvec sse2_mm = {
  { 0x00000000,0x80000000,0x00000000,0x80000000 }
};
#endif

const double sqrt3=sqrt(3.0);
const double hsqrt3=0.5*sqrt3;
const Complex zeta3(-0.5,hsqrt3);

// Build zeta table, returning the floor of the square root of m.
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

void ImplicitConvolution::convolve(Complex **F, multiplier *pmult,
                                   unsigned int i, unsigned int offset)
{ 
  if(indexsize >= 1) index[indexsize-1]=i;
  
  unsigned int C=max(A,B);
  Complex *P[C];
  for(unsigned int a=0; a < C; ++a)
    P[a]=F[a]+offset;
  
  // Backwards FFT (even indices):
  for(unsigned int a=0; a < A; ++a) {
    BackwardsO->fft(P[a],U[a]);
  }
  
  (*pmult)(U,m,indexsize,index,0,threads); // multiply even indices

  switch(A) {
    case 1: premult<premult1>(P); break;
    case 2: premult<premult2>(P); break;
    case 3: premult<premult3>(P); break;
    case 4: premult<premult4>(P); break;
    default: premult<general>(P);
  }

  if(outofplace) { // out-of-place FFTs: U[A-1] is free if A > B.
    Complex *W[A];
    W[A-1]=U[A-1];
    for(unsigned int a=1; a < A; ++a) 
      W[a-1]=P[a];

    for(unsigned int a=A; a-- > 0;) // Loop from A-1 to 0.
      BackwardsO->fft(P[a],W[a]);
    (*pmult)(W,m,indexsize,index,1,threads); // multiply odd indices
    
    // Return to original space
    Complex *lastW=W[A-1];
    for(unsigned int b=0; b < B; ++b) {
      Complex *Pb=P[b];
      ForwardsO->fft(W[b],Pb);
      ForwardsO->fft(U[b],lastW);
      postmultadd(Pb,lastW);
    }
    
  } else { 
    // in-place FFTs: this could be optimised for B > A.

    // Backwards FFT (odd indices):
    for(unsigned int a=0; a < A; ++a)
      Backwards->fft(P[a]);
    (*pmult)(P,m,indexsize,index,1,threads); //multiply odd indices

    // TODO: after the first postmultadd, u[0] is free, which means
    // that the remaining B-1 FFTs of u[i] could be done out-of-place.
    
    // Return to original space:
    for(unsigned int b=0; b < B; ++b) {
      Complex *fb=P[b];
      Complex *ub=U[b];
      Forwards->fft(fb);
      Forwards->fft(ub);
      postmultadd(fb,ub);
    }
  }
}

template<class T>
inline void ImplicitConvolution::
premult(Complex **F, unsigned int k, Vec& Zetak)
{
  for(unsigned int a=0; a < A; ++a) {
    Complex *fka=F[a]+k;
    STORE(fka,ZMULT(Zetak,LOAD(fka)));
  }
}

template<>
inline void ImplicitConvolution::
premult<premult1>(Complex **F, unsigned int k, Vec& Zetak)
{
  Complex *fk0=F[0]+k;
  Vec Fk0=LOAD(fk0);
  STORE(fk0,ZMULT(Zetak,Fk0));
}

template<>
inline void ImplicitConvolution::
premult<premult2>(Complex **F, unsigned int k, Vec& Zetak)
{
  Complex *fk0=F[0]+k;
  Complex *fk1=F[1]+k;
  Vec Fk0=LOAD(fk0);
  Vec Fk1=LOAD(fk1);
  STORE(fk0,ZMULT(Zetak,Fk0));
  STORE(fk1,ZMULT(Zetak,Fk1));
}

template<>
inline void ImplicitConvolution::
premult<premult3>(Complex **F, unsigned int k, Vec& Zetak)
{
  Complex *fk0=F[0]+k;
  Complex *fk1=F[1]+k;
  Complex *fk2=F[2]+k;
  Vec Fk0=LOAD(fk0);
  Vec Fk1=LOAD(fk1);
  Vec Fk2=LOAD(fk2);
  STORE(fk0,ZMULT(Zetak,Fk0));
  STORE(fk1,ZMULT(Zetak,Fk1));
  STORE(fk2,ZMULT(Zetak,Fk2));
}

template<>
inline void ImplicitConvolution::
premult<premult4>(Complex **F, unsigned int k, Vec& Zetak)
{
  Complex *fk0=F[0]+k;
  Complex *fk1=F[1]+k;
  Complex *fk2=F[2]+k;
  Complex *fk3=F[3]+k;
  Vec Fk0=LOAD(fk0);
  Vec Fk1=LOAD(fk1);
  Vec Fk2=LOAD(fk2);
  Vec Fk3=LOAD(fk3);
  STORE(fk0,ZMULT(Zetak,Fk0));
  STORE(fk1,ZMULT(Zetak,Fk1));
  STORE(fk2,ZMULT(Zetak,Fk2));
  STORE(fk3,ZMULT(Zetak,Fk3));
}

// multiply by root of unity to prepare for inverse FFT for odd modes
template<class T>
void ImplicitConvolution::premult(Complex **F)
{  
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
      Vec Zeta=LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(CONJ(Zeta),Zeta);
      for(unsigned int k=K; k < stop; ++k) {
        Vec Zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
        premult<T>(F,k,Zetak);
      }
    }
    );
}

// multiply by root of unity to prepare and add for inverse FFT for odd modes
void ImplicitConvolution::postmultadd(Complex *f, Complex *u)
{
  double ninv=0.5/m;
  Vec Ninv=LOAD(ninv);
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      unsigned int stop=min(K+s,m);
      Complex *ZetaL0=ZetaL-K;
      Vec Zeta=Ninv*LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(CONJ(Zeta),Zeta);
      for(unsigned int k=K; k < stop; ++k) {
        Vec Zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
        Complex *fki=f+k;
        STORE(fki,ZMULTC(Zetak,LOAD(fki))+Ninv*LOAD(u+k));
      }
    }
    );
}

void ImplicitHConvolution::premult(Complex ** F,
                                   unsigned int offset,
                                   Complex* f1c)
{
  unsigned int C=max(A,B);
  Complex *P0[C];
  Complex *S=0; // only used for even quantities

  Vec Mhalf=LOAD(-0.5);
  Vec HSqrt3=LOAD(hsqrt3);
  
  bool even=m == 2*c;
  if(even)
    S=new Complex[A];
  unsigned int m1=m-1;
  for(unsigned int a=0; a < A; ++a)
    P0[a]=F[a]+offset;

  if(even) {
    unsigned int a=1/s;
    Vec Zeta=LOAD(ZetaH+a);
    Vec X=UNPACKL(Zeta,Zeta);
    Vec Y=UNPACKH(CONJ(Zeta),Zeta);
    Vec zeta1=ZMULT(X,Y,LOAD(ZetaL+1-s*a));
    for(unsigned int a=0; a < A; ++a) {
      Complex *fa=P0[a];
      Vec Fa=LOAD(fa+1);
      Vec Fb=LOAD(fa+m1);
      Vec B=Fb*Mhalf+CONJ(Fa);
      Fb *= HSqrt3;
      Vec A=ZMULTC(zeta1,UNPACKL(B,Fb)); // Optimize?
      B=ZMULTIC(zeta1,UNPACKH(B,Fb));
      STORE(f1c+a,CONJ(A+B));
        
      double re=fa[c].re;
      S[a]=Complex(2.0*re,re+sqrt3*fa[c].im);
    }
  }
  
  unsigned int c1=c+1;
  unsigned int d=c1/2;
  unsigned int a=c1/s;
  Vec Zeta=LOAD(ZetaH+a);
  Vec X=UNPACKL(Zeta,Zeta);
  Vec Y=UNPACKH(CONJ(Zeta),Zeta);
  Vec Zetac1=ZMULT(X,Y,LOAD(ZetaL+c1-s*a));
  PARALLEL(
    for(unsigned int K=0; K <= d; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,d+1);
      Vec Zeta=LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(CONJ(Zeta),Zeta);
      for(unsigned int a=0; a < A; ++a) {
        Complex *fa=P0[a];
        Complex *fm=fa+m;
        Complex *fpc1=fa+c1;
        Complex *fmc1=fm-c1;
        Complex *ua=U[a];
        Complex *upc1=ua+c1;
        for(unsigned int k=max(1,K); k < stop; ++k) {
          Vec zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
          Vec Zetak=ZMULTC(zetak,Zetac1);
          
          Vec Fa=LOAD(fa+k);
          Vec FA=LOAD(fpc1-k);
          Vec FB=LOAD(fmc1+k);
          Vec Fb=LOAD(fm-k);
          
          Vec B=Fb*Mhalf+CONJ(Fa);
          STORE(fa+k,Fa+CONJ(Fb));
          Fb *= HSqrt3;
          Vec A=ZMULTC(zetak,UNPACKL(B,Fb));
          B=ZMULTIC(zetak,UNPACKH(B,Fb));
        
          Vec D=FB*Mhalf+CONJ(FA);
          STORE(fpc1-k,FA+CONJ(FB));
          FB *= HSqrt3;
          Vec C=ZMULTC(Zetak,UNPACKL(D,FB));
          D=ZMULTIC(Zetak,UNPACKH(D,FB));

          STORE(ua+k,A-B);
          STORE(upc1-k,C-D);
          STORE(fm-k,CONJ(C+D));
          STORE(fmc1+k,CONJ(A+B));
        }
      }
    }
    );
    
  if(even) {
    for(unsigned int a=0; a < A; ++a) {
      Complex *fa=P0[a];
      Complex *ua=U[a];
      fa[c]=S[a].re;
      ua[c]=S[a].im;
    }
    delete[] S;
  }
}

void ImplicitHConvolution::postmultadd(Complex **F, Complex *f1c, Complex **U)
{
  double ninv=1.0/(3.0*m);
  Vec Ninv=LOAD(ninv);
  bool even=m == 2*c;

  Vec Mhalf=LOAD(-0.5);
  Vec HSqrt3=LOAD(hsqrt3);

  unsigned int m1=m-1;  
  unsigned int c1=c+1;
  unsigned int d=c1/2;
  unsigned int a=c1/s;
  Vec Zeta=LOAD(ZetaH+a);
  Vec X=UNPACKL(Zeta,Zeta);
  Vec Y=UNPACKH(CONJ(Zeta),Zeta);
  Vec Zetac1=ZMULT(X,Y,LOAD(ZetaL+c1-s*a));

  if(even && m > 2) {
    unsigned int a=1/s;
    Vec Zeta=LOAD(ZetaH+a);
    Vec X=UNPACKL(Zeta,Zeta);
    Vec Y=UNPACKH(CONJ(Zeta),Zeta);
    Vec zeta1=Ninv*ZMULT(X,Y,LOAD(ZetaL+1-s*a));
    Vec Zeta1=ZMULTC(zeta1,Zetac1);
    for(unsigned int b=0; b < B; ++b) {
      Complex *f0=F[b];
      Vec F0=LOAD(f0+1)*Ninv;
      Vec F1=ZMULTC(zeta1,LOAD(f1c+b));
      Vec F2=ZMULT(zeta1,LOAD(U[b]+1));
      Vec S=F1+F2;
      F2=CONJ(F0+Mhalf*S)-HSqrt3*FLIP(F1-F2);
      STORE(f0+1,F0+S);
      F0=LOAD(f0+c)*Ninv;
      F1=ZMULTC(Zeta1,LOAD(f0+m1));
      STORE(f0+m1,F2);
      F2=ZMULT(Zeta1,LOAD(U[b]+c));
      S=F1+F2;
      STORE(f0+c,F0+S);
      STORE(f0+m-c,CONJ(F0+Mhalf*S)-HSqrt3*FLIP(F1-F2));
    }
  }
  
  unsigned int D=c-d;
  PARALLEL(
    for(unsigned int K=0; K <= D; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,D+1);
      Vec Zeta=Ninv*LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(CONJ(Zeta),Zeta);
      for(unsigned int b=0; b < B; ++b) {
        Complex *fb=F[b];
        Complex *fm=fb+m;
        Complex *fpc1=fb+c1;
        Complex *fmc1=fm-c1;
        Complex *ub=U[b];
        Complex *upc1=ub+c1;
        for(unsigned int k=max(even+1,K); k < stop; ++k) {
          Vec zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
          Vec Zetak=ZMULTC(zetak,Zetac1);
          
          Vec F0=LOAD(fb+k)*Ninv;
          Vec F1=ZMULTC(zetak,LOAD(fmc1+k));
          Vec F2=ZMULT(zetak,LOAD(U[b]+k));
          Vec S=F1+F2;
          F2=CONJ(F0+Mhalf*S)-HSqrt3*FLIP(F1-F2);
            
          Vec FA=LOAD(fpc1-k)*Ninv;
          Vec FB=ZMULTC(Zetak,LOAD(fm-k));
          Vec FC=ZMULT(Zetak,LOAD(upc1-k));
          Vec T=FB+FC;
            
          STORE(fb+k,F0+S);
          STORE(fpc1-k,FA+T);
          STORE(fmc1+k,CONJ(FA+Mhalf*T)-HSqrt3*FLIP(FB-FC));
          STORE(fm-k,F2);
        }  
      }
    }
    );

  
  if(d == D+1) {
    unsigned int a=d/s;
    Vec Zeta=Ninv*LOAD(ZetaH+a);
    Vec X=UNPACKL(Zeta,Zeta);
    Vec Y=UNPACKH(CONJ(Zeta),Zeta);
    Vec Zetak=ZMULT(X,Y,LOAD(ZetaL+d-s*a));
    if(d == 1 && even) {
      for(unsigned int b=0; b < B; ++b) {
        Complex *f0=F[b];
        Vec F0=LOAD(f0+d)*Ninv;
        Vec F1=ZMULTC(Zetak,LOAD(f1c+b));
        Vec F2=ZMULT(Zetak,LOAD(U[b]+d));
        Vec S=F1+F2;
        STORE(f0+d,F0+S);
        STORE(f0+m-d,CONJ(F0+Mhalf*S)-HSqrt3*FLIP(F1-F2));
      }
    } else {
      for(unsigned int b=0; b < B; ++b) {
        Complex *f0=F[b];
        Vec F0=LOAD(f0+d)*Ninv;
        Vec F1=ZMULTC(Zetak,LOAD(f0+m-d));
        Vec F2=ZMULT(Zetak,LOAD(U[b]+d));
        Vec S=F1+F2;
        STORE(f0+d,F0+S);
        STORE(f0+m-d,CONJ(F0+Mhalf*S)-HSqrt3*FLIP(F1-F2));
      }
    }
  }
}

void ImplicitHConvolution::convolve(Complex **F, realmultiplier *pmult,
                                    unsigned int i, unsigned int offset)
{
  if(indexsize >= 1) index[indexsize-1]=i;
  
  // Set problem-size variables and pointers:
  unsigned int C=max(A,B);

  Complex *C0[C], *C1[C]; // inputs to complex2real FFTs
  double  *D0[A], *D1[A]; // outputs of complex2real FFTs
  Complex **c0=C0, **c1=C1;
  double **d0=D0, **d1=D1;

  unsigned int start=m-1-c; // c-1 (c) for m=even (odd)
  for(unsigned int a=0; a < C; ++a) {
    Complex *f=F[a]+offset;
    c0[a]=f;
    c1[a]=f+start;
    U[a][0]=compact ? f->re : f->re-f[m].re; // Nyquist
  }
  
  if(outofplace) { 
    for(unsigned int a=0; a < A-1; ++a) {
      d0[a]=(double *) c0[a+1];
      d1[a]=(double *) c1[a+1];
    }
    d0[A-1]=(double *) U[A-1];
    d1[A-1]=(double *) U[A-1];
  } else {
    d0=(double **) c0;
    d1=(double **) c1;
  }

  bool even=m == 2*c;
  Complex *c1c=NULL;
  if(even) c1c=ComplexAlign(C);

  premult(F,offset,c1c);

  // Complex-to-real FFTs and pmults:
  Complex *S=new Complex[B];

  // r=-1:
  for(unsigned int a=0; a < A; ++a)
    cr->fft(U[a]); // in-place transform
  (*pmult)((double **) U,m,indexsize,index,-1,threads);

  // r=0:
  double T[A]; // deal with overlap between r=0 and r=1
  for(unsigned int a=A; a-- > 0;) { // Loop from A-1 to 0.
    Complex *c0a=c0[a];
    T[a]=c0a[0].re; // r=0, k=0
    if(!compact)
      c0a[0].re += 2.0*c0a[m].re; // Nyquist
    cro->fft(c0a,d0[a]);
  }
  (*pmult)(d0,m,indexsize,index,0,threads);
  for(unsigned int b=0; b < B; ++b)
    S[b]=((Complex *) d0[b])[start];   // r=0, k=start
    
  // r=1:
  for(unsigned int a=A; a-- > 0;) { // Loop from A-1 to 0.
    Complex *c1a=c1[a];
    c1a[0]=compact ? T[a] : T[a]-c1a[c+1].re; // r=1, k=0 with Nyquist
    if(even) {
      Complex tmp=c1c[a];
      c1c[a]=c1a[1];  // r=0, k=c
      c1a[1]=tmp;     // r=1, k=1
    }
    cro->fft(c1[a],d1[a]);
  }
  (*pmult)(d1,m,indexsize,index,1,threads);

  const double ninv=1.0/(3.0*m);
  
  // Real-to-complex FFTs and postmultadd:
  Complex *c2[B];
  for(unsigned int b=B; b-- > 0;) { // Loop from B-1 to 0
    c2[b]=U[b+outofplace];
    rco->fft(U[b],c2[b]); // r=2
  }
  for(unsigned int b=0; b < B; ++b) {
    Complex *c0b=c0[b];
    Complex *c1b=c1[b];
    rco->fft(d1[b],c1b); // r=1
    double R=c1b[0].re;
    ((Complex *) d0[b])[start]=S[b]; // r=0, k=c-1 (c) for m=even (odd)
    if(even) {
      Complex tmp=c1c[b];
      c1c[b]=c1b[1]; // r=1, k=1
      c1b[1]=tmp;    // r=0, k=c
    }
    rco->fft(d0[b],c0b); // r=0
    if(!compact) c0b[m]=0.0; // Zero Nyquist mode, for Hermitian symmetry.
    c0b[0]=(c0b[0].re+R+c2[b][0].re)*ninv;
  }
  postmultadd(c0,c1c,c2);
    
  delete[] S;
  if(even) deleteAlign(c1c);
}

void fftpad::expand(Complex *f, Complex *u)
{
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
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
    }
    );
}
  
void fftpad::backwards(Complex *f, Complex *u)
{
  expand(f,u);
  Backwards->fft(f);
  Backwards->fft(u);
}

void fftpad::reduce(Complex *f, Complex *u)
{
  double ninv=0.5/m;
  Vec Ninv=LOAD(ninv);
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
      Vec H=Ninv*LOAD(ZetaH+K/s);
      for(unsigned int k=K; k < stop; ++k) {
        Vec Zetak=ZMULT(H,LOAD(ZetaL0+k));
        Vec X=UNPACKL(Zetak,Zetak);
        Vec Y=UNPACKH(Zetak,CONJ(Zetak));
        unsigned int kstride=k*stride;
        Complex *uk=u+kstride;
        Complex *fk=f+kstride;
        for(unsigned int i=0; i < M; ++i)
          STORE(fk+i,LOAD(fk+i)*Ninv+ZMULT(X,Y,LOAD(uk+i)));
      }
    }
    );
}

void fftpad::forwards(Complex *f, Complex *u)
{
  Forwards->fft(f);
  Forwards->fft(u);
  reduce(f,u);
}

void fft0pad::expand(Complex *f, Complex *u)
{
  Complex *fm1stride=f+(m-1)*stride;
  for(unsigned int i=0; i < M; ++i)
    u[i]=fm1stride[i];
    
  Vec Mhalf=LOAD(-0.5);
  Vec Mhsqrt3=LOAD(-hsqrt3);
  unsigned int stop=s;
  
  for(unsigned int K=0; K < m; K += s) {
    Complex *ZetaL0=ZetaL-K;
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
        Vec A=LOAD(fmk+i);
        Vec B=LOAD(f+i);
        Vec Z=B*Mhalf+A;
        STORE(f+i,LOAD(fk+i));
        STORE(fk+i,B+A);
        B *= Mhsqrt3;
        A=ZMULT(X,Y,UNPACKL(Z,B));
        B=ZMULTI(X,Y,UNPACKH(Z,B));
        STORE(fmk+i,A+B);
        STORE(uk+i,CONJ(A-B));
      }
    }
    stop=min(stop+s,m);
  }
}

void fft0pad::Backwards1(Complex *f, Complex *u)
{
  Complex *umstride=u+m*stride;
  Complex *fm1stride=f+(m-1)*stride;
  for(unsigned int i=0; i < M; ++i) {
    umstride[i]=fm1stride[i]; // Store extra value here.
    fm1stride[i]=u[i];
  }
  Backwards->fft(fm1stride);
}

void fft0pad::backwards(Complex *f, Complex *u)
{
  expand(f,u);
  Backwards->fft(f);
  Backwards1(f,u);
  Backwards->fft(u);
}

void fft0pad::reduce(Complex *f, Complex *u)
{
  Complex *umstride=u+m*stride;
  double ninv=1.0/(3.0*m);
  for(unsigned int i=0; i < M; ++i)
    umstride[i]=(umstride[i]+f[i]+u[i])*ninv;

  Vec Ninv=LOAD(ninv);
  Vec Mhalf=LOAD(-0.5);
  Vec HSqrt3=LOAD(hsqrt3);
  Complex *fm1stride=f+(m-1)*stride;
  
  unsigned int stop=s;
  for(unsigned int K=0; K < m; K += s) {
    Complex *ZetaL0=ZetaL-K;
    Vec H=LOAD(ZetaH+K/s)*Ninv;
    for(unsigned int k=max(1,K); k < stop; ++k) {
      Vec Zetak=ZMULT(H,LOAD(ZetaL0+k));
      Vec X=UNPACKL(Zetak,Zetak);
      Vec Y=UNPACKH(CONJ(Zetak),Zetak);
      unsigned int kstride=k*stride;
      Complex *fk=f+kstride;
      Complex *fm1k=fm1stride+kstride;
      Complex *uk=u+kstride;
      for(unsigned int i=0; i < M; ++i) {
        Vec F0=LOAD(fk+i)*Ninv;
        Vec F1=ZMULT(X,-Y,LOAD(fm1k+i));
        Vec F2=ZMULT(X,Y,LOAD(uk+i));
        Vec S=F1+F2;
        STORE(fk+i-stride,F0+Mhalf*S+HSqrt3*ZMULTI(F1-F2));
        STORE(fm1k+i,F0+S);
      }
    }
    stop=min(stop+s,m);
  }
    
  for(unsigned int i=0; i < M; ++i)
    fm1stride[i]=umstride[i];
}

void fft0pad::Forwards0(Complex *f)
{
  Forwards->fft(f+(m-1)*stride);
}
  
void fft0pad::Forwards1(Complex *f, Complex *u)
{
  Complex *umstride=u+m*stride;
  unsigned int m1stride=(m-1)*stride;
  Complex *fm1stride=f+m1stride;
  for(unsigned int i=0; i < M; ++i) {
    Complex temp=umstride[i];
    umstride[i]=fm1stride[i];
    fm1stride[i]=temp;
  }
}
  
void fft0pad::forwards(Complex *f, Complex *u)
{
  Forwards0(f);  
  Forwards1(f,u);  
  Forwards->fft(f);
  Forwards->fft(u);
  reduce(f,u);
}

void fft1pad::expand(Complex *f, Complex *u)
{
  Complex *fmstride=f+m*stride;
  for(unsigned int i=0; i < M; ++i) {
    Complex Nyquist=f[i];
    f[i]=fmstride[i]+2.0*Nyquist;
    u[i]=fmstride[i] -= Nyquist;
  }
    
  Vec Mhalf=LOAD(-0.5);
  Vec Mhsqrt3=LOAD(-hsqrt3);
  unsigned int inc=s;
  PARALLEL(
    for(unsigned int K=0; K < m; K += inc) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
      Vec Zeta=LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(CONJ(Zeta),Zeta);
      for(unsigned int k=max(1,K); k < stop; ++k) {
        Vec zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
        unsigned int kstride=k*stride;
        Complex *uk=u+kstride;
        Complex *fk=f+kstride;
        Complex *fmk=fmstride+kstride;
        for(unsigned int i=0; i < M; ++i) {
          Vec Fa=LOAD(fk+i);
          Vec Fb=LOAD(fmk+i);
        
          Vec B=Fa*Mhalf+Fb;
          STORE(fk+i,Fa+Fb);
          Fa *= Mhsqrt3;
          Vec A=ZMULT(zetak,UNPACKL(B,Fa));
          B=ZMULTI(zetak,UNPACKH(B,Fa));
          STORE(fmk+i,A+B);
          STORE(uk+i,CONJ(A-B));
        }
      }
    }
    );
}

void fft1pad::Backwards1(Complex *f, Complex *u)
{
  Backwards->fft(f+m*stride);
}

void fft1pad::reduce(Complex *f, Complex *u)
{
  Complex *fmstride=f+m*stride;

  double ninv=1.0/(3.0*m);
  for(unsigned int i=0; i < M; ++i) {
    Complex f0=f[i];
    Complex f1=fmstride[i];
    Complex f2=u[i];
    f[i]=0.0; // Zero Nyquist mode, for Hermitian symmetry.
    fmstride[i]=(f0+f1+f2)*ninv;
  }
  Vec Ninv=LOAD(ninv);
  Vec Mhalf=LOAD(-0.5);
  Vec HSqrt3=LOAD(hsqrt3);
  
  unsigned int inc=s;
  PARALLEL(
    for(unsigned int K=0; K < m; K += inc) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
      Vec Zeta=Ninv*LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(CONJ(Zeta),Zeta);
      for(unsigned int k=max(1,K); k < stop; ++k) {
        Vec zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
        unsigned int kstride=k*stride;
        Complex *fk=f+kstride;
        Complex *fmk=fmstride+kstride;
        Complex *uk=u+kstride;
        for(unsigned int i=0; i < M; ++i) {
          Vec F0=LOAD(fk+i)*Ninv;
          Vec F1=ZMULTC(zetak,LOAD(fmk+i));
          Vec F2=ZMULT(zetak,LOAD(uk+i));
          Vec S=F1+F2;
          STORE(fk+i,F0+Mhalf*S+HSqrt3*ZMULTI(F1-F2));
          STORE(fmk+i,F0+S);
        }
      }
    }
    );
}

void fft1pad::Forwards0(Complex *f)
{
  Forwards->fft(f+m*stride);
}

void fft1pad::Forwards1(Complex *f, Complex *u)
{
}

void fft1pad::forwards(Complex *f, Complex *u)
{
  Forwards->fft(f);
  Forwards->fft(f+m*stride);
  Forwards->fft(u);
  reduce(f,u);
}

// a[0][k]=sum_i a[i][k]*b[i][k]*c[i][k]
void ImplicitHTConvolution::mult(double *a, double *b, double **C,
                                 unsigned int offset)
{
  unsigned int twom=2*m;
  if(M == 1) { // a[k]=a[k]*b[k]*c[k]
    double *C0=C[0]+offset;
#ifdef __SSE2__
    PARALLEL(
      for(unsigned int k=0; k < twom; k += 2) {
        double *ak=a+k;
        STORE(ak,LOAD(ak)*LOAD(b+k)*LOAD(C0+k));
      }
      );
#else
    PARALLEL(
      for(unsigned int k=0; k < twom; ++k)
        a[k] *= b[k]*C0[k];
      );
#endif
  } else if(M == 2) {
    double *a1=a+stride;
    double *b1=b+stride;
    double *C0=C[0]+offset;
    double *C1=C[1]+offset;
#ifdef __SSE2__
    PARALLEL(
      for(unsigned int k=0; k < twom; k += 2) {
        double *ak=a+k;
        STORE(ak,LOAD(ak)*LOAD(b+k)*LOAD(C0+k)+
              LOAD(a1+k)*LOAD(b1+k)*LOAD(C1+k));
      }  
      );
#else
    PARALLEL(
      for(unsigned int k=0; k < twom; ++k)
        a[k]=a[k]*b[k]*C0[k]+a1[k]*b1[k]*C1[k];
      );
#endif
  } else if(M == 3) {
    double *a1=a+stride;
    double *a2=a1+stride;
    double *b1=b+stride;
    double *b2=b1+stride;
    double *C0=C[0]+offset;
    double *C1=C[1]+offset;
    double *C2=C[2]+offset;
#ifdef __SSE2__
    PARALLEL(
      for(unsigned int k=0; k < twom; k += 2) {
        double *ak=a+k;
        STORE(ak,LOAD(ak)*LOAD(b+k)*LOAD(C0+k)+
              LOAD(a1+k)*LOAD(b1+k)*LOAD(C1+k)+
              LOAD(a2+k)*LOAD(b2+k)*LOAD(C2+k));
      }
      );
#else
    PARALLEL(
      for(unsigned int k=0; k < twom; ++k)
        a[k]=a[k]*b[k]*C0[k]+a1[k]*b1[k]*C1[k]+a2[k]*b2[k]*C2[k];
      );
#endif
  } else {
    double *A=a-offset;
    double *B=b-offset;
    double *C0=C[0];
    unsigned int stop=twom+offset;
#ifdef __SSE2__
    PARALLEL(   
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
      );
#else
    PARALLEL(
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
      );
#endif
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
      }  
      );
      
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
  Vec Ninv=LOAD(ninv);
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
      Vec Zeta=Ninv*LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(CONJ(Zeta),Zeta);
      for(unsigned int k=K; k < stop; ++k) {
        Vec Zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
        Complex *fk=f+k;
        STORE(fk,ZMULTC(Zetak,LOAD(u+k))+Ninv*LOAD(fk));
      }
    }
    );
}

// a[k]=a[k]*b[k]*b[k]
void ImplicitHFGGConvolution::mult(double *a, double *b)
{
  unsigned int twom=2*m;
#ifdef __SSE2__
  PARALLEL(
    for(unsigned int k=0; k < twom; k += 2) {
      double *ak=a+k;
      STORE(ak,LOAD(ak)*LOAD(ak)*LOAD(b+k));
    }
    );
#else
  PARALLEL(
    for(unsigned int k=0; k < twom; ++k) {
      double ak=a[k];
      a[k]=ak*ak*b[k];
    }
    );
#endif
}

void ImplicitHFGGConvolution::convolve(Complex *f, Complex *g,
                                       Complex *u, Complex *v)
{
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
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
    }  
    );

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

  Vec Ninv=LOAD(ninv);
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
      Vec Zeta=Ninv*LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(CONJ(Zeta),Zeta);
      for(unsigned int k=K; k < stop; ++k) {
        Vec Zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
        Complex *fk=f+k;
        STORE(fk,ZMULTC(Zetak,LOAD(u+k))+Ninv*LOAD(fk));
      }
    }
    );
}

// a[k]=a[k]^3
void ImplicitHFFFConvolution::mult(double *a)
{
  unsigned int twom=2*m;
  PARALLEL(
    for(unsigned int k=0; k < twom; k += 2) {
      double *p=a+k;
      Vec ak=LOAD(p);
      STORE(p,ak*ak*ak);
    }        
    );
}
  
void ImplicitHFFFConvolution::convolve(Complex *f, Complex *u)
{
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
      Vec Zeta=LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(CONJ(Zeta),Zeta);
      for(unsigned int k=K; k < stop; ++k)
        STORE(u+k,ZMULT(ZMULT(X,Y,LOAD(ZetaL0+k)),LOAD(f+k)));
    }  
    );
    
  u[m]=0.0;
  cr->fft(u);
  mult((double *) u);
  rc->fft(u);

  f[m]=0.0;
  cr->fft(f);
  mult((double *) f);
  rc->fft(f);
  double ninv=0.25/m;
  Vec Ninv=LOAD(ninv);
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
      Vec Zeta=Ninv*LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(CONJ(Zeta),Zeta);
      for(unsigned int k=K; k < stop; ++k) {
        Complex *p=f+k;
        STORE(p,ZMULTC(ZMULT(X,Y,LOAD(ZetaL0+k)),LOAD(u+k))+Ninv*LOAD(p));
      } 
    }
    );
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
    }
    );
    
  Backwards->fft(f);
  Backwards->fft(u);
}

void fft0bipad::forwards(Complex *f, Complex *u)
{
  Forwards->fft(f);
  Forwards->fft(u);

  double ninv=0.25/m;
  unsigned int twom=2*m;
  Vec Ninv=LOAD(ninv);
  PARALLEL(
    for(unsigned int K=0; K < twom; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,twom);
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
    }
    );
}

// This multiplication routine is for binary convolutions and takes two inputs
// of size m.
// F[0][j] *= conj(F[0][j]);
void multautocorrelation(Complex **F, unsigned int m,
                         const unsigned int indexsize,
                         const unsigned int *index,
                         unsigned int r, unsigned int threads)
{
  Complex* F0=F[0];
  
#ifdef __SSE2__
  PARALLEL(
    for(unsigned int j=0; j < m; ++j) {
      Complex *p=F0+j;
      STORE(p,ZMULT(LOAD(p),CONJ(LOAD(p))));
    }
    );
#else
  PARALLEL(
    for(unsigned int j=0; j < m; ++j)
      F0[j] *= conj(F0[j]);
    );
#endif
}

void multcorrelation(Complex **F, unsigned int m,
                     const unsigned int indexsize,
                     const unsigned int *index,
                     unsigned int r, unsigned int threads)
{
  Complex* F0=F[0];
  Complex* F1=F[1];
  
#ifdef __SSE2__
  PARALLEL(
    for(unsigned int j=0; j < m; ++j) {
      Complex *p=F0+j;
      Complex *q=F1+j;
      STORE(p,ZMULT(LOAD(p),CONJ(LOAD(q))));
    }
    );
#else
  PARALLEL(
    for(unsigned int j=0; j < m; ++j)
      F0[j] *= conj(F1[j]);
    );
#endif
}

// Unscramble indices, returning spatial index for remainder r at position j.
inline unsigned innerindex(unsigned j, int r) {return 2*j+r;}
  
// This multiplication routine is for binary convolutions and takes two inputs
// of size m.
// F[0][j] *= F[1][j];
void multbinary(Complex **F, unsigned int m,
                const unsigned int indexsize,
                const unsigned int *index,
                unsigned int r, unsigned int threads)
{
  Complex* F0=F[0];
  Complex* F1=F[1];
  
#if 0 // Spatial indices are available, if needed.
  size_t n=indexsize;
  for(unsigned int j=0; j < m; ++j) {
    for(unsigned int d=0; d < n; ++d)
      cout << index[d] << ",";
    cout << innerindex(j,r) << endl;
  }
#endif  
      
#ifdef __SSE2__
  PARALLEL(
    for(unsigned int j=0; j < m; ++j) {
      Complex *p=F0+j;
      STORE(p,ZMULT(LOAD(p),LOAD(F1+j)));
    }
    );
#else
  PARALLEL(
    for(unsigned int j=0; j < m; ++j)
      F0[j] *= F1[j];
    );
#endif
}

// F[0][j] *= F[0][j];
void multautoconvolution(Complex **F, unsigned int m,
                         const unsigned int indexsize,
                         const unsigned int *index,
                         unsigned int r, unsigned int threads)
{
  Complex* F0=F[0];
  
#ifdef __SSE2__
  PARALLEL(
    for(unsigned int j=0; j < m; ++j) {
      Complex *p=F0+j;
      STORE(p,ZMULT(LOAD(p),LOAD(p)));
    }
    );
#else
  PARALLEL(
    for(unsigned int j=0; j < m; ++j)
      F0[j] *= F0[j];
    );
#endif
}

// Unscramble indices, returning spatial index for remainder r at position j.
inline unsigned innerindex(unsigned j, int r, unsigned int m) {
  int x=3*j+r;
  return x >= 0 ? x : 3*m-1;
}
  
// This multiplication routine is for binary Hermitian convolutions and takes
// two inputs.
// F[0][j] *= F[1][j];
void multbinary(double **F, unsigned int m,
                const unsigned int indexsize,
                const unsigned int *index,
                unsigned int r, unsigned int threads)
{
  double* F0=F[0];
  double* F1=F[1];
  
#if 0 // Spatial indices are available, if needed.
  //size_t n=index.size();
  size_t n=indexsize;
  for(unsigned int j=0; j < m; ++j) {
    for(unsigned int d=0; d < n; ++d)
      cout << index[d] << ",";
    cout << innerindex(j,r,m) << endl;
  }
#endif
      
#ifdef __SSE2__
  unsigned int m1=m-1;
  PARALLEL(
    for(unsigned int j=0; j < m1; j += 2) {
      double *p=F0+j;
      STORE(p,LOAD(p)*LOAD(F1+j));
    }
    if(m % 2)
      F0[m1] *= F1[m1];
    );
#else
  PARALLEL(
    for(unsigned int j=0; j < m; ++j)
      F0[j] *= F1[j];
    );
#endif
}

// F[0][j]=F[0][j]*F[2][j]+F[1][j]*F[3][j]
void multbinary2(Complex **F, unsigned int m,
                 const unsigned int indexsize,
                 const unsigned int *index,
                 unsigned int r, unsigned int threads)
{
  Complex* F0=F[0];
  Complex* F1=F[1];
  Complex* F2=F[2];
  Complex* F3=F[3];
  
#ifdef __SSE2__
  PARALLEL(
    for(unsigned int j=0; j < m; ++j) {
      Complex *F0j=F0+j;
      STORE(F0j,ZMULT(LOAD(F0j),LOAD(F2+j))
            +ZMULT(LOAD(F1+j),LOAD(F3+j)));
    }
    );
#else
  PARALLEL(
    for(unsigned int j=0; j < m; ++j)
      F0[j]=F0[j]*F2[j]+F1[j]*F3[j];
    );
#endif
}

// F[0][j]=F[0][j]*F[2][j]+F[1][j]*F[3][j]
void multbinary2(double **F, unsigned int m,
                 const unsigned int indexsize,
                 const unsigned int *index,
                 unsigned int r, unsigned int threads)
{
  double* F0=F[0];
  double* F1=F[1];
  double* F2=F[2];
  double* F3=F[3];
  
#ifdef __SSE2__
  unsigned int m1=m-1;
  PARALLEL(
    for(unsigned int j=0; j < m1; j += 2) {
      double *F0j=F0+j;
      STORE(F0j,LOAD(F0j)*LOAD(F2+j)+LOAD(F1+j)*LOAD(F3+j));
    }
    );
  if(m % 2)
    F0[m1]=F0[m1]*F2[m1]+F1[m1]*F3[m1];
#else
  PARALLEL(
    for(unsigned int j=0; j < m; ++j)
      F0[j]=F0[j]*F2[j]+F1[j]*F3[j];
    );
#endif
}

// F[0][j]=F[0][j]*F[3][j]+F[1][j]*F[4][j]+F[2][j]*F[5][j];
void multbinary3(Complex **F, unsigned int m,
                 const unsigned int indexsize,
                 const unsigned int *index,
                 unsigned int r, unsigned int threads)
{
  Complex* F0=F[0];
  Complex* F1=F[1];
  Complex* F2=F[2];
  Complex* F3=F[3];
  Complex* F4=F[4];
  Complex* F5=F[5];
  
#ifdef __SSE2__
  PARALLEL(
    for(unsigned int j=0; j < m; ++j) {
      Complex *F0j=F0+j;
      STORE(F0j,ZMULT(LOAD(F0j),LOAD(F3+j))
            +ZMULT(LOAD(F1+j),LOAD(F4+j))
            +ZMULT(LOAD(F2+j),LOAD(F5+j))
        );
    }
    );
#else
  PARALLEL(
    for(unsigned int j=0; j < m; ++j)
      F0[j]=F0[j]*F3[j]+F1[j]*F4[j]+F2[j]*F5[j];
    );

#endif
}

// F[0][j]=F[0][j]*F[4][j]+F[1][j]*F[5][j]+F[2][j]*F[6][j]+F[3][j]*F[7][j];
void multbinary4(Complex **F, unsigned int m,
                 const unsigned int indexsize,           
                 const unsigned int *index,
                 unsigned int r, unsigned int threads)
{
  Complex* F0=F[0];
  Complex* F1=F[1];
  Complex* F2=F[2];
  Complex* F3=F[3];
  Complex* F4=F[4];
  Complex* F5=F[5];
  Complex* F6=F[6];
  Complex* F7=F[7];
  
#ifdef __SSE2__
  PARALLEL(
    for(unsigned int j=0; j < m; ++j) {
      Complex *F0j=F0+j;
      STORE(F0j,ZMULT(LOAD(F0j),LOAD(F4+j))
            +ZMULT(LOAD(F1+j),LOAD(F5+j))
            +ZMULT(LOAD(F2+j),LOAD(F6+j))
            +ZMULT(LOAD(F3+j),LOAD(F7+j))
        );
    }
    );
#else
  PARALLEL(
    for(unsigned int j=0; j < m; ++j)
      F0[j]=F0[j]*F4[j]+F1[j]*F5[j]+F2[j]*F6[j]+F3[j]*F7[j];
    );

#endif
}

// F[0][j]=F[0][j]*F[8][j]+F[1][j]*F[9][j]+F[2][j]*F[10][j]+F[3][j]*F[11][j]+
//         F[4][j]*F[12][j]+F[5][j]*F[13][j]+F[6][j]*F[14][j]+F[7][j]*F[15][j];
void multbinary8(Complex **F, unsigned int m,
                 const unsigned int indexsize,
                 const unsigned int *index,
                 unsigned int r, unsigned int threads)
{
  Complex* F0=F[0];
  Complex* F1=F[1];
  Complex* F2=F[2];
  Complex* F3=F[3];
  Complex* F4=F[4];
  Complex* F5=F[5];
  Complex* F6=F[6];
  Complex* F7=F[7];
  Complex* F8=F[8];
  Complex* F9=F[9];
  Complex* F10=F[10];
  Complex* F11=F[11];
  Complex* F12=F[12];
  Complex* F13=F[13];
  Complex* F14=F[14];
  Complex* F15=F[15];
    
#ifdef __SSE2__
  PARALLEL(
    for(unsigned int j=0; j < m; ++j) {
      Complex *F0j=F0+j;
      STORE(F0j,
            ZMULT(LOAD(F0j),LOAD(F8+j))
            +ZMULT(LOAD(F1+j),LOAD(F9+j))
            +ZMULT(LOAD(F2+j),LOAD(F10+j))
            +ZMULT(LOAD(F3+j),LOAD(F11+j))
            +ZMULT(LOAD(F4+j),LOAD(F12+j))
            +ZMULT(LOAD(F5+j),LOAD(F13+j))
            +ZMULT(LOAD(F6+j),LOAD(F14+j))
            +ZMULT(LOAD(F7+j),LOAD(F15+j))
        );
    }
    );
#else
  PARALLEL(
    for(unsigned int j=0; j < m; ++j)
      F0[j]=F0[j]*F8[j]+F1[j]*F9[j]+F2[j]*F10[j]+F3[j]*F11[j]
        +F4[j]*F12[j]+F5[j]*F13[j]+F6[j]*F14[j]+F7[j]*F15[j];
    );
#endif
}

} // namespace fftwpp
