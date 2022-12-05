#include "convolution.h"

using namespace std;
using namespace utils;

namespace fftwpp {

#ifdef __SSE2__
const union uvec sse2_pm = {
  { 0x00000000,0x00000000,0x00000000,0x80000000 }
};

/*
const union uvec sse2_mp = {
  { 0x00000000,0x80000000,0x00000000,0x00000000 }
};
*/

const union uvec sse2_mm = {
  { 0x00000000,0x80000000,0x00000000,0x80000000 }
};
#endif

const double sqrt3=sqrt(3.0);
const double hsqrt3=0.5*sqrt3;
const Complex zeta3(-0.5,hsqrt3);
const double twopi=2.0*M_PI;

// Build zeta table, returning the floor of the square root of m.
size_t BuildZeta(double arg, size_t m,
                       Complex *&ZetaH, Complex *&ZetaL, size_t threads,
                       size_t s)
{
  if(s == 0) s=(int) sqrt((double) m);
  size_t t=m/s;
  if(s*t < m) ++t;
  ZetaH=ComplexAlign(t);

#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif
  for(size_t a=0; a < t; ++a) {
    double theta=s*a*arg;
    ZetaH[a]=Complex(cos(theta),sin(theta));
  }
  ZetaL=ComplexAlign(s);
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif
  for(size_t b=0; b < s; ++b) {
    double theta=b*arg;
    ZetaL[b]=Complex(cos(theta),sin(theta));
  }
  return s;
}

size_t BuildZeta(size_t n, size_t m,
                       Complex *&ZetaH, Complex *&ZetaL, size_t threads,
                       size_t s)
{
  return BuildZeta(twopi/n,m,ZetaH,ZetaL,threads,s);
}

void ImplicitConvolution::convolve(Complex **F, multiplier *pmult,
                                   size_t i, size_t offset)
{
  if(indexsize >= 1) index[indexsize-1]=i;

  size_t C=max(A,B);
  Complex *P[C];
  for(size_t a=0; a < C; ++a)
    P[a]=F[a]+offset;

  // Backwards FFT (even indices):
  for(size_t a=0; a < A; ++a) {
    BackwardsO->fft(P[a],U[a]);
  }

  if(A >= B)
    (*pmult)(U,m,indexsize,index,0,threads); // multiply even indices

  switch(A) {
    case 1: pretransform<pretransform1>(P); break;
    case 2: pretransform<pretransform2>(P); break;
    case 3: pretransform<pretransform3>(P); break;
    case 4: pretransform<pretransform4>(P); break;
    default: pretransform<general>(P);
  }

  if(A > B) { // U[A-1] is free
    Complex *W[A];
    W[A-1]=U[A-1];
    for(size_t a=1; a < A; ++a)
      W[a-1]=P[a];

    for(size_t a=A; a-- > 0;) // Loop from A-1 to 0.
      BackwardsO->fft(P[a],W[a]);

    (*pmult)(W,m,indexsize,index,1,threads); // multiply odd indices

    // Return to original space
    Complex *lastW=W[A-1];
    for(size_t b=0; b < B; ++b) {
      Complex *Pb=P[b];
      ForwardsO->fft(W[b],Pb);
      ForwardsO->fft(U[b],lastW);
      posttransform(Pb,lastW);
    }

  } else if(A < B) { // U[B-1] is free
    Complex *W[B];
    W[B-1]=U[B-1];
    for(size_t b=1; b < B; ++b)
      W[b-1]=P[b];

    for(size_t a=A; a-- > 0;) // Loop from A-1 to 0.
      BackwardsO->fft(P[a],W[a]);

    (*pmult)(W,m,indexsize,index,1,threads); // multiply odd indices

    // Return to original space
    for(size_t b=0; b < B; ++b)
      ForwardsO->fft(W[b],P[b]);

    (*pmult)(U,m,indexsize,index,0,threads); // multiply even indices

    Complex *f0=P[0];
    Complex *u0=U[0];
    Forwards->fft(u0);
    posttransform(f0,u0);
    for(size_t b=1; b < B; ++b) {
      Complex *fb=P[b];
      Complex *ub=U[b];
      Complex *u0=U[0];
      ForwardsO->fft(ub,u0);
      posttransform(fb,u0);
    }

  } else { // A == B
    // Backwards FFT (odd indices):
    for(size_t a=0; a < A; ++a)
      Backwards->fft(P[a]);
    (*pmult)(P,m,indexsize,index,1,threads); //multiply odd indices

    // Return to original space:
    Complex *f0=P[0];
    Complex *u0=U[0];
    Forwards->fft(f0);
    Forwards->fft(u0);
    posttransform(f0,u0);
    for(size_t b=1; b < B; ++b) {
      Complex *fb=P[b];
      Complex *ub=U[b];
      Complex *u0=U[0];
      Forwards->fft(fb);
      ForwardsO->fft(ub,u0);
      posttransform(fb,u0);
    }
  }
}

template<class T>
inline void ImplicitConvolution::
pretransform(Complex **F, size_t k, Vec& Zetak)
{
  for(size_t a=0; a < A; ++a) {
    Complex *fka=F[a]+k;
    STORE(fka,ZMULT(Zetak,LOAD(fka)));
  }
}

template<>
inline void ImplicitConvolution::
pretransform<pretransform1>(Complex **F, size_t k, Vec& Zetak)
{
  Complex *fk0=F[0]+k;
  Vec Fk0=LOAD(fk0);
  STORE(fk0,ZMULT(Zetak,Fk0));
}

template<>
inline void ImplicitConvolution::
pretransform<pretransform2>(Complex **F, size_t k, Vec& Zetak)
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
pretransform<pretransform3>(Complex **F, size_t k, Vec& Zetak)
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
pretransform<pretransform4>(Complex **F, size_t k, Vec& Zetak)
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
void ImplicitConvolution::pretransform(Complex **F)
{
  PARALLELIF(
    m > threshold,
    for(size_t K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      size_t stop=min(K+s,m);
      Vec Zeta=LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=CONJ(UNPACKH(Zeta,Zeta));
      for(size_t k=K; k < stop; ++k) {
        Vec Zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
        pretransform<T>(F,k,Zetak);
      }
    }
    );
}

// multiply by root of unity to prepare and add for inverse FFT for odd modes
void ImplicitConvolution::posttransform(Complex *f, Complex *u)
{
  PARALLELIF(
    m > threshold,
    for(size_t K=0; K < m; K += s) {
      size_t stop=min(K+s,m);
      Complex *ZetaL0=ZetaL-K;
      Vec Zeta=LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=CONJ(UNPACKH(Zeta,Zeta));
      for(size_t k=K; k < stop; ++k) {
        Vec Zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
        Complex *fki=f+k;
        STORE(fki,ZCMULT(Zetak,LOAD(fki))+LOAD(u+k));
      }
    }
    );
}

void ImplicitHConvolution::pretransform(Complex *F, Complex *f1c, Complex *U)
{
  Vec Mhalf=LOAD2(-0.5);
  Vec HSqrt3=LOAD2(hsqrt3);

  double Re=0.0, Im=0.0;

  size_t m1=m-1;

  U[0]=compact ? F->re : F->re-F[m].re; // Nyquist

  if(even) {
    size_t a=1/s;
    Vec Zeta=LOAD(ZetaH+a);
    Vec X=UNPACKL(Zeta,Zeta);
    Vec Y=CONJ(UNPACKH(Zeta,Zeta));
    Vec zeta1=ZMULT(X,Y,LOAD(ZetaL+1-s*a));
    Vec Fa=LOAD(F+1);
    Vec Fb=LOAD(F+m1);
    Vec B=Fb*Mhalf+CONJ(Fa);
    Fb *= HSqrt3;
    Vec A=ZCMULT(zeta1,UNPACKL(B,Fb)); // Optimize?
    B=ZCMULTI(zeta1,UNPACKH(B,Fb));
    STORE(f1c,CONJ(A+B));

    double re=F[c].re;
    Re=2.0*re;
    Im=re+sqrt3*F[c].im;
  }

  size_t c1=c+1;
  size_t d=c1/2;
  size_t a=c1/s;
  Vec Zeta=LOAD(ZetaH+a);
  Vec X=UNPACKL(Zeta,Zeta);
  Vec Y=CONJ(UNPACKH(Zeta,Zeta));
  Vec Zetac1=ZMULT(X,Y,LOAD(ZetaL+c1-s*a));
  PARALLELIF(
    d > threshold,
    for(size_t K=0; K <= d; K += s) {
      Complex *ZetaL0=ZetaL-K;
      size_t stop=min(K+s,d+1);
      Vec Zeta=LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=CONJ(UNPACKH(Zeta,Zeta));
      Complex *fm=F+m;
      Complex *fpc1=F+c1;
      Complex *fmc1=fm-c1;
      Complex *upc1=U+c1;
      for(size_t k=max(1,K); k < stop; ++k) {
        Vec zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
        Vec Zetak=ZCMULT(zetak,Zetac1);

        Vec Fa=LOAD(F+k);
        Vec FA=LOAD(fpc1-k);
        Vec FB=LOAD(fmc1+k);
        Vec Fb=LOAD(fm-k);

        Vec b=Fb*Mhalf+CONJ(Fa);
        STORE(F+k,Fa+CONJ(Fb));
        Fb *= HSqrt3;
        Vec a=ZCMULT(zetak,UNPACKL(b,Fb));
        b=ZCMULTI(zetak,UNPACKH(b,Fb));

        STORE(fmc1+k,CONJ(a+b));
        STORE(U+k,a-b);

        b=FB*Mhalf+CONJ(FA);
        STORE(fpc1-k,FA+CONJ(FB));
        FB *= HSqrt3;
        a=ZCMULT(Zetak,UNPACKL(b,FB));
        b=ZCMULTI(Zetak,UNPACKH(b,FB));

        STORE(upc1-k,a-b);
        STORE(fm-k,CONJ(a+b));
      }
    });

  if(even) {
    F[c]=Re;
    U[c]=Im;
  }
}

void ImplicitHConvolution::posttransform(Complex *F, const Complex& w,
                                         Complex *U)
{
  Vec Mhalf=LOAD2(-0.5);
  Vec HSqrt3=LOAD2(hsqrt3);

  size_t m1=m-1;
  size_t c1=c+1;
  size_t d=c1/2;
  size_t a=c1/s;
  Vec Zeta=LOAD(ZetaH+a);
  Vec X=UNPACKL(Zeta,Zeta);
  Vec Y=CONJ(UNPACKH(Zeta,Zeta));
  Vec Zetac1=ZMULT(X,Y,LOAD(ZetaL+c1-s*a));

  if(even && m > 2) {
    size_t a=1/s;
    Vec Zeta=LOAD(ZetaH+a);
    Vec X=UNPACKL(Zeta,Zeta);
    Vec Y=CONJ(UNPACKH(Zeta,Zeta));
    Vec zeta1=ZMULT(X,Y,LOAD(ZetaL+1-s*a));
    Vec Zeta1=ZCMULT(zeta1,Zetac1);
    Complex *f0=F;
    Vec F0=LOAD(f0+1);
    Vec F1=ZCMULT(zeta1,LOAD(&w));
    Vec F2=ZMULT(zeta1,LOAD(U+1));
    Vec S=F1+F2;
    F2=CONJ(F0+Mhalf*S)-HSqrt3*FLIP(F1-F2);
    STORE(f0+1,F0+S);
    F0=LOAD(f0+c);
    F1=ZCMULT(Zeta1,LOAD(f0+m1));
    STORE(f0+m1,F2);
    F2=ZMULT(Zeta1,LOAD(U+c));
    STORE(f0+c,F0+F1+F2);
  }

  size_t D=c-d;
  PARALLELIF(
    D > threshold,
    for(size_t K=0; K <= D; K += s) {
      Complex *ZetaL0=ZetaL-K;
      size_t stop=min(K+s,D+1);
      Vec Zeta=LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=CONJ(UNPACKH(Zeta,Zeta));
      Complex *fm=F+m;
      Complex *fpc1=F+c1;
      Complex *fmc1=fm-c1;
      Complex *upc1=U+c1;
      for(size_t k=max(even+1,K); k < stop; ++k) {
        Vec zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
        Vec Zetak=ZCMULT(zetak,Zetac1);

        Vec F0=LOAD(F+k);
        Vec F1=ZCMULT(zetak,LOAD(fmc1+k));
        Vec F2=ZMULT(zetak,LOAD(U+k));
        Vec S=F1+F2;
        F2=CONJ(F0+Mhalf*S)-HSqrt3*FLIP(F1-F2);

        Vec FA=LOAD(fpc1-k);
        Vec FB=ZCMULT(Zetak,LOAD(fm-k));
        Vec FC=ZMULT(Zetak,LOAD(upc1-k));
        Vec T=FB+FC;

        STORE(F+k,F0+S);
        STORE(fpc1-k,FA+T);
        STORE(fmc1+k,CONJ(FA+Mhalf*T)-HSqrt3*FLIP(FB-FC));
        STORE(fm-k,F2);
      }
    });


  if(d == D+1) {
    size_t a=d/s;
    Vec Zeta=LOAD(ZetaH+a);
    Vec X=UNPACKL(Zeta,Zeta);
    Vec Y=CONJ(UNPACKH(Zeta,Zeta));
    Vec Zetak=ZMULT(X,Y,LOAD(ZetaL+d-s*a));
    Vec F0=LOAD(F+d);
    Vec F1=ZCMULT(Zetak,LOAD(d == 1 && even ? &w : F+m-d));
    Vec F2=ZMULT(Zetak,LOAD(U+d));
    Vec S=F1+F2;
    STORE(F+d,F0+S);
    STORE(F+m-d,CONJ(F0+Mhalf*S)-HSqrt3*FLIP(F1-F2));
  }
}

void ImplicitHConvolution::convolve(Complex **F, realmultiplier *pmult,
                                    size_t i, size_t offset)
{
  if(indexsize >= 1) index[indexsize-1]=i;

  // Set problem-size variables and pointers:
  size_t C=max(A,B);

  Complex *C0[C], *C1[C], *C2[C]; // inputs to complex2real FFTs
  double  *D0[C], *D1[C], *D2[C]; // outputs of complex2real FFTs
  Complex **c0=C0, **c1=C1, **c2=C2;
  double **d0=D0, **d1=D1, **d2=D2;

  size_t start=m-1-c; // c-1 (c) for m=even (odd)
  for(size_t a=0; a < C; ++a) {
    Complex *f=F[a]+offset;
    c0[a]=f;
    c1[a]=f+start;
  }

  if(A != B) {
    for(size_t a=0; a < C-1; ++a) {
      d0[a]=(double *) c0[a+1];
      d1[a]=(double *) c1[a+1];
    }
    if(A > B) {
      d0[A-1]=(double *) U[A-1];
      d1[A-1]=(double *) U[A-1];
      d2=(double **) U;
      for(size_t b=0; b < B; ++b)
        c2[b]=U[b+1];
    } else {
      d0[B-1]=(double *) U[0];
      d1[B-1]=(double *) U[0];
      for(size_t b=0; b < B-1; ++b)
        c2[b]=U[b+1];
      c2[B-1]=U[0];
      for(size_t b=0; b < B; ++b)
        d2[b]=(double *) c2[b];
    }
  } else {
    c2=U;
    d0=(double **) c0;
    d1=(double **) c1;
    d2=(double **) c2;
  }

  // Complex-to-real FFTs and pmults:

  double Re[B],Im[B];

  // r=-1 (backwards):
  if(A >= B) {
    for(size_t a=0; a < A-1; ++a) {
      pretransform(c0[a],w+a,U[A-1]);
      cro->fft(U[A-1],U[a]);
    }
    pretransform(c0[A-1],w+A-1,U[A-1]);
    cr->fft(U[A-1]);
    (*pmult)((double **) U,m,indexsize,index,-1,threads);
  } else {
    for(size_t a=A; a-- > 0;) {// Loop from A-1 to 0.
      pretransform(c0[a],w+a,U[a]);
      cro->fft(U[a],d2[a]);
    }
  }

  // r=0:
  double T[A];
  for(size_t a=A; a-- > 0;) { // Loop from A-1 to 0.
    Complex *c0a=c0[a];
    T[a]=c0a[0].re; // r=0, k=0
    if(!compact)
      c0a[0].re += 2.0*c0a[m].re; // Nyquist
    crO->fft(c0a,d0[a]);
  }
  (*pmult)(d0,m,indexsize,index,0,threads);

  for(size_t b=0; b < B; ++b) {
    Complex *c0b=c0[b];
    rcO->fft(d0[b],c0b);
    if(!compact) c0b[m]=0.0; // Zero Nyquist mode, for Hermitian symmetry.
    Complex z=c0[b][start];  // r=0, k=start
    Re[b]=z.re;
    Im[b]=z.im;
  }

  if(even) {
    for(size_t a=C; a-- > 0;) { // Loop from C-1 to 0.
      Complex *c1a=c1[a];
      Complex tmp=w[a];
      w[a].re=c1a[1].re; // r=0, k=c
      c1a[1]=tmp;          // r=1, k=1
    }
  }

  // r=1:
  for(size_t a=A; a-- > 0;) { // Loop from A-1 to 0.
    Complex *c1a=c1[a];
    c1a[0]=compact ? T[a] : T[a]-c1a[c+1].re; // r=1, k=0 with Nyquist
    crO->fft(c1[a],d1[a]);
  }
  (*pmult)(d1,m,indexsize,index,1,threads);

  for(size_t b=0; b < B; ++b) {
    Complex *c1b=c1[b];
    rcO->fft(d1[b],c1b); // r=1
    if(even) {
      double tmp=w[b].re;
      w[b]=c1b[1]; // r=1, k=1
      c1b[1]=tmp;    // r=0, k=c
    }
  }

  // r=-1 (forwards):
  if(A > B) {
    for(size_t b=0; b < B; ++b) {
      rco->fft(d2[b],U[A-1]);
      double R=c1[b][0].re;
      c0[b][start]=Complex(Re[b],Im[b]); // r=0, k=c-1 (c) for m=even (odd)
      c0[b][0]=c0[b][0].re+R+U[A-1][0].re;
      posttransform(c0[b],w[b],U[A-1]);
    }
  } else {
    if(A < B)
      (*pmult)(d2,m,indexsize,index,-1,threads);

    rc->fft(c2[0]);
    double R=c1[0][0].re;
    c0[0][start]=Complex(Re[0],Im[0]); // r=0, k=c-1 (c) for m=even (odd)
    c0[0][0]=c0[0][0].re+R+c2[0][0].re;
    posttransform(c0[0],w[0],c2[0]);

    for(size_t b=1; b < B; ++b) {
      rco->fft(d2[b],c2[0]);
      double R=c1[b][0].re;
      c0[b][start]=Complex(Re[b],Im[b]); // r=0, k=c-1 (c) for m=even (odd)
      c0[b][0]=c0[b][0].re+R+c2[0][0].re;
      posttransform(c0[b],w[b],c2[0]);
    }
  }
}

void fftpad::expand(Complex *f, Complex *u)
{
  PARALLELIF(
    m*M > threshold,
    for(size_t K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      size_t stop=min(K+s,m);
      Vec H=LOAD(ZetaH+K/s);
      for(size_t k=K; k < stop; ++k) {
        Vec Zetak=ZMULT(H,LOAD(ZetaL0+k));
        Vec X=UNPACKL(Zetak,Zetak);
        Vec Y=CONJ(UNPACKH(Zetak,Zetak));
        size_t kstride=k*stride;
        Complex *fk=f+kstride;
        Complex *uk=u+kstride;
        for(size_t i=0; i < M; ++i)
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
  PARALLELIF(
    m*M > threshold,
    for(size_t K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      size_t stop=min(K+s,m);
      Vec H=LOAD(ZetaH+K/s);
      for(size_t k=K; k < stop; ++k) {
        Vec Zetak=ZMULT(H,LOAD(ZetaL0+k));
        Vec X=UNPACKL(Zetak,Zetak);
        Vec Y=UNPACKH(CONJ(Zetak),Zetak);
        size_t kstride=k*stride;
        Complex *uk=u+kstride;
        Complex *fk=f+kstride;
        for(size_t i=0; i < M; ++i)
          STORE(fk+i,LOAD(fk+i)+ZMULT(X,Y,LOAD(uk+i)));
      }
    });
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
  PARALLELIF(
    M > threshold,
    for(size_t i=0; i < M; ++i)
      u[i]=fm1stride[i];
    );

  Vec Mhalf=LOAD2(-0.5);
  Vec Mhsqrt3=LOAD2(-hsqrt3);
  size_t stop=s;

  for(size_t K=0; K < m; K += s) {
    Complex *ZetaL0=ZetaL-K;
    Vec H=LOAD(ZetaH+K/s);
    for(size_t k=max(1,K); k < stop; ++k) {
      Vec Zetak=ZMULT(H,LOAD(ZetaL0+k));
      Vec X=UNPACKL(Zetak,Zetak);
      Vec Y=CONJ(UNPACKH(Zetak,Zetak));
      size_t kstride=k*stride;
      Complex *uk=u+kstride;
      Complex *fk=f+kstride;
      Complex *fmk=fm1stride+kstride;
      for(size_t i=0; i < M; ++i) {
        Vec A=LOAD(fmk+i);
        Vec B=LOAD(f+i);
        Vec Z=B*Mhalf+A;
        STORE(f+i,LOAD(fk+i));
        STORE(fk+i,B+A);
        B *= Mhsqrt3;
        A=ZMULT(X,Y,UNPACKL(Z,B));
        B=ZMULTI(X,Y,CONJ(UNPACKH(Z,B)));
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

  PARALLELIF(
    M > threshold,
    for(size_t i=0; i < M; ++i) {
      umstride[i]=fm1stride[i]; // Store extra value here.
      fm1stride[i]=u[i];
    });
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
  PARALLELIF(
    M > threshold,
  for(size_t i=0; i < M; ++i)
    umstride[i]=umstride[i]+f[i]+u[i];
    );

  Vec Mhalf=LOAD2(-0.5);
  Vec HSqrt3=LOAD2(hsqrt3);
  Complex *fm1stride=f+(m-1)*stride;

  size_t stop=s;
  for(size_t K=0; K < m; K += s) {
    Complex *ZetaL0=ZetaL-K;
    Vec H=LOAD(ZetaH+K/s);
    for(size_t k=max(1,K); k < stop; ++k) {
      Vec Zetak=ZMULT(H,LOAD(ZetaL0+k));
      Vec X=UNPACKL(Zetak,Zetak);
      Vec Y=CONJ(UNPACKH(Zetak,Zetak));
      size_t kstride=k*stride;
      Complex *fk=f+kstride;
      Complex *fm1k=fm1stride+kstride;
      Complex *uk=u+kstride;
      for(size_t i=0; i < M; ++i) {
        Vec F0=LOAD(fk+i);
        Vec F1=ZMULT(X,-Y,LOAD(fm1k+i));
        Vec F2=ZMULT(X,Y,LOAD(uk+i));
        Vec S=F1+F2;
        STORE(fk+i-stride,F0+Mhalf*S+HSqrt3*ZMULTI(F1-F2));
        STORE(fm1k+i,F0+S);
      }
    }
    stop=min(stop+s,m);
  }

  for(size_t i=0; i < M; ++i)
    fm1stride[i]=umstride[i];
}

void fft0pad::Forwards0(Complex *f)
{
  Forwards->fft(f+(m-1)*stride);
}

void fft0pad::Forwards1(Complex *f, Complex *u)
{
  Complex *umstride=u+m*stride;
  size_t m1stride=(m-1)*stride;
  Complex *fm1stride=f+m1stride;
  PARALLELIF(
    M > threshold,
  for(size_t i=0; i < M; ++i) {
    Complex temp=umstride[i];
    umstride[i]=fm1stride[i];
    fm1stride[i]=temp;
  });
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
  PARALLELIF(
    M > threshold,
  for(size_t i=0; i < M; ++i) {
    Complex Nyquist=f[i];
    f[i]=fmstride[i]+2.0*Nyquist;
    u[i]=fmstride[i] -= Nyquist;
  });

  Vec Mhalf=LOAD2(-0.5);
  Vec Mhsqrt3=LOAD2(-hsqrt3);
  size_t inc=s;
  PARALLELIF(
    m*M > threshold,
    for(size_t K=0; K < m; K += inc) {
      Complex *ZetaL0=ZetaL-K;
      size_t stop=min(K+s,m);
      Vec Zeta=LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=CONJ(UNPACKH(Zeta,Zeta));
      for(size_t k=max(1,K); k < stop; ++k) {
        Vec zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
        size_t kstride=k*stride;
        Complex *uk=u+kstride;
        Complex *fk=f+kstride;
        Complex *fmk=fmstride+kstride;
        for(size_t i=0; i < M; ++i) {
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
    });
}

void fft1pad::Backwards1(Complex *f, Complex *u)
{
  Backwards->fft(f+m*stride);
}

void fft1pad::reduce(Complex *f, Complex *u)
{
  Complex *fmstride=f+m*stride;

  for(size_t i=0; i < M; ++i) {
    Complex f0=f[i];
    Complex f1=fmstride[i];
    Complex f2=u[i];
    f[i]=0.0; // Zero Nyquist mode, for Hermitian symmetry.
    fmstride[i]=(f0+f1+f2);
  }
  Vec Mhalf=LOAD2(-0.5);
  Vec HSqrt3=LOAD2(hsqrt3);

  size_t inc=s;
  PARALLELIF(
    m*M > threshold,
    for(size_t K=0; K < m; K += inc) {
      Complex *ZetaL0=ZetaL-K;
      size_t stop=min(K+s,m);
      Vec Zeta=LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=CONJ(UNPACKH(Zeta,Zeta));
      for(size_t k=max(1,K); k < stop; ++k) {
        Vec zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
        size_t kstride=k*stride;
        Complex *fk=f+kstride;
        Complex *fmk=fmstride+kstride;
        Complex *uk=u+kstride;
        for(size_t i=0; i < M; ++i) {
          Vec F0=LOAD(fk+i);
          Vec F1=ZCMULT(zetak,LOAD(fmk+i));
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
                                 size_t offset)
{
  size_t twom=2*m;
  if(M == 1) { // a[k]=a[k]*b[k]*c[k]
    double *C0=C[0]+offset;
#ifdef __SSE2__
    PARALLEL(
      for(size_t k=0; k < twom; k += 2) {
        double *ak=a+k;
        STORE(ak,LOAD(ak)*LOAD(b+k)*LOAD(C0+k));
      }
      );
#else
    PARALLEL(
      for(size_t k=0; k < twom; ++k)
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
      for(size_t k=0; k < twom; k += 2) {
        double *ak=a+k;
        STORE(ak,LOAD(ak)*LOAD(b+k)*LOAD(C0+k)+
              LOAD(a1+k)*LOAD(b1+k)*LOAD(C1+k));
      }
      );
#else
    PARALLEL(
      for(size_t k=0; k < twom; ++k)
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
      for(size_t k=0; k < twom; k += 2) {
        double *ak=a+k;
        STORE(ak,LOAD(ak)*LOAD(b+k)*LOAD(C0+k)+
              LOAD(a1+k)*LOAD(b1+k)*LOAD(C1+k)+
              LOAD(a2+k)*LOAD(b2+k)*LOAD(C2+k));
      }
      );
#else
    PARALLEL(
      for(size_t k=0; k < twom; ++k)
        a[k]=a[k]*b[k]*C0[k]+a1[k]*b1[k]*C1[k]+a2[k]*b2[k]*C2[k];
      );
#endif
  } else {
    double *A=a-offset;
    double *B=b-offset;
    double *C0=C[0];
    size_t stop=twom+offset;
#ifdef __SSE2__
    PARALLEL(
      for(size_t k=offset; k < stop; k += 2) {
        double *p=A+k;
        double *q=B+k;
        Vec sum=LOAD(p)*LOAD(q)*LOAD(C0+k);
        for(size_t i=1; i < M; ++i) {
          size_t istride=i*stride;
          sum += LOAD(p+istride)*LOAD(q+istride)*LOAD(C[i]+k);
        }
        STORE(p,sum);
      }
      );
#else
    PARALLEL(
      for(size_t k=offset; k < stop; ++k) {
        double *p=A+k;
        double *q=B+k;
        double sum=(*p)*(*q)*C0[k];
        for(size_t i=1; i < M; ++i) {
          size_t istride=i*stride;
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
                                     size_t offset)
{
  // 8M-3 of 8M FFTs are out-of-place
  Complex *w=W[0];

  size_t m1=m+1;
  for(size_t i=0; i < M; ++i) {
    Complex *fi=F[i]+offset;
    Complex *gi=G[i]+offset;
    Complex *hi=H[i]+offset;
    size_t im1=i*m1;
    Complex *ui=u+im1;
    Complex *vi=v+im1;
    Complex *wi=w+im1;
    if(i+1 < M) {
      ui += m1;
      vi += m1;
      wi += m1;
    }
    PARALLEL(
      for(size_t K=0; K < m; K += s) {
        Complex *ZetaL0=ZetaL-K;
        size_t stop=min(K+s,m);
        Vec Zeta=LOAD(ZetaH+K/s);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=CONJ(UNPACKH(Zeta,Zeta));
        for(size_t k=K; k < stop; ++k) {
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

  for(size_t i=0; i < M; ++i) {
    Complex *fi=F[i]+offset;
    Complex *gi=G[i]+offset;
    Complex *hi=H[i]+offset;
    size_t im1=i*m1;
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

  PARALLEL(
    for(size_t K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      size_t stop=min(K+s,m);
      Vec Zeta=LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=CONJ(UNPACKH(Zeta,Zeta));
      for(size_t k=K; k < stop; ++k) {
        Vec Zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
        Complex *fk=f+k;
        STORE(fk,ZCMULT(Zetak,LOAD(u+k))+LOAD(fk));
      }
    }
    );
}

// a[k]=a[k]*b[k]*b[k]
void ImplicitHFGGConvolution::mult(double *a, double *b)
{
  size_t twom=2*m;
#ifdef __SSE2__
  PARALLEL(
    for(size_t k=0; k < twom; k += 2) {
      double *ak=a+k;
      STORE(ak,LOAD(ak)*LOAD(ak)*LOAD(b+k));
    }
    );
#else
  PARALLEL(
    for(size_t k=0; k < twom; ++k) {
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
    for(size_t K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      size_t stop=min(K+s,m);
      Vec Zeta=LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=CONJ(UNPACKH(Zeta,Zeta));
      for(size_t k=K; k < stop; ++k) {
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

  PARALLEL(
    for(size_t K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      size_t stop=min(K+s,m);
      Vec Zeta=LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=CONJ(UNPACKH(Zeta,Zeta));
      for(size_t k=K; k < stop; ++k) {
        Vec Zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
        Complex *fk=f+k;
        STORE(fk,ZCMULT(Zetak,LOAD(u+k))+LOAD(fk));
      }
    }
    );
}

// a[k]=a[k]^3
void ImplicitHFFFConvolution::mult(double *a)
{
  size_t twom=2*m;
  PARALLEL(
    for(size_t k=0; k < twom; k += 2) {
      double *p=a+k;
      Vec ak=LOAD(p);
      STORE(p,ak*ak*ak);
    }
    );
}

void ImplicitHFFFConvolution::convolve(Complex *f, Complex *u)
{
  PARALLEL(
    for(size_t K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      size_t stop=min(K+s,m);
      Vec Zeta=LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=CONJ(UNPACKH(Zeta,Zeta));
      for(size_t k=K; k < stop; ++k)
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
  PARALLEL(
    for(size_t K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      size_t stop=min(K+s,m);
      Vec Zeta=LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=CONJ(UNPACKH(Zeta,Zeta));
      for(size_t k=K; k < stop; ++k) {
        Complex *p=f+k;
        STORE(p,ZCMULT(ZMULT(X,Y,LOAD(ZetaL0+k)),LOAD(u+k))+LOAD(p));
      }
    }
    );
}

void fft0bipad::backwards(Complex *f, Complex *u)
{
  for(size_t i=0; i < M; ++i)
    f[i]=0.0;
  for(size_t i=0; i < M; ++i)
    u[i]=0.0;

  size_t twom=2*m;
  PARALLEL(
    for(size_t K=0; K < twom; K += s) {
      Complex *ZetaL0=ZetaL-K;
      size_t stop=min(K+s,twom);
      Vec H=-LOAD(ZetaH+K/s);
      for(size_t k=max(1,K); k < stop; ++k) {
        Vec Zetak=ZMULT(H,LOAD(ZetaL0+k));
        Vec X=CONJ(UNPACKL(Zetak,Zetak));
        Vec Y=UNPACKH(Zetak,Zetak);
        size_t kstride=k*stride;
        Complex *fk=f+kstride;
        Complex *uk=u+kstride;
        for(size_t i=0; i < M; ++i)
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

  size_t twom=2*m;
  PARALLEL(
    for(size_t K=0; K < twom; K += s) {
      Complex *ZetaL0=ZetaL-K;
      size_t stop=min(K+s,twom);
      Vec H=LOAD(ZetaH+K/s);
      for(size_t k=max(1,K); k < stop; ++k) {
        Vec Zetak=ZMULT(H,LOAD(ZetaL0+k));
        Vec X=CONJ(UNPACKL(Zetak,Zetak));
        Vec Y=UNPACKH(Zetak,Zetak);
        size_t kstride=k*stride;
        Complex *uk=u+kstride;
        Complex *fk=f+kstride;
        for(size_t i=0; i < M; ++i) {
          Complex *p=fk+i;
          STORE(p,LOAD(p)+ZMULTI(X,Y,LOAD(uk+i)));
        }
      }
    }
    );
}

// This multiplication routine is for binary convolutions and takes two inputs
// of size m.
// F[0][j] *= conj(F[0][j]);
void multautocorrelation(Complex **F, size_t m,
                         const size_t indexsize,
                         const size_t *index,
                         size_t r, size_t threads)
{
  Complex* F0=F[0];

#ifdef __SSE2__
  PARALLELIF(
    m > threshold,
    for(size_t j=0; j < m; ++j) {
      Complex *p=F0+j;
      STORE(p,ZMULT(LOAD(p),CONJ(LOAD(p))));
    }
    );
#else
  PARALLELIF(
    m > threshold,
    for(size_t j=0; j < m; ++j)
      F0[j] *= conj(F0[j]);
    );
#endif
}

void multcorrelation(Complex **F, size_t m,
                     const size_t indexsize,
                     const size_t *index,
                     size_t r, size_t threads)
{
  Complex* F0=F[0];
  Complex* F1=F[1];

#ifdef __SSE2__
  PARALLELIF(
    m > threshold,
    for(size_t j=0; j < m; ++j) {
      Complex *p=F0+j;
      Complex *q=F1+j;
      STORE(p,ZMULT(LOAD(p),CONJ(LOAD(q))));
    }
    );
#else
  PARALLELIF(
    m > threshold,
    for(size_t j=0; j < m; ++j)
      F0[j] *= conj(F1[j]);
    );
#endif
}

// Unscramble indices, returning spatial index for remainder r at position j.
inline size_t innerindex(size_t j, int r) {return 2*j+r;}

// This multiplication routine is for binary convolutions and takes two inputs
// of size m.
// F[0][j] *= F[1][j];
void multbinary(Complex **F, size_t m,
                const size_t indexsize,
                const size_t *index,
                size_t r, size_t threads)
{
  Complex* F0=F[0];
  Complex* F1=F[1];

#if 0 // Spatial indices are available, if needed.
  size_t n=indexsize;
  for(size_t j=0; j < m; ++j) {
    for(size_t d=0; d < n; ++d)
      cout << index[d] << ",";
    cout << innerindex(j,r) << endl;
  }
#endif

#ifdef __SSE2__
  PARALLELIF(
    m > threshold,
    for(size_t j=0; j < m; ++j) {
      Complex *p=F0+j;
      STORE(p,ZMULT(LOAD(p),LOAD(F1+j)));
    }
    );
#else
  PARALLELIF(
    m > threshold,
    for(size_t j=0; j < m; ++j)
      F0[j] *= F1[j];
    );
#endif
}

// F[0][j] *= F[0][j];
void multautoconvolution(Complex **F, size_t m,
                         const size_t indexsize,
                         const size_t *index,
                         size_t r, size_t threads)
{
  Complex* F0=F[0];

#ifdef __SSE2__
  PARALLELIF(
    m > threshold,
    for(size_t j=0; j < m; ++j) {
      Complex *p=F0+j;
      STORE(p,ZMULT(LOAD(p),LOAD(p)));
    }
    );
#else
  PARALLELIF(
    m > threshold,
    for(size_t j=0; j < m; ++j)
      F0[j] *= F0[j];
    );
#endif
}

// Unscramble indices, returning spatial index for remainder r at position j.
inline size_t innerindex(size_t j, int r, size_t m) {
  int x=3*j+r;
  return x >= 0 ? x : 3*m-1;
}

// This multiplication routine is for binary Hermitian convolutions and takes
// two inputs.
// F[0][j] *= F[1][j];
void multbinary(double **F, size_t m,
                const size_t indexsize,
                const size_t *index,
                size_t r, size_t threads)
{
  double* F0=F[0];
  double* F1=F[1];

#if 0 // Spatial indices are available, if needed.
  //size_t n=index.size();
  size_t n=indexsize;
  for(size_t j=0; j < m; ++j) {
    for(size_t d=0; d < n; ++d)
      cout << index[d] << ",";
    cout << innerindex(j,r,m) << endl;
  }
#endif

#ifdef __SSE2__
  size_t m1=m-1;
  PARALLELIF(
    m1 > threshold,
    for(size_t j=0; j < m1; j += 2) {
      double *p=F0+j;
      STORE(p,LOAD(p)*LOAD(F1+j));
    }
    if(m % 2)
      F0[m1] *= F1[m1];
    );
#else
  PARALLELIF(
    m > threshold,
    for(size_t j=0; j < m; ++j)
      F0[j] *= F1[j];
    );
#endif
}

// F[0][j]=F[0][j]*F[2][j]+F[1][j]*F[3][j]
void multbinary2(Complex **F, size_t m,
                 const size_t indexsize,
                 const size_t *index,
                 size_t r, size_t threads)
{
  Complex* F0=F[0];
  Complex* F1=F[1];
  Complex* F2=F[2];
  Complex* F3=F[3];

#ifdef __SSE2__
  PARALLELIF(
    m > threshold,
    for(size_t j=0; j < m; ++j) {
      Complex *F0j=F0+j;
      STORE(F0j,ZMULT(LOAD(F0j),LOAD(F2+j))
            +ZMULT(LOAD(F1+j),LOAD(F3+j)));
    }
    );
#else
  PARALLELIF(
    m > threshold,
    for(size_t j=0; j < m; ++j)
      F0[j]=F0[j]*F2[j]+F1[j]*F3[j];
    );
#endif
}

// F[0][j]=F[0][j]*F[2][j]+F[1][j]*F[3][j]
void multbinary2(double **F, size_t m,
                 const size_t indexsize,
                 const size_t *index,
                 size_t r, size_t threads)
{
  double* F0=F[0];
  double* F1=F[1];
  double* F2=F[2];
  double* F3=F[3];

#ifdef __SSE2__
  size_t m1=m-1;
  PARALLELIF(
    m1 > threshold,
    for(size_t j=0; j < m1; j += 2) {
      double *F0j=F0+j;
      STORE(F0j,LOAD(F0j)*LOAD(F2+j)+LOAD(F1+j)*LOAD(F3+j));
    }
    );
  if(m % 2)
    F0[m1]=F0[m1]*F2[m1]+F1[m1]*F3[m1];
#else
  PARALLELIF(
    m > threshold,
    for(size_t j=0; j < m; ++j)
      F0[j]=F0[j]*F2[j]+F1[j]*F3[j];
    );
#endif
}

// F[0][j]=F[0][j]*F[3][j]+F[1][j]*F[4][j]+F[2][j]*F[5][j];
void multbinary3(Complex **F, size_t m,
                 const size_t indexsize,
                 const size_t *index,
                 size_t r, size_t threads)
{
  Complex* F0=F[0];
  Complex* F1=F[1];
  Complex* F2=F[2];
  Complex* F3=F[3];
  Complex* F4=F[4];
  Complex* F5=F[5];

#ifdef __SSE2__
  PARALLELIF(
    m > threshold,
    for(size_t j=0; j < m; ++j) {
      Complex *F0j=F0+j;
      STORE(F0j,ZMULT(LOAD(F0j),LOAD(F3+j))
            +ZMULT(LOAD(F1+j),LOAD(F4+j))
            +ZMULT(LOAD(F2+j),LOAD(F5+j))
        );
    }
    );
#else
   PARALLELIF(
    m > threshold,
    for(size_t j=0; j < m; ++j)
      F0[j]=F0[j]*F3[j]+F1[j]*F4[j]+F2[j]*F5[j];
    );

#endif
}

// F[0][j]=F[0][j]*F[4][j]+F[1][j]*F[5][j]+F[2][j]*F[6][j]+F[3][j]*F[7][j];
void multbinary4(Complex **F, size_t m,
                 const size_t indexsize,
                 const size_t *index,
                 size_t r, size_t threads)
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
   PARALLELIF(
    m > threshold,
    for(size_t j=0; j < m; ++j) {
      Complex *F0j=F0+j;
      STORE(F0j,ZMULT(LOAD(F0j),LOAD(F4+j))
            +ZMULT(LOAD(F1+j),LOAD(F5+j))
            +ZMULT(LOAD(F2+j),LOAD(F6+j))
            +ZMULT(LOAD(F3+j),LOAD(F7+j))
        );
    }
    );
#else
  PARALLELIF(
    m > threshold,
    for(size_t j=0; j < m; ++j)
      F0[j]=F0[j]*F4[j]+F1[j]*F5[j]+F2[j]*F6[j]+F3[j]*F7[j];
    );

#endif
}

// F[0][j]=F[0][j]*F[8][j]+F[1][j]*F[9][j]+F[2][j]*F[10][j]+F[3][j]*F[11][j]+
//         F[4][j]*F[12][j]+F[5][j]*F[13][j]+F[6][j]*F[14][j]+F[7][j]*F[15][j];
void multbinary8(Complex **F, size_t m,
                 const size_t indexsize,
                 const size_t *index,
                 size_t r, size_t threads)
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
  PARALLELIF(
    m > threshold,
    for(size_t j=0; j < m; ++j) {
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
  PARALLELIF(
    m > threshold,
    for(size_t j=0; j < m; ++j)
      F0[j]=F0[j]*F8[j]+F1[j]*F9[j]+F2[j]*F10[j]+F3[j]*F11[j]
        +F4[j]*F12[j]+F5[j]*F13[j]+F6[j]*F14[j]+F7[j]*F15[j];
    );
#endif
}

// This 2D version of the scheme of Basdevant, J. Comp. Phys, 50, 1983
// requires only 4 FFTs per stage.
void multadvection2(double **F, size_t m,
                    const size_t indexsize,
                    const size_t *index,
                    size_t r, size_t threads)
{
  double* F0=F[0];
  double* F1=F[1];

#ifdef __SSE2__
  size_t m1=m-1;
  PARALLELIF(
    m > threshold,
    for(size_t j=0; j < m1; j += 2) {
      double *F0j=F0+j;
      double *F1j=F1+j;
      Vec u=LOAD(F0j);
      Vec v=LOAD(F1j);
      STORE(F0j,v*v-u*u);
      STORE(F1j,u*v);
    }
    );
  if(m % 2) {
    double u=F0[m1];
    double v=F1[m1];
    F0[m1]=v*v-u*u;
    F1[m1]=u*v;
  }
#else
  PARALLELIF(
    m > threshold,
  for(size_t j=0; j < m; ++j) {
    double u=F0[j];
    double v=F1[j];
    F0[j]=v*v-u*u;
    F1[j]=u*v;
  })
#endif
}

} // namespace fftwpp
