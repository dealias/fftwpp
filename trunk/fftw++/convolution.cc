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

void ImplicitConvolution::convolve(Complex **F,
                                   multiplier *pmult, unsigned int offset)
{ 
  unsigned int C=max(A,B);
  Complex *P[C];
  for(unsigned int i=0; i < A; ++i)
    P[i]=F[i]+offset;
  
  // Backwards FFT (even indices):
  for(unsigned int i=0; i < A; ++i) {
    //std::cout << P[i] << " " << U[i] << std::endl;
    BackwardsO->fft(P[i],U[i]);
  }
  (*pmult)(U,m,threads); // multiply even indices

  switch(A) {
  case 1: premult<premult1>(P); break;
  case 2: premult<premult2>(P); break;
  case 3: premult<premult3>(P); break;
  case 4: premult<premult4>(P); break;
  default: premult<general>(P);
  }

  if(out_of_place) { // out-of-place FFTs: U[A-1] is free if A > B.
    Complex *W[A];
    W[A-1]=U[A-1];
    for(unsigned int i=1; i < A; ++i) 
      W[i-1]=P[i];

    for(unsigned int i=A; i-- > 0;) // Loop from A-1 to 0.
      BackwardsO->fft(P[i],W[i]);
    (*pmult)(W,m,threads); // multiply odd indices
    
    // Return to original space
    Complex *lastW=W[A-1];
    for(unsigned int i=0; i < B; ++i) {
      Complex *Pi=P[i];
      ForwardsO->fft(W[i],Pi);
      ForwardsO->fft(U[i],lastW);
      postmultadd(Pi,lastW);
    }
    
  } else { 
    // in-place FFTs: this could be optimised for B > A.

    // Backwards FFT (odd indices):
    for(unsigned int i=0; i < A; ++i)
      Backwards->fft(P[i]);
    (*pmult)(P,m,threads); // multiply odd indices

    // Return to original space:
    for(unsigned int i=0; i < B; ++i) {
      Complex *fi=P[i];
      Complex *ui=U[i];
      Forwards->fft(fi);
      Forwards->fft(ui);
      postmultadd(fi,ui);
    }
  }
}

#ifdef __SSE2__
template<class T>
inline void ImplicitConvolution::
premult(Complex **F, unsigned int k, Vec& Zetak)
{
  for(unsigned int i=0; i < A; ++i) {
    Complex *fki=F[i]+k;
    STORE(fki,ZMULT(Zetak,LOAD(fki)));
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
#endif    

// multiply by root of unity to prepare for inverse FFT for odd modes
template<class T>
void ImplicitConvolution::premult(Complex **F)
{  
#ifdef __SSE2__
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
#else
  premult_nonSSE2(F);
#endif
}

void ImplicitConvolution::premult_nonSSE2(Complex **F)
{
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
      Complex *p=ZetaH+K/s;
      double Hre=p->re;
      double Him=p->im;
      for(unsigned int k=K; k < stop; ++k) {
        Complex L=*(ZetaL0+k);
        double Re=Hre*L.re-Him*L.im;
        double Im=Hre*L.im+Him*L.re;
        for(unsigned int i=0; i < A; ++i) {
          Complex *Fki=F[i]+k;
          Complex fk=*Fki;
          Fki->re=Re*fk.re-Im*fk.im;
          Fki->im=Im*fk.re+Re*fk.im;
        }
      }
    }
    );
}
  
// multiply by root of unity to prepare and add for inverse FFT for odd modes
void ImplicitConvolution::postmultadd(Complex *f, Complex *u)
{
  double ninv=0.5/m;
#ifdef __SSE2__
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
#else
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      unsigned int stop=min(K+s,m);
      Complex *ZetaL0=ZetaL-K;
      Complex *p=ZetaH+K/s;
      double Hre=ninv*p->re;
      double Him=ninv*p->im;
      for(unsigned int k=K; k < stop; ++k) {
        Complex *P=f+k;
        Complex fk=*P;
        Complex U=*(u+k);
        Complex L=*(ZetaL0+k);
        double Re=Hre*L.re-Him*L.im;
        double Im=Hre*L.im+Him*L.re;
        P->re=ninv*U.re+Re*fk.re+Im*fk.im;
        P->im=ninv*U.im-Im*fk.re+Re*fk.im;
      }         
    }
   );
#endif
}

void ImplicitHConvolution::premult(Complex ** F,
				   unsigned int offset,
				   Complex* f1c)
{
  unsigned int C=max(A,B);
  Complex *P0[C];
  Complex *S; // only used for even quantities


#ifdef __SSE2__
  Vec Mhalf=LOAD(-0.5);
  Vec HSqrt3=LOAD(hsqrt3);
#else
#endif
    
  bool even=m == 2*c;
  if(even)
    S = new Complex[A];
  unsigned int m1=m-1;
  for(unsigned int i=0; i < A; ++i) {
    Complex *f=F[i]+offset;
    P0[i]=f;
    Complex *ui=U[i];
    ui[0]=f->re;
  }

  if(even) {
    unsigned int a=1/s;
    Vec Zeta=LOAD(ZetaH+a);
    Vec X=UNPACKL(Zeta,Zeta);
    Vec Y=UNPACKH(CONJ(Zeta),Zeta);
    Vec zeta1=ZMULT(X,Y,LOAD(ZetaL+1-s*a));
    
    for(unsigned int i=0; i < A; ++i) {
      Complex *fi=P0[i];
      Vec Fa=LOAD(fi+1);
      Vec Fb=LOAD(fi+m1);
      Vec B=Fb*Mhalf+CONJ(Fa);
      Fb *= HSqrt3;
      Vec A=ZMULTC(zeta1,UNPACKL(B,Fb));
      B=ZMULTIC(zeta1,UNPACKH(B,Fb));
      STORE(f1c+i,CONJ(A+B));
        
      double a=fi[c].re;
      S[i]=Complex(2.0*a,a+sqrt3*fi[c].im);
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
      for(unsigned int i=0; i < A; ++i) {
        Complex *fi=P0[i];
        Complex *fm=fi+m;
        Complex *fpc1=fi+c1;
        Complex *fmc1=fm-c1;
        Complex *ui=U[i];
        Complex *upc1=ui+c1;
        for(unsigned int k=max(1,K); k < stop; ++k) {
          Vec zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
          Vec Zetak=ZMULTC(zetak,Zetac1);
          
          Vec Fa=LOAD(fi+k);
          Vec FA=LOAD(fpc1-k);
          Vec FB=LOAD(fmc1+k);
          Vec Fb=LOAD(fm-k);
          
          Vec B=Fb*Mhalf+CONJ(Fa);
          STORE(fi+k,Fa+CONJ(Fb));
          Fb *= HSqrt3;
          Vec A=ZMULTC(zetak,UNPACKL(B,Fb));
          B=ZMULTIC(zetak,UNPACKH(B,Fb));
        
          Vec D=FB*Mhalf+CONJ(FA);
          STORE(fpc1-k,FA+CONJ(FB));
          FB *= HSqrt3;
          Vec C=ZMULTC(Zetak,UNPACKL(D,FB));
          D=ZMULTIC(Zetak,UNPACKH(D,FB));

          STORE(ui+k,A-B);
          STORE(upc1-k,C-D);
          STORE(fm-k,CONJ(C+D));
          STORE(fmc1+k,CONJ(A+B));
        }
      }
    }
    );
    
  if(even) {
    for(unsigned int i=0; i < A; ++i) {
      Complex *fi=P0[i];
      Complex *ui=U[i];
      fi[c]=S[i].re;
      ui[c]=S[i].im;
    }
    delete[] S;
  }
}

// Out of place version (with f1 stored in a separate array cr2B):
// Assume that cr0 and cr1 are contiguous (and, in fact, overlapping).
void ImplicitHConvolution::postmultadd(Complex **cr2, Complex **cr0, 
				       Complex **cr2B)
{
  double ninv=1.0/(3.0*m);
#ifdef __SSE2__
  Vec Ninv=LOAD(ninv);
  Vec Mhalf=LOAD(-0.5);
  Vec HSqrt3=LOAD(hsqrt3);
  PARALLEL(
	   for(unsigned int K=0; K <= c; K += s) {
	     Complex *ZetaL0=ZetaL-K;
	     unsigned int stop=min(K+s,c+1);
	     Vec Zeta=Ninv*LOAD(ZetaH+K/s);
	     Vec X=UNPACKL(Zeta,Zeta);
	     Vec Y=UNPACKH(CONJ(Zeta),Zeta);
	     for(unsigned int i=0; i < B; ++i) {
	       Complex *f0=cr0[i];
	       Complex *fm=f0+m;
	       Complex *ui=cr2[i];
	       Complex *cr2Bi=cr2B[i];
	       for(unsigned int k=max(1,K); k < stop; ++k) {
		 Vec Zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
		 Vec F0=LOAD(f0+k)*Ninv;
		 Vec F1=ZMULTC(Zetak,LOAD(cr2Bi+k));
		 Vec F2=ZMULT(Zetak,LOAD(ui+k));
		 Vec S=F1+F2;
		 STORE(f0+k,F0+S);
		 STORE(fm-k,CONJ(F0+Mhalf*S)-HSqrt3*FLIP(F1-F2));
	       }
	     }  
	   }
	   );
#endif
}

void ImplicitHConvolution::postmultadd0(Complex **crm, Complex **cr0, 
                                        Complex **crp, Complex *f1c)
{
  double ninv=1.0/(3.0*m);
  Vec Ninv=LOAD(ninv);
  bool even=m == 2*c;

#ifdef __SSE2__
  Vec Mhalf=LOAD(-0.5);
  Vec HSqrt3=LOAD(hsqrt3);
#else
#endif

  unsigned int m1=m-1;  
  unsigned int c1=c+1;
  unsigned int d=c1/2;
  unsigned int a=c1/s;
  Vec Zeta=LOAD(ZetaH+a);
  Vec X=UNPACKL(Zeta,Zeta);
  Vec Y=UNPACKH(CONJ(Zeta),Zeta);
  Vec Zetac1=ZMULT(X,Y,LOAD(ZetaL+c1-s*a));

#ifdef __SSE2__      
    if(even && m > 2) {
      unsigned int a=1/s;
      Vec Zeta=LOAD(ZetaH+a);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(CONJ(Zeta),Zeta);
      Vec zeta1=Ninv*ZMULT(X,Y,LOAD(ZetaL+1-s*a));
      Vec Zeta1=ZMULTC(zeta1,Zetac1);
      for(unsigned int i=0; i < B; ++i) {
        Complex *f0=cr0[i];
        Vec F0=LOAD(f0+1)*Ninv;
        Vec F1=ZMULTC(zeta1,LOAD(f1c+i));
        Vec F2=ZMULT(zeta1,LOAD(crm[i]+1));
        Vec S=F1+F2;
        F2=CONJ(F0+Mhalf*S)-HSqrt3*FLIP(F1-F2);
        STORE(f0+1,F0+S);
        F0=LOAD(f0+c)*Ninv;
        F1=ZMULTC(Zeta1,LOAD(f0+m1));
        STORE(f0+m1,F2);
        F2=ZMULT(Zeta1,LOAD(crm[i]+c));
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
        for(unsigned int i=0; i < B; ++i) {
          Complex *fi=cr0[i];
          Complex *fm=fi+m;
          Complex *fpc1=fi+c1;
          Complex *fmc1=fm-c1;
          Complex *ui=crm[i];
          Complex *upc1=ui+c1;
          for(unsigned int k=max(even+1,K); k < stop; ++k) {
            Vec zetak=ZMULT(X,Y,LOAD(ZetaL0+k));
            Vec Zetak=ZMULTC(zetak,Zetac1);
          
            Vec F0=LOAD(fi+k)*Ninv;
            Vec F1=ZMULTC(zetak,LOAD(fmc1+k));
            Vec F2=ZMULT(zetak,LOAD(crm[i]+k));
            Vec S=F1+F2;
            F2=CONJ(F0+Mhalf*S)-HSqrt3*FLIP(F1-F2);
            
            Vec FA=LOAD(fpc1-k)*Ninv;
            Vec FB=ZMULTC(Zetak,LOAD(fm-k));
            Vec FC=ZMULT(Zetak,LOAD(upc1-k));
            Vec T=FB+FC;
            
            STORE(fi+k,F0+S);
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
        for(unsigned int i=0; i < B; ++i) {
          Complex *f0=cr0[i];
          Vec F0=LOAD(f0+d)*Ninv;
          Vec F1=ZMULTC(Zetak,LOAD(f1c+i));
          Vec F2=ZMULT(Zetak,LOAD(crm[i]+d));
          Vec S=F1+F2;
          STORE(f0+d,F0+S);
          STORE(f0+m-d,CONJ(F0+Mhalf*S)-HSqrt3*FLIP(F1-F2));
        }
      } else {
        for(unsigned int i=0; i < B; ++i) {
          Complex *f0=cr0[i];
          Vec F0=LOAD(f0+d)*Ninv;
          Vec F1=ZMULTC(Zetak,LOAD(f0+m-d));
          Vec F2=ZMULT(Zetak,LOAD(U[i]+d));
          Vec S=F1+F2;
          STORE(f0+d,F0+S);
          STORE(f0+m-d,CONJ(F0+Mhalf*S)-HSqrt3*FLIP(F1-F2));
        }
      }
    }
#endif  
}



void ImplicitHConvolution::convolve(Complex **F, 
                                    realmultiplier *pmult, unsigned int offset)
{
  // Set problem-size variables and pointers:

  unsigned int C=max(A,B);
  Complex cr1c[C];

  Complex *cr0[A], *cr1[A], *cr2[A]; // inputs to complex2real FFTs
  double  *dr0[A], *dr1[A], *dr2[A]; // outputs of complex2real FFTs

  bool even=m == 2*c;
  unsigned int start=m-1-c;
  for(unsigned int i=0; i < A; ++i) {
    Complex *f=F[i]+offset;
    cr0[i]=f;
    cr1[i]=f+start;
    Complex *ui=U[i];
    ui[0]=f->re;
    cr2[i]=U[i];
  }

  const bool out_of_place=A >= 2*B;
  
  if(out_of_place) {
    for(unsigned int i=0; i < A-1; ++i) {
      dr0[i]=(double *) cr0[i+1];
      dr1[i]=(double *) cr1[i+1];
    }
    dr0[A-1]=(double *) cr2[A-1];
    dr1[A-1]=(double *) cr2[A-1];
    
  } else {
    for(unsigned int i=0; i < A; ++i) {
      dr0[i]=(double *) cr0[i];
      dr1[i]=(double *) cr1[i];
    }
  }
  // r=2 is always in-place for the cr transform (TODO: perhaps change
  // the order of premult to get around this?)
  for(unsigned int i=0; i < A; ++i)
    dr2[i]=(double *) cr2[i];


  // premult:
  premult(F,offset,cr1c);


  // Complex-to-real FFTs and pmults:
  Complex S[B];
  {
    // step 1:
    // r=2 (aka r=-1):
    for(unsigned int i=0; i < A; ++i)
      cr->fft(cr2[i],dr2[i]);
    (*pmult)(dr2,m,threads);

    // r=0:
    double T[A]; // deal with overlap between r=0 and r=1
    for(unsigned int i=A; i-- > 0;) { // Loop from A-1 to 0.
      Complex *cr0i=cr0[i];
      T[i]=cr0i[0].re; // r=0, k=0
      (out_of_place ? cro : cr)->fft(cr0i,dr0[i]); 
    }
    (*pmult)(dr0,m,threads);
    // The last complex in r=0 is the first complex in r=1, so we have
    // to save it.
    for(unsigned int i=0; i < B; ++i) {
      Complex *cd=(Complex *)dr0[i];
      S[i]=cd[start];   // r=0, k=c-1 (c) for m=even (odd)
    }
    
    // r=1:
    for(unsigned int i=A; i-- > 0;) { // Loop from A-1 to 0.
      Complex *cr1i=cr1[i];
      cr1i[0]=T[i];       // r=1, k=0
      if(even) {
	Complex tmp=cr1c[i];
	cr1c[i]=cr1i[1];  // r=0, k=c
	cr1i[1]=tmp;      // r=1, k=1
      }
      (out_of_place ? cro : cr)->fft(cr1[i],dr1[i]);
    }
    (*pmult)(dr1,m,threads);
  }

  // Real-to-complex FFTs and postmultadd:
  if(out_of_place) {
    // Put dr1 into the second half of cr0:
    Complex **cr2B=cr2+B;

    const double ninv=1.0/(3.0*m);
    
    // Return to original space:
    for(unsigned int i=0; i < B; ++i) {
      Complex *cr0i=cr0[i];

      // Put dr2 into the second half of cr0, save for postmultadd:
      cr2[i]=cr0[i+B];
      Complex *cr2i=cr2[i];
      Complex *cr2Bi=cr2B[i];
      
      double *dr0i=dr0[i];
      double *dr1i=dr1[i];
      double *dr2i=dr2[i];


      // r=1:
      rco->fft(dr1i,cr2Bi); // first: transform r=1
      // r=0:
      Complex *cd=(Complex *)dr0i;
      cd[start]=S[i]; // r=0, k=c-1 (c) for m=even (odd)
      rco->fft(dr0i,cr0i); // third: transform r=0
      // r=2 (=-1):
      rco->fft(dr2i,cr2i); // fourth: transorm r=2

      cr0i[0]=(cr0i[0].re+cr2Bi[0].re+cr2i[0].re)*ninv;
    }
    postmultadd(cr2,cr0,cr2B);

  } else { // FFTs are all in-place.
    // Return to original space:
    const double ninv=1.0/(3.0*m);
    for(unsigned int i=0; i < B; ++i) {
      Complex *cr0i=cr0[i];
      Complex *cr1i=cr1[i];
      Complex *cr2i=cr2[i];
      rc->fft(cr2i);
      rc->fft(cr1i);
      double R=cr1i[0].re;
      cr0i[start]=S[i]; // r=0, k=c-1 (c) for m=even (odd)
      if(even) {
        Complex tmp=cr1c[i];
        cr1c[i]=cr1i[1]; // r=1, k=1
        cr1i[1]=tmp;    //  r=0, k=c
      }
      rc->fft(cr0i);
      cr0i[0]=(cr0i[0].re+R+cr2i[0].re)*ninv;
    }
    postmultadd0(cr2,cr0,cr1,cr1c);
  }
}


void fftpad::expand(Complex *f, Complex *u)
{
#ifdef __SSE2__
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
#else
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
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
    }
    );
#endif
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
#ifdef __SSE2__
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
#else
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
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
    }
    );
#endif
}

void fftpad::forwards(Complex *f, Complex *u)
{
  Forwards->fft(f);
  Forwards->fft(u);
  reduce(f,u);
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
        Vec F0=LOAD(fk+i)*Ninv;
        Vec F1=ZMULT(X,-Y,LOAD(fm1k+i));
        Vec F2=ZMULT(X,Y,LOAD(uk+i));
        Vec S=F1+F2;
        STORE(fk+i-stride,F0+Mhalf*S+HSqrt3*ZMULTI(F1-F2));
        STORE(fm1k+i,F0+S);
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

void fft0padwide::backwards(Complex *f, Complex *u)
{
  unsigned int mstride=m*stride;
  Complex *fmstride=f+mstride;
  for(unsigned int i=0; i < M; ++i)
    u[i]=f[i]=fmstride[i];
    
#ifdef __SSE2__      
  Vec Mhalf=LOAD(-0.5);
  Vec Mhsqrt3=LOAD(-hsqrt3);
#endif    
  unsigned int inc=s;
  PARALLEL(
    for(unsigned int K=0; K < m; K += inc) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
#ifdef __SSE2__
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
    }
    );
  
  Backwards->fft(f);
  Backwards->fft(fmstride);
  Backwards->fft(u);
}

void fft0padwide::forwards(Complex *f, Complex *u)
{
  unsigned int mstride=m*stride;
  Complex *fmstride=f+mstride;
  
  Forwards->fft(f);
  Forwards->fft(fmstride);
  Forwards->fft(u);

  double ninv=1.0/(3.0*m);
  for(unsigned int i=0; i < M; ++i)
    fmstride[i]=(f[i]+fmstride[i]+u[i])*ninv;
#ifdef __SSE2__      
  Vec Ninv=LOAD(ninv);
  Vec Mhalf=LOAD(-0.5);
  Vec HSqrt3=LOAD(hsqrt3);
#endif    
  
  unsigned int inc=s;
  PARALLEL(
    for(unsigned int K=0; K < m; K += inc) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
#ifdef __SSE2__      
      Vec Zeta=Ninv*LOAD(ZetaH+K/s);
      Vec X=UNPACKL(Zeta,Zeta);
      Vec Y=UNPACKH(CONJ(Zeta),Zeta);
#else
      Complex H=ninv*ZetaH[K/s];
#endif    
      for(unsigned int k=max(1,K); k < stop; ++k) {
#ifdef __SSE2__      
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
    }
    );
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
#ifdef __SSE2__
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
#else
    PARALLEL(
      for(unsigned int K=0; K < m; K += s) {
        Complex *ZetaL0=ZetaL-K;
        unsigned int stop=min(K+s,m);
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
      }  
      );
#endif
      
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
#else
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
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
    }
    );
#endif
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
#ifdef __SSE2__
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
#else
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
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
    }  
    );
#endif

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
#else
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
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
    }
    );
#endif
}

// a[k]=a[k]^3
void ImplicitHFFFConvolution::mult(double *a)
{
  unsigned int twom=2*m;
#ifdef __SSE2__
  PARALLEL(
    for(unsigned int k=0; k < twom; k += 2) {
      double *p=a+k;
      Vec ak=LOAD(p);
      STORE(p,ak*ak*ak);
    }        
    );
#else
  PARALLEL(
    for(unsigned int k=0; k < twom; ++k) {
      double ak=a[k];
      a[k]=ak*ak*ak;
    }
    );
#endif
}
  
void ImplicitHFFFConvolution::convolve(Complex *f, Complex *u)
{
#ifdef __SSE2__
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
#else
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
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
    }  
    );
#endif  
    
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
#else
  PARALLEL(
    for(unsigned int K=0; K < m; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,m);
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
    }
    );
#endif
}

void fft0bipad::backwards(Complex *f, Complex *u)
{
  for(unsigned int i=0; i < M; ++i)
    f[i]=0.0;
  for(unsigned int i=0; i < M; ++i)
    u[i]=0.0;
    
  unsigned int twom=2*m;
#ifdef __SSE2__
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
#else
  PARALLEL(
    for(unsigned int K=0; K < twom; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,twom);
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
    }
    );
#endif
    
  Backwards->fft(f);
  Backwards->fft(u);
}

void fft0bipad::forwards(Complex *f, Complex *u)
{
  Forwards->fft(f);
  Forwards->fft(u);

  double ninv=0.25/m;
  unsigned int twom=2*m;
#ifdef __SSE2__
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
#else
  PARALLEL(
    for(unsigned int K=0; K < twom; K += s) {
      Complex *ZetaL0=ZetaL-K;
      unsigned int stop=min(K+s,twom);
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
    }
    );
#endif
}

// This multiplication routine is for binary convolutions and takes two inputs
// of size m.
// F[0][j] *= conj(F[0][j]);
void mult_autocorrelation(Complex **F, unsigned int m, unsigned int threads)
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

// This multiplication routine is for binary convolutions and takes two inputs
// of size m.
// F[0][j] *= F[1][j];
void multbinary(Complex **F, unsigned int m, unsigned int threads)
{
  Complex* F0=F[0];
  Complex* F1=F[1];
  
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
void mult_autoconvolution(Complex **F, unsigned int m, unsigned int threads)
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

// This multiplication routine is for binary Hermitian convolutions and takes
// two inputs.
// F[0][j] *= F[1][j];
void multbinary(double **F, unsigned int m, unsigned int threads)
{
  double* F0=F[0];
  double* F1=F[1];
  
  unsigned int m1=m-1;
#ifdef __SSE2__
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
void multbinary2(Complex **F, unsigned int m, unsigned int threads) {
  Complex* F0=F[0];
  Complex* F1=F[1];
  Complex* F2=F[2];
  Complex* F3=F[3];
  
 #ifdef __SSE2__
   PARALLEL(for(unsigned int j=0; j < m; ++j) {
       Complex *F0j=F0+j;
       STORE(F0j,ZMULT(LOAD(F0j),LOAD(F2+j))
             +ZMULT(LOAD(F1+j),LOAD(F3+j)));
     });
 #else
  // TODO: Compare to optimized version
  PARALLEL(for(unsigned int j=0; j < m; ++j) {
      F0[j]=F0[j]*F2[j] + F1[j]*F3[j];
    });
  #endif
}

// F[0][j]=F[0][j]*F[2][j]+F[1][j]*F[3][j]
void multbinary2(double **F, unsigned int m, unsigned int threads) {
  double* F0=F[0];
  double* F1=F[1];
  double* F2=F[2];
  double* F3=F[3];
  
  unsigned int m1=m-1;
 #ifdef __SSE2__
   PARALLEL(
     for(unsigned int j=0; j < m1; j += 2) {
       double *F0j=F0+j;
       STORE(F0j,LOAD(F0j)*LOAD(F2+j)+LOAD(F1+j)*LOAD(F3+j));
     }
     );
   if(m % 2)
     F0[m1]=F0[m1]*F2[m1]+F1[m1]*F3[m1];
 #else
   // TODO: Compare to optimized version
  PARALLEL(
    for(unsigned int j=0; j < m; ++j) {
      F0[j]=F0[j]*F2[j] + F1[j]*F3[j];
    }
    );
  #endif
}

// F[0][j]=F[0][j]*F[1][j]+F[2][j]*F[3][j]+F[4][j]*F[5][j];
void multbinary3(Complex **F, unsigned int m, unsigned int threads) {
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
      STORE(F0j,ZMULT(LOAD(F0j),LOAD(F1+j))
            +ZMULT(LOAD(F2+j),LOAD(F3+j))
	    +ZMULT(LOAD(F4+j),LOAD(F5+j))
        );
    }
    );
#else
  PARALLEL(
    for(unsigned int j=0; j < m; ++j)
      F0[j]=F0[j]*F1[j]+F2[j]*F3[j]+F4[j]*F5[j];
    );

#endif
}

// F[0][j]=F[0][j]*F[1][j]+F[2][j]*F[3][j]+F[4][j]*F[5][j]+F[6][j]*F[7][j];
void multbinary4(Complex **F, unsigned int m, unsigned int threads) {
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
      STORE(F0j,ZMULT(LOAD(F0j),LOAD(F1+j))
            +ZMULT(LOAD(F2+j),LOAD(F3+j))
	    +ZMULT(LOAD(F4+j),LOAD(F5+j))
	    +ZMULT(LOAD(F6+j),LOAD(F7+j))
        );
    }
    );
#else
  PARALLEL(
    for(unsigned int j=0; j < m; ++j)
      F0[j]=F0[j]*F1[j]+F2[j]*F3[j]+F4[j]*F5[j]+F6[j]*F7[j];
    );

#endif
}

void multbinary8(Complex **F, unsigned int m, unsigned int threads) {
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
	    ZMULT(LOAD(F0j),LOAD(F1+j))
            +ZMULT(LOAD(F2+j),LOAD(F3+j))
	    +ZMULT(LOAD(F4+j),LOAD(F5+j))
	    +ZMULT(LOAD(F6+j),LOAD(F7+j))
	    +ZMULT(LOAD(F8+j),LOAD(F9+j))
	    +ZMULT(LOAD(F10+j),LOAD(F11+j))
	    +ZMULT(LOAD(F12+j),LOAD(F13+j))
	    +ZMULT(LOAD(F14+j),LOAD(F15+j))
        );
    }
    );
#else
  PARALLEL(
    for(unsigned int j=0; j < m; ++j)
      F0[j]=F0[j]*F1[j]+F2[j]*F3[j]+F4[j]*F5[j]+F6[j]*F7[j]
        +F8[j]*F9[j]+F10[j]*F11[j]+F12[j]*F13[j]+F14[j]*F15[j];
    );
#endif
}

} // namespace fftwpp
