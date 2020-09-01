#include "Complex.h"
#include "convolution.h"
#include "explicit.h"

namespace fftwpp {

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

  Vec Ninv=LOAD(ninv);
  PARALLEL(
    for(unsigned int k=0; k < n; ++k)
      STORE(f+k,Ninv*ZMULT(LOAD(f+k),LOAD(g+k)));
    );
  forwards(f);
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
  PARALLEL(
    for(unsigned int k=0; k < n; ++k)
      F[k] *= G[k]*ninv;
    );

  forwards(f);
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
  Vec Ninv=LOAD(1.0/n);
  PARALLEL(
    for(unsigned int k=0; k < n; ++k)
      STORE(f+k,Ninv*ZMULT(LOAD(f+k),LOAD(g+k)));
    );
  forwards(f);
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
    stop=std::min(k+s,nx);
    ZetaL0=ZetaL-k;
  }
}

void ExplicitHConvolution2::backwards(Complex *f, bool shift)
{
  if(prune) {
    xBackwards->fft(f);
    if(nx % 2 == 0) {
      if(shift) fftw::Shift(f,nx,ny,threads);
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
      fftw::Shift(f,nx,ny,threads);
    } else oddShift(nx,ny,f,1,s,ZetaH,ZetaL);
    xForwards->fft(f);
  } else
    Forwards->fft0(f);
}

void ExplicitHConvolution2::convolve(Complex **F, Complex **G, bool symmetrize)
{
  unsigned int xorigin=nx/2;
  unsigned int nyp=ny/2+1;

  for(unsigned int s=0; s < M; ++s) {
    Complex *f=F[s];
    if(symmetrize) HermitianSymmetrizeX(mx,nyp,xorigin,f);
    pad(f);
    backwards(f,false);

    Complex *g=G[s];
    if(symmetrize) HermitianSymmetrizeX(mx,nyp,xorigin,g);
    pad(g);
    backwards(g,false);
  }

  double ninv=1.0/(nx*ny);
  unsigned int nyp2=2*nyp;

  double *f=(double *) F[0];
  double *g=(double *) G[0];

  if(M == 1) {
    PARALLEL(
      for(unsigned int i=0; i < nx; ++i) {
        unsigned int nyp2i=nyp2*i;
        unsigned int stop=nyp2i+ny;
        for(unsigned int j=nyp2i; j < stop; ++j)
          f[j] *= g[j]*ninv;
      }
      );
  } else if(M == 2) {
    double *f1=(double *) F[1];
    double *g1=(double *) G[1];
    PARALLEL(
      for(unsigned int i=0; i < nx; ++i) {
        unsigned int nyp2i=nyp2*i;
        unsigned int stop=nyp2i+ny;
        for(unsigned int j=nyp2i; j < stop; ++j)
          f[j]=(f[j]*g[j]+f1[j]*g1[j])*ninv;
      }
      );
  } else {
    PARALLEL(
      for(unsigned int i=0; i < nx; ++i) {
        unsigned int nyp2i=nyp2*i;
        unsigned int stop=nyp2i+ny;
        for(unsigned int j=nyp2i; j < stop; ++j) {
          double sum=f[j]*g[j];
          for(unsigned int s=1; s < M; ++s)
            sum += ((double *) F[s])[j]*((double *) G[s])[j];
          f[j]=sum*ninv;
        }
      }
      );
  }

  forwards(F[0]);
}

void ExplicitConvolution3::backwards(Complex *f)
{
  unsigned int nyz=ny*nz;
  if(prune) {
    for(unsigned int i=0; i < mx; ++i)
      yBackwards->fft(f+i*nyz);
    for(unsigned int j=0; j < ny; ++j)
      xBackwards->fft(f+j*nz);
    zBackwards->fft(f);
  } else
    Backwards->fft(f);
}

void ExplicitConvolution3::forwards(Complex *f)
{
  if(prune) {
    zForwards->fft(f);
    for(unsigned int j=0; j < ny; ++j)
      xForwards->fft(f+j*nz);
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
  Vec Ninv=LOAD(1.0/n);
  PARALLEL(
    for(unsigned int k=0; k < n; ++k)
      STORE(f+k,Ninv*ZMULT(LOAD(f+k),LOAD(g+k)));
    );
  forwards(f);
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
  PARALLEL(
    for(unsigned int k=0; k < n; ++k)
      F[k] *= G[k]*H[k]*ninv;
    );

  forwards(f);
}

void ExplicitHTConvolution2::backwards(Complex *f, bool shift)
{
  if(prune) {
    xBackwards->fft(f);
    if(nx % 2 == 0) {
      if(shift) fftw::Shift(f,nx,ny,threads);
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
      if(shift) fftw::Shift(f,nx,ny,threads);
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

  PARALLEL(
    for(unsigned int i=0; i < nx; ++i) {
      unsigned int nyp2i=nyp2*i;
      unsigned int stop=nyp2i+ny;
      for(unsigned int j=nyp2i; j < stop; ++j)
        F[j] *= G[j]*H[j]*ninv;
    }
    );

  forwards(f,false);
}

}
