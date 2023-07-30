#include "Complex.h"
#include "convolution.h"
#include "explicit.h"

namespace fftwpp {

// This multiplication routine is for binary convolutions and takes two inputs
// of size m.
// F[0][j] *= F[1][j];
void multbinary(Complex **F, size_t m, size_t n, size_t threads)
{
  Complex* F0=F[0];
  Complex* F1=F[1];

  double ninv=1.0/n;

  PARALLELIF(
    m > threshold,
    for(size_t j=0; j < m; ++j)
      F0[j] *= ninv*F1[j];
    );
}

// This multiplication routine is for binary Hermitian convolutions and takes
// two inputs of size n.
// F[0][j] *= F[1][j];
void multbinary(double **F, size_t m, size_t n, size_t threads)
{
  double* F0=F[0];
  double* F1=F[1];

  double ninv=1.0/n;

  PARALLELIF(
    m > 2*threshold,
    for(size_t j=0; j < m; ++j)
      F0[j] *= ninv*F1[j];
    );
}

// This multiplication routine is for binary convolutions and takes two inputs
// of size n.
// F[0][j] *= F[1][j];
void multbinaryUnNormalized(Complex **F, size_t m, size_t, size_t threads)
{
  Complex* F0=F[0];
  Complex* F1=F[1];

  PARALLELIF(
    m > threshold,
    for(size_t j=0; j < m; ++j)
      F0[j] *= F1[j];
    );
}

// This multiplication routine is for binary Hermitian convolutions and takes
// two inputs of size n.
// F[0][j] *= F[1][j];
void multbinaryUnNormalized(double **F, size_t m, size_t, size_t threads)
{
  double* F0=F[0];
  double* F1=F[1];

  PARALLELIF(
    m > 2*threshold,
    for(size_t j=0; j < m; ++j)
      F0[j] *= F1[j];
    );
}

void ExplicitConvolution::backwards(Complex *F, Complex *G)
{
  Backwards->fft(F,G);
}

void ExplicitConvolution::forwards(Complex *G, Complex *F)
{
  Forwards->fft(G,F);
}

void ExplicitConvolution::convolve(Complex **F, Multiplier *mult, Complex **G)
{
  if(G == NULL) G=F;

  const size_t A=2;
  const size_t B=1;

  for(size_t a=0; a < A; ++a) {
    pad(F[a]);
    backwards(F[a],G[a]);
  }

  (*mult)(G,n,n,threads);

  for(size_t b=0; b < B; ++b)
    forwards(G[b],F[b]);
}

void ExplicitConvolution::convolve(Complex *f, Complex *g)
{
  pad(f);
  backwards(f,f);

  pad(g);
  backwards(g,g);

  double ninv=1.0/n;

  Vec Ninv=LOAD2(ninv);
  PARALLELIF(
    n > threshold,
    for(size_t k=0; k < n; ++k)
      STORE(f+k,Ninv*ZMULT(LOAD(f+k),LOAD(g+k)));
    );
  forwards(f,f);
}

void ExplicitHConvolution::backwards(Complex *F, double *G)
{
  cr->fft(F,G);
}

void ExplicitHConvolution::forwards(double *G, Complex *F)
{
  rc->fft(G,F);
}

void ExplicitHConvolution::convolve(Complex **F, Realmultiplier *mult, double **G)
{
  if(G == NULL) G=(double **) F;

  const size_t A=2;
  const size_t B=1;

  for(size_t a=0; a < A; ++a) {
    pad(F[a]);
    backwards(F[a],G[a]);
  }

  (*mult)(G,n,n,threads);

  for(size_t b=0; b < B; ++b)
    forwards(G[b],F[b]);
}

void ExplicitRConvolution::forwards(double *f, Complex *g)
{
  rc->fft(f,g);
}

void ExplicitRConvolution::backwards(Complex *g, double *f)
{
  cr->fft(g,f);
}

void ExplicitRConvolution::convolve(Complex **F, Multiplier *mult, Complex **G)
{
  if(G == NULL) G=F;

  const size_t A=2;
  const size_t B=1;

  for(size_t a=0; a < A; ++a) {
    pad((double *) (F[a]));
    forwards((double *) (F[a]),G[a]);
  }

  (*mult)(G,np,n,threads);

  for(size_t b=0; b < B; ++b)
    backwards(G[b],(double *) (F[b]));
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

  size_t n=nx*ny;
  Vec Ninv=LOAD2(1.0/n);
  PARALLELIF(
    n > threshold,
    for(size_t k=0; k < n; ++k)
      STORE(f+k,Ninv*ZMULT(LOAD(f+k),LOAD(g+k)));
    );
  forwards(f);
}

void oddShift(size_t nx, size_t ny, Complex *f, int sign,
              size_t s, Complex *ZetaH, Complex *ZetaL)
{
  size_t nyp=ny/2+1;
  int Sign=-1;
  sign=-sign;
  size_t stop=s;
  Complex *ZetaL0=ZetaL;
  for(size_t a=0, k=1; k < nx; ++a) {
    Complex H=ZetaH[a];
    for(; k < stop; ++k) {
      Complex zeta=Sign*H*ZetaL0[k];
      zeta.im *= sign;
      size_t j=nyp*k;
      size_t stop=j+nyp;
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
  size_t xorigin=nx/2;
  size_t nyp=ny/2+1;

  for(size_t s=0; s < M; ++s) {
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
  size_t nyp2=2*nyp;

  double *f=(double *) F[0];
  double *g=(double *) G[0];

  if(M == 1) {
    PARALLELIF(
      nx*nyp > threshold,
      for(size_t i=0; i < nx; ++i) {
        size_t nyp2i=nyp2*i;
        size_t stop=nyp2i+ny;
        for(size_t j=nyp2i; j < stop; ++j)
          f[j] *= g[j]*ninv;
      }
      );
  } else if(M == 2) {
    double *f1=(double *) F[1];
    double *g1=(double *) G[1];
    PARALLELIF(
      nx*nyp > threshold,
      for(size_t i=0; i < nx; ++i) {
        size_t nyp2i=nyp2*i;
        size_t stop=nyp2i+ny;
        for(size_t j=nyp2i; j < stop; ++j)
          f[j]=(f[j]*g[j]+f1[j]*g1[j])*ninv;
      }
      );
  } else {
    PARALLELIF(
      nx*(M-1) > threshold,
      for(size_t i=0; i < nx; ++i) {
        size_t nyp2i=nyp2*i;
        size_t stop=nyp2i+ny;
        for(size_t j=nyp2i; j < stop; ++j) {
          double sum=f[j]*g[j];
          for(size_t s=1; s < M; ++s)
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
  size_t nyz=ny*nz;
  if(prune) {
    for(size_t i=0; i < mx; ++i)
      yBackwards->fft(f+i*nyz);
    for(size_t j=0; j < ny; ++j)
      xBackwards->fft(f+j*nz);
    zBackwards->fft(f);
  } else
    Backwards->fft(f);
}

void ExplicitConvolution3::forwards(Complex *f)
{
  if(prune) {
    zForwards->fft(f);
    for(size_t j=0; j < ny; ++j)
      xForwards->fft(f+j*nz);
    size_t nyz=ny*nz;
    for(size_t i=0; i < mx; ++i)
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

  size_t n=nx*ny*nz;
  Vec Ninv=LOAD2(1.0/n);
  PARALLELIF(
    n > threshold,
    for(size_t k=0; k < n; ++k)
      STORE(f+k,Ninv*ZMULT(LOAD(f+k),LOAD(g+k)));
    );
  forwards(f);
}

void ExplicitHConvolution3::backwards(Complex *f)
{
  Backwards->fft(f);
}

void ExplicitHConvolution3::forwards(Complex *f)
{
  Forwards->fft0(f);
}

void ExplicitHConvolution3::convolve(Complex **F, Complex **G, bool symmetrize)
{
  size_t xorigin=nx/2;
  size_t yorigin=ny/2;
  size_t nzp=nz/2+1;

  for(size_t s=0; s < M; ++s) {
    Complex *f=F[s];

    if(symmetrize) HermitianSymmetrizeXY(mx,my,mz,xorigin,yorigin,f,ny*nzp,
                                         nzp);
    pad(f);
    backwards(f);

    Complex *g=G[s];
    if(symmetrize) HermitianSymmetrizeXY(mx,my,mz,xorigin,yorigin,g,ny*nzp,
                                         nzp);
    pad(g);
    backwards(g);
  }

  double ninv=1.0/(nx*ny*nz);
  size_t nzp2=2*nzp;

  double *f=(double *) F[0];
  double *g=(double *) G[0];

  if(M == 1) {
    PARALLELIF(
      nx > threshold,
      for(size_t i=0; i < nx; ++i) {
        size_t nzp2i=nzp2*ny*i;
        for(size_t j=0; j < ny; ++j) {
          size_t nzp2ij=nzp2i+nzp2*j;
          size_t stop=nzp2ij+nz;
          for(size_t j=nzp2ij; j < stop; ++j)
            f[j] *= g[j]*ninv;
        }
      }
      );
  }

  forwards(F[0]);
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
  PARALLELIF(
    n > threshold,
    for(size_t k=0; k < n; ++k)
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
  size_t xorigin=nx/2;
  size_t nyp=ny/2+1;

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
  size_t nyp2=2*nyp;

  PARALLELIF(
    nx*nyp > threshold,
    for(size_t i=0; i < nx; ++i) {
      size_t nyp2i=nyp2*i;
      size_t stop=nyp2i+ny;
      for(size_t j=nyp2i; j < stop; ++j)
        F[j] *= G[j]*H[j]*ninv;
    }
    );

  forwards(f,false);
}

}
