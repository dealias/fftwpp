#include "convolution.h"
#include "direct.h"

namespace fftwpp {

// Standard One Dimensional Direct Convolution
void DirectConvolution::convolve(Complex *h, Complex *f, Complex *g)
{
#if (!defined FFTWPP_SINGLE_THREAD) && defined _OPENMP
#pragma omp parallel for
#endif
  for(size_t i=0; i < m; ++i) {
    Complex sum=0.0;
    for(size_t j=0; j <= i; ++j) sum += f[j]*g[i-j];
    h[i]=sum;
  }
}

// Centered One Dimensional Direct Convolution
void DirectConvolution::Cconvolve(Complex *h, Complex *f, Complex *g)
{
#if (!defined FFTWPP_SINGLE_THREAD) && defined _OPENMP
#pragma omp parallel for
#endif
  for(size_t i=0; i < m/2; ++i) {
    Complex sum=0.0;
    for(size_t j=i+1; j < m; ++j) sum += f[j]*g[m+i-j];
    h[i+(m+1)/2]=sum;
  }
  for(size_t i=m/2; i < m; ++i) {
    Complex sum=0.0;
    for(size_t j=0; j <= i; ++j) sum += f[j]*g[i-j];
    h[i-m/2]=sum;
  }
}

void DirectConvolution::autoconvolve(Complex *h, Complex *f)
{
#if (!defined FFTWPP_SINGLE_THREAD) && defined _OPENMP
#pragma omp parallel for
#endif
  for(size_t i=0; i < m; ++i) {
    Complex sum=0.0;
    for(size_t j=0; j <= i; ++j) sum += f[j]*f[i-j];
    h[i]=sum;
  }
}

void DirectHConvolution::convolve(Complex *h, Complex *f, Complex *g)
{
#if (!defined FFTWPP_SINGLE_THREAD) && defined _OPENMP
#pragma omp parallel for
#endif
  for(size_t i=0; i < m; ++i) {
    Complex sum=0.0;
    for(size_t j=0; j <= i; ++j) sum += f[j]*g[i-j];
    for(size_t j=i+1; j < m; ++j) sum += f[j]*conj(g[j-i]);
    for(size_t j=1; j < m-i; ++j) sum += conj(f[j])*g[i+j];
    h[i]=sum;
  }
}

void DirectConvolution2::convolve(Complex *h, Complex *f, Complex *g)
{
#if (!defined FFTWPP_SINGLE_THREAD) && defined _OPENMP
#pragma omp parallel for
#endif
  for(size_t i=0; i < mx; ++i) {
    for(size_t j=0; j < my; ++j) {
      Complex sum=0.0;
      for(size_t k=0; k <= i; ++k)
        for(size_t p=0; p <= j; ++p)
          sum += f[Sx*k+p]*g[Sx*(i-k)+j-p];
      h[i*my+j]=sum;
    }
  }
}

void DirectHConvolution2::convolve(Complex *h, Complex *f, Complex *g,
                                   bool symmetrize)
{
  size_t xorigin=mx-xcompact;

  if(symmetrize) {
    HermitianSymmetrizeX(mx,my,xorigin,f);
    HermitianSymmetrizeX(mx,my,xorigin,g);
  }

  int xstart=-(int) xorigin;
  int ystart=1-(int) my;
  int xstop=mx;
  int ystop=my;
#if (!defined FFTWPP_SINGLE_THREAD) && defined _OPENMP
#pragma omp parallel for
#endif
  for(int kx=xstart; kx < xstop; ++kx) {
    for(int ky=0; ky < ystop; ++ky) {
      Complex sum=0.0;
      for(int px=xstart+!xcompact; px < xstop; ++px) {
        for(int py=ystart; py < ystop; ++py) {
          int qx=kx-px;
          if(qx >= xstart+!xcompact && qx < xstop) {
            int qy=ky-py;
            if(qy >= ystart && qy < ystop) {
              sum += ((py >= 0) ? f[(xorigin+px)*Sx+py] :
                      conj(f[(xorigin-px)*Sx-py])) *
                ((qy >= 0) ? g[(xorigin+qx)*Sx+qy] :
                 conj(g[(xorigin-qx)*Sx-qy]));
            }
          }
        }
        h[(xorigin+kx)*my+ky]=sum;
      }
    }
  }
}

void DirectConvolution3::convolve(Complex *h, Complex *f, Complex *g)
{
#if (!defined FFTWPP_SINGLE_THREAD) && defined _OPENMP
#pragma omp parallel for
#endif
  for(size_t i=0; i < mx; ++i) {
    for(size_t j=0; j < my; ++j) {
      for(size_t k=0; k < mz; ++k) {
        Complex sum=0.0;
        for(size_t r=0; r <= i; ++r)
          for(size_t p=0; p <= j; ++p)
            for(size_t q=0; q <= k; ++q)
              sum += f[r*Sx+p*Sy+q]*g[(i-r)*Sx+(j-p)*Sy+(k-q)];
        h[i*myz+j*mz+k]=sum;
      }
    }
  }
}

void DirectHConvolution3::convolve(Complex *h, Complex *f, Complex *g,
                                   bool symmetrize)
{
  size_t xorigin=mx-xcompact;
  size_t yorigin=my-ycompact;
  size_t ny=2*my-ycompact;

  if(symmetrize) {
    HermitianSymmetrizeXY(mx,my,mz,xorigin,yorigin,f);
    HermitianSymmetrizeXY(mx,my,mz,xorigin,yorigin,g);
  }

  int xstart=-(int) xorigin;
  int ystart=-(int) yorigin;
  int zstart=1-(int) mz;
  int xstop=mx;
  int ystop=my;
  int zstop=mz;
#if (!defined FFTWPP_SINGLE_THREAD) && defined _OPENMP
#pragma omp parallel for
#endif
  for(int kx=xstart; kx < xstop; ++kx) {
    for(int ky=ystart; ky < ystop; ++ky) {
      for(int kz=0; kz < zstop; ++kz) {
        Complex sum=0.0;
        for(int px=xstart+!xcompact; px < xstop; ++px) {
          for(int py=ystart+!ycompact; py < ystop; ++py) {
            for(int pz=zstart; pz < zstop; ++pz) {
              int qx=kx-px;
              if(qx >= xstart+!xcompact && qx < xstop) {
                int qy=ky-py;
                if(qy >= ystart+!ycompact && qy < ystop) {
                  int qz=kz-pz;
                  if(qz >= zstart && qz < zstop) {
                    sum += ((pz >= 0) ?
                            f[Sx*(xorigin+px)+Sy*(yorigin+py)+pz] :
                            conj(f[Sx*(xorigin-px)+Sy*(yorigin-py)-pz])) *
                      ((qz >= 0) ? g[Sx*(xorigin+qx)+Sy*(yorigin+qy)+qz] :
                       conj(g[Sx*(xorigin-qx)+Sy*(yorigin-qy)-qz]));
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

void DirectHTConvolution::convolve(Complex *h, Complex *e, Complex *f,
                                   Complex *g)
{
  int stop=m;
  int start=1-m;
#if (!defined FFTWPP_SINGLE_THREAD) && defined _OPENMP
#pragma omp parallel for
#endif
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

void DirectHTConvolution2::convolve(Complex *h, Complex *e, Complex *f,
                                    Complex *g, bool symmetrize)
{
  if(symmetrize) {
    HermitianSymmetrizeX(mx,my,mx-1,e);
    HermitianSymmetrizeX(mx,my,mx-1,f);
    HermitianSymmetrizeX(mx,my,mx-1,g);
  }

  size_t xorigin=mx-1;
  int xstart=-(int) xorigin;
  int xstop=mx;
  int ystop=my;
  int ystart=1-(int) my;
#if (!defined FFTWPP_SINGLE_THREAD) && defined _OPENMP
#pragma omp parallel for
#endif
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
