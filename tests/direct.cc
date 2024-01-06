#include "direct.h"

namespace fftwpp {

void directconvh::convolve(Complex *h, Complex *f, Complex *g)
{
  for(size_t i=0; i < m; ++i) {
    Complex sum=0.0;
    for(size_t j=0; j <= i; ++j) sum += f[j]*g[i-j];
    for(size_t j=i+1; j < m; ++j) sum += f[j]*conj(g[j-i]);
    for(size_t j=1; j < m-i; ++j) sum += conj(f[j])*g[i+j];
    h[i]=sum;
  }
}

void directconvh2::convolve(Complex *h, Complex *f, Complex *g,
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

void directconvh3::convolve(Complex *h, Complex *f, Complex *g,
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

}
