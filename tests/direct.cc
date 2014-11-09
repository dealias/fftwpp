#include "convolution.h"
#include "direct.h"

namespace fftwpp {

void DirectConvolution::convolve(Complex *h, Complex *f, Complex *g)
{
#if (!defined FFTWPP_SINGLE_THREAD) && defined _OPENMP
#pragma omp parallel for
#endif
  for(unsigned int i=0; i < m; ++i) {
    Complex sum=0.0;
    for(unsigned int j=0; j <= i; ++j) sum += f[j]*g[i-j];
    h[i]=sum;
  }
}

void DirectConvolution::autoconvolve(Complex *h, Complex *f)
{
#if (!defined FFTWPP_SINGLE_THREAD) && defined _OPENMP
#pragma omp parallel for
#endif
  for(unsigned int i=0; i < m; ++i) {
    Complex sum=0.0;
    for(unsigned int j=0; j <= i; ++j) sum += f[j]*f[i-j];
    h[i]=sum;
  }
}

void DirectHConvolution::convolve(Complex *h, Complex *f, Complex *g)
{
#if (!defined FFTWPP_SINGLE_THREAD) && defined _OPENMP
#pragma omp parallel for
#endif
  for(unsigned int i=0; i < m; ++i) {
    Complex sum=0.0;
    for(unsigned int j=0; j <= i; ++j) sum += f[j]*g[i-j];
    for(unsigned int j=i+1; j < m; ++j) sum += f[j]*conj(g[j-i]);
    for(unsigned int j=1; j < m-i; ++j) sum += conj(f[j])*g[i+j];
    h[i]=sum;
  }
}       

void DirectConvolution2::convolve(Complex *h, Complex *f, Complex *g)
{
#if (!defined FFTWPP_SINGLE_THREAD) && defined _OPENMP
#pragma omp parallel for
#endif
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

void DirectHConvolution2::convolve(Complex *h, Complex *f, Complex *g,
                                   bool symmetrize)
{
  unsigned int xorigin=mx-1;
    
  if(symmetrize) {
    HermitianSymmetrizeX(mx,my,mx-1,f);
    HermitianSymmetrizeX(mx,my,mx-1,g);
  }
    
  int xstart=-(int)xorigin;
  int ystart=1-(int) my;
  int xstop=mx;
  int ystop=my;
#if (!defined FFTWPP_SINGLE_THREAD) && defined _OPENMP
#pragma omp parallel for
#endif
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

void DirectConvolution3::convolve(Complex *h, Complex *f, Complex *g)
{
#if (!defined FFTWPP_SINGLE_THREAD) && defined _OPENMP
#pragma omp parallel for
#endif
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
    HermitianSymmetrizeXY(mx,my,mz,mx-1,my-1,f);
    HermitianSymmetrizeXY(mx,my,mz,mx-1,my-1,g);
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
    
  unsigned int xorigin=mx-1;
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
