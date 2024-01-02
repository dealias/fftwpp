#include <unistd.h>

#include "mpifftw++.h"

namespace fftwpp {

void fft2dMPI::iForward(Complex *in, Complex *out)
{
  out=Setout(in,out);
  yForward->fft(in,out);
  T->ilocalize0(out);
}

void fft2dMPI::iBackward(Complex *in, Complex *out)
{
  out=Setout(in,out);
  if(!strided) TXy->transpose(in);
  xBackward->fft(in,out);
  if(!strided) TyX->transpose(out);
  T->ilocalize1(out);
}

void fft3dMPI::iForward(Complex *in, Complex *out)
{
  out=Setout(in,out);
  if(Tyz) {
    zForward->fft(in,out);
    Tyz->ilocalize0(out);
  } else {
    unsigned int stride=d.Z*d.Y;
    unsigned int stop=d.x*stride;
    PARALLEL(
      for(unsigned int i=0; i < stop; i += stride) {
        yzForward->fft(in+i,out+i);
      }
      );
    Txy->ilocalize0(out);
  }
}

void fft3dMPI::ForwardWait0(Complex *out)
{
  if(Tyz) {
    Tyz->wait();
    unsigned int stride=d.z*d.Y;
    unsigned int stop=d.x*stride;
    PARALLEL(
      for(unsigned int i=0; i < stop; i += stride)
        yForward->fft(out+i);
      );
    Txy->ilocalize0(out);
  }
}

void fft3dMPI::iBackward(Complex *in, Complex *out)
{
  out=Setout(in,out);
  xBackward->fft(in,out);
  Txy->ilocalize1(out);
}

void fft3dMPI::BackwardWait0(Complex *out)
{
  Txy->wait();

  unsigned int stride=d.z*d.Y;
  unsigned int stop=d.x*stride;
  if(Tyz) {
    PARALLEL(
      for(unsigned int i=0; i < stop; i += stride)
        yBackward->fft(out+i);
      );
    Tyz->ilocalize1(out);
  } else {
    PARALLEL(
      for(unsigned int i=0; i < stop; i += stride)
        yzBackward->fft(out+i);
      );
  }
}

void rcfft2dMPI::Shift(double *f)
{
  if(dr.X % 2 == 0) {
    const unsigned int start=(dr.x0+1) % 2;
    const unsigned int stop=dr.x;
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif
    for(unsigned int i=start; i < stop; i += 2) {
      double *p=f+i*rdist;
      for(unsigned int j=0; j < dr.Y; ++j) {
        p[j]=-p[j];
      }
    }
  } else {
    std::cerr << "Shift is not implemented for odd X." << std::endl;
    exit(1);
  }
}

void rcfft2dMPI::iForward(double *in, Complex *out)
{
  out=Setout((Complex *) in,out);
  yForward->fft(in,out);
  T->ilocalize0(out);
}

void rcfft2dMPI::iBackward(Complex *in, double *out)
{
  out=(double *) Setout(in,(Complex *) out);
  xBackward->fft(in);
  T->ilocalize1(in);
}

void rcfft3dMPI::iForward(double *in, Complex *out)
{
  out=Setout((Complex *) in,out);
  zForward->fft(in,out);
  if(Tyz) Tyz->ilocalize0(out);
  else {
    const unsigned int stride=dc.z*dc.Y;
    const unsigned int stop=dc.x*stride;
    PARALLEL(
      for(unsigned int i=0; i < stop; i += stride)
        yForward->fft(out+i);
      )
      Txy->ilocalize0(out);
  }
}

void rcfft3dMPI::ForwardWait0(Complex *out)
{
  if(Tyz) {
    Tyz->wait();
    const unsigned int stride=dc.z*dc.Y;
    const unsigned int stop=dc.x*stride;
    PARALLEL(
      for(unsigned int i=0; i < stop; i += stride)
        yForward->fft(out+i);
      )
      Txy->ilocalize0(out);
  }
}

void rcfft3dMPI::iBackward(Complex *in, double *out)
{
  xBackward->fft(in);
  Txy->ilocalize1(in);
}

void rcfft3dMPI::BackwardWait0(Complex *in, double *out)
{
  Txy->wait();

  const unsigned int stride=dc.z*dc.Y;
  const unsigned int stop=dc.x*stride;
  PARALLEL(
    for(unsigned int i=0; i < stop; i += stride)
      yBackward->fft(in+i);
    );
  if(Tyz) Tyz->ilocalize1(in);
}

void rcfft3dMPI::Shift(double *f)
{
  if(dr.X % 2 == 0 && dr.Y % 2 == 0) {
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif
    for(unsigned int i=0; i < dr.x; ++i) {
      const unsigned int ystart=(i+dr.x0+dr.yz.x0+1) % 2;
      double *pi=f+i*dr.yz.x*rdist;
      for(unsigned int j=ystart; j < dr.yz.x; j += 2) {
        double *p=pi+j*rdist;
        for(unsigned int k=0; k < dr.Z; ++k) {
          p[k]=-p[k];
        }
      }
    }
  } else {
    std::cerr << "Shift is not implemented for odd X or odd Y."
              << std::endl;
    exit(1);
  }
}

}
