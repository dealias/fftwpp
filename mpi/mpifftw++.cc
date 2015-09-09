#include "mpifftw++.h"

namespace fftwpp {

void MPILoadWisdom(const MPI_Comm& active)
{
  int rank;
  MPI_Comm_rank(active,&rank);
  if(rank == 0)
    fftw::LoadWisdom();
  fftw_mpi_broadcast_wisdom(active);
}

void MPISaveWisdom(const MPI_Comm& active)
{
  int rank;
  MPI_Comm_rank(active,&rank);
  fftw_mpi_gather_wisdom(active);
  if(rank == 0)
    fftw::SaveWisdom();
}

void fft2dMPI::Forwards(Complex *f)
{
  yForwards->fft(f);
  T->transpose(f,true,false);
  xForwards->fft(f);
}

void fft2dMPI::Backwards(Complex *f)
{
  xBackwards->fft(f);
  T->transpose(f,false,true);
  yBackwards->fft(f);
}

void fft2dMPI::Normalize(Complex *f)
{
  // TODO: multithread
  unsigned int N=d.X*d.Y;
  unsigned int n=d.x*d.Y;
  double denom=1.0/N;
  for(unsigned int i=0; i < n; ++i) 
    f[i] *= denom;
}

void fft2dMPI::BackwardsNormalized(Complex *f)
{
  Backwards(f);
  Normalize(f);
}

void fft3dMPI::Forwards(Complex *f)
{
  unsigned int stride=d.z*d.Y;
  if(d.y < d.Y) {
    zForwards->fft(f);

    Tyz->transpose(f,true,false);

    for(unsigned int i=0; i < d.x; ++i) 
      yForwards->fft(f+i*stride);
  } else {
    for(unsigned int i=0; i < d.x; ++i) 
      yzForwards->fft(f+i*stride);
  }

  Txy->transpose(f,true,false);

  xForwards->fft(f);
}

void fft3dMPI::Backwards(Complex *f)
{
  xBackwards->fft(f);

  Txy->transpose(f,false,true);

  unsigned int stride=d.z*d.Y;
  if(d.y < d.Y) {
  for(unsigned int i=0; i < d.x; ++i)
    yBackwards->fft(f+i*stride);

  Tyz->transpose(f,false,true);

  zBackwards->fft(f);
  } else {
    for(unsigned int i=0; i < d.x; ++i) 
      yzBackwards->fft(f+i*stride);
  }
}

void fft3dMPI::Normalize(Complex *f)
{
  unsigned int N=d.X*d.Y*d.Z;
  unsigned int n=d.x*d.y*d.Z;
  double denom=1.0/N;
  for(unsigned int i=0; i < n; ++i) 
    f[i] *= denom;
}

#if 0
void rcfft2dMPI::Forwards(double *f, Complex *g)
{
  yForwards->fft(f,g);
  T->transpose(g,false,true);
  xForwards->fft(g);
}

void rcfft2dMPI::Forwards0(double *f, Complex *g)
{
  Shift(f);
  Forwards(f,g);
}

void rcfft2dMPI::Backwards(Complex *g, double *f)
{
  xBackwards->fft(g);
  T->transpose(g,true,false);
  yBackwards->fft(g,f);
}

void rcfft2dMPI::Backwards0(Complex *g, double *f)
{
  Backwards(g,f);
  Shift(f);
}

void rcfft2dMPI::BackwardsNormalized(Complex *g, double *f)
{
  Backwards(g,f);
  Normalize(f);
}

void rcfft2dMPI::Backwards0Normalized(Complex *g, double *f) 
{
  Backwards0(g,f);
  Normalize(f);
}

void rcfft2dMPI::Normalize(double *f)
{
  double norm=1.0/(dr.X*dr.Y);
  for(unsigned int i=0; i < dr.x; ++i)  {
    double *fi=&f[i*rdist];
    for(unsigned int j=0; j < dr.Y; ++j)
      fi[j] *= norm;
  }
}
void rcfft2dMPI::Shift(double *f)
{
  // Shift Fourier origin:
  for(unsigned int i=0; i < dr.x; i += 2)  {
    double *fi=&f[i*rdist];
    for(unsigned int j=0; j < dr.Y; ++j)
      fi[j] *= -1;
  }
}
#endif

} // End of namespace fftwpp
