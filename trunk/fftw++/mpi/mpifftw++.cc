#include "mpi/mpifftw++.h"

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

void fft2dMPI::Forwards(Complex *f,bool finaltranspose)
{
  xForwards->fft(f);
  if(tranfftwpp)
    T->transpose(f,false,true);
  else
    fftw_mpi_execute_r2r(intranspose,(double *)f,(double *)f);
  yForwards->fft(f);
  if(finaltranspose) {
    if(tranfftwpp)
      T->transpose(f,true,false);
    else
      fftw_mpi_execute_r2r(outtranspose,(double *)f,(double *)f);
  }
}

void fft2dMPI::Backwards(Complex *f,bool finaltranspose)
{
  if(finaltranspose) {
    if(tranfftwpp)
      T->transpose(f,false,true);
    else
      fftw_mpi_execute_r2r(intranspose,(double *)f,(double *)f);
  }
  yBackwards->fft(f);
  if(tranfftwpp)
    T->transpose(f,true,false);
  else
    fftw_mpi_execute_r2r(outtranspose,(double *)f,(double *)f);
  xBackwards->fft(f);
}

void fft2dMPI::Normalize(Complex *f)
{
  // TODO: multithread
  double overN=1.0/(d.nx*d.ny);
  for(unsigned int i=0; i < d.n; ++i) f[i] *= overN;
}

void fft2dMPI::BackwardsNormalized(Complex *f,bool finaltranspose)
{
  Backwards(f,finaltranspose);
  Normalize(f);
}

void fft3dMPI::Forwards(Complex *f,bool finaltranspose)
{
  zForwards->fft(f);

  Txy->transpose(f,true,false);

  unsigned int metastride=d.z*d.ny;
  for(unsigned int i=0; i < d.x; ++i) yForwards->fft(f+i*metastride);

  Tyz->transpose(f,true,false);

  xForwards->fft(f);

  // if(finaltranspose) Txz->transpose(f,false,true);
}

void fft3dMPI::Backwards(Complex *f,bool finaltranspose)
{
  // if(finaltranspose) T??->transpose(f,false,true);

  xBackwards->fft(f);

  Tyz->transpose(f,true,false);

  unsigned int metastride=d.z*d.ny;
  for(unsigned int i=0; i < d.x; ++i)  yBackwards->fft(f+i*metastride);

  Txy->transpose(f,true,false);

  zBackwards->fft(f);
}

void fft3dMPI::Normalize(Complex *f)
{
  double overN=1.0/(d.nx*d.ny*d.nz);
  for(unsigned int i=0; i < d.n; ++i) 
    f[i] *= overN;
}

void rcfft2dMPI::Forwards(double *f, Complex *g, bool finaltranspose)
{
  yForwards->fft(f,g);
  if(tranfftwpp) {
    T->transpose(g,false,true);
  } else {
    fftw_mpi_execute_r2r(intranspose,(double *)g,(double *)g);
  }
  xForwards->fft(g);
  // if(finaltranspose) T->transpose(f,true,false); // FIXME: enable
}

void rcfft2dMPI::Forwards0(double *f, Complex *g, bool finaltranspose)
{
  Shift(f);
  Forwards(f,g,finaltranspose);
}

void rcfft2dMPI::Backwards(Complex *g, double *f, bool finaltranspose)
{
  // if(finaltranspose) T->transpose(f,false,true); // FIXME: enable
  xBackwards->fft(g);
  if(tranfftwpp) {
    T->transpose(g,true,false);
  } else {
    fftw_mpi_execute_r2r(outtranspose,(double *)g,(double *)g);
  }
  yBackwards->fft(g,f);
}

void rcfft2dMPI::Backwards0(Complex *g, double *f, bool finaltranspose)
{
  Backwards(g,f,finaltranspose);
  Shift(f);
}

void rcfft2dMPI::BackwardsNormalized(Complex *g, double *f, 
				     bool finaltranspose)
{
  Backwards(g,f,finaltranspose);
  Normalize(f);
}

void rcfft2dMPI::Backwards0Normalized(Complex *g, double *f, 
				      bool finaltranspose)
{
  Backwards0(g,f,finaltranspose);
  Normalize(f);
}

void rcfft2dMPI::Normalize(double *f)
{
  double norm=1.0/(dr.nx*dr.ny);
  for(unsigned int i=0; i < dr.x; i++)  {
    double *fi=&f[i*rdist];
    for(unsigned int j=0; j < dr.ny; j++)
      fi[j] *= norm;
  }
}
void rcfft2dMPI::Shift(double *f)
{
  // Shift Fourier origin:
  for(unsigned int i=0; i < dr.x; i += 2)  {
    double *fi=&f[i*rdist];
    for(unsigned int j=0; j < dr.ny; j += 1)
      fi[j] *= -1;
  }
}

} // End of namespace fftwpp
