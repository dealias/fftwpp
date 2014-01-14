#include "mpi/mpifftw++.h"

namespace fftwpp {

MPI_Comm *active;

void LoadWisdom(const MPI_Comm& active)
{
  int rank;
  MPI_Comm_rank(active,&rank);
  if(rank == 0) {
    std::ifstream ifWisdom;
    ifWisdom.open(WisdomName);
    fftwpp_import_wisdom(GetWisdom,ifWisdom);
    ifWisdom.close();
  }
  fftw_mpi_broadcast_wisdom(active);
}

void SaveWisdom(const MPI_Comm& active)
{
  int rank;
  MPI_Comm_rank(active,&rank);
  fftw_mpi_gather_wisdom(active);
  if(rank == 0) {
    std::ofstream ofWisdom;
    ofWisdom.open(WisdomName);
    fftwpp_export_wisdom(PutWisdom,ofWisdom);
    ofWisdom.close();
  }
}

void MPILoadWisdom()
{
  LoadWisdom(*active);
}
  
void MPISaveWisdom()
{
  SaveWisdom(*active);
}

void cfft2MPI::Forwards(Complex *f,bool finaltranspose)
{
  xForwards->fft(f);
  T->transpose(f,false,true);
  yForwards->fft(f);
  if(finaltranspose) T->transpose(f,true,false);
}

void cfft2MPI::Backwards(Complex *f,bool finaltranspose)
{
  if(finaltranspose) T->transpose(f,false,true);
  yBackwards->fft(f);
  T->transpose(f,true,false);
  xBackwards->fft(f);
}

void cfft2MPI::Normalize(Complex *f)
{
  double overN=1.0/(d.nx*d.ny);
  for(unsigned int i=0; i < d.n; ++i) f[i] *= overN;
}

void cfft2MPI::BackwardsNormalized(Complex *f,bool finaltranspose)
{
  Backwards(f,finaltranspose);
  Normalize(f);
}

void cfft3MPI::Forwards(Complex *f,bool finaltranspose)
{
  xForwards->fft(f);
  Txy->transpose(f,false,true);
  for(unsigned int i=0; i < d.x; i++) yForwards->fft(f+i*d.ny*d.z);
  Tyz->transpose(f,false,true);
  zForwards->fft(f);
  // FIXME: Tzx->transpose ?
}

void cfft3MPI::Backwards(Complex *f,bool finaltranspose)
{
  // FIXME: Tzx->transpose ?
  zBackwards->fft(f);
  Tyz->transpose(f,true,false);
  for(unsigned int i=0; i < d.x; i++) yBackwards->fft(f+i*d.ny*d.z);
  Txy->transpose(f,true,false);
  xBackwards->fft(f);
}

void cfft3MPI::Normalize(Complex *f) 
{
  double overN=1.0/(d.nx*d.ny*d.nz);
  for(unsigned int i=0; i < d.n; ++i) f[i] *= overN;
}

} // end namespace fftwpp
