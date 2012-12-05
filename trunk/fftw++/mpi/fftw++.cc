#include <mpi.h>
#include <fftw3-mpi.h>
#include "fftw++.h"

namespace fftwpp {

bool fftw::Wise=false;
bool fftw::autothreads=false; // Must be false when using MPI.
const double fftw::twopi=2.0*acos(-1.0);
MPI_Comm *active;

// User settings:
unsigned int fftw::effort=FFTW_MEASURE;
const char *WisdomName="wisdom3.txt";
unsigned int fftw::maxthreads=1;
double fftw::testseconds=0.1; // Time limit for threading efficiency tests

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

void fftw::LoadWisdom()
{
  fftwpp::LoadWisdom(*active);
  Wise=true;
}

void fftw::SaveWisdom()
{
  fftwpp::SaveWisdom(*active);
}
  
}
