#include "mpifftw++.h"
#include "unistd.h"

namespace fftwpp {

MPI_Comm Active=MPI_COMM_NULL;

fftw_plan MPIplanner(fftw *F, Complex *in, Complex *out) 
{
 if(Active == MPI_COMM_NULL)
    return Planner(F,in,out);
  fftw_plan plan;
  int rank;
  MPI_Comm_rank(Active,&rank);
  
  if(rank == 0) {
    int size;
    MPI_Comm_size(Active,&size);
    
    static bool Wise=false;
    bool learned=false;
    if(!Wise)
      LoadWisdom();
    fftw::effort |= FFTW_WISDOM_ONLY;
    plan=F->Plan(in,out);
    fftw::effort &= !FFTW_WISDOM_ONLY;
    int length=0;
    char *experience=NULL;
    char *inspiration=NULL;
    if(!plan || !Wise) {
      if(Wise) {
        experience=fftw_export_wisdom_to_string();
        fftw_forget_wisdom();
      }
      if(!plan) {
        plan=F->Plan(in,out);
        if(plan) learned=true;
      }
      if(plan)  {
      inspiration=fftw_export_wisdom_to_string();
      length=strlen(inspiration);
      }
    }
    for(int i=1; i < size; ++i)
      MPI_Send(&length,1,MPI_INT,i,0,Active);
    if(length > 0) {
      MPI_Bcast(inspiration,length,MPI_CHAR,0,Active);
      if(Wise) {
        fftw_import_wisdom_from_string(experience);
        fftw_free(experience);
      } else Wise=true;
      fftw_free(inspiration);
    }
    int rlength[size];
    MPI_Gather(&length,1,MPI_INT,rlength,1,MPI_INT,0,Active);
    for(int i=1; i < size; ++i) {
      int length=rlength[i];
      if(length > 0) {
        learned=true;
        char inspiration[length+1];
        MPI_Recv(&inspiration,length,MPI_CHAR,i,0,Active,MPI_STATUS_IGNORE);
        inspiration[length]=0;
        fftw_import_wisdom_from_string(inspiration);
      }
    }
    if(learned) SaveWisdom();
  } else {
    int flag=false;
    MPI_Status status;
    while(true) {
      MPI_Iprobe(0,0,Active,&flag,&status);
      if(flag) break;
      usleep(10000);
    }
    int length;
    MPI_Recv(&length,1,MPI_INT,0,0,Active,MPI_STATUS_IGNORE);
    if(length > 0) {
      char inspiration[length+1];
      MPI_Bcast(inspiration,length,MPI_CHAR,0,Active);
      inspiration[length]=0;
      fftw_import_wisdom_from_string(inspiration);
    }
    fftw::effort |= FFTW_WISDOM_ONLY;
    plan=F->Plan(in,out);
    fftw::effort &= !FFTW_WISDOM_ONLY;
    char *experience=NULL;
    char *inspiration=NULL;
    if(plan)
      length=0;
    else {
      experience=fftw_export_wisdom_to_string();
      fftw_forget_wisdom();
      plan=F->Plan(in,out);
      if(plan) {
      inspiration=fftw_export_wisdom_to_string();
      length=strlen(inspiration);
      } else length=0;
    }
    MPI_Gather(&length,1,MPI_INT,NULL,1,MPI_INT,0,Active);
    if(length > 0) {
      MPI_Send(inspiration,length,MPI_CHAR,0,0,Active);
      fftw_import_wisdom_from_string(experience);
      fftw_free(experience);
      fftw_free(inspiration);
    }
  }
  return plan;
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
  unsigned int N=d.X*d.Y;
  unsigned int n=d.x*d.Y;
  double denom=1.0/N;
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif
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
  
  if(d.z > 0) 
    Txy->transpose(f,true,false);
  
  xForwards->fft(f);
}

void fft3dMPI::Backwards(Complex *f)
{
  xBackwards->fft(f);

  if(d.z > 0) 
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
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(yzthreads)
#endif
  for(unsigned int i=0; i < n; ++i) 
    f[i] *= denom;
}

void fft3dMPI::BackwardsNormalized(Complex *f)
{
  Backwards(f);
  Normalize(f);
}

void rcfft2dMPI::Forwards(double *f, Complex *g)
{
  yForwards->fft(f,g);
  T->transpose(g,true,false);
  xForwards->fft(g);
}

void rcfft2dMPI::Backwards(Complex *g, double *f)
{
  xBackwards->fft(g);
  T->transpose(g,true,false); // FIXME: correct?
  yBackwards->fft(f,g);
}

} // End of namespace fftwpp
