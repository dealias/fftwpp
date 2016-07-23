#include "mpifftw++.h"
#include "unistd.h"

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

fftw_plan MPIplanner(fftw *F, Complex *in, Complex *out) 
{
  if(utils::Active == MPI_COMM_NULL)
    return Planner(F,in,out);
  fftw_plan plan;
  int rank;
  MPI_Comm_rank(utils::Active,&rank);
  
  if(rank == 0) {
    int size;
    MPI_Comm_size(utils::Active,&size);
    
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
      MPI_Send(&length,1,MPI_INT,i,0,utils::Active);
    if(length > 0) {
      MPI_Bcast(inspiration,length,MPI_CHAR,0,utils::Active);
      if(Wise) {
        fftw_import_wisdom_from_string(experience);
        fftw_free(experience);
      } else Wise=true;
      fftw_free(inspiration);
    }
    int rlength[size];
    MPI_Gather(&length,1,MPI_INT,rlength,1,MPI_INT,0,utils::Active);
    for(int i=1; i < size; ++i) {
      int length=rlength[i];
      if(length > 0) {
        learned=true;
        char inspiration[length+1];
        MPI_Recv(&inspiration,length,MPI_CHAR,i,0,utils::Active,MPI_STATUS_IGNORE);
        inspiration[length]=0;
        fftw_import_wisdom_from_string(inspiration);
      }
    }
    if(learned) SaveWisdom();
  } else {
    int flag=false;
    MPI_Status status;
    while(true) {
      MPI_Iprobe(0,0,utils::Active,&flag,&status);
      if(flag) break;
      usleep(10000);
    }
    int length;
    MPI_Recv(&length,1,MPI_INT,0,0,utils::Active,MPI_STATUS_IGNORE);
    if(length > 0) {
      char inspiration[length+1];
      MPI_Bcast(inspiration,length,MPI_CHAR,0,utils::Active);
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
    MPI_Gather(&length,1,MPI_INT,NULL,1,MPI_INT,0,utils::Active);
    if(length > 0) {
      MPI_Send(inspiration,length,MPI_CHAR,0,0,utils::Active);
      fftw_import_wisdom_from_string(experience);
      fftw_free(experience);
      fftw_free(inspiration);
    }
  }
  return plan;
}

}

namespace utils {
MPI_Comm Active=MPI_COMM_NULL;

void setMPIplanner()
{
  fftwpp::fftw::planner=fftwpp::MPIplanner;
}

}
