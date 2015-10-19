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

void fft2dMPI::Forward(Complex *in, Complex *out)
{
  out=Setout(in,out);
  yForward->fft(in,out);
  T->transpose(out,true,false);
  xForward->fft(out);
}

void fft2dMPI::Backward(Complex *in, Complex *out)
{
  out=Setout(in,out);
  xBackward->fft(in,out);
  T->transpose(out,false,true);
  yBackward->fft(out);
}

void fft3dMPI::Forward(Complex *in, Complex *out)
{
  out=Setout(in,out);
  unsigned int stride=d.z*d.Y;
  unsigned int stop=d.x*stride;
  if(Tyz) {
    zForward->fft(in,out);
    Tyz->transpose(out,true,false);
    PARALLEL(
      for(unsigned int i=0; i < stop; i += stride) 
        yForward->fft(out+i);
      );
  } else {
    PARALLEL(
      for(unsigned int i=0; i < stop; i += stride) {
        yzForward->fft(in+i,out+i);
      }
      );
  }
  
  if(Txy)
    Txy->transpose(out,true,false);
  
  xForward->fft(out);
}

void fft3dMPI::Backward(Complex *in, Complex *out)
{
  out=Setout(in,out);
  unsigned int stride=d.z*d.Y;
  unsigned int stop=d.x*stride;
  xBackward->fft(in,out);

  if(Txy)
    Txy->transpose(out,false,true);

  if(Tyz) {
    PARALLEL(
      for(unsigned int i=0; i < stop; i += stride) 
        yBackward->fft(out+i);
      );
    Tyz->transpose(out,false,true);
    zBackward->fft(out);
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
    const unsigned int ydist=dr.Y; // FIXME: inplace?
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif
    for(unsigned int i=start; i < stop; i += 2) {
      double *p=f+i*ydist;
      for(unsigned int j=0; j < dr.Y; ++j) {
        p[j]=-p[j];
      }
    }
  } else {
    std::cerr << "Shift is not implemented for odd X." << std::endl;
    exit(1);
  }
}

void rcfft2dMPI::Forward(double *in, Complex *out)
{
  out=Setout((Complex *) in,out);
  yForward->fft(in,out);
  T->transpose(out,true,false);
  xForward->fft(out);
}

void rcfft2dMPI::Backward(Complex *in, double *out=NULL)
{
  out=(double *) Setout(in,(Complex *) out);
  xBackward->fft(in);
  T->transpose(in,false,true);
  yBackward->fft(in,out);
}

void rcfft2dMPI::Forward0(double *in, Complex *out)
{
  Shift(in);
  Forward(in,out);
}

void rcfft2dMPI::Backward0(Complex *in, double *out)
{
  Backward(in,out);
  Shift(out);
}

void rcfft3dMPI::Forward(double *in, Complex *out)
{
  out=Setout((Complex *) in,out);
  zForward->fft(in,out);
  if(Tyz) Tyz->transpose(out,true,false);
  const unsigned int stride=dc.z*dc.Y;
  const unsigned int stop=dc.x*stride;
  PARALLEL(
    for(unsigned int i=0; i < stop; i += stride) 
      yForward->fft(out+i);
    )
 if(Txy) Txy->transpose(out,true,false);
  xForward->fft(out);
}

void rcfft3dMPI::Backward(Complex *in, double *out=NULL)
{
  out=(double *) Setout(in,(Complex *) out);
  xBackward->fft(in);
  if(Txy) Txy->transpose(in,false,true);
  const unsigned int stride=dc.z*dc.Y;
  const unsigned int stop=dc.x*stride;
  PARALLEL(
    for(unsigned int i=0; i < stop; i += stride) 
      yBackward->fft(in+i);
    );
  if(Tyz) Tyz->transpose(in,false,true);
  zBackward->fft(in,out);
}

void rcfft3dMPI::Shift(double *f)
{
  if(dr.X % 2 == 0 && dr.Y % 2 == 0) {
    const unsigned int dist=dr.Z; // FIXME: inplace?
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif
    for(unsigned int i=0; i < dr.x; ++i) {
      const unsigned int ystart=(i+dr.x0+dr.yz.x0+1) % 2;
      double *pi=f+i*dr.yz.x*dist;
      for(unsigned int j=ystart; j < dr.yz.x; j += 2) {
        double *p=pi+j*dist;
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

void rcfft3dMPI::Forward0(double *in, Complex *out)
{
  Shift(in);
  Forward(in,out);
}
  
void rcfft3dMPI::Backward0(Complex *in, double *out=NULL)
{
  Backward(in,out);
  Shift(out);
}

} // End of namespace fftwpp
