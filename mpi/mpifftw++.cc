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
  if(d.yz.x < d.Y) {
    zForward->fft(in,out);
    Tyz->transpose(out,true,false);
    yForward->fft(out);
  } else {
    unsigned int stride=d.Y*d.z;
    unsigned int stop=d.x*stride;
    PARALLEL(
      for(unsigned int i=0; i < stop; i += stride)
        yzForward->fft(in+i,out+i);
      );
  }
  
  if(Txy)
    Txy->transpose(out,true,false);
  
  xForward->fft(out);
}

void fft3dMPI::Backward(Complex *in, Complex *out)
{
  out=Setout(in,out);
  xBackward->fft(in,out);

  if(Txy)
    Txy->transpose(out,false,true);

  if(d.yz.x < d.Y) {
    yBackward->fft(out);
    Tyz->transpose(out,false,true);
    zBackward->fft(out);
  } else {
    unsigned int stride=d.z*d.Y;
    unsigned int stop=d.x*stride;
    for(unsigned int i=0; i < stop; i += stride)
      yzBackward->fft(out+i);
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

void rcfft2dMPI::Forward(double *f, Complex *g)
{
  yForward->fft(f,g);
  T->transpose(g,true,false);
  xForward->fft(g);
}

void rcfft2dMPI::Forward0(double *f, Complex *g)
{
  Shift(f);
  Forward(f,g);
}

void rcfft2dMPI::Backward(Complex *g, double *f)
{
  xBackward->fft(g);
  T->transpose(g,false,true);
  yBackward->fft(g,f);
}

void rcfft2dMPI::Backward0(Complex *g, double *f)
{
  Backward(g,f);
  Shift(f);
}

void rcfft2dMPI::Normalize(double *f)
{
  // FIXME: deal with in-place.
  unsigned int N=dr.X*dr.Y;
  unsigned int n=dr.x*dr.Y;
  double denom=1.0/N;
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif
  for(unsigned int i=0; i < n; ++i) 
    f[i] *= denom;
}

void rcfft2dMPI::BackwardNormalized(Complex *g, double *f)
{
  Backward(g,f);
  Normalize(f);
}

void rcfft2dMPI::Backward0Normalized(Complex *g, double *f)
{
  BackwardNormalized(g,f);
  Shift(f);
}

void rcfft3dMPI::Forward(double *f, Complex *g)
{
  // FIXME: deal with in-place
  zForward->fft(f,g);
  if(Tyz) Tyz->transpose(g,true,false);
  const unsigned int stride=dc.z*dc.Y;
  for(unsigned int i=0; i < dc.x; ++i) 
    yForward->fft(g+i*stride);
 if(Txy) Txy->transpose(g,true,false);
  xForward->fft(g);
}

void rcfft3dMPI::Backward(Complex *g, double *f)
{
  // FIXME: deal with in-place
  xBackward->fft(g);
  if(Txy) Txy->transpose(g,false,true);
  const unsigned int stride=dc.z*dc.Y;
  for(unsigned int i=0; i < dc.x; ++i) 
    yBackward->fft(g+i*stride);
  if(Tyz) Tyz->transpose(g,false,true);
  zBackward->fft(g,f);
}

void rcfft3dMPI::Normalize(double *f)
{
  // FIXME: deal with in-place
  unsigned int N=dr.X*dr.Y*dr.Z;
  unsigned int n=dr.x*dr.yz.x*dr.Z;
  double denom=1.0/N;
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(yzthreads)
#endif
  for(unsigned int i=0; i < n; ++i) 
    f[i] *= denom;
}

void rcfft3dMPI::BackwardNormalized(Complex *g, double *f)
{
  Backward(g,f);
  Normalize(f);
}

void rcfft3dMPI::Shift(double *f)
{
  if(dr.X % 2 == 0 && dr.Y % 2 == 0) {
    const unsigned int dist=dr.Z; // FIXME: inplace?
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(xythreads)
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

void rcfft3dMPI::Forward0(double *f, Complex *g)
{
  Shift(f);
  Forward(f,g);
}
  
void rcfft3dMPI::Backward0(Complex *g, double *f)
{
  Backward(g,f);
  Shift(f);
}

void rcfft3dMPI::Backward0Normalized(Complex *g, double *f)
{
  BackwardNormalized(g,f);
  Shift(f);
}

} // End of namespace fftwpp
