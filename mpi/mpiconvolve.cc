#include "mpiconvolve.h"
#include "mpigroup.h"
#include "parallel.h"

using namespace utils;

namespace fftwpp {

// Enforce 3D Hermiticity using given (x,y > 0,z=0) and (x >= 0,y=0,z=0) data.
// u is an optional work array of size d.X.
void HermitianSymmetrizeXY(split3& d, Complex *f, Complex *u, size_t threads)
{
  size_t mx=utils::ceilquotient(d.X,2);
  size_t my=utils::ceilquotient(d.Y,2);
  bool xcompact=mx-d.X/2;
  bool ycompact=my-d.Y/2;

  int rank,size;
  MPI_Comm_size(d.communicator,&size);
  if(size == 1) {
    if(mx > 0 && my > 0 && d.Z > 0)
      HermitianSymmetrizeXY(mx,my,d.Z,d.X/2,d.Y/2,f);
    return;
  }
  unsigned int xextra=!xcompact;
  unsigned int yextra=!ycompact;
  unsigned int yorigin=my-ycompact;
  unsigned int nx=d.X-xextra;
  unsigned int y0=d.xy.y0;
  unsigned int dy=d.xy.y;
  unsigned int j0=y0 == 0 ? yextra : 0;
  unsigned int start=(yorigin > y0) ? yorigin-y0 : 0;

  if(d.XYplane == NULL) {
    d.XYplane=new MPI_Comm;
    MPI_Comm_split(d.communicator,d.z0 == 0,0,d.XYplane);
    if(d.z0 != 0) return;
    MPI_Comm_rank(*d.XYplane,&rank);
    MPI_Comm_size(*d.XYplane,&size);
    d.reflect=new int[dy];
    unsigned int n[size];
    unsigned int start[size];
    n[rank]=dy;
    start[rank]=y0;
    MPI_Allgather(MPI_IN_PLACE,0,MPI_UNSIGNED,n,1,MPI_UNSIGNED,*d.XYplane);
    MPI_Allgather(MPI_IN_PLACE,0,MPI_UNSIGNED,start,1,MPI_UNSIGNED,*d.XYplane);
    if(rank == 0) {
      int process[d.Y];
      for(int p=0; p < size; ++p) {
        unsigned int stop=start[p]+n[p];
        for(unsigned int j=start[p]; j < stop; ++j)
          process[j]=p;
      }

      for(unsigned int j=j0; j < dy; ++j)
        d.reflect[j]=process[2*yorigin-y0-j];
      for(int p=1; p < size; ++p) {
        for(unsigned int j=start[p] == 0 ? yextra : 0; j < n[p];
            ++j)
          MPI_Send(process+2*yorigin-start[p]-j,1,MPI_INT,p,j,
                   *d.XYplane);
      }
    } else {
      for(unsigned int j=0; j < dy; ++j)
        MPI_Recv(d.reflect+j,1,MPI_INT,0,j,*d.XYplane,MPI_STATUS_IGNORE);
    }
  }
  if(d.z0 != 0) return;
  MPI_Comm_rank(*d.XYplane,&rank);

  bool allocateu;
  if(u)
    allocateu=false;
  else {
    allocateu=true;
    u=ComplexAlign(nx);
  }

  unsigned int stride=dy*d.z;
  for(unsigned int j=start; j < dy; ++j) {
    for(unsigned int i=0; i < nx; ++i)
      u[i]=conj(f[stride*(d.X-1-i)+d.z*j]);
    int J=d.reflect[j];
    if(J != rank)
      MPI_Send(u,2*nx,MPI_DOUBLE,J,0,*d.XYplane);
    else {
      if(y0+j != yorigin) {
        int offset=d.z*(2*(yorigin-y0)-j);
        for(unsigned int i=0; i < nx; ++i) {
          unsigned int N=stride*(i+xextra)+offset;
          if(N < d.n)
            f[N]=u[i];
          else {
            if(rank == 0)
              std::cerr << "Invalid index in HermitianSymmetrizeXY."
                        << std::endl;
            exit(-3);
          }
        }
      } else {
        unsigned int origin=stride*(mx-xcompact)+d.z*j;
        f[origin].im=0.0;
        unsigned int mxstride=mx*stride;
        for(unsigned int i=stride; i < mxstride; i += stride)
          f[origin-i]=conj(f[origin+i]);
      }
    }
  }

  for(unsigned int j=std::min(dy,start); j-- > j0;) {
    int J=d.reflect[j];
    if(J != rank) {
      MPI_Recv(u,2*nx,MPI_DOUBLE,J,0,*d.XYplane,MPI_STATUS_IGNORE);
      for(unsigned int i=0; i < nx; ++i)
        f[stride*(i+xextra)+d.z*j]=u[i];
    }
  }

  // Zero out Nyquist modes
  if(xextra) {
    PARALLELIF(
      d.y*d.z > threshold,
      for(size_t j=0; j < d.y; ++j) {
        for(size_t k=0; k < d.z; ++k) {
          f[d.z*j+k]=0.0;
        }
      });
  }

  if(yextra && d.y0 == 0) {
    PARALLELIF(
      d.X*d.z > threshold,
      for(size_t i=0; i < d.X; ++i) {
        for(size_t k=0; k < d.z; ++k) {
          f[d.y*d.z*i+k]=0.0;
        }
      });
  }

  if(allocateu)
    deleteAlign(u);
}

}
