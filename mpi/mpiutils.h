#ifndef __mpiutils_h__
#define __mpiutils_h__ 1

#include "mpifftw++.h"

namespace fftwpp {

template<class ftype>
void show(ftype *f, unsigned int, unsigned int ny,
          unsigned int x0, unsigned int y0, unsigned int x1, unsigned int y1)
{
  for(unsigned int i=x0; i < x1; ++i) {
    for(unsigned int j=y0; j < y1; ++j) {
      std::cout << f[ny*i+j]  << "\t";
    }
    std::cout << std::endl;
  }
}
 
// FIXME: documentation
template<class ftype>
void accumulate_splitx(const ftype *part,  ftype *whole,
		       const unsigned int X,
		       const unsigned int Y,
		       const unsigned int x0,
		       const unsigned int y0,
		       const unsigned int x,
		       const unsigned int y,
		       const bool transposed, 
		       const MPI_Comm& communicator)
{
  MPI_Status stat;
  int size, rank;
  MPI_Comm_size(communicator, &size);
  MPI_Comm_rank(communicator, &rank);

  if(rank == 0) {
    // First copy rank 0's part into the whole
    if(!transposed) {
      // x . Y
      copyfromblock(part, whole, x, Y, Y);
    } else {
      // X . y
      copyfromblock(part, whole, X, y, Y);
    }

    for(int p = 1; p < size; ++p) {
      unsigned int dims[6];
      MPI_Recv(&dims, 6, MPI_UNSIGNED, p, 0, communicator, &stat);

      unsigned int X = dims[0], Y = dims[1];
      unsigned int x0 = dims[2], y0 = dims[3];
      unsigned int x = dims[4], y = dims[5];
      unsigned int n = !transposed ? x * Y :  X * y;
      if(n > 0) {
        ftype *C = new ftype[n];
        MPI_Recv(C, sizeof(ftype) * n, MPI_BYTE, p, 0, communicator, &stat);
	if(!transposed) {
	  copyfromblock(C, whole + x0 * Y, x, Y, Y);
	} else {
	  copyfromblock(C, whole + y0, X, y, Y);
	}
        delete [] C;
      }
    }
  } else {
    unsigned int dims[]={X, Y, x0, y0, x, y};
    MPI_Send(&dims, 6, MPI_UNSIGNED, 0, 0, communicator);
    unsigned int n = !transposed ? x * Y :  X * y;
    if(n > 0)
      MPI_Send(part, n * sizeof(ftype), MPI_BYTE, 0, 0, communicator);
  }
}

template<class ftype>
void accumulate_splitx(const ftype *part,  ftype *whole,
		       const splitx split,
		       const bool transposed, 
		       const MPI_Comm& communicator)
{
  unsigned int X = split.nx;
  unsigned int Y = split.ny;
  unsigned int x0 = split.x0;
  unsigned int y0 = split.y0;
  unsigned int x = split.x;
  unsigned int y = split.y;

  accumulate_splitx(part, whole, X, Y, x0, y0, x, y, transposed,communicator);
}

// output the contents of a 2D array
template<class ftype>
void show(ftype *f, unsigned int nx, unsigned int ny,
          unsigned int x0, unsigned int y0,
          unsigned int x1, unsigned int y1, const MPI_Comm& communicator)
          
{  
  MPI_Status stat;
  int size,rank;
  MPI_Comm_size(communicator,&size);
  MPI_Comm_rank(communicator,&rank);
  
  if(rank == 0) {
    std::cout << "process " << 0 << ":" <<  std::endl;
    show(f,nx,ny,x0,y0,x1,y1);
    
    for(int p=1; p < size; p++) {
      unsigned int dims[6];
      MPI_Recv(&dims,6,MPI_UNSIGNED,p,0,communicator,&stat);

      unsigned int nx=dims[0], ny=dims[1];
      unsigned int x0=dims[2], y0=dims[3];
      unsigned int x1=dims[4], y1=dims[5];
      unsigned int n=nx*ny;
      std::cout << "process " << p << ":" <<  std::endl;
      if(n > 0) {
        ftype *C=new ftype[n];
        MPI_Recv(C,sizeof(ftype)*n,MPI_BYTE,p,0,communicator,&stat);
      
        show(C,nx,ny,x0,y0,x1,y1);
        delete [] C;
      }
    }
  } else {
    unsigned int dims[]={nx,ny,x0,y0,x1,y1};
    MPI_Send(&dims,6,MPI_UNSIGNED,0,0,communicator);
    unsigned int n=nx*ny;
    if(n > 0)
      MPI_Send(f,n*sizeof(ftype),MPI_BYTE,0,0,communicator);
  }
}
  
template<class ftype>
void show(ftype *f, unsigned int nx, unsigned int ny,
          const MPI_Comm& communicator)
{ 
  show(f,nx,ny,0,0,nx,ny,communicator);
}

template<class ftype>
void show(ftype *f, unsigned int nx, unsigned int ny, unsigned int nz,
          unsigned int x0, unsigned int y0, unsigned int z0,
          unsigned int x1, unsigned int y1, unsigned int z1)
{
  for(unsigned int i=x0; i < x1; ++i) {
    for(unsigned int j=y0; j < y1; ++j) {
      for(unsigned int k=z0; k < z1; ++k) {
        std::cout << f[nz*(ny*i+j)+k]  << "\t";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

// output the contents of a 3D array
template<class ftype>
void show(ftype *f, unsigned int nx, unsigned int ny, unsigned int nz,
          unsigned int x0, unsigned int y0, unsigned int z0,
          unsigned int x1, unsigned int y1, unsigned int z1,
          const MPI_Comm& communicator)
{
  MPI_Status stat;
  int size,rank;
  MPI_Comm_size(communicator,&size);
  MPI_Comm_rank(communicator,&rank);
  
  if(rank == 0) {
    std::cout << "process " << 0 << ":" <<  std::endl;
    show(f,nx,ny,nz,x0,y0,z0,x1,y1,z1);
    
    for(int p=1; p < size; p++) {
      unsigned int dims[9];
      MPI_Recv(&dims,9,MPI_UNSIGNED,p,0,communicator,&stat);

      unsigned int nx=dims[0], ny=dims[1], nz=dims[2];
      unsigned int x0=dims[3], y0=dims[4], z0=dims[5];
      unsigned int x1=dims[6], y1=dims[7], z1=dims[8];
      unsigned int n=nx*ny*nz;
      if(n > 0) {
        ftype *C=new ftype[n];
        MPI_Recv(C,n*sizeof(ftype),MPI_BYTE,p,0,communicator,&stat);
      
        std::cout << "process " << p << ":" <<  std::endl;
        show(C,nx,ny,nz,x0,y0,z0,x1,y1,z1);
        delete [] C;
      }
    }
  } else {
    unsigned int dims[]={nx,ny,nz,x0,y0,z0,x1,y1,z1};
    MPI_Send(&dims,9,MPI_UNSIGNED,0,0,communicator);
    unsigned int n=nx*ny*nz;
    if(n > 0)
      MPI_Send(f,n*sizeof(ftype),MPI_BYTE,0,0,communicator);
  }
}

template<class ftype>
void show(ftype *f, unsigned int nx, unsigned int ny, unsigned int nz,
          const MPI_Comm& communicator)
{ 
  show(f,nx,ny,nz,0,0,0,nx,ny,nz,communicator);
}

// hash-check for 2D arrays
int hash(Complex *f, unsigned int nx, unsigned int ny,
         const MPI_Comm& communicator);

// return a hash of the contents of a 3D complex array
int hash(Complex *f, unsigned int nx, unsigned int ny, unsigned int nz,
         MPI_Comm communicator);

} // namespace fftwpp

#endif
  
