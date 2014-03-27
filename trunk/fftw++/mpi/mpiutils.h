#ifndef __mpiutils_h__
#define __mpiutils_h__ 1

namespace fftwpp {

template<class ftype>
void show(ftype *f, unsigned int nx, unsigned int ny,
          unsigned int x0, unsigned int y0)
{
  for(unsigned int i=x0; i < nx; ++i) {
    for(unsigned int j=y0; j < ny; ++j) {
      std::cout << f[ny*i+j]  << "\t";
    }
    std::cout << std::endl;
  }
}

// output the contents of a 2D array
template<class ftype>
void show(ftype *f, unsigned int nx, unsigned int ny,
          const MPI_Comm& communicator, unsigned int x0, unsigned int y0)
          
{ 
  MPI_Status stat;
  int size,rank;
  MPI_Comm_size(communicator,&size);
  MPI_Comm_rank(communicator,&rank);
  
  if(rank == 0) {
    std::cout << "process " << 0 << ":" <<  std::endl;
    show(f,nx,ny,x0,y0);
    
    for(int p=1; p < size; p++) {
      unsigned int dims[4];
      MPI_Recv(&dims,4,MPI_UNSIGNED,p,0,communicator,&stat);

      unsigned int nx=dims[0], ny=dims[1];
      unsigned int x0=dims[2], y0=dims[3];
      unsigned int n=nx*ny;
      ftype *C=new ftype[n];
      MPI_Recv(C,sizeof(ftype)*n,MPI_BYTE,p,0,communicator,&stat);
      
      std::cout << "process " << p << ":" <<  std::endl;
      show(C,nx,ny,x0,y0);
      delete [] C;
    }
  } else {
    unsigned int dims[]={nx,ny,x0,y0};
    MPI_Send(&dims,4,MPI_UNSIGNED,0,0,communicator);
    MPI_Send(f,nx*ny*sizeof(ftype),MPI_BYTE,0,0,communicator);
  }
}
  
template<class ftype>
void show(ftype *f, unsigned int nx, unsigned int ny,
          const MPI_Comm& communicator)
{ 
  show(f,nx,ny,communicator,0,0);
}

template<class ftype>
void show(ftype *f, unsigned int nx, unsigned int ny, unsigned int nz,
          unsigned int x0, unsigned int y0, unsigned int z0)
{
  for(unsigned int i=x0; i < nx; ++i) {
    for(unsigned int j=y0; j < ny; ++j) {
      for(unsigned int k=z0; k < nz; ++k) {
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
          const MPI_Comm& communicator,
          unsigned int x0, unsigned int y0, unsigned int z0)
{
  MPI_Status stat;
  int size,rank;
  MPI_Comm_size(communicator,&size);
  MPI_Comm_rank(communicator,&rank);
  
  if(rank == 0) {
    std::cout << "process " << 0 << ":" <<  std::endl;
    show(f,nx,ny,nz,x0,y0,z0);
    
    for(int p=1; p < size; p++) {
      unsigned int dims[6];
      MPI_Recv(&dims,6,MPI_UNSIGNED,p,0,communicator,&stat);

      unsigned int nx=dims[0], ny=dims[1], nz=dims[2];
      unsigned int x0=dims[3], y0=dims[4], z0=dims[5];
      int n=nx*ny*nz;
      ftype *C=new ftype[n];
      MPI_Recv(C,n*sizeof(ftype),MPI_BYTE,p,0,communicator,&stat);
      
      std::cout << "process " << p << ":" <<  std::endl;
      show(C,nx,ny,nz,x0,y0,z0);
      delete [] C;
    }
  } else {
    unsigned int dims[]={nx,ny,nz,x0,y0,z0};
    MPI_Send(&dims,6,MPI_UNSIGNED,0,0,communicator);
    MPI_Send(f,nx*ny*nz*sizeof(ftype),MPI_BYTE,0,0,communicator);
  }
}

template<class ftype>
void show(ftype *f, unsigned int nx, unsigned int ny, unsigned int nz,
          const MPI_Comm& communicator)
{ 
  show(f,nx,ny,nz,communicator,0,0,0);
}

// hash-check for 2D arrays
int hash(Complex *f, unsigned int nx, unsigned int ny,
         const MPI_Comm& communicator);

// return a hash of the contents of a 3D complex array
int hash(Complex *f, unsigned int nx, unsigned int ny, unsigned int nz,
         MPI_Comm communicator);

} // namespace fftwpp

#endif
  
