#ifndef __mpiutils_h__
#define __mpiutils_h__ 1

namespace fftwpp {

template<class ftype>
void show(ftype *f, unsigned int nx, unsigned int ny)
{
  unsigned int c=0;
  for(unsigned int i=0; i < nx; ++i) {
    for(unsigned int j=0; j < ny; ++j) {
      std::cout << f[c++]  << "\t";
    }
    std::cout << std::endl;
  }
}

// output the contents of a 2D array
template<class ftype>
void show(ftype *f, unsigned int nx, unsigned int ny,
          const MPI_Comm& communicator)
{ 
  MPI_Status stat;
  int size,rank;
  MPI_Comm_size(communicator,&size);
  MPI_Comm_rank(communicator,&rank);
  
  if(rank == 0) {
    std::cout << "process " << 0 << ":" <<  std::endl;
    show(f,nx,ny);
    
    for(int p=1; p < size; p++) {
      unsigned int dims[2];
      MPI_Recv(&dims,2,MPI_UNSIGNED,p,0,communicator,&stat);

      unsigned int px=dims[0], py=dims[1];
      unsigned int n=px*py;
      ftype *C=new ftype[n];
      MPI_Recv(C,sizeof(ftype)*n,MPI_BYTE,p,0,communicator,&stat);
      
      std::cout << "process " << p << ":" <<  std::endl;
      show(C,px,py);
      delete [] C;
    }
  } else {
    unsigned int dims[]={nx,ny};
    MPI_Send(&dims,2,MPI_UNSIGNED,0,0,communicator);
    MPI_Send(f,nx*ny*sizeof(ftype),MPI_BYTE,0,0,communicator);
  }
}
  
template<class ftype>
void show(ftype *f, unsigned int nx, unsigned int ny, unsigned int nz)
{
  unsigned int c=0;
  for(unsigned int i=0; i < nx; ++i) {
    for(unsigned int j=0; j < ny; ++j) {
      for(unsigned int k=0; k < nz; ++k) {
        std::cout << f[c++]  << "\t";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

// output the contents of a 3D array
template<class ftype>
void show(ftype *f, unsigned int nx, unsigned int ny, unsigned int nz,
          const MPI_Comm& communicator)
{
  MPI_Status stat;
  int size,rank;
  MPI_Comm_size(communicator,&size);
  MPI_Comm_rank(communicator,&rank);
  
  if(rank == 0) {
    std::cout << "process " << 0 << ":" <<  std::endl;
    show(f,nx,ny,nz);
    
    for(int p=1; p < size; p++) {
      unsigned int dims[3];
      MPI_Recv(&dims,3,MPI_UNSIGNED,p,0,communicator,&stat);

      unsigned int px=dims[0], py=dims[1], pz=dims[2];
      int n=px*py*pz;
      ftype *C=new ftype[n];
      MPI_Recv(C,n*sizeof(ftype),MPI_BYTE,p,0,communicator,&stat);
      
      std::cout << "process " << p << ":" <<  std::endl;
      show(C,px,py,pz);
      delete [] C;
    }
  } else {
    unsigned int dims[]={nx,ny,nz};
    MPI_Send(&dims,3,MPI_UNSIGNED,0,0,communicator);
    MPI_Send(f,nx*ny*nz*sizeof(ftype),MPI_BYTE,0,0,communicator);
  }
}

// hash-check for 2D arrays
int hash(Complex *f, unsigned int nx, unsigned int ny,
         const MPI_Comm& communicator);

// return a hash of the contents of a 3D complex array
int hash(Complex *f, unsigned int nx, unsigned int ny, unsigned int nz,
         MPI_Comm communicator);

} // namespace fftwpp

#endif
  
