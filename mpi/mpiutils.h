#ifndef __mpiutils_h__
#define __mpiutils_h__ 1

namespace fftwpp {

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
    unsigned int c=0;
    for(unsigned int i=0; i < nx; i++) {
      for(unsigned int j=0; j < ny; j++) {
        std::cout << f[c++] << "\t";
      }
      std::cout << std::endl;
    }
    
    for(int p=1; p < size; p++) {
      int source=p, tag=p;
      unsigned int pdims[2];
      MPI_Recv(&pdims,2,MPI_UNSIGNED,source,tag,communicator,&stat);

      unsigned int px=pdims[0], py=pdims[1];
      unsigned int n=px*py;
      ftype *C=new ftype[n];
      tag += size;
      MPI_Recv(C,sizeof(ftype)*n,MPI_BYTE,source,tag,communicator,&stat);
      
      std::cout << "process " << p << ":" <<  std::endl;
      unsigned int c=0;
      for(unsigned int i=0; i < px; i++) {
        for(unsigned int j=0; j < py; j++) {
          std::cout << C[c++] << "\t";
        }
        std::cout << std::endl;
      }
      delete [] C;
    }
  } else {
    int dest=0, tag=rank;
    unsigned int dims[]={nx,ny};
    MPI_Send(&dims,2,MPI_UNSIGNED,dest,tag,communicator);
    int n=nx*ny;
    tag += size;
    MPI_Send(f,sizeof(ftype)*n,MPI_BYTE,dest,tag,communicator);
  }
}
  
// output the contents of a 3D complex array
template<class ftype>
void show(ftype *f, unsigned int nx, unsigned int ny, unsigned int nz,
          const MPI_Comm& communicator)
{
  MPI_Status stat;
  int size,rank;
  MPI_Comm_size(communicator,&size);
  MPI_Comm_rank(communicator,&rank);
  
  if(rank ==0) {
    std::cout << "process " << 0 << ":" <<  std::endl;
    unsigned c=0;
    for(unsigned int i=0; i < nx; i++) {
      for(unsigned int j=0; j < ny; j++) {
        for(unsigned int k=0; k < nz; k++) {
          std::cout << f[c++] << "\t";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    
    for(int p=1; p < size; p++) {
      int source=p, tag=p;
      unsigned int pdims[3];
      MPI_Recv(&pdims,3,MPI_UNSIGNED,source,tag,communicator,&stat);

      unsigned int px=pdims[0], py=pdims[1], pz=pdims[2];
      int n=px*pz*pz;
      ftype *C=new ftype[n];
      tag += size;
      MPI_Recv(C,n*sizeof(ftype),MPI_BYTE,source,tag,communicator,&stat);
      
      std::cout << "process " << p << ":" <<  std::endl;
      unsigned int c=0;
      for(unsigned int i=0; i < px; i++) {
        for(unsigned int j=0; j < py; j++) {
          for(unsigned int k=0; k < pz; k++) {
            std::cout << C[c++]  << "\t";
          }
          std::cout << std::endl;
        }
        std::cout << std::endl;
      }
      delete [] C;
    }
  } else {
    int dest=0, tag=rank;
    unsigned int dims[]={nx,ny,nz};
    MPI_Send(&dims,3,MPI_UNSIGNED,dest,tag,communicator);
    int n=nx*ny*nz;
    tag += size;
    MPI_Send(f,n*sizeof(ftype),MPI_BYTE,dest,tag,communicator);
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
  
