#include <mpi.h>
#include "../Complex.h"

namespace fftwpp {

void show(Complex *f, unsigned int nx, unsigned int ny,
          const MPI_Comm& communicator=MPI_COMM_WORLD)
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
      Complex *C=new Complex[n];
      tag += size;
      MPI_Recv(C,2*n,MPI_DOUBLE,source,tag,communicator,&stat);
      
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
    MPI_Send(f,2*n,MPI_DOUBLE,dest,tag,communicator);
  }
}

void show(Complex *f, unsigned int nx, unsigned int ny, unsigned int nz,
          const MPI_Comm& communicator=MPI_COMM_WORLD)
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
      Complex *C=new Complex[n];
      tag += size;
      MPI_Recv(C,2*n,MPI_DOUBLE,source,tag,communicator,&stat);
      
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
    MPI_Send(f,2*n,MPI_DOUBLE,dest,tag,communicator);
  }
}

int hash(Complex *f, unsigned int nx, unsigned int ny,
         const MPI_Comm& communicator=MPI_COMM_WORLD)
{ 
  MPI_Barrier(communicator);
  MPI_Status stat;
  int hash=0;
  int size,rank;
  MPI_Comm_size(communicator,&size);
  MPI_Comm_rank(communicator,&rank);
  
  if(rank == 0) {
    unsigned c=0;
    for(unsigned int j=0; j < ny; j++) {
      for(unsigned int i=0; i < nx; i++) {
	c=i*ny+j;
	hash= (hash+(324723947+(int)(f[c].re+0.5)))^93485734985;
	hash= (hash+(324723947+(int)(f[c].im+0.5)))^93485734985;
      }
    }

    for(int p=1; p < size; p++) {
      int source=p, tag=p;
      unsigned int pdims[2];
      MPI_Recv(&pdims,2,MPI_UNSIGNED,source,tag,communicator,&stat);

      unsigned int px=pdims[0], py=pdims[1];
      unsigned int n=px*py;
      Complex *C=new Complex[n];
      tag += size;
      MPI_Recv(C,2*n,MPI_DOUBLE,source,tag,communicator,&stat);
      
      unsigned int c=0;
      for(unsigned int j=0; j < py; j++) {
	for(unsigned int i=0; i < px; i++) {
	  c=i*py+j;
	  hash= (hash+(324723947+(int)(C[c].re+0.5)))^93485734985;
	  hash= (hash+(324723947+(int)(C[c].im+0.5)))^93485734985;
        }
      }
      delete [] C;
    }
    
    for(int p=1; p < size; p++) {
      int tag=p+2*size;
      MPI_Send(&hash,1,MPI_INT,p,tag,communicator);
    }
  } else {
    int dest=0, tag=rank;
    unsigned int dims[]={nx,ny};
    MPI_Send(&dims,2,MPI_UNSIGNED,dest,tag,communicator);
    int n=nx*ny;
    tag += size;
    MPI_Send(f,2*n,MPI_DOUBLE,dest,tag,communicator);

    tag += size;
    MPI_Recv(&hash,1,MPI_INT,0,tag,communicator,&stat);
  }
  return hash;
}

int hash(Complex *f, unsigned int nx, unsigned int ny, unsigned int nz,
         MPI_Comm communicator=MPI_COMM_WORLD)
{
  MPI_Status stat;
  int hash=0;
  int size,rank;
  MPI_Comm_size(communicator,&size);
  MPI_Comm_rank(communicator,&rank);
  
  if(rank ==0) {
    unsigned c=0;
    for(unsigned int j=0; j < ny; j++) {
      for(unsigned int i=0; i < nx; i++) {
	for(unsigned int k=0; k < nz; k++) {
	  c=(i*ny+j)*nz+k;
	  hash= (hash+(324723947+(int)(f[c].re+0.5)))^93485734985;
	  hash= (hash+(324723947+(int)(f[c].im+0.5)))^93485734985;
        }
      }
    }
    
    for(int p=1; p < size; p++) {
      int source=p, tag=p;
      unsigned int pdims[3];
      MPI_Recv(&pdims,3,MPI_UNSIGNED,source,tag,communicator,&stat);

      unsigned int px=pdims[0], py=pdims[1], pz=pdims[2];
      int n=px*pz*pz;
      Complex *C=new Complex[n];
      tag += size;
      MPI_Recv(C,2*n,MPI_DOUBLE,source,tag,communicator,&stat);
      
      unsigned int c=0;
      for(unsigned int j=0; j < py; j++) {
	for(unsigned int i=0; i < px; i++) {
	  for(unsigned int k=0; k < pz; k++) {
	    c=(i*py+j)*pz+k;
	    hash= (hash+(324723947+(int)(C[c].re+0.5)))^93485734985;
	    hash= (hash+(324723947+(int)(C[c].im+0.5)))^93485734985;
          }
        }
      }
      delete [] C;
    }

    for(int p=1; p < size; p++) {
      int tag=p+2*size;
      MPI_Send(&hash,1,MPI_INT,p,tag,communicator);
    }
  } else {
    int dest=0, tag=rank;
    unsigned int dims[]={nx,ny,nz};
    MPI_Send(&dims,3,MPI_UNSIGNED,dest,tag,communicator);
    int n=nx*ny*nz;
    tag += size;
    MPI_Send(f,2*n,MPI_DOUBLE,dest,tag,communicator);
    
    tag += size;
    MPI_Recv(&hash,1,MPI_INT,0,tag,communicator,&stat);
  }
  return hash;
}

}
