#include "mpi/mpiconvolution.h"

namespace fftwpp {

// output the contents of a 2D complex array
void show(Complex *f, unsigned int nx, unsigned int ny, const MPIgroup& group)
{ 
  MPI_Status stat;
  
  if(group.rank == 0) {
    std::cout << "process " << 0 << ":" <<  std::endl;
    unsigned c=0;
    for(unsigned int i=0; i < nx; i++) {
      for(unsigned int j=0; j < ny; j++) {
        std::cout << f[c++] << "\t";
      }
      std::cout << std::endl;
    }
    
    for(int p=1; p < group.size; p++) {
      int source=p, tag=p;
      unsigned int pdims[2];
      MPI_Recv(&pdims,2,MPI_UNSIGNED,source,tag,group.active,&stat);

      unsigned int px=pdims[0], py=pdims[1];
      unsigned int n=px*py;
      Complex *C=new Complex[n];
      tag += group.size;
      MPI_Recv(C,2*n,MPI_DOUBLE,source,tag,group.active,&stat);
      
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
    int dest=0, tag=group.rank;
    unsigned int dims[]={nx,ny};
    MPI_Send(&dims,2,MPI_UNSIGNED,dest,tag,group.active);
    int n=nx*ny;
    tag += group.size;
    MPI_Send(f,2*n,MPI_DOUBLE,dest,tag,group.active);
  }
}

// output the contents of a 3D complex array
void show(Complex *f, unsigned int nx, unsigned int ny, unsigned int nz,
          const MPIgroup& group)
{
  MPI_Status stat;
  
  if(group.rank ==0) {
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
    
    for(int p=1; p < group.size; p++) {
      int source=p, tag=p;
      unsigned int pdims[3];
      MPI_Recv(&pdims,3,MPI_UNSIGNED,source,tag,group.active,&stat);

      unsigned int px=pdims[0], py=pdims[1], pz=pdims[2];
      int n=px*pz*pz;
      Complex *C=new Complex[n];
      tag += group.size;
      MPI_Recv(C,2*n,MPI_DOUBLE,source,tag,group.active,&stat);
      
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
    int dest=0, tag=group.rank;
    unsigned int dims[]={nx,ny,nz};
    MPI_Send(&dims,3,MPI_UNSIGNED,dest,tag,group.active);
    int n=nx*ny*nz;
    tag += group.size;
    MPI_Send(f,2*n,MPI_DOUBLE,dest,tag,group.active);
  }
}

// hash-check for 2D arrays
int hash(Complex *f, unsigned int nx, unsigned int ny, const MPIgroup& group)
{ 
  MPI_Barrier(group.active);
  MPI_Status stat;
  int hash=0;
  
  if(group.rank == 0) {
    unsigned c=0;
    for(unsigned int j=0; j < ny; j++) {
      for(unsigned int i=0; i < nx; i++) {
	c=i*ny+j;
	hash= (hash+(324723947+(int)(f[c].re+0.5)))^93485734985;
	hash= (hash+(324723947+(int)(f[c].im+0.5)))^93485734985;
      }
    }

    for(int p=1; p < group.size; p++) {
      int source=p, tag=p;
      unsigned int pdims[2];
      MPI_Recv(&pdims,2,MPI_UNSIGNED,source,tag,group.active,&stat);

      unsigned int px=pdims[0], py=pdims[1];
      unsigned int n=px*py;
      Complex *C=new Complex[n];
      tag += group.size;
      MPI_Recv(C,2*n,MPI_DOUBLE,source,tag,group.active,&stat);
      
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
    
    for(int p=1; p < group.size; p++) {
      int tag=p+2*group.size;
      MPI_Send(&hash,1,MPI_INT,p,tag,group.active);
    }
  } else {
    int dest=0, tag=group.rank;
    unsigned int dims[]={nx,ny};
    MPI_Send(&dims,2,MPI_UNSIGNED,dest,tag,group.active);
    int n=nx*ny;
    tag += group.size;
    MPI_Send(f,2*n,MPI_DOUBLE,dest,tag,group.active);

    tag += group.size;
    MPI_Recv(&hash,1,MPI_INT,0,tag,group.active,&stat);
  }
  return hash;
}

// return a hash of the contents of a 3D complex array
int hash(Complex *f, unsigned int nx, unsigned int ny, unsigned int nz,
         const MPIgroup& group)
{
  MPI_Status stat;
  int hash=0;
  
  if(group.rank ==0) {
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
    
    for(int p=1; p < group.size; p++) {
      int source=p, tag=p;
      unsigned int pdims[3];
      MPI_Recv(&pdims,3,MPI_UNSIGNED,source,tag,group.active,&stat);

      unsigned int px=pdims[0], py=pdims[1], pz=pdims[2];
      int n=px*pz*pz;
      Complex *C=new Complex[n];
      tag += group.size;
      MPI_Recv(C,2*n,MPI_DOUBLE,source,tag,group.active,&stat);
      
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

    for(int p=1; p < group.size; p++) {
      int tag=p+2*group.size;
      MPI_Send(&hash,1,MPI_INT,p,tag,group.active);
    }
  } else {
    int dest=0, tag=group.rank;
    unsigned int dims[]={nx,ny,nz};
    MPI_Send(&dims,3,MPI_UNSIGNED,dest,tag,group.active);
    int n=nx*ny*nz;
    tag += group.size;
    MPI_Send(f,2*n,MPI_DOUBLE,dest,tag,group.active);
    
    tag += group.size;
    MPI_Recv(&hash,1,MPI_INT,0,tag,group.active,&stat);
  }
  return hash;
}

}
