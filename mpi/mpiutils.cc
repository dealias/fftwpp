#include "mpi/mpiconvolution.h"

// Globally transpose an N x M matrix distributed over the second dimension.
// Here "in" is a local N x m matrix and "out" is a local n x M matrix.
// Both N and M must be divisible by the number of processors.
// If m and n are both 1 then "in" may be NULL, in which case an in-place
// transpose on "out" is performed.
// An additional local transposition is applied to the input (output) matrix if
// intranspose=true (false).
void transpose(Complex *in, Complex *out, unsigned int N, unsigned int m,
               unsigned int n, unsigned int M, bool intransposed,
               MPI_Comm& communicator, MPI_Request *request)
{
  bool outofplace=in || m > 1 || n > 1;
   
  MPI_Datatype block;
  MPI_Datatype Block;

  MPI_Type_vector(n,2*m,2*M,MPI_DOUBLE,&block);
  MPI_Type_create_resized(block,0,2*m*sizeof(double),&Block);
  MPI_Type_commit(&Block);
  void *inbuf=outofplace ? in : MPI_IN_PLACE;
  
  MPI_Request Request;
  
  if(intransposed)
    MPI_Ialltoall(inbuf,2*n*m,MPI_DOUBLE,out,1,Block,communicator,
                  request ? request : &Request);
  else
    MPI_Ialltoall(inbuf,1,Block,out,2*n*m,MPI_DOUBLE,communicator,
                  request ? request : &Request);
#if MPI_VERSION >= 3
  MPI_Status status;
  if(!request)
    MPI_Wait(&Request,&status);
#endif  
  
  MPI_Type_free(&Block);
  MPI_Type_free(&block);
}

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
