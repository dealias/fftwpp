#ifndef __mpiutils_h__
#define __mpiutils_h__ 1

namespace fftwpp {

// output the contents of a 2D complex array
void show(Complex *f, unsigned int nx, unsigned int ny,
          const MPI_Comm& communicator=MPI_COMM_WORLD);
  
// output the contents of a 3D complex array
void show(Complex *f, unsigned int nx, unsigned int ny, unsigned int nz,
          const MPI_Comm& communicator=MPI_COMM_WORLD);

// hash-check for 2D arrays
int hash(Complex *f, unsigned int nx, unsigned int ny,
         const MPI_Comm& communicator=MPI_COMM_WORLD);

// return a hash of the contents of a 3D complex array
int hash(Complex *f, unsigned int nx, unsigned int ny, unsigned int nz,
         MPI_Comm communicator=MPI_COMM_WORLD);

}

#endif
  
