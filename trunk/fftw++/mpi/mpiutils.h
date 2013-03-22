#ifndef __mpiutils_h__
#define __mpiutils_h__ 1

#include "mpiconvolution.h"

void transpose(Complex *in, Complex *out, unsigned int N, unsigned int m,
               unsigned int n, unsigned int M, bool intransposed,
               MPI_Comm& communicator, MPI_Request *request=NULL);

#if MPI_VERSION < 3
inline int MPI_Ialltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                         void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                         MPI_Comm comm, MPI_Request *)
{
  return MPI_Alltoall(sendbuf,sendcount,sendtype,recvbuf,recvcount,recvtype,
                      comm);
}
inline void Wait(MPI_Request *)
{ 
}
#else
inline void Wait(MPI_Request *request)
{ 
  MPI_Status status;
  MPI_Wait(request,&status);
}
#endif

#endif
