#ifndef __mpitranspose_h__
#define __mpitranspose_h__ 1

#include <iostream>
#include <cstdlib>

#define ALLTOALL 1

void fill1_comm_sched(int *sched, int which_pe, int npes);

// Globally transpose an N x M matrix of blocks of L complex elements
// distributed over the second dimension.
// Here "in" is a local N x m matrix and "out" is a local n x M matrix.
// Currently, both N and M must be divisible by the number of processors.
// work is a temporary array of size N*m*L.
class transpose {
private:
  unsigned int N,m,n,M;
  unsigned int L;
  Complex *work;
  MPI_Comm communicator;
  bool allocated;
  MPI_Request *request;
  MPI_Request Request;
  MPI_Request srequest;
  int size;
  int rank;
  int splitsize;
  int splitrank;
  int *sched;
  MPI_Status status;
  MPI_Comm split;
public: //temp  
  unsigned int b;

public:
  transpose(unsigned int N, unsigned int m, unsigned int n,
            unsigned int M, unsigned int L=1, Complex *work=NULL,
            MPI_Comm communicator=MPI_COMM_WORLD) : 
    N(N), m(m), n(n), M(M), L(L), work(work), communicator(communicator),
    allocated(false) {
    b=1;
    
    MPI_Comm_size(communicator,&size);
    if(size == 1) return;
    MPI_Comm_rank(communicator,&rank);
    if(rank == 0)
      std::cout << "b=" << b << ", ALLTOALL=" << ALLTOALL << std::endl;
    
    if(N % size != 0 || M % size != 0) {
      if(rank == 0)
        std::cout << 
          "ERROR: Matrix dimensions must be divisible by number of processors" 
                  << std::endl << std::endl; 
      MPI_Finalize();
      exit(0);
    }

    if(work == NULL) {
      this->work=new Complex[N*m*L];
      allocated=true;
    }
    
    if(b > (unsigned int) size) b=size;
    if(b == 1) {
      split=communicator;
      splitsize=size;
      splitrank=rank;
    } else {
      unsigned int q=size/b;
      MPI_Comm_split(communicator,rank/q,0,&split);
      MPI_Comm_size(split,&splitsize);
      MPI_Comm_rank(split,&splitrank);
    }
    
#if ALLTOALL    
    request=new MPI_Request[b-1];
#else    
    request=new MPI_Request[std::max((unsigned int) splitsize-1,b-1)];
    sched=new int[splitsize];
    fill1_comm_sched(sched,splitrank,splitsize);
#endif    
  }
  
  ~transpose() {
    if(size == 1) return;
    if(allocated) delete[] work;
    
#if !ALLTOALL
    delete[] sched;
#endif
    delete[] request;
  }
  
// Globally transpose data, applying an additional local transposition
// to the input.
  void inTransposed(Complex *data);
  
  void inwait(Complex *data);
  
// Globally transpose data, applying an additional local transposition
// to the output.
  void outTransposed(Complex *data);
  
  void InTransposed(Complex *data) {
    inTransposed(data);
    inwait(data);
  }
  
  void OutTransposed(Complex *data) {
    outTransposed(data);
    outwait(data);
  }
  
  void outwait(Complex *data);
};

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
