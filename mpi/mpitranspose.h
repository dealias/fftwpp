#ifndef __mpitranspose_h__
#define __mpitranspose_h__ 1

#include <iostream>
#include <cstdlib>
#include <cstring>

#include <fftw3.h>
#include <fftw++.h>

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
  int size;
  int rank;
  int splitsize;
  int splitrank;
  int split2size;
  int split2rank;
  int *sched, *sched2;
  MPI_Comm split;
  MPI_Comm split2;
  fftw_plan T1,T2,T3;
public: //temp  
  unsigned int a,b;

public:
  transpose(Complex *data, unsigned int N, unsigned int m, unsigned int n,
            unsigned int M, unsigned int L=1, Complex *work=NULL,
            MPI_Comm communicator=MPI_COMM_WORLD) : 
    N(N), m(m), n(n), M(M), L(L), work(work), communicator(communicator),
    allocated(false) {
    a=2;
    bool alltoall=true;
    
    MPI_Comm_size(communicator,&size);
    if(size == 1) return;
    
    if(a >= (unsigned int) size) a=1;
    b=size/a;
      
    MPI_Comm_rank(communicator,&rank);
    if(rank == 0)
      std::cout << "a=" << a << ", alltoall=" << alltoall << std::endl;
    
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
    
    T1=fftwpp::plan_transpose(n*a,b,m*L,data,this->work);
    T2=fftwpp::plan_transpose(n*b,a,m*L,data,this->work);
    T3=fftwpp::plan_transpose(N,m,L,data,this->work);
    
    if(a == 1) {
      split=communicator;
      splitsize=split2size=size;
      splitrank=rank;
    } else {
      MPI_Comm_split(communicator,rank/b,0,&split);
      MPI_Comm_size(split,&splitsize);
      MPI_Comm_rank(split,&splitrank);
      
      MPI_Comm_split(communicator,rank % b,0,&split2);
      MPI_Comm_size(split2,&split2size);
      MPI_Comm_rank(split2,&split2rank);
    }
    
    if(alltoall) {
      request=new MPI_Request[1];
      sched=sched2=NULL;
    } else {
      request=new MPI_Request[std::max(splitsize,split2size)-1];
    
      sched=new int[splitsize];
      fill1_comm_sched(sched,splitrank,splitsize);
    
      if(a > 1) {
        sched2=new int[split2size];
        fill1_comm_sched(sched2,split2rank,split2size);
      } else
        sched2=NULL;
    }
  }
  
  ~transpose() {
    if(size == 1) return;
    
    if(sched2) delete[] sched2;
    if(sched) delete[] sched;
    delete[] request;
    if(allocated) delete[] work;
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
  
  void outwait(Complex *data, bool localtranspose=false);
};

#if MPI_VERSION < 3
inline int MPI_Ialltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                         void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                         MPI_Comm comm, MPI_Request *)
{
  return MPI_Alltoall(sendbuf,sendcount,sendtype,recvbuf,recvcount,recvtype,
                      comm);
}
inline void Wait(int count, MPI_Request *request, int *sched=NULL)
{ 
  if(sched)
    MPI_Waitall(count,request,MPI_STATUSES_IGNORE);
}
#else
inline void Wait(int count, MPI_Request *request, int *sched=NULL)
{ 
  if(sched)
    MPI_Waitall(count,request,MPI_STATUSES_IGNORE);
  else
    MPI_Wait(request,MPI_STATUS_IGNORE);
}
#endif

inline int Ialltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                     void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                     MPI_Comm comm, MPI_Request *request, int *sched=NULL)
{
  if(!sched)
    return MPI_Ialltoall(sendbuf,sendcount,sendtype,recvbuf,recvcount,recvtype,
                         comm,request);
  else {
    int size;
    int rank;
    MPI_Comm_size(comm,&size);
    MPI_Comm_rank(comm,&rank);
    int sendsize;
    MPI_Type_size(sendtype,&sendsize);
    sendsize *= sendcount;
    int recvsize;
    MPI_Type_size(recvtype,&recvsize);
    recvsize *= recvcount;
    for(int p=0; p < size; ++p) {
      int P=sched[p];
      if(P != rank) {
        MPI_Irecv((char *) recvbuf+P*sendsize,sendcount,sendtype,P,0,comm,
                  request+(P < rank ? P : P-1));
        MPI_Request srequest;
        MPI_Isend((char *) sendbuf+P*recvsize,recvcount,recvtype,P,0,comm,
                  &srequest);
        MPI_Request_free(&srequest);
      }
    }
  
    memcpy((char *) recvbuf+rank*recvsize,(char *) sendbuf+rank*sendsize,
           sendsize);
    return 0;
  }
}

inline int Alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                    void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                    MPI_Comm comm, MPI_Request *request, int *sched=NULL)
{
  if(!sched)
    return MPI_Alltoall(sendbuf,sendcount,sendtype,recvbuf,recvcount,recvtype,
                        comm);
  else {
    int size;
    MPI_Comm_size(comm,&size);
    Ialltoall(sendbuf,sendcount,sendtype,recvbuf,recvcount,recvtype,comm,
              request,sched);
    MPI_Waitall(size-1,request,MPI_STATUSES_IGNORE);
    return 0;
  }
}

#endif
