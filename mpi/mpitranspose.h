#ifndef __mpitranspose_h__
#define __mpitranspose_h__ 1

#include <mpi.h>
#include "../Complex.h"
#include <fftw++.h>
#include <cstring>

const int latency=4096; // Typical bandwidth saturation message size

namespace fftwpp {

inline void transposeError(const char *text) {
  std::cout << "Cannot construct " << text << " transpose plan." << std::endl;
  exit(-1);
}

void LoadWisdom(const MPI_Comm& active);
void SaveWisdom(const MPI_Comm& active);

void fill1_comm_sched(int *sched, int which_pe, int npes);

// Globally transpose an N x M matrix of blocks of L complex elements
// distributed over the second dimension.
// Here "in" is a local N x m matrix and "out" is a local n x M matrix.
// Currently, both N and M must be divisible by the number of processors.
// work is a temporary array of size N*m*L.
class mpitranspose {
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
  Transpose *Tin1,*Tin2,*Tin3;
  Transpose *Tout1,*Tout2,*Tout3;
  unsigned int a,b;
public:
  mpitranspose(Complex *data, unsigned int N, unsigned int m, unsigned int n,
               unsigned int M, unsigned int L=1,
               Complex *work=NULL, unsigned int threads=fftw::maxthreads,
               MPI_Comm communicator=MPI_COMM_WORLD) :
    N(N), m(m), n(n), M(M), L(L), work(work), communicator(communicator),
    allocated(false) {
    MPI_Comm_size(communicator,&size);
    MPI_Comm_rank(communicator,&rank);
    
    unsigned int usize=size;
    if(usize > N || usize > M || N % usize != 0 || M % usize != 0) {
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
    
    Tout3=new Transpose(data,this->work,N,m,L,threads);
    
    if(size == 1)
      return;
    
    unsigned int K=0.5*log2(size)+0.5;
    a=(unsigned int ) 1 << K;
    b=size/a;
    
    unsigned int AlltoAll=1;
    if(N*M*L*sizeof(Complex) >= latency*size*(size-a-b)) {
      a=1;
      b=size;
    } else {
      if(rank == 0)
        std::cout << "Timing:" << std::endl;
    
      a=1;
      b=size;
      init(data,true,threads);
      OutTransposed(data); // Initialize communication buffers
      if(threads > 1) {
        deallocate();
        init(data,false,threads);
        OutTransposed(data); // Initialize communication buffers
      }
    
      unsigned int A=0;
      double T0=DBL_MAX;
      bool first=true;
      for(unsigned int alltoall=threads == 1; alltoall <= 1; ++alltoall) {
        if(rank == 0) std::cout << "alltoall=" << alltoall << std::endl;
        for(a=1; a < usize; a *= 2) {
          b=size/a;
          if(!first)
            init(data,alltoall,threads);
          first=false;
          double T=time(data);
          deallocate();
          if(rank == 0) {
            std::cout << "a=" << a << ":\tT=" << T << std::endl;
            if(T < T0) {
              T0=T;
              A=a;
              AlltoAll=alltoall;
            }
          }
        }
      }

      unsigned int parm[]={A,AlltoAll};
      MPI_Bcast(&parm,2,MPI_UNSIGNED,0,communicator);
      A=parm[0];
      AlltoAll=parm[1];
      a=A;
      b=size/a;
    }
    
    if(rank == 0) std::cout << std::endl << "Using alltoall=" << AlltoAll
                            << ", a=" << a << ":" << std::endl;
    
    init(data,AlltoAll,threads);
  }
  
  double time(Complex *data) {
    double sum=0.0;
    unsigned int N=1;
    double stop=totalseconds()+fftw::testseconds;
    for(;;++N) {
      int end;
      double start=rank == 0 ? totalseconds() : 0.0;
      OutTransposed(data);
      if(rank == 0) {
        double t=totalseconds();
        double seconds=t-start;
        sum += seconds;
        end=t > stop;
      }
      MPI_Bcast(&end,1,MPI_INT,0,communicator);
      if(end)
        break;
    }
    return sum/N;
  }

  void init(Complex *data, bool alltoall, unsigned int threads) {
    Tout1=new Transpose(data,this->work,n*a,b,m*L,threads);

    if(a == 1) {
      Tin1=new Transpose(data,this->work,b,n*a,m*L,threads);
    } else {
      Tin1=new Transpose(data,this->work,m*a,b,n*L,threads);
      Tin2=new Transpose(data,this->work,m*b,a,n*L,threads);
      Tin3=new Transpose(data,this->work,M,n,L,threads);
      Tout2=new Transpose(data,this->work,n*b,a,m*L,threads);
    }
    
    if(size == 1) return;
    
    if(a == 1) {
      split=split2=communicator;
      splitsize=split2size=size;
      splitrank=split2rank=rank;
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
        sched2=sched;
    }
  }
  
  void deallocate() {
    if(size == 1) return;
    if(sched) {
      if(a > 1) delete[] sched2;
      delete[] sched;
    }
    delete[] request;
    if(a > 1) {
//      MPI_Comm_free(&split2); 
//      MPI_Comm_free(&split); 
      
      delete Tout2;
      delete Tin3;
      delete Tin2;
    }
    delete Tin1;
    delete Tout1;
    
  }
  
  ~mpitranspose() {
    deallocate();
    delete Tout3;
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

} // end namespace fftwpp

#endif
