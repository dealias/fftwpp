#ifndef __mpitranspose_h__
#define __mpitranspose_h__ 1

#include <mpi.h>
#include "../Complex.h"
#include <fftw++.h>
#include <cstring>

const double latency=4096.0; // Typical bandwidth saturation message size

namespace fftwpp {

inline void transposeError(const char *text) {
  std::cout << "Cannot construct " << text << " transpose plan." << std::endl;
  exit(-1);
}

void LoadWisdom(MPI_Comm *active);
void SaveWisdom(MPI_Comm *active);

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
  unsigned int threads;
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
  bool inflag,outflag;
  static bool overlap;
public:
  void setup(Complex *data) {
    allocated=false;
    Tin3=NULL;
    Tout3=NULL;
    
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
      this->work=ComplexAlign(N*m*L);
      allocated=true;
    }
    
    if(size == 1) {
      a=1;
      return;
    }
    
    unsigned int AlltoAll=1;
    unsigned int alimit=(N*M*L*sizeof(Complex) >= latency*size*size) ?
      2 : usize;
    
    if(rank == 0)
      std::cout << "Timing:" << std::endl;
    
    unsigned int A=0;
    double T0=DBL_MAX;
    for(unsigned int alltoall=0; alltoall <= 1; ++alltoall) {
      if(rank == 0) std::cout << "alltoall=" << alltoall << std::endl;
      for(a=1; a < alimit; a *= 2) {
        b=size/a;
        init(data,alltoall);
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
    
    if(rank == 0) std::cout << std::endl << "Using alltoall=" << AlltoAll
                            << ", a=" << a << ":" << std::endl;
    
    init(data,AlltoAll);
  }
  
  mpitranspose(unsigned int N, unsigned int m, unsigned int n,
               unsigned int M, unsigned int L,
               Complex *data, Complex *work=NULL,
               unsigned int threads=fftw::maxthreads,
               MPI_Comm communicator=MPI_COMM_WORLD) :
    N(N), m(m), n(n), M(M), L(L), work(work), threads(threads),
    communicator(communicator) {
    setup(data);
  }
  
  
  mpitranspose(unsigned int N, unsigned int m, unsigned int n,
               unsigned int M, Complex *data, Complex *work=NULL,
               unsigned int threads=fftw::maxthreads,
               MPI_Comm communicator=MPI_COMM_WORLD) :
        N(N), m(m), n(n), M(M), L(1), work(work), threads(threads),
    communicator(communicator) {
    setup(data);
  }
    
  double time(Complex *data) {
    double sum=0.0;
    unsigned int N=1;
    transpose(data,true,false); // Initialize communication buffers
    double stop=totalseconds()+fftw::testseconds;
    for(;;++N) {
      int end;
      double start=rank == 0 ? totalseconds() : 0.0;
      transpose(data,true,false);
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

  void init(Complex *data, bool alltoall) {
    Tout1=new Transpose(n*a,b,m*L,data,this->work,threads);
    Tin1=new Transpose(b,n*a,m*L,data,this->work,threads);

    if(a > 1) {
      Tin2=new Transpose(a,n*b,m*L,data,this->work,threads);
      Tout2=new Transpose(n*b,a,m*L,data,this->work,threads);
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
      int final;
      MPI_Finalized(&final);
      if(!final) {
        MPI_Comm_free(&split2); 
        MPI_Comm_free(&split); 
      }
      delete Tout2;
      delete Tin2;
    }
    delete Tin1;
    delete Tout1;
    
  }
  
  ~mpitranspose() {
    deallocate();
    if(Tout3) delete Tout3;
    if(Tin3) delete Tin3;
    if(allocated) deleteAlign(work);
  }
  
  void inphase0(Complex *data);
  void inphase1(Complex *data);
  
  void insync0(Complex *data);
  void insync1(Complex *data);
  
  void inpost(Complex *data);
  
  void outphase0(Complex *data);
  void outphase1(Complex *data);
  
  void outsync0(Complex *data);
  void outsync1(Complex *data);
  
  void nMTranspose(Complex *data);
  void NmTranspose(Complex *data);
  
 /* Globally transpose data, including local transposition
    Beginner Interface:
    
    transpose(data);              n x M -> m x N
    
    To globally transpose data without local transposition of output:
    
    transpose(data,true,false);   n x M -> N x m
     
    To globally transpose data without local transposition of input:
    
    transpose(data,false,true);   N x m -> n x M
    
    To globally transpose data without local transposition of input or output:
    
    transpose(data,false,false);  N x m -> M x n
    
    Advanced Interface:
    
    transpose1(data);
    // User computation
    wait1(data);

    Guru Interface:
    
    transpose2(data);
    // User computation 0
    wait0(data);
    // User computation 1      
    wait1(data);
*/  
  
  void Wait0(Complex *data) {
    if(inflag) {
      outsync0(data);
      outphase1(data);
    } else {
      insync0(data);
      inphase1(data);
    }
   }
  
  void Wait1(Complex *data) {
    if(inflag) {
      outsync1(data);
      if(outflag) NmTranspose(data);
    } else {
      insync1(data);
      inpost(data);
      if(!outflag) nMTranspose(data);
    }
   }
  
  void wait0(Complex *data) {
    if(overlap) Wait0(data);
  }
  void wait1(Complex *data) {
    if(overlap) Wait1(data);
  }
  
  void transpose(Complex *data, bool intranspose=true, bool outtranspose=true) {
    transpose1(data,intranspose,outtranspose);
    wait1(data);
  }
  
  void transpose1(Complex *data, bool intranspose=true, bool outtranspose=true) {
    transpose2(data,intranspose,outtranspose);
    wait0(data);
  }
  
  void transpose2(Complex *data, bool intranspose=true, bool outtranspose=true) {
    inflag=intranspose;
    outflag=outtranspose;
    if(inflag)
      outphase0(data);
    else
      inphase0(data);
    if(!overlap) {
      Wait0(data);
      Wait1(data);
    }
  }
  
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
