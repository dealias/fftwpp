#ifndef __mpitranspose_h__
#define __mpitranspose_h__ 1

using namespace std; //****

/* 
Globally transpose an N x M matrix of blocks of L complex elements
distributed over the second dimension.
Here "in" is a local N x m matrix and "out" is a local n x M matrix.
Currently, both N and M must be divisible by the number of processors.

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
   wait1();

   Guru Interface:
    
   transpose2(data);
   // User computation 0
   wait0(); // Typically longest when intranspose=false
   // User computation 1      
   wait2(); // Typically longest when intranspose=true
*/  
  
#include <mpi.h>
#include <cstring>
#include <typeinfo>
#include "../Complex.h"
#include "../fftw++.h"

namespace fftwpp {

extern double safetyfactor; // For conservative latency estimate.
extern bool overlap; // Allow overlapped communication.

template<class T>
inline void copy(T *from, T *to, unsigned int length, unsigned int threads=1)
{
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif  
for(unsigned int i=0; i < length; ++i)
  to[i]=from[i];
}

// Copy count blocks spaced stride apart to contiguous blocks in dest.
template<class T>
inline void copytoblock(T *src, T *dest,
                        unsigned int count, unsigned int length,
                        unsigned int stride, unsigned int threads=1)
{
  for(unsigned int i=0; i < count; ++i)
    copy(src+i*stride,dest+i*length,length,threads);
}

inline void transposeError(const char *text) {
  std::cout << "Cannot construct " << text << " transpose plan." << std::endl;
  exit(-1);
}

void fill1_comm_sched(int *sched, int which_pe, int npes);

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

inline int localsize(int N, int size)
{
  int n=ceilquotient(N,size);
  size=N/n;
  if(N > n*size) ++size;
  return size;
}

inline int localstart(int N, int rank, int size)
{
  return ceilquotient(N,size)*rank;
}

inline int localdimension(int N, int rank, int size)
{
  int n=ceilquotient(N,size);
  int extra=N-n*rank;
  if(extra < 0) extra=0;
  if(n > extra) n=extra;
  return n;
}

inline int Ialltoallv(void *sendbuf, int *sendcounts, int *senddisplacements,
                      void *recvbuf, int *recvcounts, int *recvdisplacements,
                      MPI_Comm comm, MPI_Request *request, int *sched=NULL)
{
  return MPI_Ialltoallv(sendbuf,sendcounts,senddisplacements,MPI_BYTE,recvbuf,
                        recvcounts,recvdisplacements,MPI_BYTE,comm,request);
}
  
inline int Ialltoall(void *sendbuf, int sendcount,
                     void *recvbuf, int recvcount,
                     MPI_Comm comm, MPI_Request *request, int *sched=NULL)
{
//  if(!sched)
//    return MPI_Ialltoall(sendbuf,sendcount,MPI_BYTE,recvbuf,recvcount,MPI_BYTE,
    return MPI_Alltoall(sendbuf,sendcount,MPI_BYTE,recvbuf,recvcount,MPI_BYTE,
//                         comm,request);
                         comm);
    /*
  else {
    int size;
    int rank;
    MPI_Comm_size(comm,&size);
    MPI_Comm_rank(comm,&rank);
    int sendsize;
    MPI_Type_size(MPI_BYTE,&sendsize);
    sendsize *= sendcount;
    int recvsize;
    MPI_Type_size(MPI_BYTE,&recvsize);
    recvsize *= recvcount;
    for(int p=0; p < size; ++p) {
      int P=sched[p];
      if(P != rank) {
        MPI_Irecv((char *) recvbuf+P*sendsize,sendcount,MPI_BYTE,P,0,comm,
                  request+(P < rank ? P : P-1));
        MPI_Request srequest;
        MPI_Isend((char *) sendbuf+P*recvsize,recvcount,MPI_BYTE,P,0,comm,
                  &srequest);
        MPI_Request_free(&srequest);
      }
    }
  
    memcpy((char *) recvbuf+rank*recvsize,(char *) sendbuf+rank*sendsize,
           sendsize);
    return 0;
  }
    */
}

inline int Alltoall(void *sendbuf, int sendcount,
                    void *recvbuf, int recvcount,
                    MPI_Comm comm, MPI_Request *request, int *sched=NULL)
{
  if(!sched)
    return MPI_Alltoall(sendbuf,sendcount,MPI_BYTE,recvbuf,recvcount,MPI_BYTE,
                        comm);
  else {
    int size;
    MPI_Comm_size(comm,&size);
    Ialltoall(sendbuf,sendcount,recvbuf,recvcount,comm,request,sched);
    MPI_Waitall(size-1,request,MPI_STATUSES_IGNORE);
    return 0;
  }
}
  
template<class T>
class mpitranspose {
//private:
public: // ****
  unsigned int N,m,n,M;
  unsigned int m0,mp;
  unsigned int L;
  T *data;
  T *work;
  unsigned int threads;
  MPI_Comm communicator;
  MPI_Comm global;
  bool allocated;
  MPI_Request *request;
  int size;
  int rank;
  int globalrank;
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
public:

  bool divisible(int size, unsigned int M, unsigned int N) {
    unsigned int usize=size;
    
    return !(usize > N || usize > M || N % usize != 0 || M % usize != 0);
  }
  
  void poll(T *sendbuf, T *recvbuf, unsigned int N) {
    unsigned int sN=sizeof(T)*N;
    MPI_Alltoall(sendbuf,sN,MPI_BYTE,recvbuf,sN,MPI_BYTE,split);
  }
  
  // Estimate typical bandwidth saturation message size
  double Latency() {
    static double latency=-1.0;
    if(size == 1) return 0.0;
    if(latency >= 0) return latency;
    
    unsigned int b=(unsigned int) sqrt(size);
    MPI_Comm_split(communicator,rank/b,0,&split);
    MPI_Comm_size(split,&splitsize);
    MPI_Comm_rank(split,&splitrank);
    
    unsigned int N1=2;
    unsigned int N2=10000;
    T send[N2*splitsize];
    T recv[N2*splitsize];
    for(unsigned int i=0; i < N2; ++i)
      send[N2*splitrank+i]=0.0;
    unsigned int M=100;
    double T1=0.0, T2=0.0;
    poll(send,recv,N1);
    poll(send,recv,N2);
    for(unsigned int i=0; i < M; ++i) {
      MPI_Barrier(split);
      double t0=totalseconds();
      poll(send,recv,N1);
      double t1=totalseconds();
      MPI_Barrier(split);
      double t2=totalseconds();
      poll(send,recv,N2);
      double t3=totalseconds();
      T1 += t1-t0;
      T2 += t3-t2;
    }
    latency=std::max(T1*(N2-N1)/(T2-T1)-N1,0.0)*sizeof(double);
    if(globalrank == 0)
      std::cout << "latency=" << latency << std::endl;
    MPI_Comm_free(&split); 
    return latency;
  }

  void setup(T *data) {
    allocated=false;
    Tin3=NULL;
    Tout3=NULL;
    
    MPI_Comm_size(communicator,&size);
    MPI_Comm_rank(communicator,&rank);
    
    MPI_Comm_rank(global,&globalrank);
    
    m0=localdimension(M,0,size);
    mp=localdimension(M,size-1,size);
/*    
    if(!divisible(size,M,N)) {
      if(globalrank == 0)
        std::cout << 
          "ERROR: Matrix dimensions must be divisible by number of processors" 
                  << std::endl << std::endl; 
      MPI_Finalize();
      exit(0);
    }
*/
    
    if(work == NULL) {
      Array::newAlign(this->work,N*m*L,sizeof(T));
      allocated=true;
    }
    
    if(size == 1) {
      a=1;
      return;
    }
    
    unsigned int AlltoAll=1;

    double latency=safetyfactor*Latency();
    unsigned int alimit=(N*M*L*sizeof(T) >= latency*size*size) ?
      2 : size;
    MPI_Bcast(&alimit,1,MPI_UNSIGNED,0,global);

    if(globalrank == 0)
      std::cout << std::endl << "Timing:" << std::endl;
    
    /*
    unsigned int A=0;
    double T0=DBL_MAX;
//    for(unsigned int alltoall=0; alltoall <= 1; ++alltoall) {
    for(unsigned int alltoall=1; alltoall <= 1; ++alltoall) {
      if(globalrank == 0) std::cout << "alltoall=" << alltoall << std::endl;
      for(a=1; a < alimit; a *= 2) {
        b=size/a;
        init(data,alltoall);
        double t=time(data);
        deallocate();
        if(globalrank == 0) {
          std::cout << "a=" << a << ":\ttime=" << t << std::endl;
          if(t < T0) {
            T0=t;
            A=a;
            AlltoAll=alltoall;
          }
        }
      }
    }
    
    unsigned int parm[]={A,AlltoAll};
    MPI_Bcast(&parm,2,MPI_UNSIGNED,0,global);
    A=parm[0];
    AlltoAll=parm[1];
    */
    
    unsigned A=1; // *****
    
    a=A;
    b=size/a;
    
    if(globalrank == 0) std::cout << std::endl << "Using alltoall=" << AlltoAll
                            << ", a=" << a << ":" << std::endl;
    
    init(data,AlltoAll);
  }
  
  mpitranspose(){}

  // Here "in" is a local N x m matrix and "out" is a local n x M matrix.
  // work is a temporary array of size N*m*L.
  mpitranspose(unsigned int N, unsigned int m, unsigned int n,
               unsigned int M, unsigned int L,
               T *data, T *work=NULL,
               unsigned int threads=fftw::maxthreads,
               MPI_Comm communicator=MPI_COMM_WORLD,
               MPI_Comm global=0) :
    N(N), m(m), n(n), M(M), L(L), work(work), threads(threads),
    communicator(communicator), global(global ? global : communicator) {
    setup(data);
  }
  
  mpitranspose(unsigned int N, unsigned int m, unsigned int n,
               unsigned int M, unsigned int L,
               T *data, MPI_Comm communicator, MPI_Comm global=0) :
    N(N), m(m), n(n), M(M), L(L), work(NULL), threads(fftw::maxthreads),
    communicator(communicator), global(global ? global : communicator) {
    setup(data);
  }
    
  mpitranspose(unsigned int N, unsigned int m, unsigned int n,
               unsigned int M, T *data, T *work=NULL,
               unsigned int threads=fftw::maxthreads,
               MPI_Comm communicator=MPI_COMM_WORLD, MPI_Comm global=0) :
        N(N), m(m), n(n), M(M), L(1), work(work), threads(threads),
        communicator(communicator), global(global ? global : communicator) {
    setup(data);
  }
    
  double time(T *data) {
    double sum=0.0;
    unsigned int N=1;
    transpose(data,true,false); // Initialize communication buffers

    double stop=totalseconds()+fftw::testseconds;
    for(;;++N) {
      int end;
      double start=rank == 0 ? totalseconds() : 0.0;
      transpose(data,true,false);
      
      
      if(globalrank == 0) {
        double t=totalseconds();
        double seconds=t-start;
        sum += seconds;
        end=t > stop;
      }
      MPI_Bcast(&end,1,MPI_INT,0,global);
      if(end)
        break;
    }

    return sum/N;
  }

  void init(T *data, bool alltoall) {
    if(n == 0) return;
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
    if(allocated)
      Array::deleteAlign(work,N*m*L);
  }
  
  void inphase0() {
    if(size == 1) return;
    /*
    unsigned int blocksize=sizeof(T)*n*(a > 1 ? b : a)*m*L;
//    Ialltoall(data,blocksize,work,blocksize,split2,request,sched2);
//    MPI_Barrier(split);
    
    std::cout << "rank=" << rank << "n=" << n << " m=" << m << " a=" << a << " b=" << b << std::endl;
    
    copy(data,work,n*M*L,threads); // ***
    Ialltoall(work,blocksize,data,blocksize,split2,request,sched2);
    return;
    */
    
  int *sendcounts=new int[size];
  int *senddisplacements=new int[size];
  int *recvcounts=new int[size];
  int *recvdisplacements=new int[size];
    
    {
      int ni;
      int mi;
      int M0=M;
      int N0=N;
      
      int Si=0;
      int Ri=0;
    
      size_t S=sizeof(T);
      
      for(int i=0; i < size; ++i) {
        ni=ceilquotient(N0,size-i);
        mi=ceilquotient(M0,size-i);
        N0 -= ni;
        M0 -= mi;
        int si=m*ni*S;
        int ri=n*mi*S;
        sendcounts[i]=si;
        recvcounts[i]=ri;
        senddisplacements[i]=Si;
        recvdisplacements[i]=Ri;
        Si += si;
        Ri += ri;
      }
    }

    Ialltoallv(data,sendcounts,senddisplacements,work,recvcounts,
               recvdisplacements,split2,request,sched2);
  }
  
  void inphase1() {
    if(a > 1) {
      Tin2->transpose(work,data); // a x n*b x m*L
      unsigned int blocksize=sizeof(T)*n*a*m*L;
      Ialltoall(data,blocksize,work,blocksize,split,request,sched);
    }
  }

  void insync0() {
    if(size == 1) return;
    Wait(split2size-1,request,sched2);
  }
  
  void insync1() {
    if(a > 1)
      Wait(splitsize-1,request,sched);
  }

  void inpost() {
    if(size == 1) return;
    // Divisible case:
    if(m0 == mp)
      Tin1->transpose(work,data); // b x n*a x m*L
    else  {
    // Indivisible case:
      unsigned int lastblock=mp*L;
      unsigned int block=m0*L;
      unsigned int cols=n*a;
      unsigned int istride=cols*block;
      unsigned int last=b-1;
      unsigned int ostride=last*block+lastblock;

      for(unsigned int j=0; j < cols; ++j) {
        Complex *dest=data+j*ostride;
        copytoblock(work+j*block,dest,last,block,istride);
        copy(work+j*lastblock+last*istride,dest+last*block,lastblock);
      }
    }
  }
  
  void outphase0() {
    if(size == 1) return;
  
    // Inner transpose a N/a x M/a matrices over each team of b processes
    Tout1->transpose(data,work); // n*a x b x m*L
    unsigned int blocksize=sizeof(T)*n*a*m*L;
    Ialltoall(work,blocksize,data,blocksize,split,request,sched);
  }
  
  void outphase1() {
    if(a > 1) {
      // Outer transpose a x a matrix of N/a x M/a blocks over a processes
      Tout2->transpose(data,work); // n*b x a x m*L
      unsigned int blocksize=sizeof(T)*n*b*m*L;
      Ialltoall(work,blocksize,data,blocksize,split2,request,sched2);
    }
  }
  
  void outsync0() {
    if(a > 1)
    Wait(splitsize-1,request,sched);
  }
  
  void outsync1() {
    if(size > 1)
      Wait(split2size-1,request,sched2);
  }
    

  void nMTranspose() {
    if(!Tin3) Tin3=new Transpose(n,M,L,data,work,threads);
    Tin3->transpose(data,work); // n X M x L
    copy(work,data,n*M*L,threads);
  }
  
  void NmTranspose() {
    if(!Tout3) Tout3=new Transpose(N,m,L,data,work,threads);
    Tout3->transpose(data,work); // N x m x L
    copy(work,data,N*m*L,threads);
  }
  
  void Wait0() {
    if(inflag) {
      outsync0();
      outphase1();
    } else {
      insync0();
      inphase1();
    }
   }
  
  void Wait1() {
    if(inflag) {
      outsync1();
      if(outflag) NmTranspose();
    } else {
      insync1();
      inpost();
      if(!outflag) nMTranspose();
    }
   }
  
  void wait0() {
    if(overlap) Wait0();
  }
  
  void wait1() {
    if(overlap) {
      if(!inflag) Wait0();
      Wait1();
    }
  }
  
  void wait2() {
    if(overlap) Wait1();
  }
  
  void transpose(T *data, bool intranspose=true, bool outtranspose=true) {
    transpose1(data,intranspose,outtranspose);
    if(overlap) {
      if(!inflag) Wait0();
      Wait1();
    }
  }
  
  void transpose1(T *data, bool intranspose=true, bool outtranspose=true) {
    inflag=intranspose;
    transpose2(data,intranspose,outtranspose);
    if(inflag)
      wait0();
  }
  
  void transpose2(T *Data, bool intranspose=true, bool outtranspose=true) {
    data=Data;
    inflag=intranspose;
    outflag=outtranspose;
    if(inflag)
      outphase0();
    else
      inphase0();
    if(!overlap) {
      Wait0();
      Wait1();
    }
  }
  
};

} // end namespace fftwpp

#endif
