#ifndef __mpitranspose_h__
#define __mpitranspose_h__ 1

/* 
   Globally transpose an N x M matrix of blocks of L words of type T.
   The out-of-place versions preserve inputs.

   Blocking in-place and out-of-place interfaces. Upper case letters denote
   global dimensions; lower case letters denote distributed dimensions: 
   
   To globally transpose without local transposition of output:
   localize0(in);      n x M -> N x m
   localize0(in,out);  n x M -> N x m
   
   To globally transpose without local transposition of input:
   localize1(in);      N x m -> n x M
   localize1(in,out);  N x m -> n x M
     
   Non-blocking interface for localize0 (and similarly for localize1):
    
   ilocalize0(in);
   // User computation
   wait();

   Double non-blocking interface:
    
   ilocalize0(in);
   // User computation 0 (typically longest)
   wait0();
   // User computation 1      
   wait1();
*/  
  
#include <mpi.h>
#include <cstring>
#include <typeinfo>
#include <cfloat>
#include "Complex.h"
#include "seconds.h"
#include "Array.h"
#include "utils.h"
#include "align.h"
#include "transposeoptions.h"
#include "fftw++.h"

namespace utils {

extern double safetyfactor; // For conservative latency estimate.
extern bool overlap; // Allow overlapped communication.
extern double testseconds; // Limit for transpose timing tests
extern mpiOptions defaultmpiOptions;

template<class T>
inline void copy(const T *from, T *to, unsigned int length,
                 unsigned int threads=1)
{
  PARALLEL(
    for(unsigned int i=0; i < length; ++i)
      to[i]=from[i];
    );
}

// Copy count blocks spaced stride apart to contiguous blocks in dest.
template<class T>
inline void copytoblock(const T *src, T *dest,
                        unsigned int count, unsigned int length,
                        unsigned int stride, unsigned int threads=1)
{
  PARALLEL(
    for(unsigned int i=0; i < count; ++i)
      copy(src+i*stride,dest+i*length,length);
    );
}

// Copy count blocks spaced stride apart from contiguous blocks in src.
template<class T>
inline void copyfromblock(const T *src, T *dest,
                          unsigned int count, unsigned int length,
                          unsigned int stride, unsigned int threads=1)
{
  PARALLEL(
    for(unsigned int i=0; i < count; ++i)
      copy(src+i*length,dest+i*stride,length);
    );
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
inline void Wait(int count, MPI_Request *request, bool schedule)
{ 
  if(schedule)
    MPI_Waitall(count,request,MPI_STATUSES_IGNORE);
}
inline void Wait(MPI_Request *request)
{ 
}
#else
inline void Wait(int count, MPI_Request *request, bool schedule)
{ 
  if(schedule)
    MPI_Waitall(count,request,MPI_STATUSES_IGNORE);
  else
    MPI_Wait(request,MPI_STATUS_IGNORE);
}
inline void Wait(MPI_Request *request)
{ 
  MPI_Wait(request,MPI_STATUS_IGNORE);
}
#endif

class localdimension {
public:
  int n;
  int start;
  
  localdimension(int N, int rank, int size) {
    n=utils::ceilquotient(N,size);
    start=n*rank;
    int extra=N-start;
    if(extra < 0) extra=0;
    if(n > extra) n=extra;
  }
};

inline int Ialltoall(void *sendbuf, int count, void *recvbuf,
                     MPI_Comm comm, MPI_Request *request, int *sched=NULL,
                     unsigned int threads=1)
{
  if(!sched)
    return MPI_Ialltoall(sendbuf == recvbuf ? MPI_IN_PLACE : sendbuf,
                         count,MPI_BYTE,recvbuf,count,MPI_BYTE,comm,
                         request);
  else {
    int size;
    int rank;
    MPI_Comm_size(comm,&size);
    MPI_Comm_rank(comm,&rank);
    MPI_Request *srequest=request+size-1;
    for(int p=0; p < size; ++p) {
      int P=sched[p];
      if(P != rank) {
        int index=P < rank ? P : P-1;
        MPI_Irecv((char *) recvbuf+P*count,count,MPI_BYTE,P,0,comm,
                  request+index);
        MPI_Isend((char *) sendbuf+P*count,count,MPI_BYTE,P,0,comm,
                  srequest+index);
      }
    }
  
    int offset=rank*count;
    copy((char *) sendbuf+offset,(char *) recvbuf+offset,count,threads);
    return 0;
  }
}

template<class T>
class mpitranspose {
private:
  unsigned int N,M,n,m;
  unsigned int L;
  T *input,*output,*work;
  MPI_Comm communicator;
  mpiOptions options;
  MPI_Comm global;
  double latency;
  MPI_Comm block;
  
  unsigned int n0,m0;
  unsigned int np,mp;
  int mlast,nlast,last;
  unsigned int threads;
  unsigned int allocated;
  MPI_Request *request;
  MPI_Request *Request;
  int size;
  int rank;
  int globalrank;
  int splitsize;
  int splitrank;
  int split2size;
  int split2rank;
  int *sched, *sched1, *sched2;
  MPI_Comm split;
  MPI_Comm split2;
  MPI_Comm splitv;
  fftwpp::Transpose *Tin1,*Tin2;
  fftwpp::Transpose *Tout1,*Tout2;
  int a,b;
  bool outflag;
  bool uniform;
  bool subblock;
  bool compact;
  bool schedule;
public:

  mpiOptions Options() {return options;}
  
  bool divisible(int size, unsigned int M, unsigned int N) {
    unsigned int usize=size;
    return usize <= N && usize <= M && N % usize == 0 && M % usize == 0;
  }
  
  void poll(T *sendbuf, T *recvbuf, unsigned int N) {
    unsigned int sN=sizeof(T)*N;
    MPI_Alltoall(sendbuf == recvbuf ? MPI_IN_PLACE : sendbuf,
                 sN,MPI_BYTE,recvbuf,sN,MPI_BYTE,split);
  }
  
  // Estimate typical bandwidth saturation message size
  double Latency() {
    if(size == 1) return 0.0;
    if(latency >= 0) return latency;
    
    int b=sqrt(size)+0.5;
    MPI_Comm_split(communicator,rank/b,0,&split);
    MPI_Comm_size(split,&splitsize);
    MPI_Comm_rank(split,&splitrank);
    
    unsigned int N1=2;
    unsigned int N2=10000;
    T send[N2*splitsize];
    T recv[N2*splitsize];
    for(unsigned int i=0; i < N2; ++i)
      send[N2*splitrank+i]=0.0;
    unsigned int M=1000;
    double T1=0.0, T2=0.0;
    poll(send,recv,N1);
    poll(send,recv,N2);
    for(unsigned int i=0; i < M; ++i) {
      MPI_Barrier(split);
      double t0=utils::totalseconds();
      poll(send,recv,N1);
      MPI_Barrier(split);
      double t1=utils::totalseconds();
      poll(send,recv,N2);
      MPI_Barrier(split);
      double t2=utils::totalseconds();
      T1 += t1-t0;
      T2 += t2-t1;
    }
    latency=std::max(T1*(N2-N1)/(T2-T1)-N1,0.0)*sizeof(T);
    if(globalrank == 0 && options.verbose)
      std::cout << std::endl << "latency=" << latency << std::endl;
    MPI_Comm_free(&split); 
    return latency;
  }

  void setup(T *data, MPI_Comm Communicator) {
    if(N < n) Array::ArrayExit("N must be >= n");
    if(M < m) Array::ArrayExit("M must be >= m");

    threads=options.threads;
    MPI_Comm_size(Communicator,&size);
    MPI_Comm_rank(Communicator,&rank);
    
    MPI_Comm_rank(global,&globalrank);
    
    n0=localdimension(N,0,size).n;
    nlast=std::min((int) utils::ceilquotient(N,n0),size)-1;
    np=localdimension(N,nlast,size).n;
    
    m0=localdimension(M,0,size).n;
    mlast=std::min((int) utils::ceilquotient(M,m0),size)-1;
    mp=localdimension(M,mlast,size).n;
    
    allocated=0;
    if(size == 1) {
      a=1;
      subblock=false;
      return;
    }
    
    int Pbar=std::min(nlast+(n0 == np),mlast+(m0 == mp));
    size=std::max(nlast+1,mlast+1);
    MPI_Comm_split(Communicator,rank < size,0,&communicator);
    
    bool Uniform=divisible(size,M,N);
    
    int start=0,stop=Uniform ? 2 : 1;
    if(options.alltoall > stop) options.alltoall=stop;
    if(options.alltoall >= 0)
      start=stop=options.alltoall;
    int Alltoall=1;
    if(options.a >= size)
      options.a=-1;
      
    if(globalrank == 0 && options.verbose)
      std::cout << std::endl << "Initializing " << N << "x" << M
                << " transpose of " << L*sizeof(T) << "-byte elements over " 
                << size << " processes." << std::endl;
      
    int alimit;
    
    if(options.a <= 0) {
      double latency=safetyfactor*Latency();
      if(globalrank == 0) {
        if(N*M*L*sizeof(T) < latency*Pbar*Pbar) {
          if(options.a < 0) {
            int n=sqrt(Pbar)+0.5;
            options.a=Pbar/n;
            alimit=options.a+1;
          } else {
            options.a=1;
            alimit=(int) (sqrt(Pbar)+1.5);
          }
        } else { // Enforce a=1 if message length > latency.
          alimit=2;
          options.a=1;
        }
      }
      MPI_Bcast(&alimit,1,MPI_UNSIGNED,0,global);
      MPI_Bcast(&options.a,1,MPI_INT,0,global);
    } else alimit=options.a+1;
    
    int astart=options.a;
    if(alimit > astart+1 || stop-start >= 1) {
      if(globalrank == 0 && options.verbose)
        std::cout << std::endl << "Timing:" << std::endl;
      
      double T0=DBL_MAX;
      for(int alltoall=start; alltoall <= stop; ++alltoall) {
        if(globalrank == 0 && options.verbose)
          std::cout << "alltoall=" << alltoall << std::endl;
        unsigned int maxscore=0;
        // Only consider a,b values that yield largest possible submatrix.
        for(a=std::max(2,astart); a < alimit; a++) {
          b=Pbar/a;
          unsigned int ab=a*b;
          unsigned int score=ab*(N/ab)*ab*(M/ab);
          if(score > maxscore) maxscore=score;
        }
        for(a=astart; a < alimit; a++) {
          b=Pbar/a;
          unsigned int ab=a*b;
          if(a > 1 && ab*(N/ab)*ab*(M/ab) < maxscore) continue;
          options.alltoall=alltoall;
          uniform=Uniform && a*b == size;
          if(start < 2 && alltoall == 2 && !Uniform) continue;
          init(data);
          double t=time(data);
          deallocate();
          if(globalrank == 0) {
            if(options.verbose)
              std::cout << "a=" << a << ":\ttime=" << t << std::endl;
            if(t < T0) {
              T0=t;
              options.a=a;
              Alltoall=alltoall;
            }
          }
        }
      }
      int parm[]={options.a,Alltoall};
      MPI_Bcast(&parm,2,MPI_INT,0,global);
      options.a=parm[0];
      options.alltoall=parm[1];
    }
    
    a=options.a;
    b=a > 1 || Uniform ? Pbar/a : Pbar+1; 
    if(b == 1) {b=a; a=1;}
    
    if(globalrank == 0 && options.verbose)
      std::cout << std::endl << "Using alltoall=" << 
        options.alltoall << ", a=" << a << ", b=" << b << ":" << std::endl;

    uniform=Uniform && a*b == size;
    init(data);
  }
  
  mpitranspose(){}

  // data and work are arrays of size max(n*M,N*m)*L.
  mpitranspose(unsigned int N, unsigned int M, unsigned int n, unsigned int m,
               unsigned int L, T *data, T *work=NULL,
               MPI_Comm communicator=MPI_COMM_WORLD,
               const mpiOptions& options=defaultmpiOptions,
               MPI_Comm global=0) :
    N(N), M(M), n(n), m(m), L(L), work(work),
    options(options), global(global ? global : communicator), latency(-1) {
    setup(data,communicator);
  }
  
  mpitranspose(unsigned int N, unsigned int M, unsigned int n, unsigned int m,
               unsigned int L, T *data, MPI_Comm communicator=MPI_COMM_WORLD,
               const mpiOptions& options=defaultmpiOptions,
               MPI_Comm global=0) :
    N(N), M(M), n(n), m(m), L(L), work(NULL),
    options(options), global(global ? global : communicator), latency(-1) {
    setup(data,communicator);
  }
    
  mpitranspose(unsigned int N, unsigned int M, unsigned int n,
               unsigned int m, T *data, T *work=NULL,
               MPI_Comm communicator=MPI_COMM_WORLD,
               const mpiOptions& options=defaultmpiOptions,
               MPI_Comm global=0) :
    N(N), M(M), n(n), m(m), L(1), work(work),
    options(options), global(global ? global : communicator), latency(-1) {
    setup(data,communicator);
  }
    
  double time(T *data) {
    double sum=0.0;
    unsigned int N=1;
    localize0(data); // Initialize communication buffers
    double stop=utils::totalseconds()+testseconds;
    for(;;++N) {
      int end;
      double start=rank == 0 ? utils::totalseconds() : 0.0;
      localize0(data);
      if(rank == 0) {
        double t=utils::totalseconds();
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

  // Return size of request array
  int Size(int start) {return size-(rank >= start ? 1 : start);}
  
  void init(T *data) {
    compact=uniform && options.alltoall == 2;
    
    if(compact) work=data;
    else {
      if(work == NULL) {
        allocated=std::max(n*M,N*m)*L;
        Array::newAlign(work,allocated,sizeof(T));
      }
    }
    
    subblock=a > 1 && rank < a*b;
    
    if(uniform) {
      Tin1=new fftwpp::Transpose(b,n*a,m*L,data,work,threads);
      Tout1=new fftwpp::Transpose(n*a,b,m*L,data,work,threads);
    } else {
      Tin1=NULL;
      Tout1=NULL;      
    }
    
    if(subblock) {
      Tin2=new fftwpp::Transpose(a,n*b,m*L,data,work,threads);
      Tout2=new fftwpp::Transpose(n*b,a,m*L,data,work,threads);
    } else {
      Tin2=NULL;
      Tout2=NULL;
    }
    
    if(uniform)
      splitv=communicator;
    else {
      last=std::min(nlast,mlast);
      MPI_Comm_split(communicator,rank < last,0,&splitv);
    }
    
    if(a == 1) {
      split=split2=communicator;
      splitsize=split2size=size;
      splitrank=split2rank=rank;
    } else {
      MPI_Comm_split(communicator,rank < a*b,0,&block);
      
      if(rank < a*b) {
        MPI_Comm_split(block,rank/b,0,&split);
        MPI_Comm_size(split,&splitsize);
        MPI_Comm_rank(split,&splitrank);
      
        MPI_Comm_split(block,rank % b,0,&split2);
        MPI_Comm_size(split2,&split2size);
        MPI_Comm_rank(split2,&split2rank);
      } else {
        split=split2=block;
        splitsize=split2size=size;
        splitrank=split2rank=rank;
      }
    }
    
    schedule=!options.alltoall || (!uniform && a > 1);
    if(schedule) {
      Request=new MPI_Request[2*(std::max(splitsize,split2size)-1)];
      if(!uniform)
        request=new MPI_Request[2*Size(a > 1 ? a*b : 0)];
    
      if(uniform || subblock) {
        sched2=new int[split2size];
        fill1_comm_sched(sched2,split2rank,split2size);
        sched1=new int[splitsize];
        fill1_comm_sched(sched1,splitrank,splitsize);
      } else
        sched1=sched2=sched;
    } else {
      Request=new MPI_Request[1];
      sched1=sched2=NULL;
      if(!uniform)
        request=new MPI_Request[2*Size(last)+1];
    }
    
    if(!uniform) {
      sched=new int[size];
      fill1_comm_sched(sched,rank,size);
    }
  }
  
  void deallocate() {
    if(size == 1) return;
    
    if(compact) work=NULL;
    else if(allocated) {
      Array::deleteAlign(work,allocated);
      work=NULL;
      allocated=0;
    }
      
    if(schedule) {
      if(uniform || subblock) {
        delete [] sched1;
        delete [] sched2;
      }
    }
    
    delete [] Request;
    if(!uniform) {
      delete [] sched;
      delete [] request;
    }
    
    if(a > 1) {
      int final;
      MPI_Finalized(&final);
      if(!final) {
        if(rank < a*b) {
          MPI_Comm_free(&split2); 
          MPI_Comm_free(&split); 
        }
        MPI_Comm_free(&block); 
      }
    }

    if(Tout2) delete Tout2;
    if(Tin2) delete Tin2;
    if(Tin1) delete Tin1;
    if(Tout1) delete Tout1;
  }
  
  ~mpitranspose() {
    deallocate();
  }
  
  int ni(int P) {return P < nlast ? n0 : (P == nlast ? np : 0);}
  int mi(int P) {return P < mlast ? m0 : (P == mlast ? mp : 0);}
  
  void Ialltoallout(void* sendbuf, void *recvbuf, int start,
                    unsigned int threads) {
    MPI_Request *srequest=request+Size(start);
    int S=sizeof(T)*L;
    int nS=n*S;
    int mS=m*S;
    int nm0=nS*m0;
    int mn0=mS*n0;
    for(int p=0; p < size; ++p) {
      int P=sched[p];
      if(P != rank && (rank >= start || P >= start)) {
        int index=rank >= start ? (P < rank ? P : P-1) : P-start;
        int count=mS*ni(P);
        if(count > 0)
          MPI_Irecv((char *) recvbuf+mn0*P,count,MPI_BYTE,P,0,communicator,
                    request+index);
        else request[index]=MPI_REQUEST_NULL;
        count=nS*mi(P);
        if(count > 0)
          MPI_Isend((char *) sendbuf+nm0*P,count,MPI_BYTE,P,0,communicator,
                    srequest+index);
        else srequest[index]=MPI_REQUEST_NULL;
      }
    }

    if(rank >= start)
      copy((char *) sendbuf+nm0*rank,(char *) recvbuf+mn0*rank,nS*mi(rank),
           threads);
  }

  void Ialltoallin(void* sendbuf, void *recvbuf, int start,
                   unsigned int threads) {
    MPI_Request *srequest=request+Size(start);
    int S=sizeof(T)*L;
    int nS=n*S;
    int mS=m*S;
    int nm0=nS*m0;
    int mn0=mS*n0;
    for(int p=0; p < size; ++p) {
      int P=sched[p];
      if(P != rank && (rank >= start || P >= start)) {
        int index=rank >= start ? (P < rank ? P : P-1) : P-start;
        int count=nS*mi(P);
        if(count > 0)
          MPI_Irecv((char *) recvbuf+nm0*P,nS*mi(P),MPI_BYTE,P,0,communicator,
                    request+index);
        else request[index]=MPI_REQUEST_NULL;
        count=mS*ni(P);
        if(count > 0)
          MPI_Isend((char *) sendbuf+mn0*P,mS*ni(P),MPI_BYTE,P,0,communicator,
                    srequest+index);
        else srequest[index]=MPI_REQUEST_NULL;
      }
    }

    if(rank >= start)
      copy((char *) sendbuf+mn0*rank,(char *) recvbuf+nm0*rank,mS*ni(rank),
           threads);
  }

// inphase: N x m -> n x M
  void inphase0() {
    if(rank >= size) return;
    if(size == 1) {
      if(input != output)
        copy(input,output,N*m*L,threads);
      return;
    }
    if(compact) work=output;
    if(uniform || subblock)
      Ialltoall(input,n*m*sizeof(T)*(a > 1 ? b : a)*L,work,split2,Request,
                sched2,threads);
    if(!uniform) {
      if(schedule) Ialltoallin(input,work,a > 1 ? a*b : 0,threads);
      else {
        if(rank < last)
          Ialltoall(input,n*m*sizeof(T)*L,work,splitv,request+2*Size(last),NULL,threads);
        Ialltoallin(input,work,last,threads);        
      }
    }
  }
  
  void insync0() {
    if(size == 1 || rank >= size) return;
    if(uniform || subblock)
      Wait(2*(split2size-1),Request,schedule);
    if(!uniform) {
      if(schedule)
        Wait(2*Size(a > 1 ? a*b : 0),request,schedule);
      else
        Wait(2*Size(last)+(rank < last ? 1 : 0),request,true);
    }
  }
  
  void inphase1() {
    if(rank >= size) return;
    if(subblock) {
      Tin2->transpose(work,output); // a x n*b x m*L
      Ialltoall(output,n*m*sizeof(T)*a*L,work,split,Request,sched1,threads);
    }
  }

  void insync1() {
    if(rank >= size) return;
    if(subblock)
      Wait(2*(splitsize-1),Request,schedule);
  }

  void inpost() {
    if(size == 1 || rank >= size) return;
    if(uniform)
      Tin1->transpose(work,output); // b x n*a x m*L
    else {
      if(subblock) {
        unsigned int block=m0*L;
        unsigned int cols=n*a;
        unsigned int istride=cols*block;
        unsigned int ostride=b*block;
        unsigned int extra=(M-m0*a*b)*L;

        PARALLEL(
          for(unsigned int i=0; i < n; ++i) {
            unsigned int ai=a*i;
            T *src=work+ai*block;
            T *dest=output+ai*ostride+i*extra;
            for(int j=0; j < a; ++j)
              copytoblock(src+j*block,dest+j*ostride,b,block,istride);
          });
        if(extra > 0) {
          unsigned int lastblock=mp*L;
          istride=n*block;
          ostride=mlast*block+lastblock;

          int ab=a*b;
          T *src=work+ab*istride;
          T *dest=output+ab*block;
          int count=mlast-ab;
          if(count > 0) {
            PARALLEL(
              for(unsigned int j=0; j < n; ++j)
                copytoblock(src+j*block,dest+j*ostride,count,block,istride);
              );
          }
          T *src2=work+mlast*istride;
          T *dest2=output+mlast*block;
          PARALLEL(
            for(unsigned int j=0; j < n; ++j)
              copy(src2+j*lastblock,dest2+j*ostride,lastblock);
            );
        }
      } else {
        unsigned int lastblock=mp*L;
        unsigned int block=m0*L;
        unsigned int istride=n*block;
        unsigned int mlastblock=mlast*block;
        unsigned int ostride=mlastblock+lastblock;
        T *work2=work+mlast*istride;

        PARALLEL(
          for(unsigned int j=0; j < n; ++j) {
            T *dest=output+j*ostride;
            copytoblock(work+j*block,dest,mlast,block,istride);
            copy(work2+j*lastblock,dest+mlastblock,lastblock);
          });
      }
    }
  }
  
// outphase: n x M -> N x m
  void outphase0() {
    if(rank >= size) return;
    if(size == 1) {
      if(input != output)
        copy(input,output,n*M*L,threads);
      return;
    }
    if(compact) work=output;
    // Inner transpose a N/a x M/a matrices over each team of b processes
    if(uniform)
      Tout1->transpose(input,work); // n*a x b x m*L
    else {
      if(subblock) {
        unsigned int block=m0*L;
        unsigned int cols=n*a;
        unsigned int istride=cols*block;
        unsigned int ostride=b*block;
        unsigned int extra=(M-m0*a*b)*L;

        PARALLEL(
          for(unsigned int i=0; i < n; ++i) {
            unsigned int ai=a*i;
            T *src=input+ai*ostride+i*extra;
            T *dest=work+ai*block;
            for(int j=0; j < a; ++j)
              copyfromblock(src+j*ostride,dest+j*block,b,block,istride);
          });
        if(extra > 0) {
          unsigned int lastblock=mp*L;
          istride=n*block;
          ostride=mlast*block+lastblock;
          int ab=a*b;
          int count=mlast-ab;
          T *src=input+ab*block;
          T *dest=work+ab*istride;

          if(count > 0) {
            PARALLEL(
              for(unsigned int j=0; j < n; ++j)
                copyfromblock(src+j*ostride,dest+j*block,count,block,istride);
              );
          }
          
          T *src2=input+mlast*block;
          T *dest2=work+mlast*istride;
          PARALLEL(
            for(unsigned int j=0; j < n; ++j)
              copy(src2+j*ostride,dest2+j*lastblock,lastblock);
            );
        }
      } else {
        unsigned int lastblock=mp*L;
        unsigned int block=m0*L;
        unsigned int istride=n*block;
        unsigned int ostride=mlast*block+lastblock;
        unsigned int mlastblock=mlast*block;
        T *dest=work+mlast*istride;

        PARALLEL(
          for(unsigned int j=0; j < n; ++j) {
            T *src=input+j*ostride;
            copyfromblock(src,work+j*block,mlast,block,istride);
            copy(src+mlastblock,dest+j*lastblock,lastblock);
          });
      }
    }
    if(subblock)
      Ialltoall(work,n*m*sizeof(T)*a*L,output,split,Request,sched1,threads);
    else outphase();
  }             
  
  void outsync0() {
    if(rank >= size) return;
    if(subblock)
      Wait(2*(splitsize-1),Request,schedule);
    else outsync();
  }
  
  void outphase() {
    if(size == 1 || rank >= size) return;
    // Outer transpose a x a matrix of N/a x M/a blocks over a processes
    if(subblock)
      Tout2->transpose(output,work); // n*b x a x m*L
    if(!uniform) {
      if(schedule) Ialltoallout(work,output,a > 1 ? a*b : 0,threads);
      else {
        if(rank < last)
          Ialltoall(work,n*m*sizeof(T)*L,output,splitv,request+2*Size(last),
                    NULL,threads);
        Ialltoallout(work,output,last,threads);        
      }
    }
    if(uniform || subblock)
      Ialltoall(work,n*m*sizeof(T)*(a > 1 ? b : a)*L,output,split2,Request,
                sched2,threads);
  }
  
  void outphase1() {
    if(subblock) outphase();
  }
  
  void outsync() {
    if(size == 1 || rank >= size) return;
    if(!uniform) {
      if(schedule)
        Wait(2*Size(a > 1 ? a*b : 0),request,schedule);
      else
        Wait(2*Size(last)+(rank < last ? 1 : 0),request,true);
    }
    if(uniform || subblock)
      Wait(2*(split2size-1),Request,schedule);
  }
  
  void outsync1() {
    if(subblock) outsync();
  }

  void Wait0() {
    if(outflag) {
      outsync0();
      outphase1();
    } else {
      insync0();
      inphase1();
    }
  }
  
  void Wait1() {
    if(outflag)
      outsync1();
    else {
      insync1();
      inpost();
    }
  }
  
  void wait0() {
    if(overlap) Wait0();
  }
  
  void wait1() {
    if(overlap) Wait1();
  }
  
  void wait() {
    if(overlap) {
      Wait0();
      Wait1();
    }
  }
  
  void localize0(T *in, T *out=0)
  {
    ilocalize0(in,out);
    wait();
  }
  
  void localize1(T *in, T *out=0)
  {
    ilocalize1(in,out);
    wait();
  }
  
  void ilocalize0(T *in, T *out=0)
  {
    if(!out) out=in;
    input=in;
    output=out;
    outphase0();
    outflag=true;
    if(!overlap) {
      Wait0();
      Wait1();
    }
  }
  void ilocalize1(T *in, T *out=0)
  {
    if(!out) out=in;
    input=in;
    output=out;
    inphase0();
    outflag=false;
    if(!overlap) {
      Wait0();
      Wait1();
    }
  }
  
};

}

#endif
