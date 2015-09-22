#ifndef __mpitranspose_h__
#define __mpitranspose_h__ 1

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
inline void copy(const T *from, T *to, unsigned int length,
		 unsigned int threads=1)
{
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif  
  for(unsigned int i=0; i < length; ++i)
    to[i]=from[i];
}

// Copy count blocks spaced stride apart to contiguous blocks in dest.
template<class T>
inline void copytoblock(const T *src, T *dest,
                        unsigned int count, unsigned int length,
                        unsigned int stride, unsigned int threads=1)
{
  for(unsigned int i=0; i < count; ++i)
    copy(src+i*stride,dest+i*length,length,threads);
}

// Copy count blocks spaced stride apart from contiguous blocks in src.
template<class T>
inline void copyfromblock(const T *src, T *dest,
                          unsigned int count, unsigned int length,
                          unsigned int stride, unsigned int threads=1)
{
  for(unsigned int i=0; i < count; ++i)
    copy(src+i*length,dest+i*stride,length,threads);
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
inline int MPI_Ialltoallv(void *sendbuf, int *sendcounts,
                          int *senddisplacements, MPI_Datatype sendtype,
                          void *recvbuf, int *recvcounts,
                          int *recvdisplacements, MPI_Datatype recvtype,
                          MPI_Comm comm, MPI_Request *)
{
  return MPI_Alltoallv(sendbuf,sendcounts,senddisplacements,sendtype,
                       recvbuf,recvcounts,recvdisplacements,recvtype,comm);
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
  if(!sched)
    return MPI_Ialltoallv(sendbuf,sendcounts,senddisplacements,MPI_BYTE,
                          recvbuf,recvcounts,recvdisplacements,MPI_BYTE,comm,
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
        MPI_Irecv((char *) recvbuf+recvdisplacements[P],recvcounts[P],
		  MPI_BYTE,P,0,comm,request+index);
        MPI_Isend((char *) sendbuf+senddisplacements[P],sendcounts[P],
		  MPI_BYTE,P,0,comm,srequest+index);
      }
    }

    copy((char *) sendbuf+senddisplacements[rank],
         (char *) recvbuf+recvdisplacements[rank],sendcounts[rank]);
    return 0;
  }
}
  
inline int Ialltoall(void *sendbuf, int count,
                     void *recvbuf,
                     MPI_Comm comm, MPI_Request *request, int *sched=NULL)
{
  if(!sched)
    return MPI_Ialltoall(sendbuf,count,MPI_BYTE,recvbuf,count,MPI_BYTE,comm,
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
    copy((char *) sendbuf+offset,(char *) recvbuf+offset,count);
    return 0;
  }
}

struct mpioptions {
  unsigned int threads;
  int a; // Block divisor (-1=sqrt(size), 0=Tune)
  int alltoall; // -1=Tune, 0=Optimized, 1=MPI
  mpioptions(unsigned int threads=fftw::maxthreads, unsigned int a=0,
             unsigned int alltoall=-1) :
    threads(threads), a(a), alltoall(alltoall) {}
};
    
static const mpioptions defaultmpioptions;
  
template<class T>
class mpitranspose {
private:
  unsigned int N,m,n,M;
  unsigned int L;
  T *data;
  T *work;
  mpioptions options;
  MPI_Comm communicator;
  MPI_Comm global;
  MPI_Comm block;
  
  unsigned int n0,m0;
  unsigned int np,mp;
  int mlast,nlast;
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
  Transpose *Tin1,*Tin2,*Tin3;
  Transpose *Tout1,*Tout2,*Tout3;
  int a,b;
  bool inflag,outflag;
  bool uniform;
  int *sendcounts,*senddisplacements;
  int *recvcounts,*recvdisplacements;
public:

  mpioptions Options() {return options;}
  
  bool divisible(int size, unsigned int M, unsigned int N) {
    unsigned int usize=size;
    return usize <= N && usize <= M && N % usize == 0 && M % usize == 0;
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
      std::cout << std::endl << "latency=" << latency << std::endl;
    MPI_Comm_free(&split); 
    return latency;
  }

  void setup(T *data) {
    threads=options.threads;
    Tin3=NULL;
    Tout3=NULL;
    
    MPI_Comm_size(communicator,&size);
    MPI_Comm_rank(communicator,&rank);
    
    MPI_Comm_rank(global,&globalrank);
    
    m0=localdimension(M,0,size);
    mlast=ceilquotient(M,m0)-1;
    mp=localdimension(M,mlast,size);
    
    n0=localdimension(N,0,size);
    nlast=ceilquotient(N,n0)-1;
    np=localdimension(N,nlast,size);
    
    if(work == NULL) {
      allocated=std::max(N*m,n*M)*L;
      Array::newAlign(this->work,allocated,sizeof(T));
    } else allocated=0;
    
    if(size == 1) {
      a=1;
      return;
    }
    
    bool Uniform=divisible(size,M,N);

    int start=0,stop=1;
    if(options.alltoall >= 0)
      start=stop=options.alltoall;
    int Alltoall=1;
    if(options.a >= size || options.a < 0) {
      int n=sqrt(size)+0.5;
      options.a=size/n;
    }

    if(globalrank == 0)
      std::cout << std::endl << "Initializing " << N << "x" << M
                << " transpose of " << L*sizeof(T) << "-byte elements over " 
                << size << " processes." << std::endl;
      
    int alimit;
    if(options.a <= 0) { // Restrict divisor range based on latency estimate
      options.a=1;
      double latency=safetyfactor*Latency();
      alimit=(N*M*L*sizeof(T) < latency*size*size) ?
        (int) (sqrt(size)+1.5) : 2;
      MPI_Bcast(&alimit,1,MPI_UNSIGNED,0,global);
    } else alimit=options.a+1;
    int astart=options.a;
      
    if(alimit > astart+1 || stop-start >= 1) {
      if(globalrank == 0)
        std::cout << std::endl << "Timing:" << std::endl;
      
      double T0=DBL_MAX;
      for(int alltoall=start; alltoall <= stop; ++alltoall) {
        if(globalrank == 0) std::cout << "alltoall=" << alltoall << std::endl;
        for(a=astart; a < alimit; a++) {
          b=std::min(nlast+(n0 == np),mlast+(m0 == mp))/a;
          uniform=Uniform && a*b == size;
          options.alltoall=alltoall;
          init(data);
          double t=time(data);
          deallocate();
          if(globalrank == 0) {
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
    b=std::min(nlast+(n0 == np),mlast+(m0 == mp))/a;
    if(b == 1) a=1;
//    if(a == 1) b=size;
    uniform=Uniform && a*b == size;
    
    if(globalrank == 0)
      std::cout << std::endl << "Using alltoall=" << 
        options.alltoall << ", a=" << a << ", b=" << b <<
        ":" << std::endl;
    init(data);
  }
  
  mpitranspose(){}

  // Here "in" is a local N x m matrix and "out" is a local n x M matrix.
  // work is a temporary array of size N*m*L.
  mpitranspose(unsigned int N, unsigned int m, unsigned int n,
               unsigned int M, unsigned int L,
               T *data, T *work=NULL,
               mpioptions options=defaultmpioptions,
               MPI_Comm communicator=MPI_COMM_WORLD,
               MPI_Comm global=0) :
    N(N), m(m), n(n), M(M), L(L), work(work), options(options),
    communicator(communicator), global(global ? global : communicator) {
    setup(data);
  }
  
  mpitranspose(unsigned int N, unsigned int m, unsigned int n,
               unsigned int M, unsigned int L,
               T *data, MPI_Comm communicator, MPI_Comm global=0) :
    N(N), m(m), n(n), M(M), L(L), work(NULL),
    communicator(communicator), global(global ? global : communicator) {
    setup(data);
  }
    
  mpitranspose(unsigned int N, unsigned int m, unsigned int n,
               unsigned int M, T *data, T *work=NULL,
               mpioptions options=defaultmpioptions,
               MPI_Comm communicator=MPI_COMM_WORLD, MPI_Comm global=0) :
    N(N), m(m), n(n), M(M), L(1), work(work), options(options),
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

  void init(T *data) {
    Tout1=uniform || (a > 1 && rank < a*b) ? 
      new Transpose(n*a,b,m*L,data,this->work,threads) : NULL;
    Tin1=uniform ? new Transpose(b,n*a,m*L,data,this->work,threads) : NULL;
    
    if(a > 1 && rank < a*b) {
      Tin2=new Transpose(a,n*b,m*L,data,this->work,threads);
      Tout2=new Transpose(n*b,a,m*L,data,this->work,threads);
    } else {Tin2=Tout2=NULL;}
    
    if(size == 1) return;
    
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
    
    sendcounts=NULL;
    if(options.alltoall) {
      if(!uniform && a == 1) {
        int Size=size;
        sendcounts=new int[Size];
        senddisplacements=new int[Size];
        recvcounts=new int[Size];
        recvdisplacements=new int[Size];
        int S=sizeof(T)*L;
        fillindices(S,Size);
      }
    }
    if(!options.alltoall || (!uniform && a > 1)) {
      request=new MPI_Request[2*(size-1)];
      Request=new MPI_Request[2*(std::max(splitsize,split2size)-1)];
    
      sched=new int[size];
      fill1_comm_sched(sched,rank,size);
    
      if(a > 1) {
        sched2=new int[split2size];
        fill1_comm_sched(sched2,split2rank,split2size);
        sched1=new int[splitsize];
        fill1_comm_sched(sched1,splitrank,splitsize);
      } else
        sched1=sched2=sched;
    } else {
      request=new MPI_Request[1];
      Request=new MPI_Request[1];
      sched=sched1=sched2=NULL;
    }
  }
  
  void deallocate() {
    if(size == 1) return;
    if(sched) {
      if(a > 1) {
        delete[] sched1;
        delete[] sched2;
      }
      delete[] sched;
    }
    delete[] request;
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

    if(sendcounts) {
      delete [] sendcounts;
      delete [] senddisplacements;
      delete [] recvcounts;
      delete [] recvdisplacements;
    }
  }
  
  ~mpitranspose() {
    deallocate();
    if(Tout3) delete Tout3;
    if(Tin3) delete Tin3;
    MPI_Barrier(global);
    if(allocated)
      Array::deleteAlign(work,allocated);
  }
  
  void fillindices(int S, int size) {
    int nS=n*S;
    int mS=m*S;
    int nm0=nS*m0;
    int mn0=mS*n0;
    for(int i=0; i < size; ++i) {
      sendcounts[i]=i < mlast ? nm0 : (i == mlast ? nS*mp : 0);
      recvcounts[i]=i < nlast ? mn0 : (i == nlast ? mS*np : 0);
      senddisplacements[i]=nm0*i;
      recvdisplacements[i]=mn0*i;
    }
  }
  
  int ni(int P) {return P < nlast ? n0 : (P == nlast ? np : 0);}
  int mi(int P) {return P < mlast ? m0 : (P == mlast ? mp : 0);}
  
  void Ialltoallout(void* sendbuf, void *recvbuf, int start) {
    MPI_Request *srequest=rank >= start ? request+size-1 : request+size-start;
    int S=sizeof(T)*L;
    int nS=n*S;
    int mS=m*S;
    int nm0=nS*m0;
    int mn0=mS*n0;
    for(int p=0; p < size; ++p) {
      int P=sched[p];
      if(P != rank && (rank >= start || P >= start)) {
        int index=rank >= start ? (P < rank ? P : P-1) : P-start;
        MPI_Irecv((char *) recvbuf+mn0*P,mS*ni(P),MPI_BYTE,P,0,communicator,
                  request+index);
        MPI_Isend((char *) sendbuf+nm0*P,nS*mi(P),MPI_BYTE,P,0,communicator,
                  srequest+index);
      }
    }

    if(rank >= start)
      copy((char *) sendbuf+mn0*rank,(char *) recvbuf+nm0*rank,nS*mi(rank));
  }

  void Ialltoallin(void* sendbuf, void *recvbuf, int start) {
    MPI_Request *srequest=rank >= start ? request+size-1 : request+size-start;
    int S=sizeof(T)*L;
    int nS=n*S;
    int mS=m*S;
    int nm0=nS*m0;
    int mn0=mS*n0;
    for(int p=0; p < size; ++p) {
      int P=sched[p];
      if(P != rank && (rank >= start || P >= start)) {
        int index=rank >= start ? (P < rank ? P : P-1) : P-start;
        MPI_Irecv((char *) recvbuf+nm0*P,nS*mi(P),MPI_BYTE,P,0,communicator,
                  request+index);
        MPI_Isend((char *) sendbuf+mn0*P,mS*ni(P),MPI_BYTE,P,0,communicator,
                  srequest+index);
      }
    }

    if(rank >= start)
      copy((char *) sendbuf+mn0*rank,(char *) recvbuf+nm0*rank,mS*ni(rank));
  }

  void inphase0() {
    if(size == 1) return;
    if(uniform || (a > 1 && rank < a*b)) {
      size_t S=sizeof(T)*(a > 1 ? b : a)*L;
      Ialltoall(data,n*m*S,work,split2,Request,sched2);
    }
    if(!uniform) {
      if(sched) Ialltoallin(data,work,a > 1 ? a*b : 0);
      else MPI_Ialltoallv(data,recvcounts,recvdisplacements,MPI_BYTE,
                          work,sendcounts,senddisplacements,MPI_BYTE,
                          communicator,request);
    }
  }
  
  void inphase1() {
    if(a > 1 && rank < a*b) {
      size_t S=sizeof(T)*a*L;
      Tin2->transpose(work,data); // a x n*b x m*L
      Ialltoall(data,n*m*S,work,split,Request,sched1);
    }
  }

  void insync0() {
    if(size == 1) return;
    if(uniform || (a > 1 && rank < a*b)) {
      Wait(2*(split2size-1),Request,sched2);
      if(!uniform)
        Wait(2*(size-(a > 1 ? a*b : 0)),request,sched2);
    } else
      Wait(2*(size-1),request,sched2);
  }
  
  void insync1() {
    if(a > 1 && rank < a*b)
      Wait(2*(splitsize-1),Request,sched1);
  }

  void inpost() {
    if(size == 1) return;
    if(uniform)
      Tin1->transpose(work,data); // b x n*a x m*L
    else {
      if(a > 1 && rank < a*b) {
        unsigned int block=m0*L;
        unsigned int cols=n*a;
        unsigned int istride=cols*block;
        unsigned int ostride=b*block;
        unsigned int extra=(M-m0*a*b)*L;

        for(unsigned int i=0; i < n; ++i) {
          for(int j=0; j < a; ++j)
            copytoblock(work+(a*i+j)*block,data+(a*i+j)*ostride+i*extra,b,
                        block,istride);
        }
        if(extra > 0) {
          unsigned int lastblock=mp*L;
          istride=n*block;
          ostride=mlast*block+lastblock;

          for(unsigned int j=0; j < n; ++j) {
            T *dest=data+j*ostride;
            if(mlast > a*b)
              copytoblock(work+j*block+istride*a*b,dest+block*a*b,mlast-a*b,
                          block,istride);
            copy(work+j*lastblock+mlast*istride,dest+mlast*block,lastblock);
          }
        }
      } else {
        unsigned int lastblock=mp*L;
        unsigned int block=m0*L;
        unsigned int istride=n*block;
        unsigned int ostride=mlast*block+lastblock;

        for(unsigned int j=0; j < n; ++j) {
          T *dest=data+j*ostride;
          copytoblock(work+j*block,dest,mlast,block,istride);
          copy(work+j*lastblock+mlast*istride,dest+mlast*block,lastblock);
        }
      }
    }
  }
  
  void outphase0() {
    if(size == 1) return;
    size_t S=sizeof(T)*a*L;
    if(uniform || (a > 1 && rank < a*b)) {
      // Inner transpose a N/a x M/a matrices over each team of b processes
      Tout1->transpose(data,work); // n*a x b x m*L
      Ialltoall(work,n*m*S,data,split,Request,sched1);
    }
    if(!uniform) {
      if(a > 1 && rank < a*b) {
        unsigned int block=m0*L;
        unsigned int cols=n*a;
        unsigned int istride=cols*block;
        unsigned int ostride=b*block;
        unsigned int extra=(M-m0*a*b)*L;

        for(unsigned int i=0; i < n; ++i) {
          for(int j=0; j < a; ++j)
            copyfromblock(data+(a*i+j)*ostride+i*extra,work+(a*i+j)*block,b,
                          block,istride);
        }
        if(extra > 0) {
          unsigned int lastblock=mp*L;
          istride=n*block;
          ostride=mlast*block+lastblock;

          for(unsigned int j=0; j < n; ++j) {
            T *src=data+j*ostride;
            if(mlast > a*b)
              copyfromblock(src+block*a*b,work+j*block+istride*a*b,mlast-a*b,
                            block,istride);
            copy(src+mlast*block,work+j*lastblock+mlast*istride,lastblock);
          }
        }
      } else {
        unsigned int lastblock=mp*L;
        unsigned int block=m0*L;
        unsigned int istride=n*block;
        unsigned int ostride=mlast*block+lastblock;

        for(unsigned int j=0; j < n; ++j) {
          T *src=data+j*ostride;
          copyfromblock(src,work+j*block,mlast,block,istride);
          copy(src+mlast*block,work+j*lastblock+mlast*istride,lastblock);
        }
      }
      
      if(sched) Ialltoallout(work,data,a > 1 ? a*b : 0);
      else MPI_Ialltoallv(work,sendcounts,senddisplacements,MPI_BYTE,
                          data,recvcounts,recvdisplacements,MPI_BYTE,
                          communicator,request);
    }
  }
  
  void outphase1() {
    if(a > 1 && rank < a*b) {
      size_t S=sizeof(T)*b*L;
      // Outer transpose a x a matrix of N/a x M/a blocks over a processes
      Tout2->transpose(data,work); // n*b x a x m*L
      Ialltoall(work,n*m*S,data,split2,Request,sched2);
    }
  }
  
  void outsync0() {
    if(a > 1 && rank < a*b)
      Wait(2*(splitsize-1),Request,sched1);
  }
  
  void outsync1() {
    if(size == 1) return;
    if(uniform || (a > 1 && rank < a*b)) {
      Wait(2*(split2size-1),Request,sched2);
      if(!uniform) 
        Wait(2*(size-(a > 1 ? a*b : 0)),request,sched2);
    } else
      Wait(2*(size-1),request,sched2);
  }

  void nMTranspose() {
    if(n == 0) return;
    if(!Tin3) Tin3=new Transpose(n,M,L,data,work,threads);
    Tin3->transpose(data,work); // n X M x L
    copy(work,data,n*M*L,threads);
  }
  
  void NmTranspose() {
    if(m == 0) return;
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
