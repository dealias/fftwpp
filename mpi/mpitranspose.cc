#include "mpitranspose.h"

using namespace std;

namespace fftwpp {

inline void copy(Complex *from, Complex *to, unsigned int length)
{
  unsigned int size=length*sizeof(Complex);
  memcpy(to,from,size);
}

// Out-of-place transpose an n x m matrix of blocks of size length from src 
// to dest.
inline void Transpose(char *src, char *dest,
                      unsigned int n, unsigned int m, unsigned int length)
{
  if(n > 1 && m > 1) {
    unsigned int nstride=n*length;
    unsigned int mstride=m*length;
#pragma omp parallel for num_threads(4)
    for(unsigned int i=0; i < nstride; i += length) {
      char *srci=src+i*m;
      char *desti=dest+i;
      for(unsigned int j=0; j < mstride; j += length)
        memcpy(desti+j*n,srci+j,length);
    }
  } else
    memcpy(dest,src,n*m*length);
}

// Out-of-place transpose an n x m matrix of blocks of size length from src 
// (with row stride srcstride) to dest (with row stride deststride).
inline void Transpose(char *src, char *dest,
                      unsigned int n, unsigned int m, unsigned int length,
                      unsigned int srcstride, unsigned deststride,
                      unsigned int threads=4)
{
  const unsigned int limit=1024;
      
  if(n*m*length > limit) {
    if(threads > 1) {
//      unsigned int K=log2(threads)/2;
//      unsigned int a=min(n,(unsigned int ) 1 << K);
//      unsigned int b=min(m,threads/a);
      unsigned int b=min(m,threads);
      unsigned int a=min(n,threads/b);
      unsigned int N=n/a;
      unsigned int M=m/b;
      unsigned int Nlength=N*length;
      unsigned int Mlength=M*length;
#pragma omp parallel for num_threads(a)
      for(unsigned int i=0; i < a; ++i) {
        unsigned int I=i*Nlength;
#pragma omp parallel for num_threads(b)
        for(unsigned int j=0; j < b; ++j) {
          unsigned int J=j*Mlength;
          Transpose(src+srcstride*I+J,dest+deststride*J+I,N,M,length,
                    srcstride,deststride,1);
        }
      }
    } else {
      unsigned int D=2;
      unsigned int a=min(n,D);
      unsigned int b=min(m,D);
      unsigned int N=n/a;
      unsigned int M=m/b;
      unsigned int Mlength=M*length;
      unsigned int Nlength=N*length;
      for(unsigned int i=0; i < a; ++i) {
        unsigned int I=i*Nlength;
        for(unsigned int j=0; j < b; ++j) {
          unsigned int J=j*Mlength;
          Transpose(src+srcstride*I+J,dest+deststride*J+I,N,M,length,
                    srcstride,deststride,1);
        }
      }
    }
  } else if(n > 1) {
    unsigned int nlength=n*length;
    unsigned int mlength=m*length;
    for(unsigned int i=0; i < nlength; i += length) {
      char *srci=src+i*srcstride;
      char *desti=dest+i;
      for(unsigned int j=0; j < mlength; j += length)
        memcpy(desti+j*deststride,srci+j,length);
    }
  } else {
    unsigned int mlength=m*length;
    for(unsigned int j=0; j < mlength; j += length)
      memcpy(dest+j*deststride,src+j,length);
  }
}

inline void Transpose(Complex *src, Complex *dest,
                      unsigned int n, unsigned int m, unsigned int length)
{
 Transpose((char *) src,(char *) dest,n,m,length*sizeof(Complex),m,n);
//  Transpose((char *) src,(char *) dest,n,m,length*sizeof(Complex));
}

void transpose::inTransposed(Complex *data)
{
  if(size == 1) return;
  
  // Phase 1: Inner transpose each individual N/a x M/a block over b processes
  if(a > 1) {
//    Transpose(data,work,N,m,L);
    Tout3->transpose(data,work);
  
//    Transpose(work,data,m*a,b,n*L);
    Tin1->transpose(work,data);
  }

  unsigned int blocksize=n*a*m*L;
  Ialltoall(data,2*blocksize,MPI_DOUBLE,work,2*blocksize,MPI_DOUBLE,split,
            request,sched);
}
  
void transpose::inwait(Complex *data)
{
  if(size == 1) return;
  Wait(splitsize-1,request,sched);
  
  // Phase 2: Outer transpose b x b matrix of N/a x M/a blocks over a processes
  if(a > 1) {
//    Transpose(work,data,m*b,a,n*L);
    Tin2->transpose(work,data);
    unsigned int blocksize=n*b*m*L;
    Alltoall(data,2*blocksize,MPI_DOUBLE,work,2*blocksize,MPI_DOUBLE,split2,
             request,sched2);
//  Transpose(work,data,M,n,L);
    Tin3->transpose(work,data);
  } else
//    Transpose(work,data,b,n*a,m*L);
    Tin1->transpose(work,data);
}  
    
void transpose::outTransposed(Complex *data)
{
  if(size == 1) return;
  
  // Phase 1: Inner transpose each individual N/a x M/a block over b processes
//  Transpose(data,work,n*a,b,m*L);
  Tout1->transpose(data,work);
  unsigned int blocksize=n*a*m*L;
  Ialltoall(work,2*blocksize,MPI_DOUBLE,data,2*blocksize,MPI_DOUBLE,split,
            request,sched);
}

void transpose::outwait(Complex *data, bool localtranspose) 
{
  if(size > 1) {
    Wait(splitsize-1,request,sched);
  
    // Phase 2: Outer transpose b x b matrix of N/a x M/a blocks over a processes
    if(a > 1) {
//      Transpose(data,work,n*b,a,m*L);
      Tout2->transpose(data,work);
      unsigned int blocksize=n*b*m*L;
      Alltoall(work,2*blocksize,MPI_DOUBLE,data,2*blocksize,MPI_DOUBLE,split2,
               request,sched2);
    }
  }
  if(localtranspose) {
//    Transpose(data,work,N,m,L);
    Tout3->transpose(data,work);
    copy(work,data,N*m*L);
  }
}
  
/* Given a process which_pe and a number of processes npes, fills
   the array sched[npes] with a sequence of processes to communicate
   with for a deadlock-free, optimum-overlap all-to-all communication.
   (All processes must call this routine to get their own schedules.)
   The schedule can be re-ordered arbitrarily as long as all processes
   apply the same permutation to their schedules.

   The algorithm here is based upon the one described in:
   J. A. M. Schreuder, "Constructing timetables for sport
   competitions," Mathematical Programming Study 13, pp. 58-67 (1980). 
   In a sport competition, you have N teams and want every team to
   play every other team in as short a time as possible (maximum overlap
   between games).  This timetabling problem is therefore identical
   to that of an all-to-all communications problem.  In our case, there
   is one wrinkle: as part of the schedule, the process must do
   some data transfer with itself (local data movement), analogous
   to a requirement that each team "play itself" in addition to other
   teams.  With this wrinkle, it turns out that an optimal timetable
   (N parallel games) can be constructed for any N, not just for even
   N as in the original problem described by Schreuder.
*/
void fill1_comm_sched(int *sched, int which_pe, int npes)
{
  int pe, i, n, s = 0;
//  assert(which_pe >= 0 && which_pe < npes);
  if (npes % 2 == 0) {
    n = npes;
    sched[s++] = which_pe;
  }
  else
    n = npes + 1;
  for (pe = 0; pe < n - 1; ++pe) {
    if (npes % 2 == 0) {
      if (pe == which_pe) sched[s++] = npes - 1;
      else if (npes - 1 == which_pe) sched[s++] = pe;
    }
    else if (pe == which_pe) sched[s++] = pe;

    if (pe != which_pe && which_pe < n - 1) {
      i = (pe - which_pe + (n - 1)) % (n - 1);
      if (i < n/2)
        sched[s++] = (pe + i) % (n - 1);
	       
      i = (which_pe - pe + (n - 1)) % (n - 1);
      if (i < n/2)
        sched[s++] = (pe - i + (n - 1)) % (n - 1);
    }
  }
//  assert(s == npes);
}

} // end namespace fftwpp
