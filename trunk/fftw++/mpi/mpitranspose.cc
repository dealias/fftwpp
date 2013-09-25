using namespace std;

// mpic++ -O3 -fopenmp -g testmpi.cc fftw++.cc mpitranspose.cc -I ../ -lfftw3_mpi -lfftw3  -lm
// mpirun -n 2 a.out

#include <mpi.h>
#include "../Complex.h"

#include "mpi/mpitranspose.h"

inline void copy(Complex *from, Complex *to, unsigned int length)
{
  unsigned int size=length*sizeof(Complex);
  memcpy(to,from,size);
}

// Out-of-place transpose an n x m matrix of blocks of size length from src 
// to dest.
inline void Transpose(Complex *src, Complex *dest,
                      unsigned int n, unsigned int m, unsigned int length)
{
  if(n > 1 && m > 1) {
    unsigned int size=length*sizeof(Complex);
    unsigned int nstride=n*length;
    unsigned int mstride=m*length;
    for(unsigned int i=0; i < nstride; i += length) {
      Complex *srci=src+i*m;
      Complex *desti=dest+i;
      for(unsigned int j=0; j < mstride; j += length)
        memcpy(desti+j*n,srci+j,size);
    }
  } else
    copy(src,dest,n*m*length);
}

inline void copy(Complex *src, Complex *dest,
                 unsigned int count, unsigned int length,
                 unsigned int srcstride, unsigned int deststride)
{
  unsigned int size=length*sizeof(Complex);
  for(unsigned int i=0; i < count; ++i)
    memcpy(dest+i*deststride,src+i*srcstride,size);
}

void transpose::inTransposed(Complex *data)
{
  if(size == 1) return;
  
  // Outer transpose N/a x M/a matrix of a x a blocks
  Complex *in, *out;
  unsigned int Lm=L*m;
  unsigned int length=n*Lm;
  if(a > 1) {
    unsigned int stride=N/a*Lm;
    unsigned int q=length*size/a;
    for(unsigned int p=0; p < q; p += length)
      copy(data+p,work+a*p,a,length,stride,length);
    out=work;
    in=data;
  } else {
    out=data;
    in=work;
  }
    
  unsigned int doubles=2*a*length;
  Ialltoall(out,doubles,MPI_DOUBLE,in,doubles,MPI_DOUBLE,split,request,sched);
}
  
void transpose::inwait(Complex *data)
{
  if(size == 1) return;
  Wait(splitsize-1,request,sched);

  // Inner transpose each individual a x a block
  unsigned int Lm=L*m;
  if(a > 1) {
    unsigned int length=n*Lm;
    unsigned int stride=N/a*Lm;
    unsigned int lengthsize=length*sizeof(Complex);
    unsigned int q=size/a;
    for(unsigned int p=0; p < q; ++p) {
      unsigned int lengthp=length*p;
      Complex *workp=work+lengthp;
      Complex *datap=data+a*lengthp;
      for(unsigned int i=0; i < a; ++i)
        memcpy(workp+stride*i,datap+length*i,lengthsize);
    }
    
    Alltoall(work,2*stride,MPI_DOUBLE,data,2*stride,MPI_DOUBLE,split2,
             request,sched2);
  }
    
  unsigned int LM=L*M;
  if(n > 1) {
    if(a > 1) memcpy(work,data,LM*n*sizeof(Complex));
    unsigned int stop=Lm*size;
    unsigned int msize=Lm*sizeof(Complex);
    for(unsigned int i=0; i < n; ++i) {
      Complex *in=data+LM*i;
      Complex *out=work+Lm*i;
      for(unsigned int p=0; p < stop; p += Lm)
        memcpy(in+p,out+p*n,msize);
    }
  } else if(a == 1)
    memcpy(data,work,LM*n*sizeof(Complex));
}  
    
void transpose::outTransposed(Complex *data)
{
  if(size == 1) return;
  
  // Phase 1: Inner transpose each individual N/a x M/a block over b processes
//  Transpose(data,work,n*a,b,m*L);
  fftw_execute_dft(T1,(fftw_complex *) data,(fftw_complex *) work);

  unsigned int blocksize=n*a*m*L;
  Ialltoall(work,2*blocksize,MPI_DOUBLE,data,2*blocksize,MPI_DOUBLE,split,
            request,sched);
}

void transpose::outwait(Complex *data, bool localtranspose) 
{
  if(size == 1) return;
  Wait(splitsize-1,request,sched);
  
  if(a > 1) {
//    Transpose(data,work,n*b,a,m*L);
    fftw_execute_dft(T2,(fftw_complex *) data,(fftw_complex *) work);
  // Phase 2: Outer transpose b x b matrix of N/a x M/a blocks over a processes
    unsigned int blocksize=n*b*m*L;
    Alltoall(work,2*blocksize,MPI_DOUBLE,data,2*blocksize,MPI_DOUBLE,split2,
             request,sched2);
  }
  if(localtranspose) {
    fftw_execute_dft(T3,(fftw_complex *) data,(fftw_complex *) work);
//  Transpose(data,work,N,m,L);
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
