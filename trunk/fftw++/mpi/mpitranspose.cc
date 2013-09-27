#include "mpitranspose.h"

namespace fftwpp {

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

void transpose::inTransposed(Complex *data)
{
  if(size == 1) return;
  
  // Phase 1: Inner transpose each individual N/a x M/a block over b processes
  if(a > 1) {
//  Transpose(data,work,N,m,L);
    fftw_execute_dft(Tout3,(fftw_complex *) data,(fftw_complex *) work);
  
//  Transpose(work,data,m*a,b,n*L);
    fftw_execute_dft(Tin1,(fftw_complex *) work,(fftw_complex *) data);
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
    fftw_execute_dft(Tin2,(fftw_complex *) work,(fftw_complex *) data);
    unsigned int blocksize=n*b*m*L;
    Alltoall(data,2*blocksize,MPI_DOUBLE,work,2*blocksize,MPI_DOUBLE,split2,
             request,sched2);
//  Transpose(work,data,M,n,L);
    fftw_execute_dft(Tin3,(fftw_complex *) work,(fftw_complex *) data);
  } else
//    Transpose(work,data,b,n*a,m*L);
    fftw_execute_dft(Tin1,(fftw_complex *) work,(fftw_complex *) data);
}  
    
void transpose::outTransposed(Complex *data)
{
  if(size == 1) return;
  
  // Phase 1: Inner transpose each individual N/a x M/a block over b processes
//  Transpose(data,work,n*a,b,m*L);
  fftw_execute_dft(Tout1,(fftw_complex *) data,(fftw_complex *) work);

  unsigned int blocksize=n*a*m*L;
  Ialltoall(work,2*blocksize,MPI_DOUBLE,data,2*blocksize,MPI_DOUBLE,split,
            request,sched);
}

void transpose::outwait(Complex *data, bool localtranspose) 
{
  if(size == 1) return;
  Wait(splitsize-1,request,sched);
  
  // Phase 2: Outer transpose b x b matrix of N/a x M/a blocks over a processes
  if(a > 1) {
//    Transpose(data,work,n*b,a,m*L);
    fftw_execute_dft(Tout2,(fftw_complex *) data,(fftw_complex *) work);
    unsigned int blocksize=n*b*m*L;
    Alltoall(work,2*blocksize,MPI_DOUBLE,data,2*blocksize,MPI_DOUBLE,split2,
             request,sched2);
  }
  if(localtranspose) {
    fftw_execute_dft(Tout3,(fftw_complex *) data,(fftw_complex *) work);
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

} // end namespace fftwpp
