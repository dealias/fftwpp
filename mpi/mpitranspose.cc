#include "mpitranspose.h"

namespace fftwpp {

inline void copy(Complex *from, Complex *to, unsigned int length)
{
  unsigned int size=length*sizeof(Complex);
  memcpy(to,from,size);
}

void mpitranspose::inTransposed(Complex *data)
{
  if(size == 1) return;
  
  // Phase 1: Inner transpose each individual N/a x M/a matrix over b processes
  unsigned int blocksize=n*a*m*L;
  if(a > 1) {
    Tout3->transpose(data,work); // N x m x L
    Tin1->transpose(work,data); // m*a x b x n*L
    Alltoall(data,2*blocksize,MPI_DOUBLE,work,2*blocksize,MPI_DOUBLE,split,
              request,sched);
    
  // Phase 2: Outer transpose a x a matrix of N/a x M/a blocks over a processes
    Tin2->transpose(work,data); // m*b x a n*L
    unsigned int blocksize=n*b*m*L;
    Ialltoall(data,2*blocksize,MPI_DOUBLE,work,2*blocksize,MPI_DOUBLE,split2,
             request,sched2);
  } else
    Ialltoall(data,2*blocksize,MPI_DOUBLE,work,2*blocksize,MPI_DOUBLE,split,
              request,sched);
}
  
void mpitranspose::inwait(Complex *data)
{
  if(size == 1) return;
  Wait(split2size-1,request,sched2);
}

void mpitranspose::inpost(Complex *data)
{
  if(size == 1) return;
  if(a > 1) {
    Tin3->transpose(work,data); // M x n x L
  } else
    Tin1->transpose(work,data); // b x n*a x m*L
}  
    
void mpitranspose::outTransposed(Complex *data)
{
  if(size == 1) return;
  
  // Phase 1: Inner transpose each individual N/a x M/a matrix over b processes
  Tout1->transpose(data,work); // n*a x b x m*L
  unsigned int blocksize=n*a*m*L;
  if(a > 1) {
    Alltoall(work,2*blocksize,MPI_DOUBLE,data,2*blocksize,MPI_DOUBLE,split,
             request,sched);
    // Phase 2: Outer transpose a x a matrix of N/a x M/a blocks over a processes
    Tout2->transpose(data,work); // n*b x a x m*L
    unsigned int blocksize=n*b*m*L;
    Ialltoall(work,2*blocksize,MPI_DOUBLE,data,2*blocksize,MPI_DOUBLE,split2,
             request,sched2);
  }
  else
    Ialltoall(work,2*blocksize,MPI_DOUBLE,data,2*blocksize,MPI_DOUBLE,split,
              request,sched);
}

void mpitranspose::outwait(Complex *data, bool localtranspose) 
{
  if(size > 1)
    Wait(split2size-1,request,sched2);
}
  
void mpitranspose::outpost(Complex *data, bool localtranspose) 
{
   if(localtranspose) {
    Tout3->transpose(data,work); // N x m x L
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
