// mpic++ -O3 -fopenmp -g testmpi.cc fftw++.cc mpitranspose.cc -I ../ -lfftw3_mpi -lfftw3  -lm
// mpirun -n 2 a.out

#include <mpi.h>
#include "../Complex.h"

#include "mpi/mpitranspose.h"

void transpose::inTransposed(Complex *data)
{
  if(size == 1) return;
  
  // Outer transpose N/b x M/b matrix of b x b blocks
  Complex *in, *out;
  unsigned int Lm=L*m;
  unsigned int length=n*Lm;
  if(b > 1) {
    unsigned int stride=N/b*Lm;
    unsigned int lengthsize=length*sizeof(Complex);
    unsigned int q=size/b;
    for(unsigned int p=0; p < q; ++p) {
      unsigned int lengthp=length*p;
      Complex *datap=data+lengthp;
      Complex *workp=work+b*lengthp;
      for(unsigned int i=0; i < b; ++i)
        memcpy(workp+length*i,datap+stride*i,lengthsize);
    }
    out=work;
    in=data;
  } else {
    out=data;
    in=work;
  }
    
  unsigned int doubles=2*b*length;
  Ialltoall(out,doubles,MPI_DOUBLE,in,doubles,MPI_DOUBLE,split,request,sched);
}
  
void transpose::inwait(Complex *data)
{
  if(size == 1) return;
  Wait(splitsize-1,request,sched);

  // Inner transpose each individual b x b block
  unsigned int Lm=L*m;
  if(b > 1) {
    unsigned int length=n*Lm;
    unsigned int stride=N/b*Lm;
    unsigned int lengthsize=length*sizeof(Complex);
    unsigned int q=size/b;
    for(unsigned int p=0; p < q; ++p) {
      unsigned int lengthp=length*p;
      Complex *workp=work+lengthp;
      Complex *datap=data+b*lengthp;
      for(unsigned int i=0; i < b; ++i)
        memcpy(workp+stride*i,datap+length*i,lengthsize);
    }
    
    Alltoall(work,2*stride,MPI_DOUBLE,data,2*stride,MPI_DOUBLE,split2,
             request,sched2);
  }
    
  unsigned int LM=L*M;
  if(n > 1) {
    if(b > 1) memcpy(work,data,LM*n*sizeof(Complex));
    unsigned int stop=Lm*size;
    unsigned int msize=Lm*sizeof(Complex);
    for(unsigned int i=0; i < n; ++i) {
      Complex *in=data+LM*i;
      Complex *out=work+Lm*i;
      for(unsigned int p=0; p < stop; p += Lm)
        memcpy(in+p,out+p*n,msize);
    }
  } else if(b == 1)
    memcpy(data,work,LM*n*sizeof(Complex));
}  
    
void transpose::outTransposed(Complex *data)
{
  if(size == 1) return;
  
  // Inner transpose each individual b x b block
  unsigned int Lm=L*m;
  unsigned int LM=L*M;
  if(n > 1) {
    unsigned int stop=Lm*size;
    unsigned int msize=Lm*sizeof(Complex);
    for(unsigned int i=0; i < n; ++i) {
      Complex *outi=data+LM*i;
      Complex *ini=work+Lm*i;
      for(unsigned int p=0; p < stop; p += Lm)
        memcpy(ini+p*n,outi+p,msize);
    }
    if(b > 1) memcpy(data,work,LM*n*sizeof(Complex));    
  } else if(b == 1)
    memcpy(work,data,LM*n*sizeof(Complex));

  unsigned int length=n*Lm;
  Complex *in,*out;
  if(b > 1) {
    unsigned int q=size/b;
    unsigned int stride=N/b*Lm;
    Alltoall(data,2*stride,MPI_DOUBLE,work,2*stride,MPI_DOUBLE,split2,
             request,sched2);
      
    unsigned int lengthsize=length*sizeof(Complex);
    for(unsigned int p=0; p < q; ++p) {
      unsigned int lengthp=length*p;
      Complex *workp=work+lengthp;
      Complex *datap=data+b*lengthp;
      for(unsigned int i=0; i < b; ++i)
        memcpy(datap+length*i,workp+stride*i,lengthsize);
    }
    out=data;
    in=work;
  } else {
    out=work;
    in=data;
  }
      
  // Outer transpose N/b x M/b matrix of b x b blocks
  
  unsigned int doubles=2*b*length;
  Ialltoall(out,doubles,MPI_DOUBLE,in,doubles,MPI_DOUBLE,split,request,sched);
}

void transpose::outwait(Complex *data) 
{
  if(size == 1) return;
  Wait(splitsize-1,request,sched);
    
  unsigned int Lm=L*m;
  if(b > 1) {
    unsigned int stride=N/b*Lm;
    unsigned int length=n*Lm;
    unsigned int lengthsize=length*sizeof(Complex);
    unsigned int q=size/b;
    for(unsigned int p=0; p < q; ++p) {
      unsigned int lengthp=length*p;
      Complex *datap=data+lengthp;
      Complex *workp=work+b*lengthp;
      for(unsigned int i=0; i < b; ++i)
        memcpy(datap+stride*i,workp+length*i,lengthsize);
    }
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
