#include "mpitranspose.h"
#include "cmult-sse2.h"

namespace utils {

double safetyfactor=2.0;
bool overlap=true;
double testseconds=0.2;
mpiOptions defaultmpiOptions;

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

}
