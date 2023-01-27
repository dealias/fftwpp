/* cmpifftw++.cc - C callable MPI FFTW++ wrapper.
 *
 * Authors: 
 * Malcolm Roberts <malcolm.i.w.roberts@gmail.com>
 */


#ifndef _CMPIFFTWPP_H_
#define _CMPIFFTWPP_H_

#include <mpi.h>
#include "cmpifftw++.h"
#include "mpifftw++.h"


extern "C" {

  namespace fftwpp {
    
    fft2dMPI* mpifftwpp_create_fft2d(MPI_Comm comm,
				     double __complex__ *in,
				     double __complex__ *out,
				     unsigned int nx,
				     unsigned int ny) {
      utils::MPIgroup group(MPI_COMM_WORLD,ny);
      utils::split d(nx,ny,group.active);

      /* Test for best block divisor: */
      int divisor = 0;

      /* Test for best alltoall routine: */
      int alltoall = -1; 

      auto opts = utils::mpiOptions(divisor,
				    alltoall,
				    1, //utils::defaultmpithreads,
				    0);
    
      return new fft2dMPI(d,
			  (Complex*)in,
			  (Complex*)out,
			  opts);
    };

  } /* namespace fftwpp */
  
} /* extern C */

        
        

#endif 
