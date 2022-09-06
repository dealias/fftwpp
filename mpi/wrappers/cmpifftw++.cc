/* cmpifftw++.cc - C callable MPI FFTW++ wrapper.
 *
 * Authors: 
 * Malcolm Roberts <malcolm.i.w.roberts@gmail.com>
 */


#ifndef _CMPIFFTWPP_H_
#define _CMPIFFTWPP_H_

#include <mpi.h>
#include "../mpifftw++.h"
#include "cmpifftw++.h"

#ifdef  __cplusplus
namespace fftwpp { extern "C" {
#endif


fft2dMPI *mpifftwpp_create_fft2d(MPI_Comm &comm,
                                   double __complex__ *in,
                                   double __complex__ *out,
                                   unsigned int nx,
                                   unsigned int ny) {
    utils::MPIgroup group(MPI_COMM_WORLD,ny);
    utils::split d(nx,ny,group.active);

    // Test for best block divisor:
    int divisor = 0;

    // Test for best alltoall routine:
    int alltoall = -1; 

    auto opts = utils::mpiOptions(divisor,
                                  alltoall,
                                  utils::defaultmpithreads,
                                  0);
    
    return new fft2dMPI(d,
                        (Complex*)in,
                        (Complex*)out,
                        opts);
};

        
        
#ifdef  __cplusplus
} }
#endif

#endif
