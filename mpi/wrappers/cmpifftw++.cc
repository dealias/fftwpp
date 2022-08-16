/* cmpifftw++.cc - C callable MPI FFTW++ wrapper.
 *
 * Authors: 
 * Malcolm Roberts <malcolm.i.w.roberts@gmail.com>
 */


#ifndef _CMPIFFTWPP_H_
#define _CMPIFFTWPP_H_

#include <mpi.h>

#ifdef  __cplusplus
namespace fftwpp { extern "C" {
#endif
        typedef struct split split;
        split* mpifftwpp_create_split(unsigned int X,
                                      unsigned int Y,
                                      MPI_Comm comm);
        
        typedef struct fft2dMPI fft2dMPI;
        

        
        
#ifdef  __cplusplus
} }
#endif

#endif
