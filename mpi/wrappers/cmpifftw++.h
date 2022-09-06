/* cmpifftw++.h - C callable MPI FFTW++ wrapper
 *
 * Authors: 
 * Malcolm Roberts <malcolm.i.w.roberts@gmail.com>
 */


#ifndef _CMPIFFTWPP_H_
#define _CMPIFFTWPP_H_

#ifdef  __cplusplus
namespace fftwpp { extern "C" {
#endif

        typedef struct fft2dMPI fft2dMPI;
        fft2dMPI *mpifftwpp_create_fft2d(MPI_Comm &comm,
                                         double __complex__ *in,
                                         double __complex__ *out,
                                         unsigned int nx,
                                         unsigned int ny);
        
#ifdef  __cplusplus
} }
#endif

#endif
