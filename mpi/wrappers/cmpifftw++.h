/* cmpifftw++.h - C callable MPI FFTW++ wrapper
 *
 * Authors: 
 * Malcolm Roberts <malcolm.i.w.roberts@gmail.com>
 */


#ifndef _CMPIFFTWPP_H_
#define _CMPIFFTWPP_H_

#include <mpi.h>
#include "cmpifftw++.h"


#ifdef  __cplusplus
namespace mpifftwpp { extern "C" {

#endif
        typedef struct MPIgroup MPIgroup;
        MPIgroup* mpifftwpp_create_group_1d(const MPI_Comm comm,
                                            const unsigned int ny);
        MPIgroup* mpifftwpp_create_group_2d(const MPI_Comm comm,
                                            const unsigned int nx,
                                            const unsigned int ny);

        void mpifftwpp_delete_group(MPIgroup* group);
        
               
        typedef struct split split;
        split* mpifftwpp_create_split(const MPI_Comm comm,
                                      const unsigned int nx,
                                      const unsigned int ny);
        void mpifftwpp_delete_split(split* dim);
        unsigned int mpifftwpp_split_n(const split* dim);
        
        typedef struct split3 split3;
        split3* mpifftwpp_create_split3(const MPIgroup* group,
                                        const unsigned int nx,
                                        const unsigned int ny,
                                        const unsigned int nz);
        void mpifftwpp_delete_split3(split3* dim);
        unsigned int mpifftwpp_split3_n(const split3* dim);
        
        typedef struct fft2dMPI fft2dMPI;
        fft2dMPI* mpifftwpp_create_fft2d(const split* dim,
                                         double __complex__ *in,
                                         double __complex__ *out);
        void mpifftwpp_delete_fft2d(fft2dMPI* fft);
        void mpifftwpp_fft2d_forward(fft2dMPI* fft,
                                     double __complex__ *in,
                                     double __complex__ *out);
        void mpifftwpp_fft2d_backward(fft2dMPI* fft,
                                      double __complex__ *in,
                                      double __complex__ *out);

        typedef struct rcfft2dMPI rcfft2dMPI;
        fft2dMPI* mpifftwpp_create_rcfft2d(const split* rdim,
                                           const split* cdim,
                                           double *in,
                                           double __complex__ *out);
        void mpifftwpp_delete_rcfft2d(rcfft2dMPI* fft);
        void mpifftwpp_rcfft2d_forward(rcfft2dMPI* fft,
                                       double *in,
                                       double __complex__ *out);
        void mpifftwpp_rcfft2d_backward(rcfft2dMPI* fft,
                                        double __complex__ *in,
                                        double *out);

        
        typedef struct fft3dMPI fft3dMPI;
        fft3dMPI* mpifftwpp_create_fft3d(const split3* dim,
                                         double __complex__ *in,
                                         double __complex__ *out);
        void mpifftwpp_delete_fft3d(fft3dMPI* fft);
        void mpifftwpp_fft3d_forward(fft3dMPI* fft,
                                     double __complex__ *in,
                                     double __complex__ *out);
        void mpifftwpp_fft3d_backward(fft3dMPI* fft,
                                      double __complex__ *in,
                                      double __complex__ *out);

        typedef struct rcfft3dMPI rcfft3dMPI;
        rcfft3dMPI* mpifftwpp_create_rcfft3d(const split3* rdim,
                                           const split3* cdim,
                                           double *in,
                                           double __complex__ *out);
        void mpifftwpp_delete_rcfft3d(rcfft3dMPI* fft);
        void mpifftwpp_rcfft3d_forward(rcfft3dMPI* fft,
                                       double *in,
                                       double __complex__ *out);
        void mpifftwpp_rcfft3d_backward(rcfft3dMPI* fft,
                                        double __complex__ *in,
                                        double *out);

        
#ifdef  __cplusplus
    } }
#endif

#endif
