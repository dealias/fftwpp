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
#include "mpiutils.h"


extern "C" {

  namespace utils {
    // FIXME: this is a 1d group.
    MPIgroup* mpifftwpp_create_group_1d(const MPI_Comm comm,
					const unsigned int ny) {
      return new utils::MPIgroup(comm, ny);
    }
    MPIgroup* mpifftwpp_create_group_2d(const MPI_Comm comm,
					const unsigned int nx,
					const unsigned int ny) {
      return new utils::MPIgroup(comm, nx, ny);
    }
    void mpifftwpp_delete_group(MPIgroup* group) {
      delete group;
    }

      
    split* mpifftwpp_create_split(const MPI_Comm comm,
				  const unsigned int nx,
				  const unsigned int ny) {
      return new utils::split(nx, ny, comm);
    }
    void mpifftwpp_delete_split(split* dim) {
      delete dim;
    }
    unsigned int mpifftwpp_split_n(const split* dim) {
      return dim->n;
    }
      
      
    split3* mpifftwpp_create_split3(const MPIgroup* group,
				    const unsigned int nx,
				    const unsigned int ny,
				    const unsigned int nz) {
      return new utils::split3(nx, ny, nz, *group);
    }
    void mpifftwpp_delete_split3(split3* dim) {
      delete dim;
    }
    unsigned int mpifftwpp_split3_n(const split3* dim) {
      return dim->n;
    }
      
  }

  
  namespace fftwpp {

    fft2dMPI* mpifftwpp_create_fft2d(const utils::split* dim,
				     double __complex__ *in,
				     double __complex__ *out) {
      /* Test for best block divisor: */
      int divisor = 0;

      /* Test for best alltoall routine: */
      int alltoall = -1;

      auto opts = utils::mpiOptions(divisor,
				    alltoall,
				    1, //utils::defaultmpithreads,
				    0);
      return new fft2dMPI(*dim,
			  (Complex*)in,
			  (Complex*)out,
			  opts);
    }

    void mpifftwpp_delete_fft2d(fft2dMPI* fft) {
      delete fft;
    }

    void mpifftwpp_fft2d_forward(fft2dMPI* fft,
				 double __complex__ *in,
				 double __complex__ *out) {
      fft->Forward((Complex*) in, (Complex*) out);
    }

    void mpifftwpp_fft2d_backward(fft2dMPI* fft,
				  double __complex__ *in,
				  double __complex__ *out) {
      fft->Backward((Complex*) in, (Complex*) out);
    }

    
    rcfft2dMPI* mpifftwpp_create_rcfft2d(const utils::split* rdim,
					 const utils::split* cdim,
					 double *in,
					 double __complex__ *out) {
      /* Test for best block divisor: */
      int divisor = 0;

      /* Test for best alltoall routine: */
      int alltoall = -1;

      auto opts = utils::mpiOptions(divisor,
				    alltoall,
				    1, //utils::defaultmpithreads,
				    0);
      return new rcfft2dMPI(*rdim,
			    *cdim,
			    in,
			    (Complex*)out,
			    opts);
    }

    void mpifftwpp_delete_rcfft2d(rcfft2dMPI* fft) {
      delete fft;
    }

    void mpifftwpp_rcfft2d_forward(rcfft2dMPI* fft,
				   double *in,
				   double __complex__ *out) {
      fft->Forward(in, (Complex*) out);
    }

    void mpifftwpp_rcfft2d_backward(rcfft2dMPI* fft,
				    double __complex__ *in,
				    double  *out) {
      fft->Backward((Complex*) in, out);
    }

    
    fft3dMPI* mpifftwpp_create_fft3d(const utils::split3* dim,
				     double __complex__ *in,
				     double __complex__ *out) {

      /* Test for best block divisor: */
      int divisor = 0;

      /* Test for best alltoall routine: */
      int alltoall = -1;
      
      auto opts = utils::mpiOptions(divisor,
				    alltoall,
				    1, //utils::defaultmpithreads,
				    0);
      
      return new fft3dMPI(*dim,
			  (Complex*)in,
			  (Complex*)out,
			  opts);
    }
    
    void mpifftwpp_delete_fft3d(fft3dMPI* fft) {
      delete fft;
    }
    
    void mpifftwpp_fft3d_forward(fft3dMPI* fft,
				 double __complex__ *in,
				 double __complex__ *out) {
      fft->Forward((Complex*) in, (Complex*) out);
    }

    void mpifftwpp_fft3d_backward(fft3dMPI* fft,
				  double __complex__ *in,
				  double __complex__ *out) {
      fft->Backward((Complex*) in, (Complex*) out);
    }

    
    rcfft3dMPI* mpifftwpp_create_rcfft3d(const utils::split3* rdim,
					 const utils::split3* cdim,
					 double *in,
					 double __complex__ *out){
      /* Test for best block divisor: */
      int divisor = 0;

      /* Test for best alltoall routine: */
      int alltoall = -1;

      auto opts = utils::mpiOptions(divisor,
				    alltoall,
				    1, //utils::defaultmpithreads,
				    0);
      return new rcfft3dMPI(*rdim,
			    *cdim,
			    in,
			    (Complex*)out,
			    opts);
    }


    void mpifftwpp_delete_rcfft3d(rcfft3dMPI* fft){
      delete fft;
    }
    void mpifftwpp_rcfft3d_forward(rcfft3dMPI* fft,
				   double *in,
				   double __complex__ *out){
      fft->Forward(in, (Complex*) out);
    }
    void mpifftwpp_rcfft3d_backward(rcfft3dMPI* fft,
				    double __complex__ *in,
				    double *out){
      fft->Backward((Complex*) in, out);
    }
    
  } /* namespace fftwpp */
  
} /* extern C */

        
        

#endif 
