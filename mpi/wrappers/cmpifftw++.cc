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
    unsigned int mpifftwpp_split_x(const split* dim) {
      return dim->x;
    }
    unsigned int mpifftwpp_split_y(const split* dim) {
      return dim->y;
    }
    unsigned int mpifftwpp_split_x0(const split* dim) {
      return dim->x0;
    }
    unsigned int mpifftwpp_split_y0(const split* dim) {
      return dim->y0;
    }
    unsigned int mpifftwpp_split_X(const split* dim) {
      return dim->X;
    }
    unsigned int mpifftwpp_split_Y(const split* dim) {
      return dim->Y;
    }

      
      void mpitfftwpp_show_complex(const double __complex__ * f,
                                   const unsigned int x,
                                   const unsigned int ny,
                                   const MPI_Comm comm) {
          utils::show((Complex*)f,x,ny,comm);
      }
      
      void mpitfftwpp_show_complex3(const double __complex__ * f,
                                    const unsigned int x,
                                    const unsigned int y,
                                    const unsigned int Z,
                                    const MPI_Comm comm) {
          utils::show((Complex*)f,x,y,Z,comm);
      }
      void mpitfftwpp_show_real(const double * f,
                                const unsigned int x,
                                const unsigned int ny,
                                const MPI_Comm comm) {
          utils::show((double*)f,x,ny,comm);
      }
      
      split3* mpifftwpp_create_split3(const MPIgroup* group,
                                      const unsigned int nx,
                                      const unsigned int ny,
                                      const unsigned int nz) {
          return new utils::split3(nx, ny, nz, *group);
      }
      void mpitfftwpp_show_real3(const double * f,
                                 const unsigned int x,
                                 const unsigned int y,
                                 const unsigned int Z,
                                 const MPI_Comm comm) {
          utils::show((double*)f,x,y,Z,comm);
      }

      
      void mpifftwpp_delete_split3(split3* dim) {
          delete dim;
    }
    unsigned int mpifftwpp_split3_n(const split3* dim) {
      return dim->n;
    }
    unsigned int mpifftwpp_split3_x(const split3* dim) {
      return dim->x;
    }
    unsigned int mpifftwpp_split3_y(const split3* dim) {
      return dim->y;
    }
    unsigned int mpifftwpp_split3_z(const split3* dim) {
      return dim->z;
    }
    unsigned int mpifftwpp_split3_X(const split3* dim) {
      return dim->X;
    }
    unsigned int mpifftwpp_split3_Y(const split3* dim) {
      return dim->Y;
    }
    unsigned int mpifftwpp_split3_Z(const split3* dim) {
      return dim->Z;
    }
    unsigned int mpifftwpp_split3_x0(const split3* dim) {
      return dim->x0;
    }
    unsigned int mpifftwpp_split3_y0(const split3* dim) {
      return dim->y0;
    }
    unsigned int mpifftwpp_split3_z0(const split3* dim) {
      return dim->z0;
    }
      unsigned int mpifftwpp_split3_xyy(const split3* dim) {
          return dim->xy.y;
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
