#include <stdio.h>
#include <complex.h>
#include <mpi.h>
#include "cfftw++.h"
#include "cmpifftw++.h"

#include <stdio.h>



int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  /* int process_Rank, size_Of_Cluster; */
  /* MPI_Comm_size(MPI_COMM_WORLD, &size_Of_Cluster); */
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if(rank == 0) {
    printf("Example of calling fftw++ convolutions from C:\n");
      }
    
  {
    if(rank == 0) {
      printf("2D FFT:\n");
    }
  
    int nx = 8;
    int ny = 8;

    MPIgroup* group = mpifftwpp_create_group_1d(MPI_COMM_WORLD, ny);

    split* dim = mpifftwpp_create_split(MPI_COMM_WORLD, nx, ny);
        
    double complex *f = create_complexAlign(mpifftwpp_split_n(dim));
    double complex *g = create_complexAlign(mpifftwpp_split_n(dim));

    fft2dMPI* fft2 = mpifftwpp_create_fft2d(dim, f, g);
    
    const unsigned int dimx = mpifftwpp_split_x(dim);
    const unsigned int dimy = mpifftwpp_split_y(dim);
    const unsigned int dimY = mpifftwpp_split_Y(dim);
    const unsigned int dimx0 = mpifftwpp_split_x0(dim);
    unsigned int c = 0;
    for(unsigned int i = 0; i < dimx; ++i) {
      unsigned int ii = dimx0 + i;
      for(unsigned int j = 0; j < dimY; j++) {
	f[c++] = ii + I *j;
      }
    }
    mpitfftwpp_show_complex(f, dimx, ny, MPI_COMM_WORLD);

    mpifftwpp_fft2d_forward(fft2, f, g);
    
    mpitfftwpp_show_complex(g, nx, dimy, MPI_COMM_WORLD);
    
    delete_complexAlign(g);
    delete_complexAlign(f);
    
    mpifftwpp_delete_fft2d(fft2);
    mpifftwpp_delete_split(dim);
    mpifftwpp_delete_group(group);
  }

  
  {
    if(rank == 0) {
      printf("2D real/complex FFT:\n");
    }
  
    int nx = 8;
    int ny = 8;
    unsigned int nyp = ny / 2 + 1;

    MPIgroup* group = mpifftwpp_create_group_1d(MPI_COMM_WORLD, ny);
    split* rdim = mpifftwpp_create_split(MPI_COMM_WORLD, nx, ny);
    split* cdim  = mpifftwpp_create_split(MPI_COMM_WORLD, nx, nyp);

    double *f = create_doubleAlign(mpifftwpp_split_n(rdim));
    double complex *g = create_complexAlign(mpifftwpp_split_n(cdim));

    rcfft2dMPI* rcfft2 = mpifftwpp_create_rcfft2d(rdim, cdim, f, g);
    
    const unsigned int rdimx = mpifftwpp_split_x(rdim);
    const unsigned int rdimY = mpifftwpp_split_Y(rdim);
    const unsigned int rdimx0 = mpifftwpp_split_x0(rdim);
    unsigned int c = 0;
    for(unsigned int i = 0; i < rdimx; ++i) {
      unsigned int ii = rdimx0 + i;
      for(unsigned int j = 0; j < rdimY; j++) {
	f[c++] = ii + j;
      }
    }
    mpitfftwpp_show_real(f, rdimx, ny, MPI_COMM_WORLD);
    
    mpifftwpp_rcfft2d_forward(rcfft2, f, g);

    const unsigned int cdimX = mpifftwpp_split_X(cdim);
    const unsigned int cdimy = mpifftwpp_split_y(cdim);
    mpitfftwpp_show_complex(g, cdimX, cdimy, MPI_COMM_WORLD);
    
    mpifftwpp_delete_rcfft2d(rcfft2);
    
    delete_complexAlign(g);
    delete_doubleAlign(f);
    
    mpifftwpp_delete_split(cdim);
    mpifftwpp_delete_split(rdim);
    mpifftwpp_delete_group(group);    
  }
  
  {
    if(rank == 0) {
      printf("3D FFT:\n");
    }

    int nx = 4;
    int ny = 4;
    int nz = 4;

    MPIgroup* group = mpifftwpp_create_group_2d(MPI_COMM_WORLD, nx, ny);
    split3* dim =  mpifftwpp_create_split3(group, nx, ny, nz);

    double complex *f = create_complexAlign(mpifftwpp_split3_n(dim));
    double complex *g = create_complexAlign(mpifftwpp_split3_n(dim));
    
    fft3dMPI* fft3 = mpifftwpp_create_fft3d(dim, f, g);
    
    const unsigned int dimx = mpifftwpp_split3_x(dim);
    const unsigned int dimy = mpifftwpp_split3_y(dim);
    const unsigned int dimz = mpifftwpp_split3_z(dim);
    const unsigned int dimX = mpifftwpp_split3_X(dim);
    const unsigned int dimZ = mpifftwpp_split3_Z(dim);
    const unsigned int dimx0 = mpifftwpp_split3_x0(dim);
    const unsigned int dimy0 = mpifftwpp_split3_y0(dim);
    const unsigned int dimxyy = mpifftwpp_split3_xyy(dim);
        
    unsigned int c=0;
    for(unsigned int i = 0; i < dimx; ++i) {
      unsigned int ii = dimx0 + i;
      for(unsigned int j = 0; j < dimy; j++) {
	unsigned int jj = dimy0 + j;
	for(unsigned int k=0; k < dimZ; k++) {
	  f[c++] = 10*k + ii + I * jj;
	}
      }
    }

    if(rank == 0) {
      printf("input:\n");
    }
    mpitfftwpp_show_complex3(f, dimx, dimy, dimZ, MPI_COMM_WORLD);

    mpifftwpp_fft3d_forward(fft3, f, g);

    if(rank == 0) {
      printf("output:\n");
    }
    mpitfftwpp_show_complex3(g, dimX, dimxyy, dimz, MPI_COMM_WORLD);
    
    delete_complexAlign(g);
    delete_complexAlign(f);
    
    mpifftwpp_delete_fft3d(fft3);
    mpifftwpp_delete_split3(dim);
    mpifftwpp_delete_group(group);
  }

  {
    if(rank == 0) {
      printf("3D real/complex FFT:\n");
    }
  
    int nx = 4;
    int ny = 4;
    int nz = 4;

    int nzp = nz / 2 + 1;
    
    MPIgroup* group = mpifftwpp_create_group_2d(MPI_COMM_WORLD, nx, ny);

    split3* rdim = mpifftwpp_create_split3(group, nx, ny, nz);
    split3* cdim = mpifftwpp_create_split3(group, nx, ny, nzp);

    double *f = create_doubleAlign(mpifftwpp_split3_n(rdim));
    double complex *g = create_complexAlign(mpifftwpp_split3_n(cdim));

    rcfft3dMPI* rcfft3 = mpifftwpp_create_rcfft3d(rdim, cdim, f, g);

    const unsigned int rdimx = mpifftwpp_split3_x(rdim);
    const unsigned int rdimy = mpifftwpp_split3_y(rdim);
    const unsigned int rdimx0 = mpifftwpp_split3_x0(rdim);
    const unsigned int rdimy0 = mpifftwpp_split3_y0(rdim);
    const unsigned int rdimZ = mpifftwpp_split3_Z(rdim);
    unsigned int c=0;
    for(unsigned int i = 0; i < rdimx; ++i) {
      unsigned int ii = rdimx0 + i;
      for(unsigned int j = 0; j < rdimy; j++) {
	unsigned int jj = rdimy0 + j;
	for(unsigned int k=0; k < rdimZ; k++) {
	  f[c++] = ii + jj + k + 1;
	}
      }
    }
    
    mpifftwpp_rcfft3d_forward(rcfft3, f, g);
    
    const unsigned int cdimX = mpifftwpp_split3_X(cdim);
    const unsigned int cdimxyy = mpifftwpp_split3_xyy(cdim);
    const unsigned int cdimz = mpifftwpp_split3_z(cdim);
    mpitfftwpp_show_complex3(g, cdimX, cdimxyy, cdimz, MPI_COMM_WORLD);
    
    delete_complexAlign(g);
    delete_doubleAlign(f);
    
    mpifftwpp_delete_split3(cdim);
    mpifftwpp_delete_split3(rdim);
    mpifftwpp_delete_group(group);
    
  }

  
  
  MPI_Finalize();
  
  return 0;

}
