#include <fftw3.h>
#include <fftw++.h>

using namespace fftwpp;

fftw_plan plan_transpose(int rows, int cols, int length, Complex *in,
                         Complex *out)
{
  fftw_iodim dims[3];

  dims[0].n  = rows;
  dims[0].is = length*cols;
  dims[0].os = length;

  dims[1].n  = cols;
  dims[1].is = length;
  dims[1].os = length*rows;

  dims[2].n  = length;
  dims[2].is = 1;
  dims[2].os = 1;

  return fftw_plan_guru_dft(0,NULL,3,dims,
                            (fftw_complex *) in,(fftw_complex *) out,
                            1,fftw::effort);
}

