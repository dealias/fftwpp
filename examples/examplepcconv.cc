#include "Complex.h"
#include "convolution.h"
#include "Array.h"

using namespace std;
using namespace Array;
using namespace fftwpp;

// Multiplcation function in physical space which is passed to
// convolution.  One can also use multbinary, defined in
// convolution.h/cc, or, for four inputs and one output,
// multbinarydot.  The offset argument is necessary for computing
// multi-dimensional convolutions.
void mult(Complex **f, unsigned int m, unsigned int M,
           unsigned int offset) {
  Complex* f0=f[0]+offset;
  Complex* f1=f[1]+offset;
  for(unsigned int j=0; j < m; ++j)
    f0[j] *= f1[j];
}

// Initialization function.
inline void init(Complex **f, unsigned int m) 
{
  for(unsigned int k=0; k < m; k++) {
    f[0][k]=Complex(k,k+1);
    f[1][k]=Complex(k,2*k+1);
  }
}

int main(int argc, char* argv[])
{
  cout << "Non-centered complex convolution:" << endl;

  // Set maximum number of threads to be used:
  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif
  
  unsigned int m=8; // Problem size
  unsigned int A=2; // Number of inputs
  unsigned int B=1; // Number of outputs

  // Allocate input arrays:
  Complex **f = new Complex *[A];
  for(unsigned int s=0; s < A; ++s)
    f[s]=ComplexAlign(m);
  
  // Set up the input:
  init(f,m);
  cout << "input:\nf[0]\tg[0]" << endl;
  for(unsigned int i=0; i < m; i++) 
    cout << f[0][i] << "\t" << f[1][i] << endl;

  // Creat a convolution object C:
  pImplicitConvolution C(m,A,B);

  // Perform the convolution:
  C.convolve(f,mult); // Use "mult" for the multiplication.
  //C.convolve(f); // Use the default binary mulitplication.
  
  // Display output:
  cout << "\noutput:\nf[0]" << endl;
  for(unsigned int i=0; i < m; i++) 
    cout << f[0][i] << endl;
    
  // Free input arrays:
  for(unsigned int s=0; s < A; ++s) 
    deleteAlign(f[s]);
  delete[] f;

  return 0;
}
