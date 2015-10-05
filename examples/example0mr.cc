#include "Array.h"
#include "fftw++.h"

using namespace std;
using namespace Array;
using namespace fftwpp;

int main()
{  
  cout << "Multiple 1D real-to-complex and complex-to-real FFTs" << endl;

  fftw::maxthreads=get_max_threads();
  
  unsigned int nx=4, ny=4;
  unsigned int nyp=ny/2+1;
  size_t align=sizeof(Complex);
  
  cout << "Out-of-place transforms:" << endl;
    
  array2<double> f(nx,ny,align);
  array2<Complex> g(nx,nyp,align);
  unsigned int dist=ny;
  
  mrcfft1d Forward(ny, // size
                   nx, // how many
                   1, // stride
                   dist, // distance between the start of each input vector
                   f, // input
                   g); // output

  mcrfft1d Backward(ny, // size (length of real output)
                    nx, // how many
                    1, // stride
                    nyp, // distance between the start of each input vector
                    g, // input
                    f); // output

  // Initialize data:
  for(unsigned int i=0; i < nx; i++) 
    for(unsigned int j=0; j < ny; j++) 
      f(i,j)=i+j;
        
  cout << endl << "input:" << endl << f;

  Forward.fft(f,g);
  
  cout << endl << "output:" << endl << g;
  
  Backward.fftNormalized(g,f);
  
  cout << endl << "back to input:" << endl << f;

}
