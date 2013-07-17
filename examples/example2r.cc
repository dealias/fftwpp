#include "Array.h"
#include "fftw++.h"
#include <cstdlib>
#include <time.h>       /* time */


// Compile with
// g++ -I .. -fopenmp example2r.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace Array;
using namespace fftwpp;

int main()
{
  fftw::maxthreads=get_max_threads();
  
  //srand(time(NULL));

  unsigned int nx=4, ny=4;
  unsigned int nyp=ny/2+1;
  //  size_t align=sizeof(Complex);
  
  Complex *pg=ComplexAlign(nx*nyp);
  double *pf=(double *)ComplexAlign(nx*ny/2);
  //double *pf=(double *)pg;

  // sign = -1
  //rcfft2d Forward(ny,f,g);
  rcfft2d Forward0(nx,ny,pf,pg);
    
  // sign = +1
  //crfft2d Backward(ny,g,f);
  crfft2d Backward0(nx,ny,pg,pf);  
  
  for(unsigned int i=0; i < nx; i++) 
    for(unsigned int j=0; j < ny; j++) 
      pf[i*nyp+j]=rand()%9+1;

  for(unsigned int i=0; i < nx; i++) {
    for(unsigned int j=0; j < ny; j++) {
      cout << pf[i*nyp+j] << " ";
    }
    cout << endl;
  }
  cout << endl;

  //  cout << f << endl;

  //Forward.fft(f,g);
  Forward0.fft0(pf,pg);

  unsigned int nyhalf=ny/2+1;
  for(unsigned int i=0; i < nx; i++) {
    for(unsigned int j=0; j < nyhalf; j++) {
      cout << pg[i*nyhalf+j] << " ";
    }
    cout << endl;
  }
  cout << endl;
      
  //  cout << g << endl;
  
  //Backward.fftNormalized(g,f);
  Backward0.fft0Normalized(pg,pf);

  for(unsigned int i=0; i < nx; i++) {
    for(unsigned int j=0; j < ny; j++) {
      cout << pf[i*nyp+j] << " ";
    }
    cout << endl;
  }
  cout << endl;
  //  cout << f << endl;
}
