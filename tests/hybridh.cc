#include "convolve.h"

#define OUTPUT 1

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

unsigned int A=2; // number of inputs
unsigned int B=1; // number of outputs
unsigned int C=1; // number of copies
unsigned int L=512; // input data length
unsigned int M=768; // minimum padded length

int main(int argc, char* argv[])
{
  fftw::maxthreads=1;//get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  optionsHybrid(argc,argv);

  ForwardBackward FB(A,B);

  cout << "Explicit:" << endl;
  // Minimal explicit padding
  fftPadHermitian fft0(L,M,FB,C,true,true);

  double mean0=fft0.report(FB);

  // Optimal explicit padding
  fftPadHermitian fft1(L,M,FB,C,true,false);
  double mean1=min(mean0,fft1.report(FB));

  // Hybrid padding
  fftPadHermitian fft(L,M,FB,C);

  double mean=fft.report(FB);

  if(mean0 > 0)
    cout << "minimal ratio=" << mean/mean0 << endl;
  cout << endl;

  if(mean1 > 0)
    cout << "optimal ratio=" << mean/mean1 << endl;
  cout << endl;

  unsigned H=ceilquotient(L,2);
  Complex *f=ComplexAlign(C*H);

  for(unsigned int c=0; c < C; ++c)
    f[c]=1;
  for(unsigned int j=1; j < H; ++j)
    for(unsigned int c=0; c < C; ++c)
      f[C*j+c]=Complex(j+1,j+1);

  Complex *F=ComplexAlign(fft.fullOutputSize());
//  (fft.*fft.Forward)(f,F,0,NULL);
  fft.W0=ComplexAlign(fft.workSizeW());
  fft.forward(f,F);

#if OUTPUT
  if(L < 50) {
    double *Fr=(double *) F;
    for(unsigned int j=0; j < fft.fullOutputSize(); ++j) {
      if(j % fft.Cm == 0) cout << endl;
      cout << Fr[j] << endl;
    }
  }
#endif

  fft.backward(F,f);

  cout << endl;
  if(L < 30) {
    cout << endl;
    cout << "Inverse:" << endl;
    double scale=1.0/fft.normalization();
    for(unsigned int j=0; j < H*C; ++j)
      cout << f[j]*scale << endl;
    cout << endl;
  }

  return 0;
}
