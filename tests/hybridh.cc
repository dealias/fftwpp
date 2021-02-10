#include "convolve.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

int main(int argc, char* argv[])
{
  fftw::maxthreads=1;//get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  L=512;
  M=768;

  optionsHybrid(argc,argv);

  ForwardBackward FB;
  Application *app=&FB;

  cout << "Explicit:" << endl;
  // Minimal explicit padding
  fftPadHermitian fft0(L,M,*app,C,true,true);

  double mean0=fft0.report(*app);

  // Optimal explicit padding
  fftPadHermitian fft1(L,M,*app,C,true,false);
  double mean1=min(mean0,fft1.report(*app));

  // Hybrid padding
  fftPadHermitian fft(L,M,*app,C);

  double mean=fft.report(*app);

  if(mean0 > 0)
    cout << "minimal ratio=" << mean/mean0 << endl;
  cout << endl;

  if(mean1 > 0)
    cout << "optimal ratio=" << mean/mean1 << endl;
  cout << endl;

  Complex *f=ComplexAlign(C*fft.length());

  unsigned int length=L/2+1;

  for(unsigned int c=0; c < C; ++c)
    f[c]=1;
  for(unsigned int j=1; j < length; ++j)
    for(unsigned int c=0; c < C; ++c)
      f[C*j+c]=Complex(j+1,j+1);

  Complex *F=ComplexAlign(fft.q*fft.worksizeF()/fft.D);// Improve
//  (fft.*fft.Forward)(f,F,0,NULL);
  fft.W0=ComplexAlign(fft.worksizeW());
  fft.forward(f,F);

  if(L < 30) {
    double *Fr=(double *) F;
    for(unsigned int j=0; j < C*fft.size(); ++j)
      cout << Fr[j] << endl;
  }

  fft.backward(F,f);

  cout << endl;
  if(L < 30) {
    cout << endl;
    cout << "Inverse:" << endl;
    unsigned int N=fft.size();
    for(unsigned int j=0; j < (L+1)/2*C; ++j)
      cout << f[j]/N << endl;
    cout << endl;
  }

  return 0;
}
