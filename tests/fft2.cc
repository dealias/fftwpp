#include "Complex.h"
#include "Array.h"
#include "fftw++.h"
#include "utils.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

// Number of iterations.
unsigned int N0=10000000;
unsigned int N=0;
unsigned int mx=4;
unsigned int my=4;

bool Direct=false, Implicit=true, Explicit=false, Pruned=false;

inline void init(array2<Complex>& f) 
{
  for(unsigned int i=0; i < mx; ++i)
    for(unsigned int j=0; j < my; j++)
      f[i][j]=Complex(i,j);
}
  
unsigned int outlimit=100;

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();
  int r=-1; // which of the 8 options do we do?  r=-1 does all of them.

  int stats=0; // Type of statistics used in timing test.

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif  
  
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"hN:m:x:y:n:T:S:r:");
    if (c == -1) break;
                
    switch (c) {
      case 0:
        break;
      case 'N':
        N=atoi(optarg);
        break;
      case 'm':
        mx=my=atoi(optarg);
        break;
      case 'x':
        mx=atoi(optarg);
        break;
      case 'y':
        my=atoi(optarg);
        break;
      case 'n':
        N0=atoi(optarg);
        break;
      case 'T':
        fftw::maxthreads=max(atoi(optarg),1);
        break;
      case 'S':
        stats=atoi(optarg);
        break;
      case 'r':
        r=atoi(optarg);
        break;
      case 'h':
      default:
        usageFFT(2);
        exit(0);
    }
  }

  if(my == 0) my=mx;

  cout << "mx=" << mx << ", my=" << my << endl;
  
  if(N == 0) {
    N=N0/mx/my;
    N = max(N, 20);
  }
  cout << "N=" << N << endl;
  
  size_t align=sizeof(Complex);

  array2<Complex> f(mx,my,align);
  array2<Complex> g(mx,my,align);

  double *T=new double[N];

  if(r == -1 || r == 0) { // conventional FFT, in-place
    fft2d Forward2(-1,f);
    fft2d Backward2(1,f);
    
    for(unsigned int i=0; i < N; ++i) {
      init(f);
      seconds();
      Forward2.fft(f);
      Backward2.fft(f);
      T[i]=0.5*seconds();
      Backward2.Normalize(f);
    }
    timings("fft2d, in-place",mx,T,N,stats);
  }
  
  if(r == -1 || r == 1) { // conventional FFT, out-of-place

    fft2d Forward2(-1,f,g);
    fft2d Backward2(1,f,g);
    
    for(unsigned int i=0; i < N; ++i) {
      init(f);
      seconds();
      Forward2.fft(f,g);
      Backward2.fft(g,f);
      T[i]=0.5*seconds();
      Backward2.Normalize(f);
    }
    timings("fft2d, out-of-place",mx,T,N,stats);
  }

  if(r == -1 || r == 2)  { // using the transpose, in-place

    Transpose Txy(mx,my,1,f(),f(),fftw::maxthreads);
    Transpose Tyx(my,mx,1,f(),f(),fftw::maxthreads);
  
    mfft1d Forwardx(mx,-1,my,1,my);
    mfft1d Backwardx(mx,1,my,1,my);

    mfft1d Forwardy(my,-1,mx,1,mx);
    mfft1d Backwardy(my,1,mx,1,mx);

    for(unsigned int i=0; i < N; ++i) {
      init(f);
      seconds();

      Forwardy.fft(f);
      Txy.transpose(f());
      Forwardx.fft(f);

      Backwardx.fft(f);
      Tyx.transpose(f());
      Backwardy.fft(f);
      T[i]=0.5*seconds();
      Backwardx.Normalize(f);
      Backwardy.Normalize(f);
    }
    timings("transpose and mfft, in-place",mx,T,N,stats);
  }

  if(r == -1 || r == 3)  { // using the transpose, out-of-place

    Transpose Txy(mx,my,1,f(),g(),fftw::maxthreads);
    Transpose Tyx(my,mx,1,f(),g(),fftw::maxthreads);
  
    mfft1d Forwardx(mx,-1,my,1,my,f,g);
    mfft1d Backwardx(mx,1,my,1,my,f,g);

    mfft1d Forwardy(my,-1,mx,1,mx,f,g);
    mfft1d Backwardy(my,1,mx,1,mx,f,g);

    for(unsigned int i=0; i < N; ++i) {
      init(f);
      seconds();

      Forwardy.fft(f,g);
      Txy.transpose(g(),f());
      Forwardx.fft(f,g);

      Backwardx.fft(g,f);
      Tyx.transpose(f(),g());
      Backwardy.fft(g,f);
      T[i]=0.5*seconds();
      Backwardx.Normalize(f);
      Backwardy.Normalize(f);
    }
    timings("transpose and mfft, out-of-place",mx,T,N,stats);
  }
  
  if(r == -1 || r == 4) { // full transpose, in-place

    Transpose Txy(mx,my,1,f(),f(),fftw::maxthreads);
    Transpose Tyx(my,mx,1,f(),f(),fftw::maxthreads);
  
    mfft1d Forwardx(mx,-1,my,1,my);
    mfft1d Backwardx(mx,1,my,1,my);

    mfft1d Forwardy(my,-1,mx,1,mx);
    mfft1d Backwardy(my,1,mx,1,mx);

    for(unsigned int i=0; i < N; ++i) {
      init(f);
      seconds();

      Forwardy.fft(f);
      Txy.transpose(f());
      Forwardx.fft(f);

      Tyx.transpose(f());
      Txy.transpose(f());

      Backwardx.fft(f);
      Tyx.transpose(f());
      Backwardy.fft(f);
      T[i]=0.5*seconds();
      Backwardx.Normalize(f);
      Backwardy.Normalize(f);
    }
    timings("2 transposes and mfft, in-place",mx,T,N,stats);
  }

  if(r == -1 || r == 5) { // full transpose, out-of-place

    Transpose Txy(mx,my,1,f(),g(),fftw::maxthreads);
    Transpose Tyx(my,mx,1,f(),g(),fftw::maxthreads);
  
    mfft1d Forwardx(mx,-1,my,1,my,f,g);
    mfft1d Backwardx(mx,1,my,1,my,f,g);

    mfft1d Forwardy(my,-1,mx,1,mx,f,g);
    mfft1d Backwardy(my,1,mx,1,mx,f,g);

    for(unsigned int i=0; i < N; ++i) {
      init(f);
      seconds();

      Forwardy.fft(f,g);
      Txy.transpose(g(),f());
      Forwardx.fft(f,g);
      
      Tyx.transpose(g(),f());
      
      Txy.transpose(f(),g());

      Backwardx.fft(g,f);
      Tyx.transpose(f(),g());
      Backwardy.fft(g,f);
      T[i]=0.5*seconds();
      Backwardx.Normalize(f);
      Backwardy.Normalize(f);
    }
    timings("2 transposes and mfft, out-of-place",mx,T,N,stats);
  }
  
  if(r == -1 || r == 6)  { // using strides, in-place

    mfft1d Forwardx(mx,-1,my,my,1,f,f);
    mfft1d Backwardx(mx,1,my,my,1,f,f);

    mfft1d Forwardy(my,-1,mx,1,mx,f,f);
    mfft1d Backwardy(my,1,mx,1,mx,f,f);

    for(unsigned int i=0; i < N; ++i) {
      init(f);
      seconds();

      Forwardy.fft(f);
      Forwardx.fft(f);

      Backwardx.fft(f);
      Backwardy.fft(f);
      T[i]=0.5*seconds();
      Backwardx.Normalize(f);
      Backwardy.Normalize(f);
    }
    timings("strided mfft in-place",mx,T,N,stats);
  }


  if(r == -1 || r == 7) { // using strides, out-of-place

    mfft1d Forwardx(mx,-1,my,my,1,f,g);
    mfft1d Backwardx(mx,1,my,my,1,f,g);

    mfft1d Forwardy(my,-1,mx,1,mx,f,g);
    mfft1d Backwardy(my,1,mx,1,mx,f,g);

    for(unsigned int i=0; i < N; ++i) {
      init(f);
      seconds();

      Forwardy.fft(f,g);
      Forwardx.fft(g,f);

      Backwardx.fft(f,g);
      Backwardy.fft(g,f);
      T[i]=0.5*seconds();
      Backwardx.Normalize(f);
      Backwardy.Normalize(f);
    }
    timings("strided mfft out-of-place",mx,T,N,stats);
  }
  
  cout << endl;
  if(mx*my < outlimit) {
    for(unsigned int i=0; i < mx; i++) {
      for(unsigned int j=0; j < my; j++)
        cout << f[i][j] << "\t";
      cout << endl;
    }
  }
  else cout << f[0][0] << endl;
    
  delete [] T;
  
  return 0;
}

