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
  M=1024;

  optionsHybrid(argc,argv);

  ForwardBackward FB;
  Application *app=&FB;


#if 1
  cout << "Explicit:" << endl;
  // Minimal explicit padding
  fftPad fft0(L,M,*app,C,true,true);

  double mean0=fft0.report(*app);

  // Optimal explicit padding
  fftPad fft1(L,M,*app,C,true,false);
  double mean1=min(mean0,fft1.report(*app));

  // Hybrid padding
  fftPad fft(L,M,*app,C);
//  fftPadCentered fft(L,M,*app,C);

  double mean=fft.report(*app);

  if(mean0 > 0)
    cout << "minimal ratio=" << mean/mean0 << endl;
  cout << endl;

  if(mean1 > 0)
    cout << "optimal ratio=" << mean/mean1 << endl;
  cout << endl;

  unsigned int M=fft.size();

  Complex *f=ComplexAlign(C*fft.length());
  // C*qm
  Complex *F=ComplexAlign(fft.q*fft.worksizeF()/fft.D);// Improve
  fft.W0=ComplexAlign(fft.worksizeW());

  unsigned int Length=L;

  for(unsigned int j=0; j < Length; ++j)
    for(unsigned int c=0; c < C; ++c)
      f[C*j+c]=Complex(j+1,j+2);

  fft.forward(f,F);

#if 0
  for(unsigned int j=0; j < fft.size(); ++j)
    for(unsigned int c=0; c < C; ++c)
      cout << F[C*j+c] << endl;
#endif

  Complex *f0=ComplexAlign(C*fft.length());
  Complex *F0=ComplexAlign(C*M);

  for(unsigned int j=0; j < fft.size(); ++j)
    for(unsigned int c=0; c < C; ++c)
      F0[C*j+c]=F[C*j+c];

  fft.backward(F0,f0);

  if(L < 30) {
    cout << endl;
    cout << "Inverse:" << endl;
    unsigned int M=fft.size();
    for(unsigned int j=0; j < C*L; ++j)
      cout << f0[j]/M << endl;
    cout << endl;
  }

  Complex *F2=ComplexAlign(M*C);
  fftPad fft2(L,M,C,M,1,1);

  for(unsigned int j=0; j < L; ++j)
    for(unsigned int c=0; c < C; ++c)
      f[C*j+c]=Complex(j+1,j+2);
  fft2.forward(f,F2);

#if 0
  cout << endl;
  for(unsigned int j=0; j < fft.size(); ++j)
    for(unsigned int c=0; c < C; ++c)
      cout << F2[C*j+c] << endl;
  cout << endl;
#endif

  double error=0.0, norm=0.0;
  double error2=0.0, norm2=0.0;

  unsigned int m=fft.m;
  unsigned int p=fft.p;
  unsigned int n=fft.n;

  for(unsigned int s=0; s < m; ++s) {
    for(unsigned int t=0; t < p; ++t) {
      for(unsigned int r=0; r < n; ++r) {
        for(unsigned int c=0; c < C; ++c) {
          unsigned int i=C*(n*(p*s+t)+r)+c;
          error += abs2(F[C*(m*(p*r+t)+s)+c]-F2[i]);
          norm += abs2(F2[i]);
        }
      }
    }
  }

  for(unsigned int j=0; j < C*L; ++j) {
    error2 += abs2(f0[j]/M-f[j]);
    norm2 += abs2(f[j]);
  }

  if(norm > 0) error=sqrt(error/norm);
  double eps=1e-12;
  if(error > eps || error2 > eps)
    cerr << endl << "WARNING: " << endl;
  cout << "forward error=" << error << endl;
  cout << "backward error=" << error2 << endl;

  exit(-1);
#endif
  {

#if 0
    {
      unsigned int Lx=L;
      unsigned int Ly=Lx;
      unsigned int Mx=M;
      unsigned int My=Mx;

      cout << "Lx=" << Lx << endl;
      cout << "Mx=" << Mx << endl;
      cout << endl;

//      fftPad fftx(Lx,Mx,Ly,Lx,2,1);
      fftPad fftx(Lx,Mx,*app,Ly);

//      fftPad ffty(Ly,My,1,Ly,2,1);
      fftPad ffty(Ly,My,FB,1);

      HybridConvolution convolvey(ffty);

      Complex **f=new Complex *[A];
      Complex **h=new Complex *[B];
      for(unsigned int a=0; a < A; ++a)
        f[a]=ComplexAlign(Lx*Ly);
      for(unsigned int b=0; b < B; ++b)
        h[b]=ComplexAlign(Lx*Ly);

      array2<Complex> f0(Lx,Ly,f[0]);
      array2<Complex> f1(Lx,Ly,f[1]);

      for(unsigned int i=0; i < Lx; ++i) {
        for(unsigned int j=0; j < Ly; ++j) {
          f0[i][j]=Complex(i,j);
          f1[i][j]=Complex(2*i,j+1);
        }
      }

      if(Lx*Ly < 200) {
        for(unsigned int i=0; i < Lx; ++i) {
          for(unsigned int j=0; j < Ly; ++j) {
            cout << f0[i][j] << " ";
          }
          cout << endl;
        }
      }
      HybridConvolution2 Convolve2(fftx,convolvey);

      unsigned int K=1000;
      double t0=totalseconds();

      for(unsigned int k=0; k < K; ++k)
        Convolve2.convolve(f,h,multbinary);

      double t=totalseconds();
      cout << (t-t0)/K << endl;
      cout << endl;

      array2<Complex> h0(Lx,Ly,h[0]);

      Complex sum=0.0;
      for(unsigned int i=0; i < Lx; ++i) {
        for(unsigned int j=0; j < Ly; ++j) {
          sum += h0[i][j];
        }
      }

      cout << "sum=" << sum << endl;
      cout << endl;

      if(Lx*Ly < 200) {
        for(unsigned int i=0; i < Lx; ++i) {
          for(unsigned int j=0; j < Ly; ++j) {
            cout << h0[i][j] << " ";
          }
          cout << endl;
        }
      }
    }
#endif

#if 0
    fftPad fft(L,M);

    unsigned int L0=fft.length();
    Complex *f=ComplexAlign(L0);
    Complex *g=ComplexAlign(L0);

    for(unsigned int j=0; j < L0; ++j) {
#if OUTPUT
      f[j]=Complex(j,j+1);
      g[j]=Complex(j,2*j+1);
#else
      f[j]=0.0;
      g[j]=0.0;
#endif
    }

    HybridConvolution Convolve(fft);

    Complex *F[]={f,g};
//  Complex *h=ComplexAlign(L0);
//  Complex *H[]={h};
#if OUTPUT
    unsigned int K=1;
#else
    unsigned int K=10000;
#endif
    double t0=totalseconds();

    for(unsigned int k=0; k < K; ++k)
      Convolve.convolve(F,F,multbinary);

    double t=totalseconds();
    cout << (t-t0)/K << endl;
    cout << endl;
#if OUTPUT
    for(unsigned int j=0; j < L; ++j)
      cout << F[0][j] << endl;
#endif

#endif

  }

  return 0;
}
