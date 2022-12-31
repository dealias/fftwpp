#include <vector>

#include "convolution.h"
#include "explicit.h"
#include "direct.h"
#include "utils.h"
#include "Array.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

// Number of iterations.
size_t N0=10000000;
size_t N=0;
size_t nx=0;
size_t ny=0;
size_t nz=0;
size_t mx=4;
size_t my=0;
size_t mz=0;

inline void init(Complex **F,
                 size_t nxp, size_t nyp, size_t nzp,
                 size_t A)
{
  if(A %2 == 0) {
    size_t M=A/2;
    double factor=1.0/sqrt((double) M);
    for(size_t s=0; s < M; ++s) {
      double S=sqrt(1.0+s);
      double ffactor=S*factor;
      double gfactor=1.0/S*factor;
      array3<Complex> f(nxp,nyp,nzp,F[s]);
      array3<Complex> g(nxp,nyp,nzp,F[M+s]);
#pragma omp parallel for
      for(size_t i=0; i < mx; ++i) {
        for(size_t j=0; j < my; j++) {
          for(size_t k=0; k < mz; k++) {
            f[i][j][k]=ffactor*Complex(i+k,j+k);
            g[i][j][k]=gfactor*Complex(2*i+k,j+1+k);
            //cout << f[i][j][k] << " ";
          }
          //cout << endl;
        }
        //cout << endl;
      }
    }
  } else {
    cerr << "Init not implemented for A=" << A << endl;
    exit(1);
  }
}

int main(int argc, char *argv[])
{
  fftw::maxthreads=get_max_threads();

  bool Direct=false;
  bool Implicit=true;
  bool Explicit=false;
  bool Output=false;
  bool Normalized=true;
  bool Pruned=false;

  double K=1.0; // Time limit (seconds)
  size_t minCount=20;

  size_t A=2; // Number of independent inputs
  size_t B=1; // Number of independent outputs

  size_t stats=0; // Type of statistics used in timing test.

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

#ifdef __GNUC__
  optind=0;
#endif
  for (;;) {
    int c = getopt(argc,argv,"hdeiptA:B:K:Om:x:y:z:n:T:uS:");
    if (c == -1) break;

    switch (c) {
      case 0:
        break;
      case 'd':
        Direct=true;
        break;
      case 'e':
        Explicit=true;
        Implicit=false;
        Pruned=false;
        break;
      case 'i':
        Implicit=true;
        Explicit=false;
        break;
      case 'p':
        Explicit=true;
        Implicit=false;
        Pruned=true;
        break;
      case 'A':
        A=atoi(optarg);
        break;
      case 'B':
        B=atoi(optarg);
        break;
      case 'K':
        K=atof(optarg);
        break;
      case 'O':
        Output=true;
        break;
      case 'm':
        mx=my=mz=atoi(optarg);
        break;
      case 'x':
        mx=atoi(optarg);
        break;
      case 'y':
        my=atoi(optarg);
        break;
      case 'z':
        mz=atoi(optarg);
        break;
      case 'u':
        Normalized=false;
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
      case 'h':
      default:
        usage(3);
        usageExplicit(3);
        exit(1);
    }
  }

  if(my == 0) my=mx;
  if(mz == 0) mz=mx;

  nx=cpadding(mx);
  ny=cpadding(my);
  nz=cpadding(mz);

  cout << "nx=" << nx << ", ny=" << ny << ", nz=" << ny << endl;
  cout << "mx=" << mx << ", my=" << my << ", mz=" << mz << endl;

  if(K == 0) minCount=1;
  cout << "K=" << K << endl;
  K *= 1.0e9;

  size_t align=ALIGNMENT;

  array3<Complex> h0;
  if(Direct) {
    h0.Allocate(mx,my,mz,align);
    if(!Normalized) {
      cerr << "-u option is incompatible with -d." << endl;
      exit(-1);
    }
  }

  int nxp=Explicit ? nx : mx;
  int nyp=Explicit ? ny : my;
  int nzp=Explicit ? nz : mz;

  if(!Implicit)
    A=2;

  if(B < 1) B=1;
  if(B > A) {
    cerr << "B=" << B << " is not yet implemented for A=" << A << endl;
    exit(1);
  }

  // Allocate input/ouput memory and set up pointers
  Complex **F=new Complex *[A];
  for(size_t a=0; a < A; ++a)
    F[a]=ComplexAlign(nxp*nyp*nzp);

  // For easy access of first element
  array3<Complex> f(mx,my,mz,F[0]);

  vector<double> T;

  if(Implicit) {
    multiplier *mult;

    switch(A) {
      case 2: mult=multbinary; break;
      case 4: mult=multbinary2; break;
      case 6: mult=multbinary3; break;
      case 8: mult=multbinary4; break;
      case 16: mult=multbinary8; break;
      default: cout << "mult for A=" << A
                    << " is not yet implemented" << endl; exit(1);
    }

    ImplicitConvolution3 C(mx,my,mz,A,B);
    cout << "Using " << C.Threads() << " threads."<< endl;
    double sum=0.0;
    while(sum <= K || T.size() < minCount) {
      init(F,mx,my,mz,A);
      cpuTimer c;
      C.convolve(F,mult);
//      C.convolve(F[0],F[1]);
      double t=c.nanoseconds();
      T.push_back(t);
      sum += t;
    }

    cout << endl;
    timings("Implicit",mx*my*mz,T.data(),T.size(),stats);
    T.clear();

    if(Normalized) {
      double norm=0.125/(mx*my*mz);
      for(size_t i=0; i < mx; i++)
        for(size_t j=0; j < my; j++)
          for(size_t k=0; k < mz; k++)
          f[i][j][k] *= norm;
    }

    if(Direct)
      for(size_t i=0; i < mx; i++)
        for(size_t j=0; j < my; j++)
          for(size_t k=0; k < mz; k++)
            h0[i][j][k]=f[i][j][k];

    if(Output)
      cout << f;
  }

  if(Explicit) {
    if(A != 2) {
      cerr << "Explicit convolutions for A=" << A
           << " are not yet implemented" << endl;
      exit(1);
    }

    ExplicitConvolution3 C(nx,ny,nz,mx,my,mz,f,Pruned);

    double sum=0.0;
    while(sum <= K || T.size() < minCount) {
      init(F,nxp,nyp,nzp,A);
      cpuTimer c;
      C.convolve(F[0],F[1]);
      double t=c.nanoseconds();
      T.push_back(t);
      sum += t;
    }

    cout << endl;
    timings(Pruned ? "Pruned" : "Explicit",mx*my*mz,T.data(),T.size(),stats);
    T.clear();

    if(Direct) {
      for(size_t i=0; i < mx; i++)
        for(size_t j=0; j < my; j++)
          for(size_t k=0; k < mz; k++)
            h0[i][j][k]=F[0][nyp*nzp*i+nzp*j+k];
    }

    if(Output) {
      for(size_t i=0; i < mx; i++) {
        for(size_t j=0; j < my; j++) {
          for(size_t k=0; k < mz; k++)
            cout << F[0][nyp*nzp*i+nzp*j+k] << " ";
          cout << endl;
        }
        cout << endl;
      }
    } else {
      Complex sum=0.0;
      for(size_t i=0; i < mx; i++)
        for(size_t j=0; j < my; j++)
          for(size_t k=0; k < mz; k++)
            sum += F[0][nyp*nzp*i+nzp*j+k];
      cout << endl << "sum=" << sum << endl;
    }
  }

  if(Direct) {
    array3<Complex> h(mx,my,mz,align);
    DirectConvolution3 C(mx,my,mz);
    init(F,mx,my,mz,2);
    cpuTimer c;
    C.convolve(h,F[0],F[1]);
    T[0]=c.nanoseconds();

    timings("Direct",mx*my*mz,T.data(),1);
    T.clear();

    if(Output) {
      for(size_t i=0; i < mx; i++) {
        for(size_t j=0; j < my; j++) {
          for(size_t k=0; k < mz; k++)
            cout << h[i][j][k] << "\t";
          cout << endl;
        }
        cout << endl;
      }
    }

    { // compare implicit or explicit version with direct verion:
      double error=0.0;
      double norm=0.0;
      for(size_t i=0; i < mx; i++) {
        for(size_t j=0; j < my; j++) {
          for(size_t k=0; k < mz; k++) {
            error += abs2(h0[i][j][k]-h[i][j][k]);
            norm += abs2(h[i][j][k]);
          }
        }
      }
      if(norm > 0) error=sqrt(error/norm);
      cout << "error=" << error << endl;
      if (error > 1e-12)
        cerr << "Caution! error=" << error << endl;
    }

  }

  for(size_t a=0; a < A; ++a)
    deleteAlign(F[a]);
  delete [] F;

  return 0;
}
