#include <vector>

#include "convolution.h"
#include "explicit.h"
#include "direct.h"
#include "utils.h"
#include "options.h"
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
size_t nxp;
size_t nyp;
size_t nzp;
bool xcompact=true;
bool ycompact=true;
bool zcompact=true;

size_t threads;

bool Explicit=false;

inline void init(Complex **F,
                 size_t mx, size_t my, size_t mz,
                 size_t nxp, size_t nyp, size_t nzp,
                 size_t A, bool xcompact, bool ycompact, bool zcompact)
{
  if(A % 2 == 0) {
    size_t M=A/2;
    size_t xoffset=Explicit ? nxp/2-mx+1 : !xcompact;
    size_t yoffset=Explicit ? nyp/2-my+1 : !ycompact;
    size_t nx=2*mx-1;
    size_t ny=2*my-1;

    double factor=1.0/sqrt((double) M);
    for(size_t s=0; s < M; ++s) {
      double S=sqrt(1.0+s);
      double ffactor=S*factor;
      double gfactor=1.0/S*factor;

      array3<Complex> f(nxp,nyp,nzp,F[s]);
      array3<Complex> g(nxp,nyp,nzp,F[M+s]);

      if(!xcompact) {
        for(size_t j=0; j < ny+!ycompact; ++j) {
          for(size_t k=0; k < mz+!zcompact; ++k) {
            f[0][j][k]=0.0;
            g[0][j][k]=0.0;
          }
        }
      }
      if(!ycompact) {
        for(size_t i=0; i < nx+!xcompact; ++i) {
          for(size_t k=0; k < mz+!zcompact; ++k) {
            f[i][0][k]=0.0;
            g[i][0][k]=0.0;
          }
        }
      }

      if(!zcompact) {
        for(size_t i=0; i < nx+!xcompact; ++i) {
          for(size_t j=0; j < ny+!ycompact; ++j) {
            f[i][j][mz]=0.0;
            g[i][j][mz]=0.0;
          }
        }
      }

      PARALLEL(
        for(size_t i=0; i < nx; ++i) {
          size_t I=i+xoffset;
          for(size_t j=0; j < ny; ++j) {
            size_t J=j+yoffset;
            for(size_t k=0; k < mz; ++k) {
              f[I][J][k]=ffactor*Complex(i+k,j+k);
              g[I][J][k]=gfactor*Complex(2*i+k,j+1+k);
            }
          }
        }
        );
    }
  } else {
    cerr << "Init not implemented for A=" << A << endl;
    exit(1);
  }
}

int main(int argc, char *argv[])
{
  threads=fftw::maxthreads=parallel::get_max_threads();

  bool Direct=false;
  bool Implicit=true;
  bool Output=false;
  bool Normalized=true;

  double K=1.0; // Time limit (seconds)
  size_t minCount=20;

  size_t A=2; // Number of independent inputs
  size_t B=1; // Number of outputs

  int stats=0; // Type of statistics used in timing test.

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

#ifdef __GNUC__
  optind=0;
#endif
  for (;;) {
    int c = getopt(argc,argv,"hdeipA:B:K:Om:x:y:z:n:T:uS:X:Y:Z:");
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
        break;
      case 'i':
        Implicit=true;
        Explicit=false;
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
      case 'T':
        threads=fftw::maxthreads=max(atoi(optarg),1);
        break;
      case 'S':
        stats=atoi(optarg);
        break;
      case 'X':
        xcompact=atoi(optarg) == 0;
        break;
      case 'Y':
        ycompact=atoi(optarg) == 0;
        break;
      case 'Z':
        zcompact=atoi(optarg) == 0;
        break;
      case 'h':
      default:
        usage(3);
        usageExplicit(3);
        usageCompact(3);
        exit(1);
    }
  }

  if(my == 0) my=mx;
  if(mz == 0) mz=mx;

  nx=hpadding(mx);
  ny=hpadding(my);
  nz=hpadding(mz);

  cout << "nx=" << nx << ", ny=" << ny << ", nz=" << ny << endl;
  cout << "mx=" << mx << ", my=" << my << ", mz=" << mz << endl;

  if(K == 0) minCount=1;
  cout << "K=" << K << endl;
  K *= 1.0e9;

  size_t align=ALIGNMENT;
  nxp=Explicit ? nx : 2*mx-xcompact;
  nyp=Explicit ? ny : 2*my-ycompact;
  nzp=Explicit ? nz/2+1 : mz+!zcompact;

  cout << "nxp=" << nxp << ", nyp=" << nyp << ", nzp=" << nzp << endl;

  if(B < 1) B=1;
  if(B > A) {
    cerr << "B=" << B << " is not yet implemented for A=" << A << endl;
    exit(1);
  }

  Complex **F=new Complex *[A];
  for(size_t a=0; a < A; ++a)
    F[a]=ComplexAlign(nxp*nyp*nzp);

  // For easy access of first element
  array3<Complex> f(nxp,nyp,nzp,F[0]);

  array3<Complex> h0;
  if(Direct) {
    h0.Allocate(mx,my,mz,align);
    if(!Normalized) {
      cerr << "-u option is incompatible with -d." << endl;
      exit(-1);
    }
  }

  vector<double> T;

  if(Implicit) {
    ImplicitHConvolution3 C(mx,my,mz,xcompact,ycompact,zcompact,A,B);
    cout << "threads=" << C.Threads() << endl << endl;

    realmultiplier *mult;
    switch(A) {
      case 2: mult=multbinary; break;
      case 4: mult=multbinary2; break;
      default: cerr << "A=" << A << " is not yet implemented" << endl; exit(1);
    }

    double sum=0.0;
    while(sum <= K || T.size() < minCount) {
      init(F,mx,my,mz,nxp,nyp,nzp,A,xcompact,ycompact,zcompact);
      cpuTimer c;
      C.convolve(F,mult);
//      C.convolve(f,g);
      double t=c.nanoseconds();
      T.push_back(t);
      sum += t;
    }

    timings("Implicit",(2*mx-1)*(2*my-1)*(2*mz-1),T.data(),T.size(),stats);
    T.clear();
    cout << endl;

    if(Normalized) {
      double norm=1.0/(27.0*mx*my*mz);
      for(size_t i=!xcompact; i < nxp; ++i) {
        for(size_t j=!ycompact; j < nyp; ++j) {
          for(size_t k=0; k < mz; k++)
          f[i][j][k] *= norm;
        }
      }
    }

    if(Direct) {
      for(size_t i=0; i < mx; i++)
        for(size_t j=0; j < my; j++)
          for(size_t k=0; k < mz; k++)
            h0[i][j][k]=f[i+!xcompact][j+!ycompact][k];
    }

    if(Output) {
      for(size_t i=!xcompact; i < nxp; ++i) {
        for(size_t j=!ycompact; j < nyp; ++j) {
          for(size_t k=0; k < mz; ++k)
            cout << f[i][j][k] << "\t";
          cout << endl;
        }
        cout << endl;
      }
    }
  }

  if(Explicit) {
    size_t M=A/2;
    ExplicitHConvolution3 C(nx,ny,nz,mx,my,mz,f,M);
    cout << "threads=" << C.Threads() << endl << endl;

    array3<Complex> g(nxp,nyp,nzp,F[1]);

    double sum=0.0;
    while(sum <= K || T.size() < minCount) {
      cpuTimer c;
      f=0.0;
      g=0.0;
      // Conservative estimate accounting for extra zero padding above.
      double Sum=c.nanoseconds()*19.0/27.0;
      init(F,mx,my,mz,nxp,nyp,nzp,A,true,true,true);
      cpuTimer c2;
      C.convolve(F,F+M);
      double t=c2.nanoseconds()+Sum;
      T.push_back(t);
      sum += t;
    }


    timings("Explicit",(2*mx-1)*(2*my-1)*(2*mz-1),T.data(),T.size(),stats);
    cout << endl;

    size_t xoffset=nx/2-mx+1;
    size_t yoffset=ny/2-my+1;

    if(Direct) {
      for(size_t i=0; i < mx; i++)
        for(size_t j=0; j < my; j++)
          for(size_t k=0; k < mz; ++k)
            h0[i][j][k]=f[xoffset+i][yoffset+j][k];
    }

    if(Output) {
      for(size_t i=xoffset; i < xoffset+2*mx-1; i++) {
        for(size_t j=yoffset; j < yoffset+2*my-1; j++) {
          for(size_t k=0; k < mz; ++k)
            cout << f[i][j][k] << "\t";
          cout << endl;
        }
        cout << endl;
      }
      cout << endl;
    }
  }

  if(Direct) {
    size_t nxp=2*mx-1;
    size_t nyp=2*my-1;

    array3<Complex> h(nxp,nyp,mz,align);

    directconvh3 C(mx,my,mz);
    init(F,mx,my,mz,nxp,nyp,mz,A,true,true,true);
    cpuTimer c;
    C.convolve(h,F[0],F[1]);
    T[0]=c.nanoseconds();

    timings("Direct",(2*mx-1)*(2*my-1)*(2*mz-1),T.data(),1);
    T.clear();
    cout << endl;

    if(Output)
      for(size_t i=0; i < nxp; ++i) {
        for(size_t j=0; j < nyp; ++j) {
          for(size_t k=0; k < mz; ++k)
            cout << h[i][j][k] << "\t";
          cout << endl;
        }
        cout << endl;
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
      if (error > 1e-12) cerr << "Caution! error=" << error << endl;
    }

  }

  for(size_t a=0; a < A; ++a)
    deleteAlign(F[a]);
  delete [] F;


  return 0;
}
