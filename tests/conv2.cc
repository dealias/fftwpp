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

unsigned int nx=0;
unsigned int ny=0;
unsigned int mx=4;
unsigned int my=0;
unsigned int nxp;
unsigned int nyp;
bool xcompact=true;
bool ycompact=true;
bool Explicit=false;

unsigned int outlimit=100;

inline void init(Complex **F,
                 unsigned int mx, unsigned int my,
                 unsigned int nxp, unsigned int nyp,
                 unsigned int A,
                 bool xcompact, bool ycompact)
{
  if(A % 2 == 0) {
    unsigned int M=A/2;

    unsigned int xoffset=Explicit ? nxp/2-mx+1 : !xcompact;
    unsigned int nx=2*mx-1;
    double factor=1.0/sqrt((double) M);
    for(unsigned int s=0; s < M; ++s) {
      double S=sqrt(1.0+s);
      double ffactor=S*factor;
      double gfactor=1.0/S*factor;
      array2<Complex> f(nxp,nyp,F[s]);
      array2<Complex> g(nxp,nyp,F[M+s]);
      if(!xcompact) {
        for(unsigned int j=0; j < my+!ycompact; j++) {
          f[0][j]=0.0;
          g[0][j]=0.0;
        }
      }
      if(!ycompact) {
        for(unsigned int i=0; i < nx+!xcompact; ++i) {
          f[i][my]=0.0;
          g[i][my]=0.0;
        }
      }
#pragma omp parallel for
      for(unsigned int i=0; i < nx; ++i) {
        unsigned int I=i+xoffset;
        for(unsigned int j=0; j < my; j++) {
          f[I][j]=ffactor*Complex(i,j);
          g[I][j]=gfactor*Complex(2*i,j+1);
        }
      }
    }
  } else {
    cerr << "Init not implemented for A=" << A << endl;
    exit(1);
  }
}

int main(int argc, char *argv[])
{
  bool Direct=false;
  bool Implicit=true;
  bool Pruned=false;
  bool Output=false;
  bool Normalized=true;

  double K=1.0; // Time limit (seconds)
  unsigned int minCount=20;

  fftw::maxthreads=get_max_threads();

  unsigned int A=2; // Number of independent inputs
  unsigned int B=1;   // Number of outputs

  unsigned int stats=0; // Type of statistics used in timing test.

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

#ifdef __GNUC__
  optind=0;
#endif
  for (;;) {
    int c = getopt(argc,argv,"hdeipA:B:N:Om:x:y:n:T:uS:X:Y:");
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
      case 'u':
        Normalized=false;
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
      case 'T':
        fftw::maxthreads=max(atoi(optarg),1);
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
      case 'h':
      default:
        usage(2);
        usageExplicit(2);
        usageCompact(2);
        exit(1);
    }
  }

  if(my == 0) my=mx;

  nx=hpadding(mx);
  ny=hpadding(my);

  cout << "nx=" << nx << ", ny=" << ny << endl;
  cout << "mx=" << mx << ", my=" << my << endl;
  cout << "K=" << K << endl;

  K *= 1.0e9;

  size_t align=ALIGNMENT;

  array2<Complex> h0;
  if(Direct) {
    h0.Allocate(mx,my,align);
    if(!Normalized) {
      cerr << "-u option is incompatible with -d." << endl;
      exit(-1);
    }
  }

  nxp=Explicit ? nx : 2*mx-xcompact;
  nyp=Explicit ? ny/2+1 : my+!ycompact;

  cout << "nxp=" << nxp << ", nyp=" << nyp << endl;


  if(B < 1) B=1;
  if(B > A) {
    cerr << "B=" << B << " is not yet implemented for A=" << A << endl;
    exit(1);
  }

  Complex **F=new Complex *[A];
  for(unsigned int a=0; a < A; ++a)
    F[a]=ComplexAlign(nxp*nyp);

  // For easy access of first element
  array2<Complex> f(nxp,nyp,F[0]);

  vector<double> T;

  if(Implicit) {
    ImplicitHConvolution2 C(mx,my,xcompact,ycompact,A,B);
    cout << "threads=" << C.Threads() << endl << endl;

    realmultiplier *mult;
    switch(A) {
      case 2: mult=multbinary; break;
      case 4: mult=multbinary2; break;
      default: cerr << "A=" << A << " is not yet implemented" << endl; exit(1);
    }

    double sum=0.0;
    while(sum <= K || T.size() < minCount) {
      init(F,mx,my,nxp,nyp,A,xcompact,ycompact);
      double t0=nanoseconds();
      C.convolve(F,mult);
//      C.convolve(f,g);
      double t=nanoseconds()-t0;
      T.push_back(t);
      sum += t;
    }

    timings("Implicit",(2*mx-1)*(2*my-1),T.data(),T.size(),stats);
    T.clear();
    cout << endl;

    if(Normalized) {
      double norm=1.0/(9.0*mx*my);
      for(unsigned int b=0; b < B; ++b) {
        for(unsigned int i=!xcompact; i < nxp; i++) {
          for(unsigned int j=0; j < nyp; j++)
            F[b][nyp*i+j] *= norm;
        }
      }
    }

    if(Direct) {
      for(unsigned int i=0; i < mx; i++)
        for(unsigned int j=0; j < my; j++)
          h0[i][j]=f[i+!xcompact][j];
    }

    if(Output) {
      for(unsigned int i=!xcompact; i < nxp; i++) {
        for(unsigned int j=0; j < my; j++)
          cout << f[i][j] << "\t";
        cout << endl;
      }
      cout << endl;
    }
  }

  if(Explicit) {
    unsigned int M=A/2;
    ExplicitHConvolution2 C(nx,ny,mx,my,f,M,Pruned);
    cout << "threads=" << C.Threads() << endl << endl;

    double sum=0.0;
    while(sum <= K || T.size() < minCount) {
      init(F,mx,my,nxp,nyp,A,true,true);
      double t0=nanoseconds();
      C.convolve(F,F+M);
      double t=nanoseconds()-t0;
      T.push_back(t);
      sum += t;
    }

    timings(Pruned ? "Pruned" : "Explicit",(2*mx-1)*(2*my-1),T.data(),T.size(),
            stats);
    T.clear();
    cout << endl;

    unsigned int offset=nx/2-mx+1;

    if(Direct) {
      for(unsigned int i=0; i < mx; i++)
        for(unsigned int j=0; j < my; j++)
          h0[i][j]=f[offset+i][j];
    }

    if(Output) {
      for(unsigned int i=offset; i < offset+2*mx-1; i++) {
        for(unsigned int j=0; j < my; j++)
          cout << f[i][j] << "\t";
        cout << endl;
      }
    }
  }

  if(Direct) {
    unsigned int nxp=2*mx-1;
    array2<Complex> h(nxp,my,align);
    DirectHConvolution2 C(mx,my);
    init(F,mx,my,nxp,my,2,true,true);
    double t0=nanoseconds();
    C.convolve(h,F[0],F[1]);
    T[0]=nanoseconds()-t0;

    timings("Direct",(2*mx-1)*(2*my-1),T.data(),1);
    T.clear();
    cout << endl;

    if(Output) {
      for(unsigned int i=0; i < nxp; i++) {
        for(unsigned int j=0; j < my; j++)
          cout << h[i][j] << "\t";
        cout << endl;
      }
    }

    { // compare implicit or explicit version with direct verion:
      double error=0.0;
      cout << endl;
      double norm=0.0;
      for(unsigned int i=0; i < mx; i++) {
        for(unsigned int j=0; j < my; j++) {
          error += abs2(h0[i][j]-h[i][j]);
          norm += abs2(h[i][j]);
        }
      }
      if(norm > 0) error=sqrt(error/norm);
      cout << "error=" << error << endl;
      if (error > 1e-12) cerr << "Caution! error=" << error << endl;
    }
  }

  for(unsigned int a=0; a < A; ++a)
    deleteAlign(F[a]);
  delete [] F;

  return 0;
}
