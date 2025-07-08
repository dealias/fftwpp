#include <vector>

#include "convolution.h"
#include "explicit.h"
#include "direct.h"
#include "utils.h"
#include "options.h"

using namespace std;
using namespace utils;
using namespace fftwpp;

// Constants used for initialization and testing.
const Complex I(0.0,1.0);
const Complex iF(sqrt(3.0),sqrt(7.0));
const Complex iG(sqrt(5.0),sqrt(11.0));

bool Test=false;

size_t A=2; // number of inputs
size_t B=1; // number of outputs

inline void init(double **F, size_t m, size_t A)
{
  for(size_t a=0; a < A; ++a) {
    double *f=F[a];
    for(size_t i=0; i < m; ++i)
      f[i]=i+1+a;
  }
}

// Pair-wise binary multiply for A=2 or A=4.
// NB: example function, not optimised or threaded.
void multA(Complex **F, size_t m,
           const size_t indexsize,
           const size_t *index,
           size_t r, size_t threads)
{
  switch(A) {
    case 2: multBinary(F,m,indexsize,index,r,threads); break;
    case 4: multBinary2(F,m,indexsize,index,r,threads); break;
    default:
      cerr << "A=" << A << " is not yet implemented" << endl;
      exit(1);
  }

  for(size_t b=1; b < B; ++b) {
    double factor=1.0+b;
    for(size_t i=0; i < m; ++i) {
      F[b][i]=factor*F[0][i];
    }
  }
}

int main(int argc, char *argv[])
{
  fftw::maxthreads=parallel::get_max_threads();

  bool Output=false;
  bool Normalized=true;
  bool Inplace=true;

  double s=1.0; // Time limit (seconds)
  size_t N=20;
  size_t m=4; // Problem size

  int stats=MEDIAN; // Type of statistics used in timing test.

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

#ifdef __GNUC__
  optind=0;
#endif
  for (;;) {
    int c = getopt(argc,argv,"hdeiptA:B:I:s:Om:n:uS:T:");
    if (c == -1) break;

    switch (c) {
      case 0:
        break;
      case 'I':
        Inplace=atoi(optarg) > 0;
        break;
      case 'p':
        break;
      case 'A':
        A=atoi(optarg);
        break;
      case 'B':
        B=atoi(optarg);
        break;
      case 's':
        s=atof(optarg);
        break;
      case 'O':
        Output=true;
        break;
      case 't':
        Test=true;
        break;
      case 'u':
        Normalized=false;
        break;
      case 'm':
        m=atoi(optarg);
        break;
      case 'T':
        fftw::maxthreads=max(atoi(optarg),1);
        break;
      case 'S':
        stats=atoi(optarg);
        break;
      case 'h':
      default:
        usage(1);
        exit(1);
    }
  }

  size_t n=cpadding(m);

  cout << "n=" << n << endl;
  cout << "m=" << m << endl;

  if(s == 0) N=1;
  cout << "s=" << s << endl;
  s *= 1.0e9;

  if(B < 1)
    B=1;

  size_t C=max(A,B);
  size_t np=n/2+1;
  double *f=doubleAlign(C*m);
  Complex *f0=ComplexAlign(C*np);
  Complex *g0=Inplace ? f0 : ComplexAlign(C*np);

  double **F=new double *[C];
  Complex **F0=new Complex *[C];
  Complex **G0=Inplace ? F0 : new Complex *[C];

  for(size_t c=0; c < C; ++c) {
    F[c]=f+m*c;
    F0[c]=f0+np*c;
  }

  if(!Inplace)
    for(size_t c=0; c < C; ++c) {
      G0[c]=g0+np*c;
    }

  vector<double> T;

  Multiplier *mult;
  if(Normalized) mult=multBinary;
  else mult=multBinaryUnNormalized;

  ExplicitRConvolution Convolve(n,m,F0[0],G0[0]);
  double sum=0.0;
  while(sum < s || T.size() < N) {
    init(F,m,A);
    cpuTimer c;
    for(size_t a=0; a < A; ++a) {
      double *f=F[a];
      double *f0=(double *) (F0[a]);
      for(size_t i=0; i < m; ++i)
        f0[i]=f[i];
    }
    Convolve.convolve(F0,mult,G0);
    for(size_t b=0; b < B; ++b) {
      double *f=(double *) (F[b]);
      double *f0=(double *) (F0[b]);
      for(size_t i=0; i < m; ++i)
        f[i]=f0[i];
    }
    double t=c.nanoseconds();
    T.push_back(t);
    sum += t;
  }

  cout << endl;
  timings("Explicit",m,T.data(),T.size(),stats);
  T.clear();

  cout << endl;
  if(Output)
    for(size_t i=0; i < m; i++) {
      double *f=(double *) (F[0]);
      cout << f[i] << endl;
    }

  cout << endl;

  delete [] F0;
  deleteAlign(f0);

  delete [] F;
  deleteAlign(f);

  return 0;
}
