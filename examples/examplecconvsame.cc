#include <map>

#include "Complex.h"
#include "convolution.h"
#include "Array.h"

// Compile with:
// g++ -I .. -fopenmp examplecconv.cc ../convolution.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace fftwpp;
using namespace parallel;

double f0[]={0,1,2,3,4};

void init(Complex *f, Complex *g, size_t m)
{
  for(size_t k=0; k < m; k++) {
    f[k]=f0[k];
    g[k]=f0[k];
  }
}

class ZetaTable {
public:
  Complex *ZetaH, *ZetaL;
  size_t s;
  ZetaTable() {}
  ZetaTable(size_t m) {
    int c=m/2;
    s=BuildZeta(twopi*c/(2*m),2*m,ZetaH,ZetaL);
  }
};

static void multbinarysame(Complex **F, size_t m,
                           const size_t indexsize,
                           const size_t *index,
                           size_t r, size_t threads)
{
  static ZetaTable zeta;
  static size_t lastm=0;

  Complex* F0=F[0];
  Complex* F1=F[1];

  size_t c=m/2;
  if(2*c == m) {
    if(r == 0) {
      double sign=1;
      PARALLEL(
        for(size_t j=0; j < m; ++j) {
          Complex *F0j=F0+j;
          Vec h=ZMULT(LOAD(F0j),LOAD(F1+j));
          STORE(F0j,sign*h);
          sign=-sign;
        }
        );
    } else {
      double sign=-1;
      PARALLEL(
        for(size_t j=0; j < m; ++j) {
          Complex *F0j=F0+j;
          Vec h=ZMULT(LOAD(F0j),LOAD(F1+j));
          STORE(F0j,sign*ZMULTI(h));
          sign=-sign;
        }
        );

    }
  } else {
    if(m != lastm) {
      lastm=m;
      static std::map<size_t,ZetaTable> list;
      std::map<size_t,ZetaTable>::iterator p=list.find(m);
      if(p == list.end()) {
        zeta=ZetaTable(m);
        list[m]=zeta;
      } else zeta=p->second;
    }

    Complex *ZetaH=zeta.ZetaH;
    Complex *ZetaL=zeta.ZetaL;
    size_t s=zeta.s;

    PARALLEL(
      for(size_t j=0; j < m; ++j) {
        Complex *F0j=F0+j;
        Vec h=ZMULT(LOAD(F0j),LOAD(F1+j));
        size_t index=2*j+r;
        size_t a=index/s;
        Vec Zeta=LOAD(ZetaH+a);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(CONJ(Zeta),Zeta);
        STORE(F0j,ZCMULT(ZMULT(X,Y,LOAD(ZetaL+index-s*a)),h));
      }
      );
  }
}

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

  // size of problem
  size_t m=sizeof(f0)/sizeof(double);

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  // allocate arrays:
  Complex *F[2];
  Complex *f=F[0]=ComplexAlign(2*m);
  Complex *g=F[1]=f+m;

  ImplicitConvolution C(m);

  cout << "1d non-centered complex convolution:" << endl;
  init(f,g,m);
  cout << "\ninput:\nf\tg" << endl;
  for(size_t i=0; i < m; i++)
    cout << f[i] << "\t" << g[i] << endl;

  C.convolve(F,multbinarysame);

  cout << "\noutput:" << endl;
  for(size_t i=0; i < m; i++) cout << f[i] << endl;

  deleteAlign(f);

  return 0;
}
