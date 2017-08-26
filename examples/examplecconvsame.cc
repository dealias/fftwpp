#include <map>

#include "Complex.h"
#include "convolution.h"
#include "Array.h"

// Compile with:
// g++ -I .. -fopenmp examplecconv.cc ../convolution.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace fftwpp;

double f0[]={0,1,2,3,4};

void init(Complex *f, Complex *g, unsigned int m) 
{
  for(unsigned int k=0; k < m; k++) {
    f[k]=f0[k];
    g[k]=f0[k];
  }
}

class ZetaTable {
public:
  Complex *ZetaH, *ZetaL;
  unsigned int s;
  ZetaTable() {}
  ZetaTable(unsigned int m) {
    int c=m/2;
    s=BuildZeta(twopi*c/(2*m),2*m,ZetaH,ZetaL);
  }
};
  
static void multbinarysame(Complex **F, unsigned int m,
                           const unsigned int indexsize,
                           const unsigned int *index,
                           unsigned int r, unsigned int threads)
{
  static ZetaTable zeta;
  static unsigned int lastm=0;

  Complex* F0=F[0];
  Complex* F1=F[1];
  
  unsigned int c=m/2;
  if(2*c == m) {
    if(r == 0) {
      double sign=1;
      PARALLEL(
        for(unsigned int j=0; j < m; ++j) {
          Complex *F0j=F0+j;
          Vec h=ZMULT(LOAD(F0j),LOAD(F1+j));
          STORE(F0j,sign*h);
          sign=-sign;
        }
        );
    } else {
      double sign=-1;
      PARALLEL(
        for(unsigned int j=0; j < m; ++j) {
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
      static std::map<unsigned int,ZetaTable> list;
      std::map<unsigned int,ZetaTable>::iterator p=list.find(m);
      if(p == list.end()) {
        zeta=ZetaTable(m);
        list[m]=zeta;
      } else zeta=p->second;
    }
  
    Complex *ZetaH=zeta.ZetaH;
    Complex *ZetaL=zeta.ZetaL;
    unsigned int s=zeta.s;
  
    PARALLEL(
      for(unsigned int j=0; j < m; ++j) {
        Complex *F0j=F0+j;
        Vec h=ZMULT(LOAD(F0j),LOAD(F1+j));
        unsigned int index=2*j+r;
        unsigned int a=index/s;
        Vec Zeta=LOAD(ZetaH+a);
        Vec X=UNPACKL(Zeta,Zeta);
        Vec Y=UNPACKH(CONJ(Zeta),Zeta);
        STORE(F0j,ZMULTC(ZMULT(X,Y,LOAD(ZetaL+index-s*a)),h));
      }
      );
  }
}

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

  // size of problem
  unsigned int m=sizeof(f0)/sizeof(double);
  
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
  for(unsigned int i=0; i < m; i++) 
    cout << f[i] << "\t" << g[i] << endl;
  
  C.convolve(F,multbinarysame);

  cout << "\noutput:" << endl;
  for(unsigned int i=0; i < m; i++) cout << f[i] << endl;
  
  deleteAlign(f);

  return 0;
}
