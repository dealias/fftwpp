#include <vector>

#include "convolve.h"
#include "timing.h"
#include "direct.h"
#include "options.h"

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

size_t A=2; // number of inputs
size_t B=1; // number of outputs

int main(int argc, char *argv[])
{
  L=8;  // input data length
  M=16; // minimum padded length

  optionsHybrid(argc,argv);

  cout << "L=" << L << endl;
  cout << "M=" << M << endl;

  if(Output || testError)
    s=0;
  if(s == 0) N=1;
  cout << "s=" << s << endl << endl;
  s *= 1.0e9;

  vector<double> T;

  Complex **f=ComplexAlign(max(A,B),L);
  for(size_t a=0; a < A; ++a) {
    Complex *fa=f[a];
    for(size_t j=0; j < L; ++j)
      fa[j]=0.0;
  }

  Complex *h=NULL;

  h=ComplexAlign(L);
  directconv<Complex> C(L);

  double sum=0.0;
  while(sum <= s || T.size() < N) {
    double t;
    cpuTimer c;
    C.convolve(h,f[0],f[1]);
    t=c.nanoseconds();
    T.push_back(t);
    sum += t;
  }

  cout << endl;
  timings("Direct",L,T.data(),T.size(),stats);
  cout << endl;
  deleteAlign(h);

  return 0;
}
