#include "convolve.h"

using namespace std;
using namespace fftwpp;

unsigned int K=1; // number of iterations
unsigned int C=1;
unsigned int S=0;

extern unsigned int L;
extern unsigned int M;

namespace utils {

void optionsHybrid(int argc, char* argv[], bool fft)
{
#ifdef __GNUC__
  optind=0;
#endif
  for (;;) {
    int c=getopt(argc,argv,"hC:D:I:K:L:M:O:S:T:m:");
    if (c == -1) break;

    switch(c) {
      case 0:
        break;
      case 'C':
        C=max(atoi(optarg),1);
        break;
      case 'D':
        DOption=max(atoi(optarg),0);
        if(DOption > 1 && DOption % 2) ++DOption;
        break;
      case 'K':
        K=atoi(optarg);
        break;
      case 'L':
        L=atoi(optarg);
        break;
      case 'I':
        IOption=atoi(optarg) > 0;
        break;
      case 'M':
        M=atoi(optarg);
        break;
      case 'S':
        S=atoi(optarg);
        break;
      case 'T':
        fftw::maxthreads=max(atoi(optarg),1);
        break;
//      case 'X':
//        Mx=atoi(optarg);
//        break;
      case 'm':
        mOption=max(atoi(optarg),0);
        break;
      case 'h':
      default:
        usageHybrid(fft);
        exit(1);
    }
  }

  cout << "L=" << L << endl;
  cout << "M=" << M << endl;
  if(fft)
    cout << "C=" << C << endl;
  cout << "K=" << K << endl;

  cout << endl;
}

}
