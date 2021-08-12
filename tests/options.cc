#include "convolve.h"

using namespace std;
using namespace fftwpp;

extern unsigned int L;
extern unsigned int M;
extern unsigned int C;
extern unsigned int S;

namespace utils {

void optionsHybrid(int argc, char* argv[])
{
#ifdef __GNUC__
  optind=0;
#endif
  for (;;) {
    int c = getopt(argc,argv,"hC:D:I:L:M:O:S:T:m:");
    if (c == -1) break;

    switch (c) {
      case 0:
        break;
      case 'C':
        C=max(atoi(optarg),1);
        break;
      case 'D':
        DOption=max(atoi(optarg),0);
        if(DOption > 1 && DOption % 2) ++DOption;
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
      case 'm':
        mOption=max(atoi(optarg),0);
        break;
      case 'h':
      default:
        usageHybrid();
        exit(1);
    }
  }

  if(S == 0) S=C;

  cout << "L=" << L << endl;
  cout << "M=" << M << endl;
  cout << "C=" << C << endl;
  cout << "S=" << S << endl;

  cout << endl;
}

}
