#include "fftw++.h"

// Compile with:
// g++ -I .. -fopenmp optimal.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace fftwpp;

int main(int argc, char *argv[])
{
  fftw::maxthreads=1;//get_max_threads();

  ofstream fout("optimal.dat");

  double eps=0.5;

  unsigned int N;

  cerr << "Determining optimal sizes for 1D complex to complex in-place FFTs...";

  if(argc == 1) {
    cerr << endl << endl << "Usage: " << argv[0] << " maxsize" << endl;
    exit(0);
  }

  N=atoi(argv[1]);
  cerr << " up to size " << N << endl;

  Complex *f=ComplexAlign(N);

  fout << "# length\tmean\tstdev" << endl;

  unsigned int pow2=1;
  for(unsigned int i=0; pow2 <= N; ++i, pow2 *= 2) {
    unsigned int pow23=pow2;
    for(unsigned int j=0; pow23 <= N; ++j, pow23 *= 3) {
      unsigned int pow235=pow23;
      for(unsigned int k=0; pow235 <= N; ++k, pow235 *= 5) {
        unsigned int n=pow235;
        for(unsigned int l=0; n <= N; ++l, n *= 7) {

          utils::statistics S;
          unsigned int K=1;

          if(n == 1) continue;

          fft1d Forward(n,-1,f);

          for(;;) {
            double t0=utils::totalseconds();
            for(unsigned int i=0; i < K; ++i)
              for(unsigned int i=0; i < n; ++i) f[i]=i;
            double t1=utils::totalseconds();
            for(unsigned int i=0; i < K; ++i) {
              for(unsigned int i=0; i < n; ++i) f[i]=i;
              Forward.fft(f);
            }
            double t=utils::totalseconds();
            S.add(((t-t1)-(t1-t0))/K);
            double mean=S.mean();
            if(K*mean < 1000.0/CLOCKS_PER_SEC) {
              K *= 2;
              S.clear();
            }
            if(S.count() >= 10 && S.stdev() < eps*mean) {
// TODO: accrue in memory first
              fout << n << "\t" << mean << "\t" << S.stdev() << endl ;
              break;
            }
          }
        }
      }
    }
  }

  deleteAlign(f);
}
