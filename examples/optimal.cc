#include "fftw++.h"
#include "seconds.h"

// Compile with:
// g++ -I .. -fopenmp optimal.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace fftwpp;
using namespace parallel;

int main(int argc, char *argv[])
{
  fftw::maxthreads=1;//get_max_threads();

  ofstream fout("optimal.dat");

  double eps=0.5;

  size_t N;

  cerr << "Determining optimal sizes for 1D complex to complex in-place FFTs...";

  if(argc == 1) {
    cerr << endl << endl << "Usage: " << argv[0] << " maxsize" << endl;
    exit(0);
  }

  N=atoi(argv[1]);
  cerr << " up to size " << N << endl;

  Complex *f=ComplexAlign(N);

  fout << "# length\tmean\tstdev" << endl;

  size_t pow2=1;
  for(size_t i=0; pow2 <= N; ++i, pow2 *= 2) {
    size_t pow23=pow2;
    for(size_t j=0; pow23 <= N; ++j, pow23 *= 3) {
      size_t pow235=pow23;
      for(size_t k=0; pow235 <= N; ++k, pow235 *= 5) {
        size_t n=pow235;
        for(size_t l=0; n <= N; ++l, n *= 7) {

          utils::statistics S;
          size_t K=1;

          if(n == 1) continue;

          fft1d Forward(n,-1,f);

          for(;;) {
            cpuTimer c;
            for(size_t i=0; i < K; ++i)
              for(size_t i=0; i < n; ++i) f[i]=i;
            double t1=c.seconds();
            for(size_t i=0; i < K; ++i) {
              for(size_t i=0; i < n; ++i) f[i]=i;
              Forward.fft(f);
            }
            double t=c.seconds();
            S.add(((t-t1)-t1)/K);
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
