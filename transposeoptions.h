#ifndef __transposeoptions_h__
#define __transposeoptions_h__ 1

namespace utils {
extern unsigned int defaultmpithreads;

struct mpiOptions {
  int a; // Block divisor: -1=sqrt(size), 0=Tune
  int alltoall; // -1=Tune, 0=Optimized, 1=MPI, 2=Compact
  unsigned int threads;
  unsigned int verbose;
  bool transposed; // Locally transpose output (input) of forward
                   // (backward) FFT
  mpiOptions(int a=0, int alltoall=-1,
             unsigned int threads=defaultmpithreads,
             unsigned int verbose=0, bool transposed=false) :
    a(a), alltoall(alltoall), threads(threads), verbose(verbose),
    transposed(transposed) {}
};

}

#endif
