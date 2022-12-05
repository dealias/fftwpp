#ifndef __transposeoptions_h__
#define __transposeoptions_h__ 1

namespace utils {
extern size_t defaultmpithreads;

struct mpiOptions {
  int a; // Block divisor: -1=sqrt(size), 0=Tune
  int alltoall; // -1=Tune, 0=Optimized, 1=MPI, 2=Inplace
  size_t threads;
  size_t verbose;
  mpiOptions(int a=0, int alltoall=-1,
             size_t threads=defaultmpithreads,
             size_t verbose=0) :
    a(a), alltoall(alltoall), threads(threads), verbose(verbose) {}
};

}

#endif
