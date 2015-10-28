#ifndef __transposeoptions_h__
#define __transposeoptions_h__ 1

namespace fftwpp {
extern unsigned int defaultmpithreads;

struct mpiOptions {
  int a; // Block divisor (-1=sqrt(size), 0=Tune)
  int alltoall; // -1=Tune, 0=Optimized, 1=MPI
  unsigned int threads;
  mpiOptions(unsigned int a=0, unsigned int alltoall=-1,
             unsigned int threads=defaultmpithreads) :
    a(a), alltoall(alltoall), threads(threads) {}
  mpiOptions(unsigned int threads) :
    a(0), alltoall(1), threads(threads) {}
};

}

#endif
