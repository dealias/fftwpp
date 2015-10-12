#ifndef __transposeoptions_h__
#define __transposeoptions_h__ 1

struct mpiOptions {
  unsigned int threads;
  int a; // Block divisor (-1=sqrt(size), 0=Tune)
  int alltoall; // -1=Tune, 0=Optimized, 1=MPI
  mpiOptions(unsigned int threads=fftw::maxthreads, unsigned int a=0,
             unsigned int alltoall=-1) :
    threads(threads), a(a), alltoall(alltoall) {}
};

#endif
