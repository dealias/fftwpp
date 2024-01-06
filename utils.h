#ifndef __fftwpputils_h__
#define __fftwpputils_h__ 1

#include <iostream>
#include "seconds.h"
#include "Complex.h"

extern double s;  // Time limit (seconds) for testing
extern size_t N;  // Minimum sample size for testing

extern size_t C;  // number of padded FFTs to compute
extern size_t S;  // stride between padded FFTs
extern int stats; // type of statistics used in timing test

#ifdef _WIN32
#include "getopt.h"
inline double cbrt(double x)
{
  if(x == 0.0) return 0.0;
  static double third=1.0/3.0;
  return x > 0.0 ? exp(third*log(x)) : -exp(third*log(-x));
}
#else
#include <getopt.h>
#endif

namespace utils {

template<class T, class S>
inline T max(const T a, const S b)
{
  return a > (T) b ? a : b;
}

template<class T>
inline T pow(T x, size_t y)
{
  if(y == 0) return 1;
  if(x == 0) return 0;

  size_t r = 1;
  while(true) {
    if(y & 1) r *= x;
    if((y >>= 1) == 0) return r;
    x *= x;
  }
}

extern void optionsHybrid(int argc, char* argv[], bool fft=false,
                          bool mpi=false);

inline void usageCommon(int n)
{
  std::cerr << "Options: " << std::endl;
  std::cerr << "-h\t\t help" << std::endl;
  std::cerr << "-m n\t\t size m" << std::endl;
  std::cerr << "-u\t\t unnormalized" << std::endl;
  std::cerr << "-s t\t\t time limit (seconds)" << std::endl;
  std::cerr << "-T n\t\t use n threads" << std::endl;
  std::cerr << "-O\t\t output result" << std::endl;
  std::cerr << "-S<int>\t\t stats used in timing test: "
            << "0=mean, 1=min, 2=max, 3=median, "
            << "4=90th percentile, 5=80th percentile, 6=50th percentile"
            << std::endl;
  if(n > 1) {
    std::cerr << "-x\t\t x size" << std::endl;
    std::cerr << "-y\t\t y size" << std::endl;
  }
  if(n > 2)
    std::cerr << "-z\t\t z size" << std::endl;
}

inline void usageDirect()
{
  std::cerr << "-i\t\t implicitly padded convolution" << std::endl;
  std::cerr << "-d\t\t direct convolution (slow)" << std::endl;
}

inline void usage(int n)
{
  usageCommon(n);
  std::cerr << "-A\t\t number of data blocks in input" << std::endl;
  std::cerr << "-B\t\t number of data blocks in output" << std::endl;
}

inline void usageInplace(int n)
{
  usageCommon(n);
  std::cerr << "-i\t\t 0=out-of-place, 1=in-place" << std::endl;
}

inline void usageTest()
{
  std::cerr << "-t\t\t accuracy test" << std::endl;
}

inline void usageExplicit(size_t n)
{
  usageDirect();
  if(n == 1)
    std::cerr << "-I b\t\t 1=inplace (default), 0=out of place" << std::endl;

  std::cerr << "-e\t\t explicitly padded convolution" << std::endl;
  if(n > 1)
    std::cerr << "-p\t\t pruned explicitly padded convolution" << std::endl;
}

inline void usageCompact(size_t n)
{
  std::cerr << "-X\t\t x Hermitian padding (0 or 1)" << std::endl;
  if(n > 1)
    std::cerr << "-Y\t\t y Hermitian padding (0 or 1)" << std::endl;
  if(n > 2)
    std::cerr << "-Z\t\t z Hermitian padding (0 or 1)" << std::endl;
}

inline void usageb()
{
  std::cerr << "-b\t\t which output block to check" << std::endl;
}

inline void usageTranspose()
{
  std::cerr << "-a<int>\t\t block divisor: -1=sqrt(size), [0]=Tune"
            << std::endl;
  std::cerr << "-s<int>\t\t alltoall: [-1]=Tune, 0=Optimized, 1=MPI, 2=compact"
            << std::endl;
  std::cerr << "-q\t\t quiet" << std::endl;
}

inline void usageShift()
{
  std::cerr << "-O<int>\t\t [0]=Standard, 1=Shift origin"
            << std::endl;
}

inline void usageFFT(int n)
{
  usageCommon(n);
  std::cerr << "-r\t\t type of run:\n"
            << "\t\t r=-1: all runs\n"
            << "\t\t r=0: in-place\n"
            << "\t\t r=1: out-of-place\n";
  if(n > 1)
    std::cerr << "\t\t r=2: transpose, in-place\n"
              << "\t\t r=3: transpose, out-of-place\n"
              << "\t\t r=4: full transpose, in-place\n"
              << "\t\t r=5: full transpose, out-of-place\n"
              << "\t\t r=6: strided, in-place\n"
              << "\t\t r=7: strided, out-of-place\n";
}

inline void usageGather()
{
  std::cerr << "Options: " << std::endl;
  std::cerr << "-h\t\t help" << std::endl;
  std::cerr << "-m\t\t size" << std::endl;
  std::cerr << "-x\t\t x size" << std::endl;
  std::cerr << "-y\t\t y size" << std::endl;
  std::cerr << "-z\t\t z size" << std::endl;
  std::cerr << "-q\t\t quiet" << std::endl;
}

inline void usageHybrid(bool fft=false, bool mpi=false)
{
  std::cerr << "Options: " << std::endl;
  std::cerr << "-a\t\t accuracy test" << std::endl;
  std::cerr << "-c\t\t use centered tranforms (if possible)" << std::endl;
  std::cerr << "-h\t\t help" << std::endl;
  std::cerr << "-m n\t\t use subtransform size n" << std::endl;
  std::cerr << "-t\t\t show times produced by optimizer" << std::endl;
  if(fft)
    std::cerr << "-C n\t\t compute n padded FFTs at a time"
              << std::endl;
  std::cerr << "-D n\t\t number n of blocks to process at a time" << std::endl;
  std::cerr << "-E\t\t compute relative error using direct convolution (sets s=0 and forces normalization)" << std::endl;
  std::cerr << "-I\t\t (0=out-of-place, 1=in-place) FFTs [by default I=1 only for multiple FFTs]" << std::endl;
  std::cerr << "-O\t\t output result (sets s=0)" << std::endl;
  std::cerr << "-R\t\t show which forward and backward routines are used" << std::endl;
  if(mpi)
    std::cerr << "-N n\t\t number of iterations" << std::endl;
  else {
    std::cerr << "-N t\t\t minimum number of iterations" << std::endl;
    std::cerr << "-s t\t\t time limit (seconds)" << std::endl;
  }
  std::cerr << "-L n\t\t number n of physical data values" << std::endl;
  std::cerr << "-M n\t\t minimal number n of padded data values" << std::endl;
  if(fft)
    std::cerr << "-S s\t\t use stride s between padded FFTs (defaults to C)" << std::endl;
  else
    std::cerr << "-S n\t\t use statistics type n (defaults to 0: MEDIAN)" << std::endl;
  std::cerr << "-T n\t\t number n of threads" << std::endl;
}

// ceilpow2(n) returns the smallest power of 2 greater than or equal
// to a positive integer n.
inline size_t ceilpow2(size_t n)
{
  --n;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  return ++n;
}

// Return the smallest power of p greater than or equal to n.
inline size_t ceilpow(size_t p, size_t n)
{
  size_t x=p;
  size_t u=1;
  size_t l=0;
  while(n > x) {
    x *= x;
    l=u;
    u *= 2;
  }
  if(n == x) return n;

  while (l < u) {
    size_t i=(l+u) >> 1;
    if(n > pow(p,i))
      l=i+1;
    else
      u=i;
  }
  return pow(p,u);
}

inline size_t padding(size_t n)
{
  std::cout << "min padded buffer=" << n << std::endl;
  // Choose next power of 2 for maximal efficiency.
  return ceilpow2(n);
}

inline size_t cpadding(size_t m)
{
  return padding(2*m-1);
}

inline size_t hpadding(size_t m)
{
  return padding(3*m-2);
}

inline size_t tpadding(size_t m)
{
  return padding(4*m-3);
}

// return real(z*w)
}

#endif
