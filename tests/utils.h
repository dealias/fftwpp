#ifndef __fftwpputils_h__
#define __fftwpputils_h__ 1

#include <iostream>
#include "seconds.h"
#include "timing.h"
#include "Complex.h"
 
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

inline void usage(int n, bool test=false, bool Explicit=true,
		  bool compact=false)
{
  std::cerr << "Options: " << std::endl;
  std::cerr << "-h\t\t help" << std::endl;
  std::cerr << "-i\t\t implicitly padded convolution" << std::endl;
  if(Explicit) {
    std::cerr << "-e\t\t explicitly padded convolution" << std::endl;
    if(n > 1)
      std::cerr << "-p\t\t pruned explicitly padded convolution" << std::endl;
  }
  std::cerr << "-d\t\t direct convolution (slow)" << std::endl;
  std::cerr << "-T\t\t number of threads" << std::endl;
  if(test) std::cerr << "-t\t\t accuracy test" << std::endl;
  std::cerr << "-N\t\t number of iterations" << std::endl;
  std::cerr << "-M\t\t number of data blocks in dot product" << std::endl;
  std::cerr << "-m\t\t size" << std::endl;
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
  
   if(compact) {
    if(n > 1) {
      std::cerr << "-X\t\t x Hermitian padding (0 or 1)" << std::endl;
      std::cerr << "-Y\t\t y Hermitian padding (0 or 1)" << std::endl;
    }
    if(n > 2)
      std::cerr << "-Z\t\t z Hermitian padding (0 or 1)" << std::endl;
  }
}

inline void usageA()
{
  std::cerr << "-A\t\t number of data blocks in input" << std::endl;
}

inline void usageB(bool littleb=true)
{
  std::cerr << "-B\t\t number of data blocks in output" << std::endl;
  if(littleb)
    std::cerr << "-b\t\t which output block to check" << std::endl;
}

inline void usageTranspose()
{
  std::cerr << "-a<int>\t\t block divisor: -1=sqrt(size), [0]=Tune"
            << std::endl;
  std::cerr << "-s<int>\t\t alltoall: [-1]=Tune, 0=Optimized, 1=MPI"
            << std::endl;
}

void fft_usage(int dim)
{
  std::cout << "Usage:\n"
	    << "-h\thelp\n"
	    << "-T\tnumber of threads\n"
	    << "-N\tnumber of iterations\n"
	    << "-m\tsize\n";
  
  std::cout << "-r\ttype of run\n"
	    << "\t\tr=-1: all runs\n"
	    << "\t\tr=0: in-place\n"
	    << "\t\tr=1: out-of-place\n";
  if(dim > 1)
    std::cout << "\t\tr=2: transpose, in-place\n"
	      << "\t\tr=3: transpose, out-of-place\n"
	      << "\t\tr=4: full transpose, in-place\n"
	      << "\t\tr=5: full transpose, out-of-place\n"
	      << "\t\tr=6: strided, in-place\n"
	      << "\t\tr=7: strided, out-of-place\n";

  std::cout << "-x\tsize in first dimension\n";
  if(dim > 1)
    std::cout << "-y\tsize in second dimension\n";
  if(dim > 2)
    std::cout << "-z\tsize in third dimension\n";
  std::cout << std::endl;
}

unsigned int padding(unsigned int n)
{
  std::cout << "min padded buffer=" << n << std::endl;
  unsigned int log2n;
  // Choose next power of 2 for maximal efficiency.
  for(log2n=0; n > ((unsigned int) 1 << log2n); log2n++);
  return 1 << log2n;
}
  
unsigned int cpadding(unsigned int m)
{
  return padding(2*m);
}

unsigned int hpadding(unsigned int m)
{
  return padding(3*m-2);
}

unsigned int tpadding(unsigned int m)
{
  return padding(4*m-3);
}

inline int hash(Complex* f, unsigned int m)
{
  int h=0;
  unsigned int i;
  for(i=0; i<m; ++i) {
    h= (h+ (324723947+(int)(f[i].im+0.5)))^93485734985;
    h= (h+ (324723947+(int)(f[i].im+0.5)))^93485734985;
  }
  return h;
}

#endif
