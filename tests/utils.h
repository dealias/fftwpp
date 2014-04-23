#ifndef __utils_h__
#define __utils_h__ 1

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
  if(compact) 
    std::cerr << "-c\t\t compact data format [1] (0=false, 1=true)" 
	      << std::endl;
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
  if(n > 1) {
    std::cerr << "-x\t\t x size" << std::endl;
    std::cerr << "-y\t\t y size" << std::endl;
  }
  if(n > 2)
    std::cerr << "-z\t\t z size" << std::endl;
}

inline void usageB()
{
  std::cerr << "-B\t\t number of data blocks in output" << std::endl;
  std::cerr << "-b\t\t which output block to check" << std::endl;
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
