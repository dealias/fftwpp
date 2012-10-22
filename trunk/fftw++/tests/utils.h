#ifndef __utils_h__
#define __utils_h__ 1

#include <iostream>
#include "seconds.h"
#include "getopt.h"
 
#ifdef _WIN32

inline double cbrt(double x) 
{
  if(x == 0.0) return 0.0;
  static double third=1.0/3.0;
  return x > 0.0 ? exp(third*log(x)) : -exp(third*log(-x));
}
#endif

// timing routines
inline double emptytime(double *T, unsigned int N)
{
  double val=0.0;
  for(unsigned int i=0; i < N; ++i) {
    seconds();
    T[i]=seconds();
  }
  for(unsigned int i=0; i < N; ++i) 
    val += T[i];
  return val/N;
}

inline double mean(double *T, unsigned int N) 
{
  double sum=0.0;
  for(unsigned int i=0; i < N; ++i)
    sum += T[i];
  return sum/N;
}

inline void stdev(double *T, unsigned int N, double mean, double &sigmaL,
           double& sigmaH) 
{
  sigmaL=0.0, sigmaH=0.0;
  for(unsigned int i=0; i < N; ++i) {
    double v=T[i]-mean;
    if(v < 0)
      sigmaL += v*v;
    if(v > 0)
      sigmaH += v*v;
  }
  
  double factor=N > 2 ? 2.0/(N-2.0) : 0.0; 
  sigmaL=sqrt(sigmaL*factor);
  sigmaH=sqrt(sigmaH*factor);
}

inline void timings(const char* text, double *T, unsigned int N)
{
  double sigmaL=0.0, sigmaH=0.0;
  double avg=mean(T,N);
  stdev(T,N,avg,sigmaL,sigmaH);
  avg -= emptytime(T,N);
  std::cout << std::endl << text << ":\n" << avg << "\t" << sigmaL << "\t" <<
    sigmaH << std::endl << std::endl;
}

inline void usage(int n, bool test=false, bool Explicit=true)
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
  if(test)
  std::cerr << "-t\t\t accuracy test" << std::endl;
  std::cerr << "-N\t\t number of iterations" << std::endl;
  std::cerr << "-M\t\t number of data blocks in dot product" << std::endl;
  std::cerr << "-m\t\t size" << std::endl;
  if(n > 1) {
    std::cerr << "-x\t\t x size" << std::endl;
    std::cerr << "-y\t\t y size" << std::endl;
  }
  if(n > 2)
    std::cerr << "-z\t\t z size" << std::endl;
  exit(1);
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
