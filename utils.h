#ifndef __utils_h__
#define __utils_h__ 1
#endif

#include <iostream>
#include <sys/time.h>

inline double seconds()
{
  static timeval lasttime;
  timeval tv;
  gettimeofday(&tv,NULL);
  double seconds=tv.tv_sec-lasttime.tv_sec+
    ((double) tv.tv_usec-lasttime.tv_usec)/1000000.0;
  lasttime=tv;
  return seconds;
}

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

inline void stdev(double *T, unsigned int N, double mean, double &sigmaL,
           double& sigmaH) 
{
  sigmaL=0.0, sigmaH=0.0;
  for(unsigned int i=0; i < N; ++i) {
    double v=T[i]-mean;
    if(v < 0)
      sigmaL += v*v;
    if(v >= 0)
      sigmaH += v*v;
  }
  
  double factor=N > 2 ? 2.0/(N-2.0) : 0.0; 
  sigmaL=sqrt(sigmaL*factor);
  sigmaH=sqrt(sigmaH*factor);
}

inline void timings(const char* text, double *T, unsigned int N)
{
  double sigmaL=0.0, sigmaH=0.0;
  double mean=0.0;
  for(unsigned int i=0; i < N; ++i)
    mean += T[i];
  mean /= N;
  stdev(T,N,mean,sigmaL,sigmaH);
  mean -= emptytime(T,N);
  std::cout << std::endl << text << ":\n" << mean << "\t" << sigmaL << "\t" <<
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
