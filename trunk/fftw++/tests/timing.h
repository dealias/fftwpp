#ifndef __timing_h__
#define __timing_h__ 1

#include "math.h"

/*
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
*/

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

inline void timings(const char* text, unsigned int m, unsigned int count,
                    double mean, double sigmaL, double sigmaH)
{
//  mean -= emptytime(T,N);
//  if(mean < 0.0) mean=0.0;
  std::cout << std::endl 
	    << text << ":\n" 
	    << m << "\t" 
	    << mean << "\t" 
	    << sigmaL << "\t" 
	    << sigmaH 
	    << std::endl << std::endl;
}

inline void timings(const char* text, unsigned int m, double *T, 
		    unsigned int N)
{
  double sigmaL=0.0, sigmaH=0.0;
  double avg=mean(T,N);
  stdev(T,N,avg,sigmaL,sigmaH);
  timings(text,m,N,avg,sigmaL,sigmaH);
}

#endif
