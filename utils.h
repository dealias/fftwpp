#ifndef __utils_h__
#define __utils_h__ 1
#endif

#include <iostream>

#ifdef _WIN32
#include <Winsock2.h>

#include <time.h>
#include <windows.h>
#include <iostream>

using namespace std;
 
#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif
 
struct timezone
{
  int  tz_minuteswest; /* minutes W of Greenwich */
  int  tz_dsttime;     /* type of dst correction */
};
 
// Definition of a gettimeofday function
 
inline int gettimeofday(struct timeval *tv, struct timezone *tz)
{
// Define a structure to receive the current Windows filetime
  FILETIME ft;
 
// Initialize the present time to 0 and the timezone to UTC
  unsigned __int64 tmpres = 0;
  static int tzflag = 0;
 
  if (NULL != tv)
  {
    GetSystemTimeAsFileTime(&ft);
 
// The GetSystemTimeAsFileTime returns the number of 100 nanosecond 
// intervals since Jan 1, 1601 in a structure. Copy the high bits to 
// the 64 bit tmpres, shift it left by 32 then or in the low 32 bits.
    tmpres |= ft.dwHighDateTime;
    tmpres <<= 32;
    tmpres |= ft.dwLowDateTime;
 
// Convert to microseconds by dividing by 10
    tmpres /= 10;
 
// The Unix epoch starts on Jan 1 1970.  Need to subtract the difference 
// in seconds from Jan 1 1601.
    tmpres -= DELTA_EPOCH_IN_MICROSECS;
 
// Finally change microseconds to seconds and place in the seconds value. 
// The modulus picks up the microseconds.
    tv->tv_sec = (long)(tmpres / 1000000UL);
    tv->tv_usec = (long)(tmpres % 1000000UL);
  }
 
  if (NULL != tz)
  {
    if (!tzflag)
    {
      _tzset();
      tzflag++;
    }
  
// Adjust for the timezone west of Greenwich
      tz->tz_minuteswest = _timezone / 60;
    tz->tz_dsttime = _daylight;
  }
 
  return 0;
}

#include "getopt.h"

inline double cbrt(double x) 
{
  if(x == 0.0) return 0.0;
  static double third=1.0/3.0;
  return x > 0.0 ? exp(third*log(x)) : -exp(third*log(-x));
}

#else
#include <sys/time.h>
#endif

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
