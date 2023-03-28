#ifndef __timing_h__
#define __timing_h__ 1

#include <math.h>
#include <algorithm> // For std::sort

#include <iostream>
#include <fstream>

const double unit=1.0e-9; // default unit is 1 nanosecond

const std::string Algorithm[]={"Median","Mean","Min","Max","P90","P80","P50"};
enum timing_algorithm {WRITETOFILE = -1, MEDIAN, MEAN, MIN, MAX, P90, P80, P50};

inline double median(double *T, size_t N) // TODO: Replace with selection algorithm
{
  std::sort(T,T+N);
  size_t h=N/2;
  return 2*h == N ? 0.5*(T[h-1]+T[h]) : T[h];
}

inline double value(double *T, size_t N, int algorithm)
{
  switch(algorithm) {
    case WRITETOFILE:
    case MEDIAN:
      return median(T,N);
    case MEAN: {
      double sum=0.0;
      for(size_t i=0; i < N; ++i)
        sum += T[i];
      return sum/N;
    }
    case MIN: {
      double min=T[0];
      for(size_t i=0; i < N; ++i) {
        if(T[i] < min)
          min=T[i];
      }
      return min;
    }
    case MAX: {
      double max=T[0];
      for(size_t i=0; i < N; ++i) {
        if(T[i] > max)
          max=T[i];
      }
      return max;
    }
    case P90: {
      std::sort(T,T+N);
      size_t start=(size_t) ceil(N*0.05);
      size_t stop=(size_t) floor(N*0.95);

      double sum=0.0;
      for(size_t i=start; i < stop; ++i)
        sum += T[i];
      return sum/(stop-start);
    }
    case P80: {
      std::sort(T,T+N);
      size_t start=(size_t) ceil(N*0.1);
      size_t stop=(size_t) floor(N*0.9);

      double sum=0.0;
      for(size_t i=start; i < stop; ++i)
        sum += T[i];
      return sum/(stop-start);
    }
    case P50: {
      std::sort(T,T+N);
      size_t start=(size_t) ceil(N*0.25);
      size_t stop=(size_t) floor(N*0.75);

      double sum=0.0;
      for(size_t i=start; i < stop; ++i)
        sum += T[i];
      return sum/(stop-start);
    }
    default:
      std::cout << "Error: invalid algorithm choice: "
                << algorithm
                << std::endl;
      exit(1);
  }
  return 0;
}

inline void stdev(double *T, size_t N,
                  double &lower, double& upper,
                  int algorithm)
{
  lower=0.0, upper=0.0;
  double sum=0.0;
  for(size_t i=0; i < N; ++i)
    sum += T[i];
  double mean=sum/N;
  for(size_t i=0; i < N; ++i) {
    double v=T[i]-mean;
    if(v < 0)
      lower += v*v;
    if(v > 0)
      upper += v*v;
  }
  double factor=N > 2 ? 2.0/(N-2.0) : 0.0;
  lower=mean-sqrt(lower*factor);
  upper=mean+sqrt(upper*factor);
}

inline void timings(const char* text, size_t m, size_t count,
                    double value, double lower, double upper)
{
  std::cout << text << ":\n"
            << m
            << "\t" << value
//            << "\t" << lower
//            << "\t" << upper
            << std::endl;
}

inline void timings(const char* text, size_t m, double *T,
                    size_t N, int algorithm=MEDIAN)
{
  double lower=0.0, upper=0.0;
  if(algorithm == WRITETOFILE) {
    std::ofstream myfile;
    myfile.open("timing.dat", std::fstream::app);
    myfile << m << "\t";
    myfile << N << "\t";
    for(size_t i=0; i < N; ++i)
      myfile << T[i]*unit << "\t";
    myfile << "\n";
  } else {
    double avg=value(T,N,algorithm)*unit;
    stdev(T,N,lower,upper,algorithm);
    std::cout << "Statistics: " << Algorithm[algorithm] << " " << "time" << std::endl;
    std::cout << "Count: " << N << std::endl << std::endl;

    timings(text,m,N,avg,lower*unit,upper*unit);
  }
}

#endif
