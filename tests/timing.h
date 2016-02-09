#ifndef __timing_h__
#define __timing_h__ 1

#include <math.h>
#include <algorithm> // For std::sort

#include <iostream>
#include <fstream>

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

enum timing_algorithm {WRITETOFILE = -1, MEAN, MIN, MAX, MEDIAN, P90, P80, P50};

inline double mean(double *T, unsigned int N, int algorithm) 
{
  switch(algorithm) {
    case WRITETOFILE: 
    case MEAN: {
      double sum=0.0;
      for(unsigned int i=0; i < N; ++i)
        sum += T[i];
      return sum/N;
      break;
    }
    case MIN: {
      double min=T[0];
      for(unsigned int i=0; i < N; ++i) {
        if(T[i] < min)
          min=T[i];
      }
      return min;
      break;
    }
    case MAX: {
      double max=T[0];
      for(unsigned int i=0; i < N; ++i) {
        if(T[i] > max)
          max=T[i];
      }
      return max;
      break;
    }
    case MEDIAN: {
      std::sort(T,T+N);
      return T[(int)ceil(N*0.5)];
      break;
    }
    case P90: {
      std::sort(T,T+N);
      unsigned int start=(int)ceil(N*0.05);
      unsigned int stop=(int)floor(N*0.95);
      
      double sum=0.0;
      for(unsigned int i=start; i < stop; ++i)
        sum += T[i];
      return sum/(stop-start);
      break;
    }
    case P80: {
      std::sort(T,T+N);
      unsigned int start=(int)ceil(N*0.1);
      unsigned int stop=(int)floor(N*0.9);
      
      double sum=0.0;
      for(unsigned int i=start; i < stop; ++i)
        sum += T[i];
      return sum/(stop-start);
      break;
    }
    case P50: {
      std::sort(T,T+N);
      unsigned int start=(int)ceil(N*0.25);
      unsigned int stop=(int)floor(N*0.75);
      
      double sum=0.0;
      for(unsigned int i=start; i < stop; ++i)
        sum += T[i];
      return sum/(stop-start);
      break;
    }
    default:
      std::cout << "Error: invalid algorithm choice: " 
                << algorithm
                << std::endl;
      exit(1);
  }
  return 0;
}

inline void stdev(double *T, unsigned int N, double mean, 
                  double &sigmaL, double& sigmaH,
                  int algorithm) 
{
  switch(algorithm) {
    case WRITETOFILE: 
    case MEAN:  {
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
      break;
    case MIN:
      sigmaL=0.0;
      sigmaH=0.0;
      break;
    case MAX:
      sigmaL=0.0;
      sigmaH=0.0;
      break;
    case MEDIAN:
    {
      // Return 68% confidence intervals
      sigmaL=mean-T[(int)ceil(N*(0.19))];
      sigmaH=T[(int)ceil(N*0.81)]-mean;
    }
    break;
    case P90:  {
      unsigned int start=(int)ceil(N*0.5);
      unsigned int stop=(int)floor(N*0.95);
      sigmaL=mean-T[start];
      sigmaH=T[stop]-mean;
    }
      break;
    case P80: {
      unsigned int start=(int)ceil(N*0.1);
      unsigned int stop=(int)floor(N*0.9);
      sigmaL=mean-T[start];
      sigmaH=T[stop]-mean;
    }
      break;
    case P50: {
      unsigned int start=(int)ceil(N*0.25);
      unsigned int stop=(int)floor(N*0.75);
      sigmaL=mean-T[start];
      sigmaH=T[stop]-mean;
    }
      break;
    default:
      std::cout << "Error: invalid algorithm choice: " 
                << algorithm
                << std::endl;
      exit(1);
  }
}

inline void timings(const char* text, unsigned int m, unsigned int count,
                    double mean, double sigmaL, double sigmaH)
{
//  mean -= emptytime(T,N);
//  if(mean < 0.0) mean=0.0;
  std::cout << text << ":\n" 
            << m << "\t" 
            << mean << "\t" 
            << sigmaL << "\t" 
            << sigmaH 
            << std::endl;
}

inline void timings(const char* text, unsigned int m, double *T, 
                    unsigned int N, int algorithm=MEAN)
{
  double sigmaL=0.0, sigmaH=0.0;
  if(algorithm == WRITETOFILE) {
    std::ofstream myfile;
    myfile.open ("timing.dat", std::fstream::app);
    myfile << m << "\t";
    myfile << N << "\t";
    for(unsigned int i=0;  i<N; ++i)
      myfile << T[i] << "\t";
    myfile << "\n";
  }

  double avg=mean(T,N,algorithm);
  stdev(T,N,avg,sigmaL,sigmaH,algorithm);
  timings(text,m,N,avg,sigmaL,sigmaH);
}

#endif
