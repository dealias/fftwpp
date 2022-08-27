#ifndef __statistics_h__
#define __statistics_h__ 1

#include <cfloat>
#include <list>

namespace utils {

class statistics {
  unsigned int N;
  double A;
  double varL;
  double varH;
  double m; // min
  double M; // max
  std::list<double> T; // Only used if constructed with median=true
  bool Median;
public:
  statistics(bool median=false) : Median(median) {clear();}
  void clear() {N=0; A=varL=varH=0.0; m=DBL_MAX; M=-m; T.clear();}
  double count() {return N;}
  double mean() {return A;}
  double min() {return m;}
  double max() {return M;}
  void add(double t) {
    m=std::min(m,t);
    M=std::max(M,t);
    ++N;
    double diff=t-A;
    A += diff/N;
    double v=diff*(t-A);
    if(diff < 0.0)
      varL += v;
    else
      varH += v;

    if(Median) {
      auto p=T.begin();
      if(N == 1 || t <= *p) T.push_front(t);
      else {
        size_t l=0;
        size_t u=N-1;
        ssize_t last=0;
        while(l < u) {
          ssize_t i=(l+u)/2;
          std::advance(p,i-last);
          last=i;
          if(t < *p) u=i;
          else {
            ++p;
            if(p == T.end() || t <= *p) {T.insert(p,t); break;}
            last=l=i+1;
          }
        }
      }
    }
  }
  double stdev(double var, double f) {
    double factor=N > f ? f/(N-f) : 0.0;
    return sqrt(var*factor);
  }
  double stdev() {
    return stdev(varL+varH,1.0);
  }
  double stdevL() {
    return stdev(varL,2.0);
  }
  double stdevH() {
    return stdev(varH,2.0);
  }
  double median() {
    if(!Median) {
      std::cerr << "Constructor requires median=true" << std::endl;
      exit(-1);
    }
    unsigned int h=N/2;
    auto p=T.begin();
    std::advance(p,h-1);
    return 2*h == N ? 0.5*(*p+*(++p)) : *(++p);
  }
  void output(const char *text, unsigned int m) {
    std::cout << text << ":\n"
              << m << "\t"
              << A << "\t"
              << stdevL() << "\t"
              << stdevH() << std::endl;
  }
};

}

#endif
