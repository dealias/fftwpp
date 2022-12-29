#include <cstring>
#include <sstream>

#include "fftw++.h"

using namespace std;
using namespace utils;

namespace utils {
int ALIGNMENT=2*sizeof(Complex); // Must be a multiple of sizeof(Complex)
}

namespace fftwpp {

const double fftw::twopi=2.0*acos(-1.0);

// User settings:
size_t fftw::effort=FFTW_MEASURE;
const char *fftw::WisdomName="wisdom3.txt";
ostringstream WisdomTemp;
size_t fftw::maxthreads=1;

fftw_plan (*fftw::planner)(fftw *f, Complex *in, Complex *out)=Planner;

const char *fftw::oddshift="Shift is not implemented for odd nx";
const char *inout=
  "constructor and call must be both in place or both out of place";

Mfft1d::Table Mfft1d::threadtable;
Mrcfft1d::Table Mrcfft1d::threadtable;
Mcrfft1d::Table Mcrfft1d::threadtable;

void LoadWisdom()
{
  static bool Wise=false;
  if(!Wise) {
    WisdomTemp << fftw::WisdomName << "_" << getpid();
    ifstream ifWisdom;
    ifWisdom.open(fftw::WisdomName);
    ostringstream wisdom;
    wisdom << ifWisdom.rdbuf();
    ifWisdom.close();
    const string& s=wisdom.str();
    fftw_import_wisdom_from_string(s.c_str());
    Wise=true;
  }
}

void SaveWisdom()
{
  char *wisdom=fftw_export_wisdom_to_string();
  ofstream ofWisdom;
  ofWisdom.open(WisdomTemp.str().c_str());
  ofWisdom << wisdom;
  fftw_free(wisdom);
  ofWisdom.close();
  rename(WisdomTemp.str().c_str(),fftw::WisdomName);
}

fftw_plan Planner(fftw *F, Complex *in, Complex *out)
{
  LoadWisdom();
  fftw::effort |= FFTW_WISDOM_ONLY;
  fftw_plan plan=F->Plan(in,out);
  fftw::effort &= ~FFTW_WISDOM_ONLY;
  if(!plan) {
    plan=F->Plan(in,out);
    SaveWisdom();
  }
  return plan;
}

size_t parallelLoop(Complex *A, size_t m, size_t threads)
{
  auto T0=std::chrono::steady_clock::now();
  PARALLEL(
    for(size_t k=0; k < m; ++k)
      A[k]=k;
    );
  PARALLEL(
    for(size_t k=0; k < m; ++k)
      A[k] *= k;
    );
  auto T1=std::chrono::steady_clock::now();

  auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>
    (T1-T0);
  return elapsed.count();
}

const size_t maxThreshold=1 << 24;
size_t threshold=0;
size_t lastThreads=SIZE_MAX;

size_t measureThreshold(size_t threads)
{
  if(threads > 1) {
    for(size_t m=1; m < maxThreshold; m *= 2) {
      Complex *A=ComplexAlign(m);
      if(!A)
        break;
      if(parallelLoop(A,m,threads) < parallelLoop(A,m,1))
        return m;
      deleteAlign(A);
    }
  }
  return maxThreshold;
}

void Threshold(size_t threads)
{
  if(threads > 1 && threads < lastThreads) {
    statistics S(true);
    for(size_t i=0; i < 10; ++i)
      S.add(measureThreshold(threads));
    threshold=S.median();
    lastThreads=threads;
  }
}

ThreadBase::ThreadBase() {threads=fftw::maxthreads;}

}

namespace utils {
size_t defaultmpithreads=1;
}
