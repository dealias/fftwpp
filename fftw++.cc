#include <cstring>
#include <sstream>
#include "fftw++.h"

using namespace std;

namespace utils {
int ALIGNMENT=2*sizeof(Complex); // Must be a multiple of sizeof(Complex)
}

namespace fftwpp {

const double fftw::twopi=2.0*acos(-1.0);

// User settings:
unsigned int fftw::effort=FFTW_MEASURE;
const char *fftw::WisdomName="wisdom3.txt";
unsigned int fftw::maxthreads=1;
double fftw::testseconds=0.2; // Time limit for threading efficiency tests

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
  ofstream ofWisdom;
  ofWisdom.open(fftw::WisdomName);
  char *wisdom=fftw_export_wisdom_to_string();
  ofWisdom << wisdom;
  fftw_free(wisdom);
  ofWisdom.close();
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

unsigned int parallelLoop(Complex *A, unsigned int m, unsigned int threads)
{
  auto T0=std::chrono::steady_clock::now();
  for(unsigned int i=0; i < 10; ++i) {
    PARALLEL(
      for(unsigned int k=0; k < m; ++k)
        A[k]=k;
      );
    PARALLEL(
      for(unsigned int k=0; k < m; ++k)
        A[k] *= k;
      );
  }
  auto T1=std::chrono::steady_clock::now();

  auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>
    (T1-T0);
  return elapsed.count();
}

unsigned int threshold=UINT_MAX-1;

unsigned int Threshold(unsigned int threads)
{
  if(threads > 1) {
    for(unsigned int m=1; m < UINT_MAX; m *= 2) {
      Complex *A=utils::ComplexAlign(m);
      if(!A)
        break;
      if(parallelLoop(A,m,threads) < parallelLoop(A,m,1))
        return m;
      utils::deleteAlign(A);
    }
  }
  return UINT_MAX;
}

void Threshold()
{
  if(threshold == UINT_MAX-1)
    threshold=Threshold(fftw::maxthreads);
}

ThreadBase::ThreadBase() {threads=fftw::maxthreads;}

}

namespace utils {
unsigned int defaultmpithreads=1;
}
