#include <cstring>
#include <sstream>

#include "fftw++.h"

using namespace std;
using namespace utils;

namespace fftwpp {

const double fftw::twopi=2.0*acos(-1.0);
bool fftw::wiser=false;

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
  if(fftw::wiser) {
    char *wisdom=fftw_export_wisdom_to_string();
    ofstream ofWisdom;
    ofWisdom.open(WisdomTemp.str().c_str());
    ofWisdom << wisdom;
    fftw_free(wisdom);
    ofWisdom.close();
    rename(WisdomTemp.str().c_str(),fftw::WisdomName);
    fftw::wiser=false;
  }
}

fftw_plan Planner(fftw *F, Complex *in, Complex *out)
{
  LoadWisdom();
  fftw::effort |= FFTW_WISDOM_ONLY;
  fftw_plan plan=F->Plan(in,out);
  fftw::effort &= ~FFTW_WISDOM_ONLY;
  if(!plan) {
    plan=F->Plan(in,out);
    static bool first=true;
    if(first) {
      atexit(SaveWisdom);
      first=false;
    }
    fftw::wiser=true;
  }
  return plan;
}

ThreadBase::ThreadBase() {threads=fftw::maxthreads;}

}

namespace utils {
size_t defaultmpithreads=1;
}
