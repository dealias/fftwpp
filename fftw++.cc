#include <cstring>
#include "fftw++.h"

namespace fftwpp {

const double fftw::twopi=2.0*acos(-1.0);

fft1d::Table fft1d::threadtable;
mfft1d::Table mfft1d::threadtable;
rcfft1d::Table rcfft1d::threadtable;
crfft1d::Table crfft1d::threadtable;
mrcfft1d::Table mrcfft1d::threadtable;
mcrfft1d::Table mcrfft1d::threadtable;
fft2d::Table fft2d::threadtable;

// User settings:
unsigned int fftw::effort=FFTW_MEASURE;
const char *fftw::WisdomName="wisdom3.txt";
size_t fftw::WisdomLength=0;
unsigned int fftw::maxthreads=1;
double fftw::testseconds=1.0; // Time limit for threading efficiency tests
void (*fftw::beforePlanner)()=BeforePlanner;
void (*fftw::afterPlanner)()=AfterPlanner;

const char *fftw::oddshift="Shift is not implemented for odd nx";
const char *inout=
  "constructor and call must be both in place or both out of place";

void fftw::LoadWisdom() {
  std::ifstream ifWisdom;
  ifWisdom.open(WisdomName);
  fftwpp_import_wisdom(GetWisdom,ifWisdom);
  ifWisdom.close();
  char *wisdom=fftw_export_wisdom_to_string();
  WisdomLength=strlen(wisdom);
  fftw_free(wisdom);
}

void fftw::SaveWisdom() {
  std::cout << "Save?" << std::endl;
  char *wisdom=fftw_export_wisdom_to_string();
  size_t len=strlen(wisdom);
  if(len != WisdomLength) {
    std::ofstream ofWisdom;
    ofWisdom.open(WisdomName);
    ofWisdom << wisdom;
    ofWisdom.close();
    WisdomLength=len;
  }
  fftw_free(wisdom);
}
  
void BeforePlanner()
{
  static bool Wise=false;
  if(!Wise) {
    fftw::LoadWisdom();
    Wise=true;
  }
}

void AfterPlanner()
{
  fftw::SaveWisdom();
}

}
