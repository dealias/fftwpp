#include "fftw++.h"

namespace fftwpp {

std::ifstream fftw::ifWisdom;
std::ofstream fftw::ofWisdom;
bool fftw::Wise=false;
const double fftw::twopi=2.0*acos(-1.0);

// User settings:
unsigned int fftw::effort=FFTW_MEASURE;
const char *fftw::WisdomName="wisdom3.txt";
}
