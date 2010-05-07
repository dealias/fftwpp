#include "fftw++.h"

namespace fftwpp {

std::ifstream fftw::ifWisdom;
std::ofstream fftw::ofWisdom;
bool fftw::Wise=false;

// User settings:
unsigned int fftw::effort=FFTW_MEASURE;
const char *fftw::WisdomName="wisdom3.txt";
}
