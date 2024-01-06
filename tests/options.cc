#include "convolve.h"
#include "timing.h"
#include "options.h"

using namespace std;
using namespace fftwpp;

size_t L,Lx,Ly,Lz;
size_t M,Mx,My,Mz;
size_t mx=0,my=0,mz=0;
size_t Dx=0,Dy=0,Dz=0;
size_t Sx=0,Sy=0;
ptrdiff_t Ix=-1,Iy=-1,Iz=-1;

bool Output=false;
bool testError=false;
bool Centered=false;
bool normalized=true;
bool Tforced=false;
bool accuracy=false;

double s=1.0; // Time limit (seconds)
size_t N=20;  // Minimum sample size for testing
size_t C=1;
size_t S=0;
int stats=MEDIAN;

namespace utils {

size_t Atoi(char *optarg, size_t min=1)
{
  size_t val=atoll(optarg);
  if(val < min) {
    usageHybrid();
    exit(-1);
  }
  return val;
}

void optionsHybrid(int argc, char *argv[], bool fft, bool mpi)
{
#ifdef __GNUC__
  optind=0;
#endif

  enum Parameters {LXYZ=256,LX,LY,LZ,MXYZ,MX,MY,MZ,SX,SY,mXYZ,mX,mY,mZ,
    DXYZ,DX,DY,DZ,IXYZ,IX,IY,IZ,T,THRESHOLD};

  int option_index = 0;
  static struct option long_options[] =
    {
      {"L", 1, 0, LXYZ},
      {"Lx", 1, 0, LX},
      {"Ly", 1, 0, LY},
      {"Lz", 1, 0, LZ},
      {"M", 1, 0, MXYZ},
      {"Mx", 1, 0, MX},
      {"My", 1, 0, MY},
      {"Mz", 1, 0, MZ},
      {"Sx", 1, 0, SX},
      {"Sy", 1, 0, SY},
      {"m", 1, 0, mXYZ},
      {"mx", 1, 0, mX},
      {"my", 1, 0, mY},
      {"mz", 1, 0, mZ},
      {"D", 1, 0, DXYZ},
      {"Dx", 1, 0, DX},
      {"Dy", 1, 0, DY},
      {"Dz", 1, 0, DZ},
      {"I", 1, 0, IXYZ},
      {"Ix", 1, 0, IX},
      {"Iy", 1, 0, IY},
      {"Iz", 1, 0, IZ},
      {"T", 1, 0, T},
      {"threshold", 1, 0, THRESHOLD},
      {0, 0, 0, 0}
    };

  for (;;) {
    int c=getopt_long_only(argc,argv,"ahC:D:I:s:L:M:N:ctEORS:T:m:u",
                           long_options,&option_index);

    if (c == -1) break;

    switch(c) {
      case 0:
        break;
      case 'C':
        C=Atoi(optarg);
        break;
      case 'D':
      case DXYZ:
        Dx=Dy=Dz=Atoi(optarg);
        break;
      case DX:
        Dx=Atoi(optarg);
        break;
      case DY:
        Dy=Atoi(optarg);
        break;
      case DZ:
        Dz=Atoi(optarg);
        break;
      case 'E':
        testError=true;
        break;
      case 's':
        s=atof(optarg);
        break;
      case 'I':
      case IXYZ:
        Ix=Iy=Iz=atoi(optarg) > 0;
        break;
      case IX:
        Ix=atoi(optarg) > 0;
        break;
      case IY:
        Iy=atoi(optarg) > 0;
        break;
      case IZ:
        Iz=atoi(optarg) > 0;
        break;
      case 'L':
      case LXYZ:
        L=Lx=Ly=Lz=Atoi(optarg);
        break;
      case LX:
        L=Lx=Atoi(optarg);
        break;
      case LY:
        Ly=Atoi(optarg);
        break;
      case LZ:
        Lz=Atoi(optarg);
        break;
      case 'M':
      case MXYZ:
        M=Mx=My=Mz=Atoi(optarg);
        break;
      case MX:
        M=Mx=Atoi(optarg);
        break;
      case MY:
        My=Atoi(optarg);
        break;
      case MZ:
        Mz=Atoi(optarg);
        break;
      case 'm':
      case mXYZ:
        mx=my=mz=Atoi(optarg,0);
        break;
      case mX:
        mx=Atoi(optarg,0);
        break;
      case mY:
        my=Atoi(optarg,0);
        break;
      case mZ:
        mz=Atoi(optarg,0);
        break;
      case 'N':
        N=atoi(optarg);
        break;
      case 'O':
        Output=true;
        break;
      case 'R':
        showRoutines=true;
        break;
      case 'S':
        if(fft)
          S=atoi(optarg);
        else
          stats=atoi(optarg);
        break;
      case SX:
        Sx=Atoi(optarg,0);
        break;
      case SY:
        Sy=Atoi(optarg,0);
        break;
      case 'T':
      case T:
        fftw::maxthreads=Atoi(optarg);
        break;
      case THRESHOLD:
        threshold=Atoi(optarg,0);
        parallel::lastThreads=0;
        break;
      case 'a':
        accuracy=true;
        break;
      case 'c':
        Centered=true;
        break;
      case 'u':
        normalized=false;
        break;
      case 't':
        showOptTimes=true;
        break;
      case 'h':
      default:
        usageHybrid(fft,mpi);
        exit(1);
    }
  }

  if(fft)
    cout << "C=" << C << endl;
}

}
