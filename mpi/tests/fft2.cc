 #include "Array.h"
#include "mpifftw++.h"
#include "utils.h"

//using namespace utils;
using namespace fftwpp;
using namespace Array;

inline void init(Complex *f, utils::split d) 
{
  unsigned int c=0;
  for(unsigned int i=0; i < d.x; ++i) {
    unsigned int ii=d.x0+i;
    for(unsigned int j=0; j < d.Y; j++) {
      f[c++]=Complex(ii,j);
    }
  }
}


int main(int argc, char* argv[])
{
  int retval = 0; // success!

  unsigned int outlimit=100;
  
#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  // Number of iterations.
  unsigned int N0=10000000;
  unsigned int N=0;
  unsigned int nx=4;
  unsigned int ny=4;
  int divisor=0; // Test for best block divisor
  int alltoall=-1; // Test for best alltoall routine

  bool inplace=true;
  
  bool quiet=false;
  bool test=false;
  
  unsigned int stats=0; // Type of statistics used in timing test.
  
  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if(rank != 0) opterr=0;
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"hL:N:a:i:m:s:x:y:n:S:T:qt");
    if (c == -1) break;
                
    switch (c) {
      case 0:
        break;
      case 'a':
        divisor=atoi(optarg);
        break;
      case 'N':
        N=atoi(optarg);
        break;
      case 'i':
        inplace=atoi(optarg);
        break;
      case 'm':
        nx=ny=atoi(optarg);
        break;
      case 's':
        alltoall=atoi(optarg);
        break;
      case 'x':
        nx=atoi(optarg);
        break;
      case 'y':
        ny=atoi(optarg);
        break;
      case 'n':
        N0=atoi(optarg);
        break;
      case 'S':
        stats=atoi(optarg);
        break;
      case 'T':
        fftw::maxthreads=atoi(optarg);
        break;
      case 'q':
        quiet=true;
        break;
      case 't':
        test=true;
        break;
      case 'h':
      default:
        if(rank == 0) {
            utils::usageInplace(2);
          utils::usageTranspose();
        }
        exit(1);
    }
  }

  if(ny == 0) ny=nx;

  if(N == 0) {
    N=N0/nx/ny;
    if(N < 10) N=10;
  }
  
  utils::MPIgroup group(MPI_COMM_WORLD,ny);

  if(group.size > 1 && provided < MPI_THREAD_FUNNELED)
    fftw::maxthreads=1;
  
  utils::defaultmpithreads=fftw::maxthreads;

  if(group.rank < group.size) { 
    bool main=group.rank == 0;
  
    if(!quiet && main) {
      std::cout << "Configuration: " 
           << group.size << " nodes X " << fftw::maxthreads 
           << " threads/node" << std::endl;
      std::cout << "Using MPI VERSION " << MPI_VERSION << std::endl;
      std::cout << "N=" << N << std::endl;
      std::cout << "nx=" << nx << ", ny=" << ny << std::endl;
    } 

    bool showresult = nx*ny < outlimit;
    
    utils::split d(nx,ny,group.active);
  
    Complex *f=utils::ComplexAlign(d.n);
    Complex *g=inplace ? f : utils::ComplexAlign(d.n);

    // Create instance of FFT
    fft2dMPI fft(d,f,g,utils::mpiOptions(divisor,alltoall,utils::defaultmpithreads,0));

    if(!quiet && group.rank == 0)
        std::cout << "Initialized after " << utils::seconds() << " seconds." << std::endl;    

    if(test) {
      init(f,d);

      if(!quiet && showresult) {
        if(main) std::cout << "\nDistributed input:" << std::endl;
        utils::show(f,d.x,ny,group.active);
      }

      size_t align=sizeof(Complex);
      array2<Complex> flocal(nx,ny,align);
      fft2d localForward(-1,flocal);
      fft2d localBackward(1,flocal);

      utils::gatherx(f,flocal(),d,1,group.active);

      if(!quiet && main) {
        std::cout << "\nGathered input:\n" << flocal << std::endl;
      }

      fft.Forward(f,g);

      if(!quiet && showresult) {
        if(main)
	  std::cout << "\nDistributed output:" << std::endl;
        utils::show(g,nx,d.y,group.active);
      }
      
      array2<Complex> fgather(nx,ny,align);
      
      utils::gathery(g,fgather(),d,1,group.active);
      
      MPI_Barrier(group.active);
      if(main) {
        localForward.fft(flocal);
        if(!quiet) {
          std::cout << "\nGathered output:\n" << fgather << std::endl;
          std::cout << "\nLocal output:\n" << flocal << std::endl;
        }
        double maxerr=0.0, norm=0.0;
        unsigned int stop=d.X*d.Y;
        for(unsigned int i=0; i < stop; i++) {
          maxerr=std::max(maxerr,abs(fgather(i)-flocal(i)));
          norm=std::max(norm,abs(flocal(i)));
        }
        std::cout << "max error: " << maxerr << std::endl;
        if(maxerr > 1e-12*norm) {
          std::cerr << "CAUTION: max error is LARGE!" << std::endl;
          retval += 1;
        }
      }

      fft.Backward(g,f);
      fft.Normalize(f);

      if(!quiet && showresult) {
        if(main) std::cout << "\nDistributed inverse:" << std::endl;
        utils::show(f,d.x,ny,group.active);
      }

      utils::gatherx(f,fgather(),d,1,group.active);
      MPI_Barrier(group.active);
      if(main) {
        localBackward.fftNormalized(flocal);
        if(!quiet) {
          std::cout << "\nGathered inverse:\n" << fgather << std::endl;
          std::cout << "\nLocal inverse:\n" << flocal << std::endl;
        }
        retval += utils::checkerror(flocal(),fgather(),d.X*d.Y);
      }

      if(!quiet && group.rank == 0) {
        std::cout << std::endl;
        if(retval == 0)
          std::cout << "pass" << std::endl;
        else
          std::cout << "FAIL" << std::endl;
      }  
  
    } else {
      if(N > 0) {
        double *T=new double[N];
        for(unsigned int i=0; i < N; ++i) {
          init(f,d);
          utils::seconds();
          fft.Forward(f,g);
          fft.Backward(g,f);
          T[i]=0.5*utils::seconds();
          fft.Normalize(f);
        }    
        if(!quiet && showresult)
	  utils::show(f,d.X,d.y,group.active);
        if(main)
	  timings("FFT timing:",nx,T,N,stats);
        delete [] T;
      }
    }

    utils::deleteAlign(f);
    if(!inplace)
      utils::deleteAlign(g);
  }
  
  MPI_Finalize();

  return retval;
}
