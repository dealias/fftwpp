#include <mpi.h>
#include "../Complex.h"
#include "../seconds.h"
#include "mpitranspose.h"
#include "utils.h"
#include "mpiutils.h"
#include <unistd.h>

using namespace std;
using namespace fftwpp;

bool test=false;
bool quiet=false;

unsigned int X=8, Y=8, Z=1;
int a=0; // Test for best block divisor
int alltoall=-1; // Test for best alltoall routine

const unsigned int showlimit=1024;
unsigned int N0=1000000;
int N=0;
bool outtranspose=false;

namespace fftwpp {
void MPILoadWisdom(const MPI_Comm& active);
void MPISaveWisdom(const MPI_Comm& active);
}

void init(Complex *data, unsigned int X, unsigned int y, unsigned int Z,
	  int xstart, int ystart) {
  for(unsigned int i=0; i < X; ++i) { 
    for(unsigned int j=0; j < y; ++j) {
      for(unsigned int k=0; k < Z; ++k) {
        data[(y*i+j)*Z+k].re=xstart+i;
        data[(y*i+j)*Z+k].im=ystart+j;
      }
    }
  }
}

inline void usage()
{
  std::cerr << "Options: " << std::endl;
  std::cerr << "-h\t\t help" << std::endl;
  std::cerr << "-T<int>\t\t number of threads" << std::endl;
  std::cerr << "-t\t\t test" << std::endl;
  std::cerr << "-q\t\t quiet" << std::endl;
  std::cerr << "-N<int>\t\t number of timing tests"
	    << std::endl;
  std::cerr << "-m<int>\t\t size" << std::endl;
  std::cerr << "-x<int>\t\t x size" << std::endl;
  std::cerr << "-y<int>\t\t y size" << std::endl;
  std::cerr << "-z<int>\t\t z size" << std::endl;
  std::cerr << "-a<int>\t\t block divisor: -1=sqrt(size), [0]=Tune" << std::endl;
  std::cerr << "-A<int>\t\t alltoall: [-1]=Tune, 0=Optimized, 1=MPI" << std::endl;
  std::cerr << "-L\t\t locally transpose output" << std::endl;
  exit(1);
}

#ifdef OLD
#include <fftw3-mpi.h>
void fftwTranspose(int rank, int size)
{
  Complex *data;
  ptrdiff_t x,xstart;
  ptrdiff_t y,ystart;
  
  fftw_mpi_init();
  
  /* get local data size and allocate */
  ptrdiff_t NN[2]={Y,X};
  unsigned int block=ceilquotient(Y,size);
  ptrdiff_t alloc=
    fftw_mpi_local_size_many_transposed(2,NN,Z,block,0,
					MPI_COMM_WORLD,&y,
					&ystart,&x,&xstart);
  if(rank == 0) {
    cout << "x=" << x << endl;
    cout << "y=" << y << endl;
    cout << "X=" << X << endl;
    cout << "Y=" << Y << endl;
    cout << "Z=" << Z << endl;
    cout << "N=" << N << endl;
  }

  
  data=ComplexAlign(alloc);
  
  if(rank == 0) cout << "\nOLD\n" << endl;
  
  fftwpp::MPILoadWisdom(MPI_COMM_WORLD);
  fftw_plan inplan=fftw_mpi_plan_many_transpose(Y,X,2*Z,block,0,
                                                (double*) data,(double*) data,
                                                MPI_COMM_WORLD,
						FFTW_MPI_TRANSPOSED_IN);
  fftw_plan outplan=fftw_mpi_plan_many_transpose(X,Y,2*Z,0,block,
                                                 (double*) data,(double*) data,
                                                 MPI_COMM_WORLD,
                                                 outtranspose ? 0 : FFTW_MPI_TRANSPOSED_OUT);
  fftwpp::MPISaveWisdom(MPI_COMM_WORLD);
  
  init(data,X,y,Z,xstart,ystart);

  bool showoutput=X*Y < showlimit && N == 1;
  if(showoutput)
    show(data,X,y*Z,MPI_COMM_WORLD);
  
  fftw::statistics Sininit,Sinwait0,Sinwait1,Sin;
  fftw::statistics Soutinit,Soutwait0,Soutwait1,Sout;

  for(int k=0; k < N; ++k) {
    double begin=0.0, Tinit0=0.0, Tinit=0.0, Twait0=0.0, Twait1=0.0;
    if(rank == 0) begin=totalseconds();
    fftw_execute(inplan);
    if(rank == 0) Tinit0=totalseconds();
    if(rank == 0) Twait0=totalseconds();
    if(rank == 0) Tinit=totalseconds();
    if(rank == 0) Twait1=totalseconds();
    if(rank == 0) {
      Sin.add(totalseconds()-begin);
      Sininit.add(Tinit0-begin);
      Sinwait0.add(Twait0-Tinit0);
      Sinwait1.add(Twait1-Tinit);
    }

    if(showoutput) {
      if(rank == 0) cout << "\nTranspose:" << endl;
      show(data,x,Y*Z,MPI_COMM_WORLD);
    }
    
    if(rank == 0) begin=totalseconds();
    fftw_execute(outplan);
    if(rank == 0) Tinit0=totalseconds();
    if(rank == 0) Twait0=totalseconds();
    if(rank == 0) Tinit=totalseconds();
    if(rank == 0) Twait1=totalseconds();
    if(rank == 0) {
      Sout.add(totalseconds()-begin);
      Soutinit.add(Tinit0-begin);
      Soutwait0.add(Twait0-Tinit0);
      Soutwait1.add(Twait1-Tinit);
    }
  }
  
  if(showoutput) {
    if(outtranspose) {
      if(rank == 0) cout << "\nOutput:\n" << endl;
      show(data,y,X*Z,MPI_COMM_WORLD);
    } else {
      if(rank == 0) cout << "\nOriginal:\n" << endl;
      show(data,X,y*Z,MPI_COMM_WORLD);
    }
  }

  if(rank == 0) {
    Sininit.output("Tininit",X);
    Sinwait0.output("Tinwait0",X);
    Sinwait1.output("Tinwait1",X);
    Sin.output("Tin",X);
    cout << endl;
    Soutinit.output("Toutinit",X);
    Soutwait0.output("Toutwait0",X);
    Soutwait1.output("Toutwait1",X);
    Sout.output("Tout",X);
  }
  
  fftw_destroy_plan(inplan);
  fftw_destroy_plan(outplan);
}
#endif  
  
int transpose(int rank, int size, int N)
{
  int retval=0;

  Complex *data;
  
  unsigned int xsize=localsize(X,size);
  unsigned int ysize=localsize(Y,size);
  size=std::max(xsize,ysize);
  
  unsigned int x=localdimension(X,rank,size);
  unsigned int y=localdimension(Y,rank,size);
  
  unsigned int xstart=localstart(X,rank,size);
  unsigned int ystart=localstart(Y,rank,size);

  MPI_Comm active; 
  MPI_Comm_split(MPI_COMM_WORLD,rank < size,0,&active);

  
  if(rank < size) {

    if(rank == 0) {
      cout << "rank=" << rank << endl;
      cout << "size=" << size << endl;
      cout << "x=" << x << endl;
      cout << "y=" << y << endl;
      cout << "X=" << X << endl;
      cout << "Y=" << Y << endl;
      cout << "Z=" << Z << endl;
      cout << "N=" << N << endl;
    }
    
    data=ComplexAlign(std::max(X*y,x*Y)*Z);
  
    init(data,X,y,Z,xstart,ystart);

    //    show(data,X,y*Z,active);
    
    // Initialize remaining plans.

//    mpitranspose<Complex> T(X,y,x,Y,Z,data,NULL,fftw::maxthreads,active);
    mpitranspose<Complex> T(X,y,x,Y,Z,data,NULL,
                            mpioptions(fftw::maxthreads,a,alltoall),active);
    init(data,X,y,Z,xstart,ystart);
    T.transpose(data,false,true);
  
    //    show(data,x,Y*Z,active);
    
    T.NmTranspose();
    init(data,X,y,Z,xstart,ystart);
    
    fftw::statistics Sininit,Sinwait0,Sinwait1,Sin;
    fftw::statistics Soutinit,Soutwait0,Soutwait1,Sout;

    bool showoutput=!quiet && (test || (!X*Y < showlimit && N == 1));

    if(showoutput) {
      if(rank == 0) 
        cout << "\nInput:" << endl;
      show(data,X,y*Z,active);
    }
    
    if(test) {
      if(rank == 0)
	cout << "\nDiagnostics and unit test.\n" << endl;

      init(data,X,y,Z,xstart,ystart);
      if(showoutput) {
	if(rank == 0) 
	  cout << "Input:" << endl;
	show(data,X,y*Z,active);
      }

      Complex *wholedata=NULL, *wholeoutput=NULL;
      if(rank == 0) {
	wholedata=new Complex[X*Y*Z];
	wholeoutput=new Complex[X*Y*Z];
      }
      accumulate_split(data,wholedata,X,Y,xstart,ystart,x,y,Z,true,active);

      if(showoutput && rank == 0) {
	cout << "\nAccumulated input data:" << endl;
	show(wholedata,X,Y,0,0,X,Y);
      }

      T.transpose(data,false,true); // N x m -> n x M

      if(showoutput) {
	if(rank == 0)
	  cout << "\nOutput:" << endl;
	  show(data,X,y*Z,active);
      }

      accumulate_split(data,wholeoutput,X,Y,xstart,ystart,x,y,Z,false,active);

      if(rank == 0) {
	if(showoutput) {
	  cout << "\nAccumulated output data:" << endl;
	  show(wholeoutput,X,Y,0,0,X,Y);
	}

	bool success=true;
	const unsigned int stop=X*Y*Z;
	for(unsigned int pos=0; pos < stop; ++pos) {
	  if(wholedata[pos] != wholeoutput[pos])
	    success=false;
	}
		
	if(success == true) {
	  cout << "\nTest succeeded." << endl;
	} else {
	  cout << "\nERROR: TEST FAILED!" << endl;
	  ++retval;
	}

      }
    } else {
      if(rank == 0)
	cout << "\nSpeed test.\n" << endl;
      for(int k=0; k < N; ++k) {
	init(data,X,y,Z,xstart,ystart);
    
	double begin=0.0, Tinit0=0.0, Tinit=0.0, Twait0=0.0, Twait1=0.0;
	if(rank == 0) begin=totalseconds();

	T.inphase0();
	if(rank == 0) Tinit0=totalseconds();
	T.insync0();
	if(rank == 0) Twait0=totalseconds();
	T.inphase1();
	if(rank == 0) Tinit=totalseconds();
	T.insync1();
	if(rank == 0) Twait1=totalseconds();
	T.inpost();

	if(rank == 0) {
	  Sin.add(totalseconds()-begin);
	  Sininit.add(Tinit0-begin);
	  Sinwait0.add(Twait0-Tinit0);
	  Sinwait1.add(Twait1-Tinit);
	}

	if(showoutput) {
	  if(rank == 0) cout << "Transpose:" << endl;
	  show(data,x,Y*Z,active);
          if(rank == 0) cout << endl;
        }
        
	if(rank == 0) begin=totalseconds();
	T.outphase0();
	if(rank == 0) Tinit0=totalseconds();
	T.outsync0();
	if(rank == 0) Twait0=totalseconds();
	T.outphase1();
	if(rank == 0) Tinit=totalseconds();
	T.outsync1();
	if(rank == 0) Twait1=totalseconds();
	if(outtranspose) T.NmTranspose();
    
	if(rank == 0) {
	  Sout.add(totalseconds()-begin);
	  Soutinit.add(Tinit0-begin);
	  Soutwait0.add(Twait0-Tinit0);
	  Soutwait1.add(Twait1-Tinit);
	}
      }
  
      if(showoutput) {
      	if(outtranspose) {
      	  if(rank == 0) cout << "\nOutput:" << endl;
      	  show(data,y,X*Z,active);
      	} else {
      	  if(rank == 0) cout << "\nOriginal:" << endl;
      	  show(data,X,y*Z,active);
      	}
      }

      if(rank == 0) {
        cout << endl;
	Sininit.output("Tininit",X);
	Sinwait0.output("Tinwait0",X);
	Sinwait1.output("Tinwait1",X);
	Sin.output("Tin",X);
	cout << endl;
	Soutinit.output("Toutinit",X);
	Soutwait0.output("Toutwait0",X);
	Soutwait1.output("Toutwait1",X);
	Sout.output("Tout",X);
      }

    }
  
  }

  return retval;
}

int main(int argc, char **argv)
{
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c=getopt(argc,argv,"hLN:A:a:m:n:T:x:y:z:qt");
    if (c == -1) break;
                
    switch (c) {
    case 0:
      break;
    case 'N':
      N=atoi(optarg);
      break;
    case 'L':
      outtranspose=true;
      break;
    case 'A':
      alltoall=atoi(optarg);
      break;
    case 'a':
      a=atoi(optarg);
      break;
    case 'm':
      X=Y=atoi(optarg);
      break;
    case 'x':
      X=atoi(optarg);
      break;
    case 'y':
      Y=atoi(optarg);
      break;
    case 'z':
      Z=atoi(optarg);
      break;
    case 'T':
      fftw::maxthreads=atoi(optarg);
      break;
    case 't':
      test=true;
      break;
    case 'q':
      quiet=true;
      break;
    case 'n':
      N0=atoi(optarg);
      break;
    case 'h':
    default:
      usage();
    }
  }

  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);

  int rank,size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  
  if(rank == 0) {
    cout << "size=" << size << endl;
    cout << "threads=" << fftw::maxthreads << endl;
  }
  
  if(test) N=1;
  else if(N == 0) {
    N=N0/(X*Y*Z);
    if(N < 10) N=10;
  }

  int retval=0;
#ifdef OLD
  fftwTranspose(rank,size);
#else
  retval=transpose(rank,size,N);
#endif  
  
  MPI_Finalize();
  return retval;
}
