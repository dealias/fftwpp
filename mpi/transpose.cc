#include <mpi.h>
#include <fftw3-mpi.h>
#include "../Complex.h"
#include "../seconds.h"
#include "mpitranspose.h"
#include "utils.h"
#include "mpiutils.h"
#include <unistd.h>

using namespace std;
using namespace fftwpp;

unsigned int X=8, Y=8, Z=1;
const unsigned int showlimit=1024;
unsigned int N0=10000000;
int N=0;
bool outtranspose=false;

namespace fftwpp {
void MPILoadWisdom(const MPI_Comm& active);
void MPISaveWisdom(const MPI_Comm& active);
}

void init(Complex *data, unsigned int X, unsigned int y, unsigned int Z,
	  ptrdiff_t ystart) {
  for(unsigned int i=0; i < X; ++i) { 
    for(unsigned int j=0; j < y; ++j) {
      for(unsigned int k=0; k < Z; ++k) {
        data[(y*i+j)*Z+k].re = i;
        data[(y*i+j)*Z+k].im = ystart+j;
      }
    }
  }
}
  
inline void usage()
{
  std::cerr << "Options: " << std::endl;
  std::cerr << "-h\t\t help" << std::endl;
  std::cerr << "-T\t\t number of threads" << std::endl;
  std::cerr << "-N\t\t number of iterations" << std::endl;
  std::cerr << "-m\t\t size" << std::endl;
  std::cerr << "-X\t\t X size" << std::endl;
  std::cerr << "-Y\t\t Y size" << std::endl;
  std::cerr << "-Z\t\t Z size" << std::endl;
  std::cerr << "-L\t\t local transpose output" << std::endl;
  exit(1);
}

void fftwTranspose(int rank, int size)
{
  Complex *data;
  ptrdiff_t x,xstart;
  ptrdiff_t y,ystart;
  
  /* get local data size and allocate */
  ptrdiff_t NN[2]={Y,X};
  unsigned int block=ceilquotient(Y,size);
  ptrdiff_t alloc=
    fftw_mpi_local_size_many_transposed(2,NN,Z,block,0,
					MPI_COMM_WORLD,&y,
					&ystart,&x,&xstart);
  if(N == 0) {
    N=N0/(X*y);
    if(N < 10) N=10;
  }
  
  if(rank == 0) {
    cout << "x=" << x << endl;
    cout << "y=" << y << endl;
    cout << "X=" << X << endl;
    cout << "Y=" << Y << endl;
    cout << "Z=" << Z << endl;
    cout << "N=" << N << endl;
    cout << endl;
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
  
  init(data,X,y,Z,ystart);

  bool showoutput=X*Y < showlimit && N == 1;
  if(showoutput)
    show(data,X,y*Z,MPI_COMM_WORLD);
  
  fftw::statistics Sininit,Sinwait0,Sinwait1,Sin,Soutinit,Soutwait0,Soutwait1,Sout;

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
      if(rank == 0) cout << "\ntranspose:\n" << endl;
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
      if(rank == 0) cout << "\nout:\n" << endl;
      show(data,y,X*Z,MPI_COMM_WORLD);
    } else {
      if(rank == 0) cout << "\noriginal:\n" << endl;
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
  
int transpose(int rank, int size, int N)
{
  int retval=0; // success!

  Complex *data;
  
  int xsize=localsize(X,size);
  int ysize=localsize(Y,size);
  size=max(xsize,ysize);
  
  cout << "Size=" << size << endl;
  
  int x = localdimension(X,rank,size);
  int y = localdimension(Y,rank,size);
  
  int xstart=localstart(X,rank,size);
  int ystart=localstart(Y,rank,size);
  
  MPI_Comm active; 
  MPI_Comm_split(MPI_COMM_WORLD,rank < size,0,&active);

  if(rank < size) {
    cout << "rank=" << rank << " size=" << size << " x=" << x << " " <<
      " y=" << y << endl;
  
    if(rank == 0) {
      cout << "x=" << x << endl;
      cout << "y=" << y << endl;
      cout << "X=" << X << endl;
      cout << "Y=" << Y << endl;
      cout << "Z=" << Z << endl;
      cout << "N=" << N << endl;
      cout << endl;
    }
  
    data=ComplexAlign(X*y*Z);
  
    init(data,X,y,Z,ystart);

    //  show(data,X,y*Z,active);
    
    // Initialize remaining plans.

    mpitranspose<Complex> T(X,y,x,Y,Z,data,NULL,fftw::maxthreads,active);
    init(data,X,y,Z,ystart);
    T.transpose(data,false,true);
  
    //  show(data,x,Y*Z,active);
  
    T.NmTranspose();
    init(data,X,y,Z,ystart);
  
    bool showoutput=X*Y < showlimit && N == 1;
    if(showoutput)
      show(data,X,y*Z,active);
  
    fftw::statistics Sininit,Sinwait0,Sinwait1,Sin,Soutinit;
    fftw::statistics Soutwait0,Soutwait1,Sout;

    if(N == 0) {
      if(rank == 0)
	cout << "Diagnostics and unit test.\n" << endl;
      bool showoutput=true; //X*Y < showlimit;

      init(data,X,y,Z,ystart);
      if(showoutput) {
	if(rank == 0) 
	  cout << "Input:\n" << endl;
	show(data,X,y*Z,active);
      }

      Complex *wholedata=NULL, *wholeoutput=NULL;
      Transpose *localtranspose=NULL;
      if(rank == 0) {
	wholedata=new Complex[X*Y*Z];
	wholeoutput=new Complex[X*Y*Z];
	localtranspose=new Transpose(X,Y,Z,wholedata);
      }
      accumulate_splitx(data,wholedata,X,Y,xstart,ystart,x,y,true,active);

      if(showoutput && rank == 0) {
	cout << "\naccumulated input data:" << endl;
	show(wholedata,X,Y,0,0,X,Y);
      }


      T.transpose(data,false,true); // false,true :  N x m -> n x M

      if(showoutput) {
	if(rank == 0)
	  cout << "\noutput:\n" << endl;
	if(outtranspose)
	  show(data,y,X*Z,active);
	else
	  show(data,X,y*Z,active);
      }

      accumulate_splitx(data,wholeoutput,X,Y,xstart,ystart,x,y,false,active);

      if(rank == 0) {
	if(outtranspose) 
	  localtranspose->transpose(wholedata);

	if(showoutput) {
	  cout << "\naccumulated output data:" << endl;
	  show(wholeoutput,X,Y,0,0,X,Y);
	  if(outtranspose) {
	    cout << "\nlocally transposed data:" << endl;
	    show(wholedata,X,Y,0,0,X,Y);
	  }
	}

	bool success=true;
	for(unsigned int i=0; i < X; ++i) {
	  for(unsigned int j=0; j < Y; ++j) {
	    unsigned int pos=i * Y + j; 
	    if(wholedata[pos] != wholeoutput[pos])
	      success=false;
	  }
	}
	
	if(success == true) {
	  cout << "\nTest succeeded." << endl;
	} else {
	  cout << "\nERROR: TEST FAILED!" << endl;
	  retval += 1;
	}

      }
      
    } else {
      if(rank == 0)
	cout << "Speed test.\n" << endl;
      for(int k=0; k < N; ++k) {
	init(data,X,y,Z,ystart);
    
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

	// if(showoutput) {
	//   if(rank == 0) cout << "\ntranspose:\n" << endl;
	//   show(data,x,Y*Z,active);
	// }

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
  
      // if(showoutput) {
      // 	if(outtranspose) {
      // 	  if(rank == 0) cout << "\nout:\n" << endl;
      // 	  show(data,y,X*Z,active);
      // 	} else {
      // 	  if(rank == 0) cout << "\noriginal:\n" << endl;
      // 	  show(data,X,y*Z,active);
      // 	}
      // }

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

    }
  
  }

  return retval;
}

int main(int argc, char **argv)
{
  bool Nset = false;

#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"hLN:m:n:T:X:Y:Z:");
    if (c == -1) break;
                
    switch (c) {
    case 0:
      break;
    case 'N':
      N=atoi(optarg);
      Nset = true;
      break;
    case 'L':
      outtranspose=true;
      break;
    case 'm':
      X=Y=atoi(optarg);
      break;
    case 'X':
      X=atoi(optarg);
      break;
    case 'Y':
      Y=atoi(optarg);
      break;
    case 'Z':
      Z=atoi(optarg);
      break;
    case 'T':
      fftw::maxthreads=atoi(optarg);
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
  //  MPI_Init(&argc,&argv);
  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  if(rank == 0) {
    cout << "size=" << size << endl;
    cout << "threads=" << fftw::maxthreads << endl;
  }
  
  fftw_mpi_init();
  
#ifdef OLD
  fftwTranspose(rank, size);
#else
  if(!Nset) {
    N=N0/(X*Y);
    if(N < 10) N=10;
  }
  transpose(rank,size, N);
#endif  
  
  MPI_Finalize();
  return 0;
}
