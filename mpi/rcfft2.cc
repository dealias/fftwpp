#include "mpifftw++.h"
#include "utils.h"
#include "mpiutils.h"

using namespace std;
using namespace fftwpp;

// Number of iterations.
unsigned int N0=10000000;
unsigned int N=0;
unsigned int mx=4;
unsigned int my=4;

inline void init(double *f, dimensions dr, bool inplace) 
{
  unsigned int rdist=inplace ? dr.ny+2: dr.ny;
  for(unsigned int i=0; i < dr.x; i++)  {
    for(unsigned int j=0; j < dr.ny; j++) {
      unsigned int ii=dr.x0+i;
      f[i*rdist+j]=ii+10*j;
    }
  }
}

inline void init(Complex *g, dimensions dc) 
{
  for(unsigned int i=0; i < dc.x; i++)  {
    for(unsigned int j=0; j < dc.ny; j++) {
      unsigned int ii=dc.x0+i;
      g[i*dc.ny+j]=Complex(ii,10*j);
    }
  }
}

unsigned int outlimit=100;

int main(int argc, char* argv[])
{
#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif
  int retval=0;
  
#ifdef __GNUC__ 
  optind=0;
#endif  
  for (;;) {
    int c = getopt(argc,argv,"hN:m:x:y:n:T:");
    if (c == -1) break;
                
    switch (c) {
      case 0:
        break;
      case 'N':
        N=atoi(optarg);
        break;
      case 'm':
        mx=my=atoi(optarg);
        break;
      case 'x':
        mx=atoi(optarg);
        break;
      case 'y':
        my=atoi(optarg);
        break;
      case 'n':
        N0=atoi(optarg);
        break;
      case 'T':
        fftw::maxthreads=atoi(optarg);
        break;
      case 'h':
      default:
        usage(2);
    }
  }

  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);


  if(my == 0) my=mx;

  if(N == 0) {
    N=N0/mx/my;
    if(N < 10) N=10;
  }
  
  MPIgroup group(MPI_COMM_WORLD,mx);

  if(group.size > 1 && provided < MPI_THREAD_FUNNELED) {
    fftw::maxthreads=1;
  } else {
    fftw_init_threads();
    fftw_mpi_init();
  }

  if(group.rank == 0) {
    cout << "provided: " << provided << endl;
    cout << "fftw::maxthreads: " << fftw::maxthreads << endl;
  }
  
  if(group.rank == 0) {
    cout << "Configuration: " 
         << group.size << " nodes X " << fftw::maxthreads 
         << " threads/node" << endl;
  }

  if(group.rank < group.size) { // If the process is unused, then do nothing.
    bool main=group.rank == 0;
    if(main) {
      cout << "N=" << N << endl;
      cout << "mx=" << mx << ", my=" << my << endl;
    }

    // Load wisdom
    MPILoadWisdom(group.active);

    // real dimensions
    dimensions dr(mx,my,group.active,group.yblock); 
    // complex dimensions
    unsigned int myp=my/2+1; // y-dimension of complex data
    dimensions dc(mx,myp,group.active,group.yblock);
  
    for(int i=0; i < group.size; ++i) {
      MPI_Barrier(group.active);
      if(i == group.rank) {
	cout << "process " << i << " dimensions:" << endl;
	dr.show();
	cout << "complex case:" << endl;
	dc.show();
	cout << endl;
      }
    }
    
    double *f;
    Complex *g=ComplexAlign(dc.n);
    f=(double *)g;
    //Array::newAlign(f,dr.n,sizeof(double));
    bool inplace=(double*) g == f;
    cout << "inplace: " << inplace  << endl;

    // Create instance of FFT
    rcfft2dMPI fft(dr,dc,f,g);
    
    bool dofinaltranspose=false;
    //dofinaltranspose=true;

    // sample output for small problems
    if(mx*my < outlimit) {
      init(f,dr,inplace);
      //init(g,dc); // FIXME: temp

      if(main) cout << "\ninput:" << endl;
      //show(f,1,dr.n,group.active);
      show(f,dr.x,dr.ny+(inplace?2:0),group.active);
      //show(g,dc.x,dc.ny,group.active);
      //show(g,1,dc.n,group.active);      

      fft.Forwards(f,g,dofinaltranspose);

      if(main) cout << "\noutput:" << endl;
       show(g,dc.nx,dc.y,group.active);
      // show(g,1,dc.n,group.active);

      fft.Backwards(g,f,dofinaltranspose);
      fft.Normalize(f);

      // if(main) cout << "\ntranposed back:" << endl;
      // show(g,dc.x,dc.ny,group.active);

      if(main) cout << "\nback to input:" << endl;
      show(f,dr.x,dr.ny+(inplace?2:0),group.active);
      //show(f,1,dr.n,group.active);
    }


    // Timing

    // double *T=new double[N];
    // for(unsigned int i=0; i < N; ++i) {
    //   init(f,dr,inplace);
    //   seconds();
    //   fft.Forwards(f,dofinaltranspose);
    //   fft.Backwards(f,dofinaltranspose);
    //   fft.Normalize(f);
    //   T[i]=seconds();
    // }
    // if(main) timings("FFT timing:",mx,T,N);
    // delete [] T;


    if((double *) g != f) deleteAlign(f);
    deleteAlign(g);
    
    // Save wisdom

    // FIXME: wisdom destroyed if load not present when using FFTW
    // transpose (or with non-active processes?) unless load present
    // here.
    MPILoadWisdom(group.active); 
    MPISaveWisdom(group.active);
  }

  MPI_Finalize();
  
  return retval;
}
