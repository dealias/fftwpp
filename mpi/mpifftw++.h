#ifndef __mpifftwpp_h__
#define __mpifftwpp_h__ 1

#include "mpigroup.h"
#include "fftw++.h"
#include "mpiutils.h"

namespace fftwpp {

// In-place and out-of-place distributed FFTs. Upper case letters denote
// global dimensions; lower case letters denote distributed dimensions: 

// 2D OpenMP/MPI complex in-place and out-of-place 
// xY -> Xy
// Fourier transform an nx*ny array, distributed first over x.
// The array must be allocated as split::n Complex words.
// The sign argument (default -1) of the constructor specfies the sign
// of the forward transform.
//
// Example:
//
// MPIgroup group(MPI_COMM_WORLD,nx);
// split d(nx,ny,group.active);
// Complex *f=ComplexAlign(d.n);
// fft2dMPI fft(d,f);
// fft.Forward(f);
// fft.Backward(f);
// fft.Normalize(f);
// deleteAlign(f);
//
// Non-blocking interface:
//    
// fft.iForward(f);
// User computation
// fft.ForwardWait(f);

class fft2dMPI : public fftw {
protected:
  utils::split d;
  mfft1d *xForward,*xBackward;
  mfft1d *yForward,*yBackward;
  Transpose *TXy,*TyX;
  bool strided;
public:
  utils::mpitranspose<Complex> *T;
  
  void init(Complex *in, Complex *out, const utils::mpiOptions& options) {
    d.Activate();
    out=CheckAlign(in,out);
    inplace=(in == out);

    yForward=new mfft1d(d.Y,sign,d.x,1,d.Y,in,out,threads);
    yBackward=new mfft1d(d.Y,-sign,d.x,1,d.Y,out,out,threads);

    T=new utils::mpitranspose<Complex>(d.X,d.Y,d.x,d.y,1,out,d.communicator,
                                       options);
    strided=d.y & (d.y-1); // Always use strides except for powers of 2
    if(strided) {
      xForward=new mfft1d(d.X,sign,d.y,d.y,1,out,out,threads);
      xBackward=new mfft1d(d.X,-sign,d.y,d.y,1,in,out,threads);
    } else {
      TXy=new Transpose(d.X,d.y,1,out,out,threads);
      TyX=new Transpose(d.y,d.X,1,out,out,threads);
      xForward=new mfft1d(d.X,sign,d.y,1,d.X,out,out,threads);
      xBackward=new mfft1d(d.X,-sign,d.y,1,d.X,in,out,threads);
    }
    
    d.Deactivate();
  }
  
  fft2dMPI(const utils::split& d, Complex *in,
           const utils::mpiOptions& options=utils::defaultmpiOptions,
           int sign=-1) : 
    fftw(2*d.x*d.Y,sign,options.threads,d.X*d.Y), d(d) {
    init(in,in,options);
  }
    
  fft2dMPI(const utils::split& d, Complex *in, Complex *out,
           const utils::mpiOptions& options=utils::defaultmpiOptions,
           int sign=-1) : 
    fftw(2*d.x*d.Y,sign,options.threads,d.X*d.Y), d(d) {
    init(in,out,options);
  }
  
  virtual ~fft2dMPI() {
    delete xBackward;
    delete xForward;
    if(!strided) {
      delete TXy;
      delete TyX;
    }
    delete T;
    delete yBackward;
    delete yForward;
  }

  virtual void iForward(Complex *in, Complex *out=NULL);
  virtual void ForwardWait(Complex *out)
  {
    T->wait();
    if(!strided) TXy->transpose(out);
    xForward->fft(out);
    if(!strided) TyX->transpose(out);
  }
  void Forward(Complex *in, Complex *out=NULL) {
    iForward(in,out);
    ForwardWait(out);
  }
  virtual void iBackward(Complex *in, Complex *out=NULL);
  virtual void BackwardWait(Complex *out) {
    T->wait();
    yBackward->fft(out);
  }
  void Backward(Complex *in, Complex *out=NULL) {
    iBackward(in,out);
    BackwardWait(out);
  }
  
};

// 3D OpenMP/MPI complex in-place and out-of-place 
// xyZ -> Xyz
// Fourier transform an nx*ny*nz array, distributed first over x and
// then over y. The array must be allocated as split3::n Complex words.
// The sign argument (default -1) of the constructor specfies the sign
// of the forward transform.
//
// Example:
//
// MPIgroup group(MPI_COMM_WORLD,nx,ny);
// split3 d(nx,ny,nz,group);
// Complex *f=ComplexAlign(d.n);
// fft3dMPI fft(d,f);
// fft.Forward(f);
// fft.Backward(f);
// fft.Normalize(f);
// deleteAlign(f);
//
// Non-blocking interface:
//    
// fft.iForward(f);
// User computation
// fft.ForwardWait(f);

// Double non-blocking interface (for pencil decomposition):
// fft.iForward(f);
// User computation 0
// fft.ForwardWait0(f);
// User computation 1
// fft.ForwardWait1(f);

class fft3dMPI : public fftw {
protected:
  utils::split3 d;
  mfft1d *xForward,*xBackward;
  mfft1d *yForward,*yBackward;
  mfft1d *zForward,*zBackward;
  fft2d *yzForward,*yzBackward;
public:
  utils::mpitranspose<Complex> *Txy,*Tyz;
  
  void init(Complex *in, Complex *out, const utils::mpiOptions& xy,
            const utils::mpiOptions &yz) {
    d.Activate();
    multithread(d.x);
    out=CheckAlign(in,out);
    inplace=(in == out);
    
    if(d.yz.x < d.Y) {
      unsigned int M=d.x*d.yz.x;
      zForward=new mfft1d(d.Z,sign,M,1,d.Z,in,out,threads);
      zBackward=new mfft1d(d.Z,-sign,M,1,d.Z,out,out,threads);
      Tyz=new utils::mpitranspose<Complex>(d.Y,d.Z,d.yz.x,d.z,1,out,
                                           d.yz.communicator,yz,
                                           d.communicator);
      yForward=new mfft1d(d.Y,sign,d.z,d.z,1,out,out,innerthreads);
      yBackward=new mfft1d(d.Y,-sign,d.z,d.z,1,out,out,innerthreads);
    } else {
      yzForward=new fft2d(d.Y,d.Z,sign,in,out,innerthreads);
      yzBackward=new fft2d(d.Y,d.Z,-sign,out,out,innerthreads);
      Tyz=NULL;
    }
    
    Txy=new utils::mpitranspose<Complex>(d.X,d.Y,d.x,d.xy.y,d.z,out,
                                         d.xy.communicator,xy,d.communicator);
    unsigned int M=d.xy.y*d.z;
    xForward=new mfft1d(d.X,sign,M,M,1,out,out,threads);
    xBackward=new mfft1d(d.X,-sign,M,M,1,in,out,threads);
    
    d.Deactivate();
  }
  
  fft3dMPI(const utils::split3& d, Complex *in, Complex *out,
           const utils::mpiOptions& xy, const utils::mpiOptions& yz,
           int sign=-1) :
    fftw(2*d.x*d.y*d.Z,sign,xy.threads,d.X*d.Y*d.Z), d(d) {init(in,out,xy,yz);}
  
  fft3dMPI(const utils::split3& d, Complex *in, Complex *out,
           const utils::mpiOptions& xy=utils::defaultmpiOptions,
           int sign=-1) :
    fftw(2*d.x*d.y*d.Z,sign,xy.threads,d.X*d.Y*d.Z), d(d) {init(in,out,xy,xy);}
    
  fft3dMPI(const utils::split3& d, Complex *in, const utils::mpiOptions& xy,
           const utils::mpiOptions& yz, int sign=-1) :
    fftw(2*d.x*d.y*d.Z,sign,xy.threads,d.X*d.Y*d.Z), d(d) {init(in,in,xy,yz);}
  
  fft3dMPI(const utils::split3& d, Complex *in,
           const utils::mpiOptions& xy=utils::defaultmpiOptions, int sign=-1) :
    fftw(2*d.x*d.y*d.Z,sign,xy.threads,d.X*d.Y*d.Z), d(d) {init(in,in,xy,xy);}
    
  virtual ~fft3dMPI() {
    delete xBackward;
    delete xForward;
    delete Txy;
    
    if(Tyz) {
      delete yBackward;
      delete yForward;
      delete Tyz;
      delete zBackward;
      delete zForward;
    } else {
      delete yzBackward;
      delete yzForward;
    }
  }

  virtual void iForward(Complex *in, Complex *out=NULL);
  virtual void ForwardWait0(Complex *out);
  virtual void ForwardWait1(Complex *out) {
    Txy->wait();
    xForward->fft(out);
  }
  void ForwardWait(Complex *out) {
    ForwardWait0(out);
    ForwardWait1(out);
  }
  void Forward(Complex *in, Complex *out=NULL) {
    iForward(in,out);
    ForwardWait(out);
  }
  
  virtual void iBackward(Complex *in, Complex *out=NULL);
  virtual void BackwardWait0(Complex *out);
  virtual void BackwardWait1(Complex *out) {
    if(Tyz) {
      Tyz->wait();
      zBackward->fft(out);
    }
  }
  void BackwardWait(Complex *out) {
    BackwardWait0(out);
    BackwardWait1(out);
  }
  void Backward(Complex *in, Complex *out=NULL) {
    iBackward(in,out);
    BackwardWait(out);
  }
  
};

// 2D OpenMP/MPI real-to-complex and complex-to-real in-place and out-of-place
// xY->Xy distributed FFT.
//
// The array in has size nx*ny, distributed in the x direction.
// The array out has size nx*(ny/2+1), distributed in the y direction. 
// The arrays must be allocated as split3::n words.
// The arrays in and out may coincide, dimensioned according to out.
//
// Basic interface:
// Forward(double *in, Complex *out=NULL);   // Fourier origin at (0,0)
// Forward0(double *in, Complex *out=NULL);  // Fourier origin at (nx/2,0);
//                                              input destroyed.
//
// Backward(Complex *in, double *out=NULL);  // Fourier origin at (0,0);
//                                              input destroyed.
// Backward0(Complex *in, double *out=NULL); // Fourier origin at (nx/2,0);
//                                              input destroyed.
// Normalize(Complex *out);
//
// Example:
//
// MPIgroup group(MPI_COMM_WORLD,ny);
// split df(nx,ny,nz,group.active);
// split dg(nx,ny,nz/2+1,group.active);
// double *f=doubleAlign(df.n);
// Complex *g=ComplexAlign(dg.n);
// rcfft2dMPI fft(df,dg,f,g);
// fft.Forward(f,g);
// fft.Backward(g,f);
// fft.Normalize(f);
// deleteAlign(g);
// deleteAlign(f);
//
// Non-blocking interface:
//
// fft.iForward(f,g);
// User computation
// fft.ForwardWait(f);

class rcfft2dMPI : public fftw {
protected:
  utils::split dr,dc; // real and complex MPI dimensions.
  mfft1d *xForward,*xBackward;
  mrcfft1d *yForward;
  mcrfft1d *yBackward;
  unsigned int rdist;
public:
  utils::mpitranspose<Complex> *T;
    
  void init(double *in, Complex *out, const utils::mpiOptions& options) {
    dc.Activate();
    out=CheckAlign((Complex *) in,out);
    inplace=((Complex *) in == out);
    
    T=new utils::mpitranspose<Complex>(dc.X,dc.Y,dc.x,dc.y,1,out,
                                       dc.communicator,options);
    size_t cdist=dr.Y/2+1;
    rdist=inplace ? 2*cdist : realsize(dr.Y,in,out);
    {
      int n=dr.Y;
      int M=dr.x;
      size_t istride=1;
      size_t ostride=1;
      yForward=new mrcfft1d(n,M,istride,ostride,rdist,cdist,in,out,threads);
      yBackward=new mcrfft1d(n,M,ostride,istride,cdist,rdist,out,in,threads);
    }

    xForward=new mfft1d(dc.X,-1,dc.y,dc.y,1,out,out,threads);
    xBackward=new mfft1d(dc.X,1,dc.y,dc.y,1,out,out,threads);
    dc.Deactivate();
  }
   
  rcfft2dMPI(const utils::split& dr, const utils::split& dc, double *in,
             Complex *out,
             const utils::mpiOptions& options=utils::defaultmpiOptions) :
    fftw(dr.x*realsize(dr.Y,in,out),-1,options.threads,dr.X*dr.Y), dr(dr),
    dc(dc) {init(in,out,options);}
    
  rcfft2dMPI(const utils::split& dr, const utils::split& dc, Complex *out,
             const utils::mpiOptions& options=utils::defaultmpiOptions) :
    fftw(dr.x*2*(dr.Y/2+1),-1,options.threads,dr.X*dr.Y), dr(dr), dc(dc)
  {init((double *) out,out,options);}
    
  virtual ~rcfft2dMPI() {
    delete xBackward;
    delete xForward;
    delete yBackward;
    delete yForward;
    delete T;
  }

  // Set Nyquist modes of even shifted transforms to zero.
  void deNyquist(Complex *f) {
    if(dr.X % 2 == 0)
      for(unsigned int j=0; j < dc.y; ++j)
        f[j]=0.0;
    
    if(dr.Y % 2 == 0 && dc.y0+dc.y == dc.Y) // Last process
      for(unsigned int i=0; i < dc.X; ++i)
        f[(i+1)*dc.y-1]=0.0;
  }
  
  void Shift(double *out);
  
  virtual void iForward(double *in, Complex *out=NULL);
  virtual void ForwardWait(Complex *out) {
    T->wait();
    xForward->fft(out);
  };
  void Forward(double *in, Complex *out=NULL) {
    iForward(in,out);
    ForwardWait(out);
  }
  void Forward0(double *in, Complex *out=NULL) {
    Shift(in);
    Forward(in,out);
  }
  
  virtual void iBackward(Complex *in, double *out=NULL);
  virtual void BackwardWait(Complex *in, double *out=NULL) {
    out=(double *) Setout(in,(Complex *) out);
    T->wait();
    yBackward->fft(in,out);
  };
  void Backward(Complex *in, double *out=NULL) {
    iBackward(in,out);
    BackwardWait(in,out);
  }
  void Backward0(Complex *in, double *out=NULL) {
    Backward(in,out);
    Shift(out);
  }
  
  void iForward(Complex *out) {iForward((double *) out,out);}
  void Forward(Complex *out) {Forward((double *) out,out);}
  void Forward0(Complex *out) {Forward0((double *) out,out);}
};
  
// 3D real-to-complex and complex-to-real in-place and out-of-place
// xyZ -> Xyz distributed FFT.
//
// The input has size nx*ny*nz, distributed in the x and y directions.
// The output has size nx*ny*(nz/2+1), distributed in the y and z directions.
// The arrays must be allocated as split3::n Complex words.
//
// Basic interface:
// Forward(double *in, Complex *out=NULL);   // Fourier origin at (0,0)
// Forward0(double *in, Complex *out=NULL);  // Fourier origin at (nx/2,ny/2,0)
//                                              input destroyed.
//
// Backward(Complex *in, double *out=NULL);  // Fourier origin at (0,0)
//                                              input destroyed.
// Backward0(Complex *in, double *out=NULL); // Fourier origin at (nx/2,ny/2,0)
//                                              input destroyed.
// Normalize(Complex *out);
//
// Example:
//
// MPIgroup group(MPI_COMM_WORLD,nx,ny);
// split3 df(nx,ny,nz,group);
// split3 dg(nx,ny,nz/2+1,group);
// double *f=doubleAlign(df.n);
// Complex *g=ComplexAlign(dg.n);
// rcfft3dMPI fft(df,dg,f,g);
// fft.Forward(f,g);
// fft.Backward(g,f);
// fft.Normalize(f);
// deleteAlign(g);
// deleteAlign(f);
//
// Non-blocking interface:
//    
// fft.iForward(f,g);
// User computation
// fft.ForwardWait(g);

// Double non-blocking interface (for pencil decomposition):
// fft.iForward(f,g);
// User computation 0
// fft.ForwardWait0(g);
// User computation 1
// fft.ForwardWait1(g);
//
class rcfft3dMPI : public fftw {
protected:
  utils::split3 dr,dc; // real and complex MPI dimensions
  mfft1d *xForward,*xBackward;
  mfft1d *yForward,*yBackward;
  mrcfft1d *zForward;
  mcrfft1d *zBackward;
  unsigned int rdist;
public:
  utils::mpitranspose<Complex> *Txy,*Tyz;
  
  void init(double *in, Complex *out, const utils::mpiOptions& xy,
            const utils::mpiOptions &yz) {
    dc.Activate();
    multithread(dc.x);
    out=CheckAlign((Complex *) in,out);
    inplace=((Complex *) in == out);

    Txy=new utils::mpitranspose<Complex>(dc.X,dc.Y,dc.x,dc.xy.y,dc.z,
                                         out,dc.xy.communicator,xy,
                                         dc.communicator);
    Tyz=dc.yz.x < dc.Y ? 
                  new utils::mpitranspose<Complex>(dc.Y,dc.Z,dc.yz.x,dc.z,1,
                                                   out,dc.yz.communicator,yz,
                                                   dc.communicator) : NULL;
    unsigned int M=dr.x*dr.yz.x;
    size_t cdist=dr.Z/2+1;
    zForward=new mrcfft1d(dr.Z,M,1,1,rdist,cdist,in,out,threads);
    zBackward=new mcrfft1d(dr.Z,M,1,1,cdist,rdist,out,in,threads);

    yForward=new mfft1d(dc.Y,-1,dc.z,dc.z,1,out,out,innerthreads);
    yBackward=new mfft1d(dc.Y,1,dc.z,dc.z,1,out,out,innerthreads);

    M=dc.xy.y*dc.z;
    xForward=new mfft1d(dc.X,-1,M,M,1,out,out,threads);
    xBackward=new mfft1d(dc.X,1,M,M,1,out,out,threads);
        
    dc.Deactivate();
  }
  
  rcfft3dMPI(const utils::split3& dr, const utils::split3& dc, double *in,
             Complex *out, const utils::mpiOptions& xy,
             const utils::mpiOptions& yz) : 
    fftw(dr.x*dr.yz.x*realsize(dr.Z,in,out),-1,xy.threads,dr.X*dr.Y*dr.Z),
    dr(dr), dc(dc), rdist(realsize(dr.Z,in,out)) {
    init(in,out,xy,yz);
  }
  
  rcfft3dMPI(const utils::split3& dr, const utils::split3& dc, double *in,
             Complex *out, 
             const utils::mpiOptions& xy=utils::defaultmpiOptions) : 
    fftw(dr.x*dr.yz.x*realsize(dr.Z,in,out),-1,xy.threads,dr.X*dr.Y*dr.Z),
    dr(dr), dc(dc), rdist(realsize(dr.Z,in,out)) {
    init(in,out,xy,xy);
  }
  
  rcfft3dMPI(const utils::split3& dr, const utils::split3& dc, Complex *out,
             const utils::mpiOptions& xy, const utils::mpiOptions& yz) : 
    fftw(dr.x*dr.yz.x*2*(dr.Z/2+1),-1,xy.threads,dr.X*dr.Y*dr.Z),
    dr(dr), dc(dc), rdist(2*(dr.Z/2+1)) {
    init((double *) out,out,xy,yz);
  }
  
  rcfft3dMPI(const utils::split3& dr, const utils::split3& dc, Complex *out,
             const utils::mpiOptions& xy=utils::defaultmpiOptions) : 
    fftw(dr.x*dr.yz.x*2*(dr.Z/2+1),-1,xy.threads,dr.X*dr.Y*dr.Z),
    dr(dr), dc(dc), rdist(2*(dr.Z/2+1)) {
    init((double *) out,out,xy,xy);
  }
  
  virtual ~rcfft3dMPI() {
    delete xBackward;
    delete xForward;
    delete yBackward;
    delete yForward;
    delete zBackward;
    delete zForward;
    
    if(Tyz) delete Tyz;
    delete Txy;
  }

  // Set Nyquist modes of even shifted transforms to zero.
  void deNyquist(Complex *f) {
    unsigned int yz=dc.xy.y*dc.z;
    if(dr.X % 2 == 0) {
      for(unsigned int k=0; k < yz; ++k)
        f[k]=0.0;
    }
    
    if(dr.Y % 2 == 0 && dc.xy.y0 == 0) {
      for(unsigned int i=0; i < dc.X; ++i) {
        unsigned int iyz=i*yz;
        for(unsigned int k=0; k < dc.z; ++k)
          f[iyz+k]=0.0;
      }
    }
        
    if(dr.Z % 2 == 0 && dc.z0+dc.z == dc.Z) // Last process
      for(unsigned int i=0; i < dc.X; ++i)
        for(unsigned int j=0; j < dc.xy.y; ++j)
          f[i*yz+(j+1)*dc.z-1]=0.0;
  }
  
  void Shift(double *out);
  virtual void iForward(double *in, Complex *out=NULL);
  virtual void ForwardWait0(Complex *out);
  virtual void ForwardWait1(Complex *out) {
    Txy->wait();
    xForward->fft(out);
  }
  void ForwardWait(Complex *out) {
    ForwardWait0(out);
    ForwardWait1(out);
  }
  void Forward(double *in, Complex *out=NULL) {
    iForward(in,out);
    ForwardWait(out);
  }
  void Forward0(double *in, Complex *out=NULL) {
    Shift(in);
    Forward(in,out);
  }
  
  virtual void iBackward(Complex *in, double *out=NULL);
  virtual void BackwardWait0(Complex *in, double *out=NULL);
  virtual void BackwardWait1(Complex *in, double *out=NULL) {
    if(Tyz) Tyz->wait();
    zBackward->fft(in,out);
  }
  void BackwardWait(Complex *in, double *out=NULL) {
    BackwardWait0(in,out);
    BackwardWait1(in,out);
  }
  void Backward(Complex *in, double *out=NULL) {
    iBackward(in,out);
    BackwardWait(in,out);
  }
  void Backward0(Complex *in, double *out=NULL) {
    Backward(in,out);
    Shift(out);
  }
  
  void Forward(Complex *out) {Forward((double *) out,out);}
  void Forward0(Complex *out) {Forward0((double *) out,out);}
};

} // end namespace fftwpp

#endif
