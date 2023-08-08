namespace fftwpp {

#ifndef __direct_h__
#define __direct_h__ 1

// Out-of-place direct 1D complex convolution.
class directconv {
protected:
  size_t m;
public:
  directconv(size_t m) : m(m) {}

  void convolve(Complex *h, Complex *f, Complex *g);
  void Cconvolve(Complex *h, Complex *f, Complex *g);
  void autoconvolve(Complex *h, Complex *f);
};

// Out-of-place direct 1D Hermitian convolution.
class directconvh {
protected:
  size_t m;
public:
  directconvh(size_t m) : m(m) {}

// Compute h= f (*) g via direct convolution, where f and g contain the m
// non-negative Fourier components of real functions (contents
// preserved). The output of m complex values is returned in the array h,
// which must be distinct from f and g.
  void convolve(Complex *h, Complex *f, Complex *g);
};

// Out-of-place direct 1D real convolution.
class directconvr {
protected:
  size_t m;
public:
  directconvr(size_t m) : m(m) {}
  void convolve(double *h, double *f, double *g);
};

// Out-of-place direct 2D complex convolution.
template<class T>
class directconv2 {
protected:
  size_t mx,my; // x and y data lengths
  size_t Sx; // x stride
public:
  directconv2(size_t mx, size_t my,
                     size_t Sx=0) : mx(mx), my(my) {
    this->Sx=Sx ? Sx : my;
  }

  void convolve(T *h, T *f, T *g)
  {
#if (!defined FFTWPP_SINGLE_THREAD) && defined _OPENMP
#pragma omp parallel for
#endif
    for(size_t i=0; i < mx; ++i) {
      for(size_t j=0; j < my; ++j) {
        T sum=0.0;
        for(size_t k=0; k <= i; ++k)
          for(size_t p=0; p <= j; ++p)
            sum += f[Sx*k+p]*g[Sx*(i-k)+j-p];
        h[i*my+j]=sum;
      }
    }
  }
};

// Out-of-place direct 2D Hermitian convolution.
class directconvh2 {
protected:
  size_t mx,my;
  bool xcompact;
  size_t Sx; // x stride
public:
  directconvh2(size_t mx, size_t my, bool xcompact=true,
                      size_t Sx=0) : mx(mx), my(my), xcompact(xcompact) {
    this->Sx=Sx ? Sx : my;
  }

  void convolve(Complex *h, Complex *f, Complex *g, bool symmetrize=true);
};

// Out-of-place direct 3D complex convolution.
class directconv3 {
protected:
  size_t mx,my,mz;
  size_t Sx; // x stride
  size_t Sy; // y stride
  size_t myz;
public:
  directconv3(size_t mx, size_t my, size_t mz,
                     size_t Sx=0, size_t Sy=0) :
    mx(mx), my(my), mz(mz), myz(my*mz) {
    this->Sy=Sy ? Sy : mz;
    this->Sx=Sx ? Sx : my*Sy;
  }

  void convolve(Complex *h, Complex *f, Complex *g);
};

// Out-of-place direct 3D Hermitian convolution.
class directconvh3 {
protected:
  size_t mx,my,mz;
  bool xcompact, ycompact;
  size_t Sx; // x stride
  size_t Sy; // y stride
public:
  directconvh3(size_t mx, size_t my, size_t mz,
                      bool xcompact=true, bool ycompact=true,
                      size_t Sx=0, size_t Sy=0) :
    mx(mx), my(my), mz(mz), xcompact(xcompact), ycompact(ycompact) {
    this->Sy=Sy ? Sy : mz;
    this->Sx=Sx ? Sx : (2*my-ycompact)*Sy;
  }

  void convolve(Complex *h, Complex *f, Complex *g, bool symmetrize=true);
};


// Out-of-place direct 1D Hermitian ternary convolution.
class directconvhT {
protected:
  size_t m;
public:
  directconvhT(size_t m) : m(m) {}

  void convolve(Complex *h, Complex *e, Complex *f, Complex *g);
};

// Out-of-place direct 2D Hermitian ternary convolution.
class directconvhT2 {
protected:
  size_t mx,my;
public:
  directconvhT2(size_t mx, size_t my) : mx(mx), my(my)
  {}

  void convolve(Complex *h, Complex *e, Complex *f, Complex *g,
                bool symmetrize=true);
};



#endif

}
