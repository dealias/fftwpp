namespace fftwpp {

#ifndef __direct_h__
#define __direct_h__ 1

// Out-of-place direct 1D complex convolution.
class DirectConvolution {
protected:
  unsigned int m;
public:  
  DirectConvolution(unsigned int m) : m(m) {}
  
  void convolve(Complex *h, Complex *f, Complex *g);
  void autoconvolve(Complex *h, Complex *f);
};

// Out-of-place direct 1D Hermitian convolution.
class DirectHConvolution {
protected:
  unsigned int m;
public:  
  DirectHConvolution(unsigned int m) : m(m) {}
  
// Compute h= f (*) g via direct convolution, where f and g contain the m
// non-negative Fourier components of real functions (contents
// preserved). The output of m complex values is returned in the array h,
// which must be distinct from f and g.
  void convolve(Complex *h, Complex *f, Complex *g);
};

// Out-of-place direct 2D complex convolution.
class DirectConvolution2 {
protected:  
  unsigned int mx,my;
public:
  DirectConvolution2(unsigned int mx, unsigned int my) : mx(mx), my(my) {}
  
  void convolve(Complex *h, Complex *f, Complex *g);
};

// Out-of-place direct 2D Hermitian convolution.
class DirectHConvolution2 {
protected:  
  unsigned int mx,my;
public:
  DirectHConvolution2(unsigned int mx, unsigned int my) : mx(mx), my(my) {}
  
  void convolve(Complex *h, Complex *f, Complex *g, bool symmetrize=true);
};

// Out-of-place direct 3D complex convolution.
class DirectConvolution3 {
protected:  
  unsigned int mx,my,mz;
  unsigned int myz;
public:
  DirectConvolution3(unsigned int mx, unsigned int my, unsigned int mz) : 
    mx(mx), my(my), mz(mz), myz(my*mz) {}
  
  void convolve(Complex *h, Complex *f, Complex *g);
};

// Out-of-place direct 3D Hermitian convolution.
class DirectHConvolution3 {
protected:  
  unsigned int mx,my,mz;
public:
  DirectHConvolution3(unsigned int mx, unsigned int my, unsigned int mz) : 
    mx(mx), my(my), mz(mz) {}
  
  void convolve(Complex *h, Complex *f, Complex *g, bool symmetrize=true);
};

// Out-of-place direct 1D Hermitian ternary convolution.
class DirectHTConvolution {
protected:  
  unsigned int m;
public:
  DirectHTConvolution(unsigned int m) : m(m) {}
  
  void convolve(Complex *h, Complex *e, Complex *f, Complex *g);
};

// Out-of-place direct 2D Hermitian ternary convolution.
class DirectHTConvolution2 {
protected:  
  unsigned int mx,my;
public:
  DirectHTConvolution2(unsigned int mx, unsigned int my) : mx(mx), my(my)
  {}
  
  void convolve(Complex *h, Complex *e, Complex *f, Complex *g,
                bool symmetrize=true);
};



#endif

}
