#ifndef __localtranspose_h__
#define __localtranspose_h__ 1


#ifndef FFTWPP_SINGLE_THREAD
#define PARALLEL(code)                                  \
  if(threads > 1) {                                     \
    _Pragma("omp parallel for num_threads(threads)")    \
      code                                              \
      } else {                                          \
    code                                                \
      }
#else
#define PARALLEL(code)                          \
  {                                             \
    code                                        \
  }
#endif

template<class T>
inline void copy(const T *from, T *to, unsigned int length,
                 unsigned int threads=1)
{
  PARALLEL(
    for(unsigned int i=0; i < length; ++i)
      to[i]=from[i];
    );
}

template<class T>
inline void localtranspose(const T *src, T *dest,
                           unsigned int n, unsigned int m, unsigned int length,
                           unsigned int threads)
{
  if(n > 1 && m > 1) {
    unsigned int nlength=n*length;
    unsigned int mlength=m*length;
    PARALLEL(
      for(unsigned int i=0; i < nlength; i += length) {
        const T *Src=src+i*m;
        T *Dest=dest+i;
        for(unsigned int j=0; j < mlength; j += length)
          copy(Src+j,Dest+j*n,length);
      });
  } else
    copy(src,dest,n*m*length,threads);
}

#endif
