#!/usr/bin/python
# fftwpp.py - Python wrapper of the C callable FFTW+ wrapper.
#
# Authors: Matthew Emmett <memmett@gmail.com>
#          Malcolm Roberts <malcolm.i.w.roberts@gmail.com>
#          Robert Joseph <rjoseph1@ualberta.ca>
#          John Bowman <bowman@ualberta.ca>

import numpy as np
import os

from numpy.ctypeslib import ndpointer
from ctypes import *

def complex_align(shape):
    dtype=np.dtype(np.complex128)
    nbytes=np.prod(shape)*dtype.itemsize
    buf=np.empty(nbytes+16,dtype=np.uint8)
    start_index=-buf.ctypes.data % 16
    return buf[start_index:start_index+nbytes].view(dtype).reshape(shape)

__all__=['Convolution' ,'HConvolution']

# Load fftwpp shared library
base=os.path.dirname(os.path.abspath(__file__))
clib=CDLL(os.path.join(base,'lib_fftwpp.so'))

def fftwpp_set_maxthreads(nthreads):
    clib.set_fftwpp_maxthreads(nthreads)

def fftwpp_get_maxthreads():
    return clib.get_fftwpp_maxthreads()

# Prototypes
clib.fftwpp_create_conv1d.restype=c_void_p
clib.fftwpp_create_conv1d.argtypes=[c_size_t]

clib.fftwpp_conv1d_delete.argtypes=[c_void_p]
clib.fftwpp_conv1d_convolve.argtypes=[c_void_p,
                                         ndpointer(dtype=np.complex128),
                                         ndpointer(dtype=np.complex128)]

clib.fftwpp_create_hconv1d.restype=c_void_p
clib.fftwpp_create_hconv1d.argtypes=[c_size_t]
clib.fftwpp_hconv1d_delete.argtypes=[c_void_p]
clib.fftwpp_hconv1d_convolve.argtypes=[c_void_p,
                                          ndpointer(dtype=np.complex128),
                                          ndpointer(dtype=np.complex128)]

clib.fftwpp_create_conv2d.restype=c_void_p
clib.fftwpp_create_conv2d.argtypes=[c_size_t,c_size_t]
clib.fftwpp_conv2d_delete.argtypes=[c_void_p]
clib.fftwpp_conv2d_convolve.argtypes=[c_void_p,
                                         ndpointer(dtype=np.complex128),
                                         ndpointer(dtype=np.complex128)]
clib.fftwpp_HermitianSymmetrize.argtypes=[ndpointer(dtype=np.complex128)]

clib.fftwpp_create_hconv2d.restype=c_void_p
clib.fftwpp_create_hconv2d.argtypes=[c_size_t,c_size_t]
clib.fftwpp_hconv2d_delete.argtypes=[c_void_p]
clib.fftwpp_hconv2d_convolve.argtypes=[c_void_p,
                                          ndpointer(dtype=np.complex128),
                                          ndpointer(dtype=np.complex128)]
clib.fftwpp_HermitianSymmetrizeX.argtypes=[c_size_t,c_size_t,c_size_t,
                                          ndpointer(dtype=np.complex128)]
clib.fftwpp_create_conv3d.restype=c_void_p
clib.fftwpp_create_conv3d.argtypes=[c_size_t,c_size_t,c_size_t]
clib.fftwpp_conv3d_delete.argtypes=[c_void_p]
clib.fftwpp_conv3d_convolve.argtypes=[c_void_p,
                                          ndpointer(dtype=np.complex128),
                                          ndpointer(dtype=np.complex128)]

clib.fftwpp_create_hconv3d.restype=c_void_p
clib.fftwpp_create_hconv3d.argtypes=[c_size_t,c_size_t,c_size_t]
clib.fftwpp_hconv3d_delete.argtypes=[c_void_p]
clib.fftwpp_hconv3d_convolve.argtypes=[c_void_p,
                                          ndpointer(dtype=np.complex128),
                                          ndpointer(dtype=np.complex128)]
clib.fftwpp_HermitianSymmetrizeXY.argtypes=[c_size_t,c_size_t,c_size_t,c_size_t,
                                             c_size_t,
                                             ndpointer(dtype=np.complex128)
                                           ]

class Convolution(object):
    """Implicitly zero-padded complex convolution class.

    :param shape: shape/number of elements in the input arrays (int,
    tuple or list)

    The length of *shape* determines the dimension of the convolution.

    One dimensional convolutions
    ----------------------------

    To perform one dimensional convolutions::

    >>> import numpy as np
    >>> import fftwpp
    >>> N=8
    >>> f=fftwpp.complex_align([N])
    >>> g=fftwpp.complex_align([N])
    >>> for i in range(len(f)):
    ...   f[i]=complex(i+1,i+3)
    ...   g[i]=complex(i+2,2*i+3)

    At this point, both ``f`` and ``g`` have shape ``(N,)``::

    >>> assert f.shape == (N,)

    Now, construct the convolution object and convolve::

    >>> c=fftwpp.Convolution(f.shape)
    >>> c.convolve(f, g)

    The convolution is now in ``f``::

    Two dimensional convolutions
    ----------------------------

    To perform two dimensional convolutions::

    >>> import numpy as np
    >>> import fftwpp
    >>> N=4
    >>> f=fftwpp.complex_align([N,N])
    >>> g=fftwpp.complex_align([N,N])
    >>> for i in range(len(f)):
    ...   for j in range(len(f[i])):
    ...     f[i][j]=complex(i+1,j+3)
    ...     g[i][j]=complex(i+2,2*j+3)

    At this point, both ``f`` and ``g`` have shape ``(N,N)``::

    >>> assert f.shape == (N,N)
    >>> assert g.shape == (N,N)

    Now, construct the convolution object and convolve::

    >>> c=fftwpp.Convolution(f.shape)
    >>> c.convolve(f,g)

    Again, the convolution is now in ``f``::


    Three dimensional convolutions
    ------------------------------

    To perform three dimensional convolutions::

    >>> import numpy as np
    >>> import fftwpp
    >>> N=4
    >>> f=fftwpp.complex_align([N,N,N])
    >>> g=fftwpp.complex_align([N,N,N])
    >>> for i in range(len(f)):
    ...   for j in range(len(f[i])):
    ...     for k in range(len(f[i][j])):
    ...       f[i][j][k]=complex(i+1,j+3+k)
    ...       g[i][j][k]=complex(i+k+1,2*j+3+k)
    ...

    At this point, both ``f`` and ``g`` have shape ``(N,N,N)``::

    >>> assert f.shape == (N,N,N)
    >>> assert g.shape == (N,N,N)

    Now, construct the convolution object and convolve::

    >>> c=fftwpp.Convolution(f.shape)
    >>> c.convolve(f,g)

    Again, the convolution is now in ``f``::

    """

    def __init__(self, shape):

        if isinstance(shape, int):
            shape=(shape,)

        self.dim=len(shape)
        self.shape=tuple(shape)

        if self.dim == 1:
            self.cptr=clib.fftwpp_create_conv1d(*shape)
            self._convolve=clib.fftwpp_conv1d_convolve
            self._delete=clib.fftwpp_conv1d_delete
        elif self.dim == 2:
            self.cptr=clib.fftwpp_create_conv2d(*shape)
            self._convolve=clib.fftwpp_conv2d_convolve
            self._delete=clib.fftwpp_conv2d_delete
        elif self.dim == 3:
            self.cptr=clib.fftwpp_create_conv3d(*shape)
            self._convolve=clib.fftwpp_conv3d_convolve
            self._delete=clib.fftwpp_conv3d_delete
        else:
            raise ValueError("invalid shape (length/dimension should be 1, 2, or 3)")

    def __del__(self):
        self._delete(self.cptr)

    def convolve(self,f,g):
        """Compute the convolution of *f* and *g*.

        The convolution is performed in-place (*f* is over-written).
        """

        assert f.shape == self.shape
        assert g.shape == self.shape

        self._convolve(self.cptr,f,g)

class HConvolution(object):
    """Implicitly zero-padded complex Hermitian-symmetric convolution class.

    :param shape: shape/number of elements in the input arrays (int,
    tuple or list)

    The length of *shape* determines the dimension of the convolution.

    One dimensional convolutions
    ----------------------------

    To perform one dimensional convolutions::

    >>> import numpy as np
    >>> import fftwpp
    >>> N=4
    >>> f=fftwpp.complex_align([N])
    >>> g=fftwpp.complex_align([N])
    >>> for i in range(len(f)):
    ...   f[i]=complex(i+1,i+3)
    ...   g[i]=complex(i+2,2*i+3)

    At this point, both ``f`` and ``g`` have shape ``(N,)``::

    >>> assert f.shape == (N,)

    Now, construct the convolution object and convolve::
    >>> c=fftwpp.HConvolution(N)
    >>> c.convolve(f,g)

    The convolution is now in ``f``::

    Two dimensional convolutions
    ----------------------------

    To perform two dimensional convolutions::

    >>> import numpy as np
    >>> import fftwpp
    >>> Nx=4
    >>> Ny=4
    >>> f=fftwpp.complex_align([2*Nx-1,Ny])
    >>> g=fftwpp.complex_align([2*Nx-1,Ny])
    >>> for i in range(len(f)):
    ...   for j in range(len(f[i])):
    ...     f[i][j]=complex(i+1,j+3)
    ...     g[i][j]=complex(i+2,2*j+3)

    Now, construct the convolution object and convolve::

    >>> c=fftwpp.HConvolution(f.shape)
    >>> c.convolve(f,g)

    Again, the convolution is now in ``f``::


    Three dimensional convolutions
    ------------------------------

    To perform three dimensional convolutions::

    >>> import numpy as np
    >>> import fftwpp
    >>> Nx=4
    >>> Ny=4
    >>> Nz=4
    >>> f=fftwpp.complex_align([2*Nx-1,2*Ny-1,Nz])
    >>> g=fftwpp.complex_align([2*Nx-1,2*Ny-1,Nz])
    >>> for i in range(len(f)):
    ...   for j in range(len(f[i])):
    ...     for k in range(len(f[i][j])):
    ...       f[i][j][k]=complex(i+1,j+3+k)
    ...       g[i][j][k]=complex(i+k+1,2*j+3+k)

    Now, construct the convolution object and convolve::

    >>> c=fftwpp.HConvolution(f.shape)
    >>> c.convolve(f,g)

    Again, the convolution is now in ``f``::
"""

    def __init__(self,shape):

        if isinstance(shape,int):
            shape=(shape,)

        self.dim=len(shape)
        self.shape=tuple(shape)

        if self.dim == 1:
            self.cptr=clib.fftwpp_create_hconv1d(c_size_t(2*shape[0]-1))
            self._convolve=clib.fftwpp_hconv1d_convolve
            self._delete=clib.fftwpp_hconv1d_delete
        elif self.dim == 2:
            self.cptr=clib.fftwpp_create_hconv2d(c_size_t(shape[0]),
                                                   c_size_t(2*shape[1]-1))
            self._convolve=clib.fftwpp_hconv2d_convolve
            self._delete=clib.fftwpp_hconv2d_delete
        elif self.dim == 3:
            self.cptr=clib.fftwpp_create_hconv3d(c_size_t(shape[0]),
                                                   c_size_t(shape[1]),
                                                   c_size_t(2*shape[2]-1))
            self._convolve=clib.fftwpp_hconv3d_convolve
            self._delete=clib.fftwpp_hconv3d_delete
        else:
            raise ValueError("invalid shape (dimension should be 1, 2, or 3)")

    def __del__(self):
        self._delete(self.cptr)

    def convolve(self,f,g):
        """Compute the convolution of *f* and *g*.

        The convolution is performed in-place (*f* is over-written).
        """
        # Enforce Hermitian symmetry on input
        if self.dim == 1:
            clib.fftwpp_HermitianSymmetrize(f)
            clib.fftwpp_HermitianSymmetrize(g)
        if self.dim == 2:
            Lx=f.shape[0]
            Hx=(Lx+1)//2
            Hy=f.shape[1]
            x0=Lx//2
            clib.fftwpp_HermitianSymmetrizeX(Hx,Hy,x0,f)
            clib.fftwpp_HermitianSymmetrizeX(Hx,Hy,x0,g)
        elif self.dim == 3:
            Lx=f.shape[0]
            Hx=(Lx+1)//2
            Ly=f.shape[1]
            Hy=(Ly+1)//2
            Hz=f.shape[2]
            x0=Lx//2
            y0=Ly//2
            clib.fftwpp_HermitianSymmetrizeXY(Hx,Hy,Hz,x0,y0,f)
            clib.fftwpp_HermitianSymmetrizeXY(Hx,Hy,Hz,x0,y0,g)

        self._convolve(self.cptr,f,g)

if __name__ == "__main__":
    import doctest,sys
    sys.exit(doctest.testmod()[0])
