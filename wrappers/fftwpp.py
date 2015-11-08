#!/usr/bin/python
# fftwpp.py - Python wrapper of the C callable FFTW+ wrapper.
#
# Authors: Matthew Emmett <memmett@gmail.com>
#          Malcolm Roberts <malcolm.i.w.roberts@gmail.com>

import numpy as np
import os

from numpy.ctypeslib import ndpointer
from ctypes import *

def complex_align(shape):
    dtype = np.dtype(np.complex128)
    nbytes = np.prod(shape) * dtype.itemsize
    buf = np.empty(nbytes + 16, dtype = np.uint8)
    start_index = -buf.ctypes.data % 16
    return buf[start_index:start_index + nbytes].view(dtype).reshape(shape)

__all__ = [ 'Convolution' , 'HConvolution' ]

# Load fftwpp shared library
base = os.path.dirname(os.path.abspath(__file__))
clib = CDLL(os.path.join(base, '_fftwpp.so'))

def fftwpp_set_maxthreads(nthreads):
    clib.set_fftwpp_maxthreads(nthreads)

def fftwpp_get_maxthreads():
    return clib.get_fftwpp_maxthreads()

# Prototypes
clib.fftwpp_create_conv1d.restype = c_void_p
clib.fftwpp_create_conv1d.argtypes = [ c_int ]
clib.fftwpp_create_conv1dAB.restype = c_void_p
clib.fftwpp_create_conv1dAB.argtypes = [ c_int , c_int, c_int ]
clib.fftwpp_conv1d_delete.argtypes = [ c_void_p ]
clib.fftwpp_conv1d_convolve.argtypes = [ c_void_p,
                                         ndpointer(dtype = np.complex128),
                                         ndpointer(dtype = np.complex128) ]
clib.fftwpp_conv1d_correlate.argtypes = [ c_void_p,
                                          ndpointer(dtype = np.complex128),
                                          ndpointer(dtype = np.complex128) ]
clib.fftwpp_conv1d_autoconvolve.argtypes = [ c_void_p,
                                             ndpointer(dtype = np.complex128)]
clib.fftwpp_conv1d_autocorrelate.argtypes = [ c_void_p,
                                              ndpointer(dtype = np.complex128)]

clib.fftwpp_create_hconv1d.restype = c_void_p
clib.fftwpp_create_hconv1d.argtypes = [ c_int ]
clib.fftwpp_hconv1d_delete.argtypes = [ c_void_p ]
clib.fftwpp_hconv1d_convolve.argtypes = [ c_void_p,
                                          ndpointer(dtype = np.complex128),
                                          ndpointer(dtype = np.complex128) ]

clib.fftwpp_create_conv2d.restype = c_void_p
clib.fftwpp_create_conv2d.argtypes = [ c_int, c_int ]
clib.fftwpp_conv2d_delete.argtypes = [ c_void_p ]
clib.fftwpp_conv2d_convolve.argtypes = [ c_void_p,
                                         ndpointer(dtype = np.complex128),
                                         ndpointer(dtype = np.complex128) ]
clib.fftwpp_conv2d_correlate.argtypes = [ c_void_p,
                                          ndpointer(dtype = np.complex128),
                                          ndpointer(dtype = np.complex128) ]

clib.fftwpp_create_hconv2d.restype = c_void_p
clib.fftwpp_create_hconv2d.argtypes = [ c_int, c_int ]
clib.fftwpp_hconv2d_delete.argtypes = [ c_void_p ]
clib.fftwpp_hconv2d_convolve.argtypes = [ c_void_p,
                                          ndpointer(dtype = np.complex128),
                                          ndpointer(dtype = np.complex128) ]

clib.fftwpp_create_conv3d.restype = c_void_p
clib.fftwpp_create_conv3d.argtypes = [ c_int, c_int, c_int ]
clib.fftwpp_conv3d_delete.argtypes = [ c_void_p ]
clib.fftwpp_conv3d_convolve.argtypes = [ c_void_p,
                                          ndpointer(dtype = np.complex128),
                                          ndpointer(dtype = np.complex128) ]
clib.fftwpp_conv3d_correlate.argtypes = [ c_void_p,
                                          ndpointer(dtype = np.complex128),
                                          ndpointer(dtype = np.complex128) ]

clib.fftwpp_create_hconv3d.restype = c_void_p
clib.fftwpp_create_hconv3d.argtypes = [ c_int, c_int, c_int ]
clib.fftwpp_hconv3d_delete.argtypes = [ c_void_p ]
clib.fftwpp_hconv3d_convolve.argtypes = [ c_void_p,
                                          ndpointer(dtype = np.complex128),
                                          ndpointer(dtype = np.complex128) ]

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
    >>> N = 8
    >>> f = fftwpp.complex_align([N])
    >>> g = fftwpp.complex_align([N])
    >>> for i in range(len(f)): f[i]=np.complex(i,i+1)
    >>> for i in range(len(g)): g[i]=np.complex(i,2*i+1)

    At this point, both ``f`` and ``g`` have shape ``(N,)``::

    >>> assert f.shape == (N,)

    Now, construct the convolution object and convolve::

    >>> c = fftwpp.Convolution(f.shape)
    >>> c.convolve(f, g)

    The convolution is now in ``f``::

    >>> np.allclose(f, [  -1.  +0.j,   -5.  +2.j,  -13.  +9.j,  -26. +24.j, \
    -45. +50.j,-71. +90.j, -105.+147.j, -148.+224.j])
    True

    Two dimensional convolutions
    ----------------------------

    To perform two dimensional convolutions::

    >>> import numpy as np
    >>> import fftwpp
    >>> N = 4
    >>> f = fftwpp.complex_align([N,N])
    >>> g = fftwpp.complex_align([N,N])
    >>> for i in range(len(f)):
    ...     for j in range(len(f[i])):
    ...             f[i][j]=np.complex(i,j)
    ...
    >>> for i in range(len(g)):
    ...     for j in range(len(g[i])):
    ...             g[i][j]=np.complex(2*i,j+1)
    ...

    At this point, both ``f`` and ``g`` have shape ``(N, N)``::

    >>> assert f.shape == (N, N)
    >>> assert g.shape == (N, N)

    Now, construct the convolution object and convolve::

    >>> c = fftwpp.Convolution(f.shape)
    >>> c.convolve(f, g)

    Again, the convolution is now in ``f``::


    Three dimensional convolutions
    ------------------------------

    To perform three dimensional convolutions::

    >>> import numpy as np
    >>> import fftwpp
    >>> N = 4
    >>> f = fftwpp.complex_align([N,N,N])
    >>> g = fftwpp.complex_align([N,N,N])
    >>> for i in range(len(f)):
    ...     for j in range(len(f[i])):
    ...             for k in range(len(f[i][j])):
    ...                     f[i][j][k]=np.complex(i+k,j+k)
    ...                     g[i][j][k]=np.complex(2*i+k,j+1+k)
    ... 

    At this point, both ``f`` and ``g`` have shape ``(N, N, N)``::

    >>> assert f.shape == (N, N, N)
    >>> assert g.shape == (N, N, N)

    Now, construct the convolution object and convolve::

    >>> c = fftwpp.Convolution(f.shape)
    >>> c.convolve(f, g)

    Again, the convolution is now in ``f``::

    """

    def __init__(self, shape):

        if isinstance(shape, int):
            shape = (shape,)

        self.dim   = len(shape)
        self.shape = tuple(shape)

        if self.dim == 1:
            self.cptr = clib.fftwpp_create_conv1d(*shape)
            self._convolve = clib.fftwpp_conv1d_convolve
            self._correlate = clib.fftwpp_conv1d_correlate
            self._delete = clib.fftwpp_conv1d_delete
        elif self.dim == 2:
            self.cptr = clib.fftwpp_create_conv2d(*shape)
            self._convolve = clib.fftwpp_conv2d_convolve
            self._correlate = clib.fftwpp_conv2d_correlate
            self._delete = clib.fftwpp_conv2d_delete
        elif self.dim == 3:
            self.cptr = clib.fftwpp_create_conv3d(*shape)
            self._convolve = clib.fftwpp_conv3d_convolve
            self._correlate = clib.fftwpp_conv3d_correlate
            self._delete = clib.fftwpp_conv3d_delete
        else:
            raise ValueError("invalid shape (length/dimension should be 1, 2, or 3)")

    def __del__(self):
        self._delete(self.cptr)

    def convolve(self, f, g):
        """Compute the convolution of *f* and *g*.

        The convolution is performed in-place (*f* is over-written).
        """

        assert f.shape == self.shape
        assert g.shape == self.shape

        self._convolve(self.cptr, f, g)

    # def correlate(self, f, g):
    #     """Compute the convolution of *f* and *g*.

    #     The convolution is performed in-place (*f* is over-written).
    #     """

    #     assert f.shape == self.shape
    #     assert g.shape == self.shape

    #     self._correlate(self.cptr, f, g)
        

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
    >>> N = 8
    >>> f = fftwpp.complex_align([N])
    >>> g = fftwpp.complex_align([N])
    >>> for i in range(len(f)):
    ...     f[i] = np.complex(i, i + 1)
    ...     g[i] = np.complex(i, 2 * i + 1)

    At this point, both ``f`` and ``g`` have shape ``(N,)``::

    >>> assert f.shape == (N,)

    Now, construct the convolution object and convolve::
    >>> c = fftwpp.HConvolution(N)
    >>> c.convolve(f, g)

    The convolution is now in ``f``::

    >>> np.allclose(f, [ 1022.  +0.j,   828. -12.j,   635. -15.j,   449.  -6.j,\
    275. +18.j,   118. +60.j,   -17.+123.j,  -125.+210.j])
    True


    Two dimensional convolutions
    ----------------------------

    To perform two dimensional convolutions::

    >>> import numpy as np
    >>> import fftwpp
    >>> Nx = 4
    >>> Ny = 4
    >>> f = fftwpp.complex_align([2 * Nx - 1, Ny])
    >>> g = fftwpp.complex_align([2 * Nx - 1, Ny])
    >>> for i in range(len(f)):
    ...     for j in range(len(f[i])):
    ...             f[i][j] = np.complex(i, j)
    ...             g[i][j] = np.complex(2 * i, j + 1)
    ... 

    Now, construct the convolution object and convolve::

    >>> c = fftwpp.HConvolution(f.shape)
    >>> c.convolve(f, g)

    Again, the convolution is now in ``f``::


    Three dimensional convolutions
    ------------------------------

    To perform three dimensional convolutions::

    >>> import numpy as np
    >>> import fftwpp
    >>> Nx = 4
    >>> Ny = 4
    >>> Nz = 4
    >>> f = fftwpp.complex_align([2 * Nx - 1, 2 * Ny - 1, Nz])
    >>> g = fftwpp.complex_align([2 * Nx - 1, 2 * Ny - 1, Nz])

    Now, construct the convolution object and convolve::

    >>> c = fftwpp.HConvolution(f.shape)
    >>> c.convolve(f, g)

    Again, the convolution is now in ``f``::

    >>> #np.allclose(f/N**3, np.fft.fftn(np.sin(x)*np.sin(5*y)*np.sin(z)))
    True

    """

    def __init__(self, shape):

        if isinstance(shape, int):
            shape = (shape,)
            
        self.dim   = len(shape)
        self.shape = tuple(shape)

        if self.dim == 1:
            self.cptr = clib.fftwpp_create_hconv1d(c_int(shape[0]))
            #self.cptr = clib.fftwpp_create_hconv1d(*shape)
            self._convolve = clib.fftwpp_hconv1d_convolve
            self._delete = clib.fftwpp_hconv1d_delete
        elif self.dim == 2:
            self.cptr = clib.fftwpp_create_hconv2d(c_int((shape[0] + 1) / 2),
                                                   c_int(shape[1]))
            self._convolve = clib.fftwpp_hconv2d_convolve
            self._delete = clib.fftwpp_hconv2d_delete
        elif self.dim == 3:
            self.cptr = clib.fftwpp_create_hconv3d(c_int((shape[0] + 1) / 2), 
                                                   c_int((shape[1] + 1) / 2), 
                                                   c_int(shape[2]))
            self._convolve = clib.fftwpp_hconv3d_convolve
            self._delete = clib.fftwpp_hconv3d_delete
        else:
            raise ValueError("invalid shape (length/dimension should be 1, 2, or 3)")

    def __del__(self):
        self._delete(self.cptr)

    def convolve(self, f, g):
        """Compute the convolution of *f* and *g*.

        The convolution is performed in-place (*f* is over-written).
        """

        assert f.shape == self.shape
        assert g.shape == self.shape

        self._convolve(self.cptr, f, g)

class AutoConvolution(object):
    """Implicitly zero-padded complex autoconvolution class.
    FIXME: doc
    """
    def __init__(self, shape):

        if isinstance(shape, int):
            shape = (shape,)

        self.dim   = len(shape)
        self.shape = tuple(shape)
        
        if self.dim == 1:
            self.cptr = clib.fftwpp_create_conv1dAB(shape[0], 1, 1)
            self._autoconvolve = clib.fftwpp_conv1d_autoconvolve
            self._autocorrelate = clib.fftwpp_conv1d_autocorrelate
            self._delete = clib.fftwpp_conv1d_delete
        elif self.dim == 2:
            self.cptr = clib.fftwpp_create_conv2dAB(shape[0], 1, 1)
            self._autoconvolve = clib.fftwpp_conv2d_autoconvolve
            self._autocorrelate = clib.fftwpp_conv2d_autocorrelate
            self._delete = clib.fftwpp_conv2d_delete
        elif self.dim == 3:
            self.cptr = clib.fftwpp_create_conv3dAB(shape[0], 1, 1)
            self._autoconvolve = clib.fftwpp_conv3d_autoconvolve
            self._autocorrelate = clib.fftwpp_conv3d_autocorrelate
            self._delete = clib.fftwpp_conv3d_delete
        else:
            raise ValueError("invalid shape (length/dimension should be 1, 2, or 3)")

    def __del__(self):
        self._delete(self.cptr)

    def autoconvolve(self, f):
        """
        Compute the autoconvolution of *f*.
        The convolution is performed in-place (*f* is over-written).
        """
        assert f.shape == self.shape
        self._autoconvolve(self.cptr, f)

    def autocorrelate(self, f):
        """
        Compute the autocorrelation of *f*.
        The correlation is performed in-place (*f* is over-written).
        """
        assert f.shape == self.shape
        self._autocorrelate(self.cptr, f)

if __name__ == "__main__":
    import doctest
    doctest.testmod()

