# fftwpp.py - Python wrapper of the C callable FFTW+ wrapper.
#
# Author: Matthew Emmett <memmett@gmail.com>
#

import numpy as np
import os

from numpy.ctypeslib import ndpointer
from ctypes import *

__all__ = [ 'Convolution' , 'HConvolution' ]

# load fftwpp shared library
base = os.path.dirname(os.path.abspath(__file__))
clib = CDLL(os.path.join(base, '_fftwpp.so'))

# prototypes
clib.fftwpp_create_conv1d.restype = c_void_p
clib.fftwpp_create_conv1d.argtypes = [ c_int ]
clib.fftwpp_conv1d_convolve.argtypes = [ c_void_p,
                                          ndpointer(dtype=np.complex128),
                                          ndpointer(dtype=np.complex128) ]

clib.fftwpp_create_hconv1d.restype = c_void_p
clib.fftwpp_create_hconv1d.argtypes = [ c_int ]
clib.fftwpp_hconv1d_convolve.argtypes = [ c_void_p,
                                          ndpointer(dtype=np.complex128),
                                          ndpointer(dtype=np.complex128) ]

clib.fftwpp_create_conv2d.restype = c_void_p
clib.fftwpp_create_conv2d.argtypes = [ c_int, c_int ]
clib.fftwpp_conv2d_convolve.argtypes = [ c_void_p,
                                         ndpointer(dtype=np.complex128),
                                         ndpointer(dtype=np.complex128) ]

clib.fftwpp_create_hconv2d.restype = c_void_p
clib.fftwpp_create_hconv2d.argtypes = [ c_int, c_int ]
clib.fftwpp_hconv2d_convolve.argtypes = [ c_void_p,
                                          ndpointer(dtype=np.complex128),
                                          ndpointer(dtype=np.complex128) ]

clib.fftwpp_create_conv3d.restype = c_void_p
clib.fftwpp_create_conv3d.argtypes = [ c_int, c_int, c_int ]
clib.fftwpp_conv3d_convolve.argtypes = [ c_void_p,
                                          ndpointer(dtype=np.complex128),
                                          ndpointer(dtype=np.complex128) ]

# clib.fftwpp_create_hconv3d.restype = c_void_p
# clib.fftwpp_create_hconv3d.argtypes = [ c_int, c_int, c_int ]
# clib.fftwpp_hconv3d_convolve.argtypes = [ c_void_p,
#                                           ndpointer(dtype=np.complex128),
#                                           ndpointer(dtype=np.complex128) ]


class Convolution(object):
    """Implicitly zero-padded complex convolution class.

    :param shape: shape/number of elements in the input arrays (int, tuple or list)

    The length of *shape* determines the dimension of the convolution.


    One dimensional convolutions
    ----------------------------

    To perform one dimensional convolutions::

    >>> import numpy as np
    >>> import fftwpp
    >>> N = 32
    >>> L = 2*np.pi
    >>> x = np.arange(0.0, L, L/N)
    >>> f = np.fft.fftn(np.sin(x))
    >>> g = np.fft.fftn(np.sin(5*x))

    At this point, both ``f`` and ``g`` have shape ``(N,)``::

    >>> assert f.shape == (N,)

    Now, construct the convolution object and convolve::

    >>> c = fftwpp.Convolution(f.shape)
    >>> c.convolve(f, g)

    The convolution is now in ``f``::

    # >>> # np.allclose(np.fft.ifft(f)/32, np.sin(x)*np.sin(5*x))
    # # True


    Two dimensional convolutions
    ----------------------------

    To perform two dimensional convolutions::

    >>> import numpy as np
    >>> import fftwpp
    >>> N = 32
    >>> L = 2*np.pi
    >>> x, y = np.mgrid[0.0:L:L/N, 0.0:L:L/N]
    >>> f = np.fft.fftn(np.sin(x))
    >>> g = np.fft.fftn(np.sin(5*y))

    At this point, both ``f`` and ``g`` have shape ``(N, N)``::

    >>> assert f.shape == (N, N)
    >>> assert g.shape == (N, N)

    Now, construct the convolution object and convolve::

    >>> c = fftwpp.Convolution(f.shape)
    >>> c.convolve(f, g)

    Again, the convolution is now in ``f``::

    >>> np.allclose(f/N**2, np.fft.fftn(np.sin(x)*np.sin(5*y)))
    True


    Three dimensional convolutions
    ------------------------------

    To perform three dimensional convolutions::

    >>> import numpy as np
    >>> import fftwpp
    >>> N = 32
    >>> L = 2*np.pi
    >>> x, y, z = np.mgrid[0.0:L:L/N, 0.0:L:L/N, 0.0:L:L/N]
    >>> f = np.fft.fftn(np.sin(x)*np.sin(z))
    >>> g = np.fft.fftn(np.sin(5*y))

    At this point, both ``f`` and ``g`` have shape ``(N, N, N)``::

    >>> assert f.shape == (N, N, N)
    >>> assert g.shape == (N, N, N)

    Now, construct the convolution object and convolve::

    >>> c = fftwpp.Convolution(f.shape)
    >>> c.convolve(f, g)

    Again, the convolution is now in ``f``::

    >>> np.allclose(f/N**3, np.fft.fftn(np.sin(x)*np.sin(5*y)*np.sin(z)))
    True


    """

    def __init__(self, shape):

        if isinstance(shape, int):
            shape = (shape,)

        self.dim   = len(shape)
        self.shape = tuple(shape)

        if self.dim == 1:
            self.cptr = clib.fftwpp_create_conv1d(*shape)
            self._convolve = clib.fftwpp_conv1d_convolve
            self._delete = clib.fftwpp_conv1d_delete
        elif self.dim == 2:
            self.cptr = clib.fftwpp_create_conv2d(*shape)
            self._convolve = clib.fftwpp_conv2d_convolve
            self._delete = clib.fftwpp_conv2d_delete
        elif self.dim == 3:
            self.cptr = clib.fftwpp_create_conv3d(*shape)
            self._convolve = clib.fftwpp_conv3d_convolve
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
        

class HConvolution(object):
    """Implicitly zero-padded complex convolution class. FIXME

    :param shape: shape/number of elements in the input arrays (int, tuple or list)

    The length of *shape* determines the dimension of the convolution.


    One dimensional convolutions
    ----------------------------

    To perform one dimensional convolutions::

    >>> import numpy as np
    >>> import fftwpp
    >>> N = 32
    >>> L = 2*np.pi
    >>> x = np.arange(0.0, L, L/N)
    >>> f = np.fft.fftn(np.sin(x))
    >>> g = np.fft.fftn(np.sin(5*x))

    At this point, both ``f`` and ``g`` have shape ``(N,)``::

    >>> assert f.shape == (N,)

    Now, construct the convolution object and convolve::
#FIXME: should be a HConvolution?
    >>> c = fftwpp.Convolution(f.shape)
    >>> c.convolve(f, g)

    The convolution is now in ``f``::

    # >>> # np.allclose(np.fft.ifft(f)/32, np.sin(x)*np.sin(5*x))
    # # True


    Two dimensional convolutions
    ----------------------------

    To perform two dimensional convolutions::

    >>> import numpy as np
    >>> import fftwpp
    >>> N = 32
    >>> L = 2*np.pi
    >>> x, y = np.mgrid[0.0:L:L/N, 0.0:L:L/N]
    >>> f = np.fft.fftn(np.sin(x))
    >>> g = np.fft.fftn(np.sin(5*y))

    At this point, both ``f`` and ``g`` have shape ``(N, N)``::

    >>> assert f.shape == (N, N)
    >>> assert g.shape == (N, N)

    Now, construct the convolution object and convolve::

    >>> c = fftwpp.Convolution(f.shape)
    >>> c.convolve(f, g)

    Again, the convolution is now in ``f``::

    >>> np.allclose(f/N**2, np.fft.fftn(np.sin(x)*np.sin(5*y)))
    True


    Three dimensional convolutions
    ------------------------------

    To perform three dimensional convolutions::

    >>> import numpy as np
    >>> import fftwpp
    >>> N = 32
    >>> L = 2*np.pi
    >>> x, y, z = np.mgrid[0.0:L:L/N, 0.0:L:L/N, 0.0:L:L/N]
    >>> f = np.fft.fftn(np.sin(x)*np.sin(z))
    >>> g = np.fft.fftn(np.sin(5*y))

    At this point, both ``f`` and ``g`` have shape ``(N, N, N)``::

    >>> assert f.shape == (N, N, N)
    >>> assert g.shape == (N, N, N)

    Now, construct the convolution object and convolve::

    >>> c = fftwpp.Convolution(f.shape)
    >>> c.convolve(f, g)

    Again, the convolution is now in ``f``::

    >>> np.allclose(f/N**3, np.fft.fftn(np.sin(x)*np.sin(5*y)*np.sin(z)))
    True


    """

    def __init__(self, shape):

        if isinstance(shape, int):
            shape = (shape,)

        self.dim   = len(shape)
        self.shape = tuple(shape)

        if self.dim == 1:
#             self.cptr = clib.fftwpp_create_hconv1d(c_int(shape[0]))
            self.cptr = clib.fftwpp_create_hconv1d(*shape)
            self._convolve = clib.fftwpp_hconv1d_convolve
            self._delete = clib.fftwpp_hconv1d_delete
        elif self.dim == 2:
#             self.cptr = clib.fftwpp_create_hconv2d(c_int(ny), c_int(ny))
            self.cptr = clib.fftwpp_create_hconv2d(*shape) # FIXME???
            self._convolve = clib.fftwpp_hconv2d_convolve
            self._delete = clib.fftwpp_hconv2d_delete
        elif self.dim == 3:
#             self.cptr = clib.fftwpp_create_hconv3d(c_int(shape[0]/2), c_int(shape[1]/2), c_int(shape[2]))
            self.cptr = clib.fftwpp_create_hconv3d(*shape)
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


if __name__ == "__main__":
    import doctest
    doctest.testmod()

