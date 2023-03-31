# FFTW++
__Library of Fast Fourier Transforms, Convolutions, and MPI Transposes built on FFTW3__

Copyright &copy; 2004-2023 by John C. Bowman, Malcolm Roberts, and Noel Murasko, University of Alberta http://fftwpp.sourceforge.net

---

FFTW++ is a C++ header/MPI transpose for Version 3 of the highly optimized [FFTW](http://www.fftw.org) Fourier Transform library. It provides a simple
interface for 1D, 2D, and 3D complex-to-complex, real-to-complex, and
complex-to-real Fast Fourier Transforms and convolutions. It takes
care of the technical aspects of memory allocation, alignment, planning,
wisdom, and communication on both serial and parallel (OpenMP/MPI)
architectures. Wrappers for multiple 1D transforms are also provided. As
with the FFTW3 library itself, both in-place and out-of-place transforms of
arbitrary size are supported.

For reproducibility of Hybrid Dealiased Convolutions, see the [test programs](#test-programs) below.

Implicit dealiasing of standard and centered Hermitian convolutions is
also implemented; in 2D and 3D implicit zero-padding substantially
reduces memory usage and computation time.  For more information, see

- _Efficient Dealiased Convolutions without Padding_,
John C. Bowman and Malcolm Roberts. SIAM Journal on Scientific
Computing, 33:1, 386-406 (2011): http://www.math.ualberta.ca/~bowman/publications/dealias.pdf

- _Multithreaded implicitly dealiased convolutions_,
Malcolm Roberts and John C. Bowman. Journal of Computational
Physics, 356, 98-114 (2018): http://www.math.ualberta.ca/~bowman/publications/dealias2.pdf

- _Hybrid Dealiasing of Complex Convolutions_,
Noel Murasko and John C. Bowman. Submitted in Februrary 2023: [arXiv:2303.17510](https://arxiv.org/abs/2303.17510)

Convenient optional shift routines that place the Fourier origin in the logical
center of the domain are provided for centered complex-to-real transforms
in 2D and 3D; see `fftw++.h` for details.

FFTW++ supports multithreaded transforms and convolutions.
The global variable fftw::maxthreads specifies the maximum number of threads
to use. The constructors invoke a short timing test to check that using
multiple threads is actually beneficial for the given problem size.
Multithreading requires linking with a multithreaded FFTW implementation
and can be disabled by adding `-DFFTWPP_SINGLE_THREAD` to `CFLAGS`.

FFTW++ can also exploit the high-performance Array class available [here](http://www.math.ualberta.ca/~bowman/Array) (version 1.56 or higher),
designed for scientific computing. The arrays in that package do
memory bounds checking in debugging mode, but can be optimized by
specifying the `-DNDEBUG` compilation option (1D arrays optimize
completely to pointer operations).

Detailed documentation is provided before each class in the `fftw++.h`
header file. The included examples illustrate how easy it is to use
FFTW in C++ with the FFTW++ header class. Use of the Array class is
optional, but encouraged. If for some reason the Array class is not
used, memory should be allocated with `ComplexAlign` (or `doubleAlign`) to
ensure that the data is optimally aligned to sizeof(Complex), to
enable the SIMD extensions.  The optional alignment check in `fftw++.h`
can be disabled with the `-DNO_CHECK_ALIGN` compiler option.

### MPI

Hybrid OpenMP/MPI versions of the convolution routines in 2 and 3
dimensions are available in the `mpi/` directory.  Parallelization is
accomplished using the adaptive hybrid OpenMP/MPI transpose routine
described in

- _Adaptive Matrix Transpose Algorithms for Distributed
Multicore Processors_, John C. Bowman and Malcolm Roberts.
Interdisciplinary Topics in Applied Mathematics, Modeling and Computational
Science, Springer Proceedings in Mathematics & Statistics 117,
97-103 (2015): http://www.math.ualberta.ca/~bowman/publications/transpose.pdf

Either a 1D ("slab") and 2D ("pencil") data decomposition is used
for the three-dimensional convolutions, depending on the number of processors.

`mpi/fftw/` contains comparison code using FFTW's parallel MPI transform
and explicit padding.

#### Examples

The following programs are provided in the examples directory:

- 1D examples using ComplexAlign allocator:
    * `example0.cc`
    * `example0r.cc`

- 1D examples using Array class:
    * `example1.cc`
    * `example1r.cc`

- 2D examples using Array class:
    * `example2.cc`
    * `example2r.cc`

- 3D examples using Array class:
    * `example3.cc`
    * `example3r.cc`

- Examples of implicitly dealiased convolutions on complex non-centered data in 1, 2, and 3 dimensions:
    * `examplecconv.cc`
    * `examplecconv2.cc`
    * `examplecconv3.cc`

- Examples of implicitly dealiased convolutions on complex Hermitian-symmetric centered data in 1, 2, and 3 dimensions:
    * `exampleconv.cc
    * `exampleconv2.cc`
    * `exampleconv3.cc`

- Local transpose (in-place or out-of-place):
    * `exampletranspose.cc`

More general types of convolutions (for example, autoconvolutions)
can be performed by defining a custom multiplier or realmultiplier
function pointer.

### Wrappers

Wrappers for the convolution routines are available for C, Fortran,
and Python. Examples are given in the wrappers/ directory. The C
wrapper may be found in `cfftw++.h` and `cfftw++.cc`, the Fortran wrapper
in `fftwpp.f90`, and the Python wrapper in `fftwpp.py`. A unit-testing
script, `test.py`, is also available. Results for the given input data
are checked with a simple hash.

Compilation uses the environment variables `CPLUS_INCLUDE_PATH` to tell
the compiler where to find `fftw3.h`, and `FORTRAN_INCLUDE_PATH` to
indicate to the compiler the location of `fftw3.f03` from FFTW.

The following programs are available in the wrappers directory:

Using C to call multi-threaded 1D, 2D, and 3D binary convolutions and
1D and 2D ternary convolutions, with and without passing work arrays,
where the operation in physical space may correspond to either a
scalar multiplication (`M == 1`) or a dot product (`M > 1`):
`cexample.c`

Using Fortran to call multi-threaded 1D, 2D, and 3D binary
convolutions, with and without passing work arrays, where the
operation in physical space may correspond to either a scalar
multiplication (`mm == 1`) or a dot product (`mm > 1`):
`fexample.f90`

Using Python to call multi-threaded 1D, 2D, and 3D binary convolutions
(for scalar multiplication (`M == 1`) and with work arrays created by the
constructor):
`pexample.py`


### MPI

Hybrid OpenMP/MPI versions of the convolution routines in 2 and 3
dimensions are available in the mpi directory.

`cconv2.cc` and `cconv3.cc` are examples of two- and three-dimensional
complex non-centered convolutions.

`conv2.cc` and `conv3.cc` are examples of two- and three-dimensional
Hermitian-symmetric complex centered convolutions.

`fft2.cc` and `fft2r.cc` are examples of two-dimensional hybrid MPI/OpenMP FFTs
using a 1D data decomposition, for complex and real data, respectively.

fft3.cc and `fft3r.cc` are examples of three-dimensional hybrid MPI/OpenMP FFTs
using a 1D (slab) or 2D (pencil) data decomposition (depending on the
number of MPI processes), for complex and real data, respectively.

`timing.py` is a script which performs timing tests for mpi-based
convolutions.

The directory mpi/explicit contains comparison code using FFTW's parallel
MPI transform and explicit padding.


### Test Programs

The following programs are provided in `tests/`, along with various
timing and error analysis scripts. [Asymptote](http://asymptote.sourceforge.io/) scripts are provided for
visualizing the output. Passing the argument `-h` to each of these programs
outputs usage information.

- 1D complex convolution test: `hybridconv`

- 1D Hermitian convolution test: `hybridconvh`

- 2D complex convolution test: `hybridconv2`

- 2D Hermitian convolution test: `hybridconvh2`

- 3D complex convolution test: `hybridconv3`

- 3D Hermitian convolution test: `hybridconvh3`

- 1D FFT: `fft1`

- 1D real FFT: `fft1r`

- 1D multiple FFT: `mfft1`

- 1D multiple real FFT: `mfft1r`

- 2D FFT: `fft2`

- 2D real FFT: `fft2r`

- 3D FFT: `fft3`

- 3D real FFT: `fft3r`

### Availability and License

To compile from Git developmental source code:
`git clone https://github.com/dealias/fftwpp`

All source files in the FFTW++ project, unless explicitly noted otherwise,
are released under version 3 (or later) of the GNU Lesser General Public
License (see the files LICENSE.LESSER and LICENSE in the top-level source
directory).

---

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

---
