#pragma once

#include <cstddef>

extern size_t L,Lx,Ly,Lz; // input data lengths
extern size_t M,Mx,My,Mz; // minimum padded lengths

extern size_t mx,my,mz; // internal FFT sizes
extern size_t Dx,Dy,Dz; // numbers of residues computed at a time
extern size_t Sx,Sy;    // strides
extern ptrdiff_t Ix,Iy,Iz;          // inplace flags

extern bool Output;
extern bool testError;
extern bool Centered;
extern bool normalized;
extern bool accuracy;
