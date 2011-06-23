/* SSE2 complex multiplication routines
   Copyright (C) 2010 John C. Bowman, University of Alberta

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
   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA. */

#ifndef __cmult_sse2_h__
#define __cmult_sse2_h__ 1

#ifdef __SSE2__      

#include "Complex.h"
#include <emmintrin.h>

namespace fftwpp {

typedef __m128d Vec;

union uvec {
  unsigned u[4];
  Vec v;
};
  
extern const union uvec sse2_pm;
extern const union uvec sse2_mm;

#if defined(__INTEL_COMPILER) || !defined(__GNUC__)
static inline Vec operator -(const Vec& a) 
{
  return _mm_xor_pd(sse2_mm.v,a);
}

static inline Vec operator +(const Vec& a, const Vec& b) 
{
  return _mm_add_pd(a,b);
}

static inline Vec operator -(const Vec& a, const Vec& b) 
{
  return _mm_sub_pd(a,b);
}

static inline Vec operator *(const Vec& a, const Vec& b) 
{
  return _mm_mul_pd(a,b);
}

static inline void operator +=(Vec& a, const Vec& b) 
{
  a=_mm_add_pd(a,b);
}

static inline void operator -=(Vec& a, const Vec& b) 
{
  a=_mm_sub_pd(a,b);
}

static inline void operator *=(Vec& a, const Vec& b) 
{
  a=_mm_mul_pd(a,b);
}
#endif

static inline Vec UNPACKL(const Vec& z, const Vec& w)
{
  return _mm_unpacklo_pd(z,w);
}

static inline Vec UNPACKH(const Vec& z, const Vec& w)
{
  return _mm_unpackhi_pd(z,w);
}

static inline Vec FLIP(const Vec& z)
{
  return _mm_shuffle_pd(z,z,1);
}

static inline Vec CONJ(const Vec& z)
{
  return _mm_xor_pd(sse2_pm.v,z);
}

// Return I*z.
static inline Vec ZMULTI(const Vec& z)
{
  return FLIP(CONJ(z));
}

// Return the complex product of z and w.
static inline Vec ZMULT(const Vec& z, const Vec& w)
{
  return w*UNPACKL(z,z)+UNPACKH(z,z)*ZMULTI(w);
}

// Return the complex product of CONJ(z) and w.
static inline Vec ZMULTC(const Vec& z, const Vec& w)
{
  return w*UNPACKL(z,z)-UNPACKH(z,z)*ZMULTI(w);
}

// Return the complex product of z and I*w.
static inline Vec ZMULTI(const Vec& z, const Vec& w)
{
  return ZMULTI(w)*UNPACKL(z,z)-UNPACKH(z,z)*w;
}

// Return the complex product of CONJ(z) and I*w.
static inline Vec ZMULTIC(const Vec& z, const Vec& w)
{
  return ZMULTI(w)*UNPACKL(z,z)+UNPACKH(z,z)*w;
}

static inline Vec ZMULT(const Vec& x, const Vec& y, const Vec& w)
{
  return x*w+y*FLIP(w);
}

static inline Vec ZMULTI(const Vec& x, const Vec& y, const Vec& w)
{
  Vec z=CONJ(w);
  return x*FLIP(z)+y*z;
}

static inline Vec LOAD(const Complex *z)
{
  return *(const Vec *) z;
}

static inline void STORE(Complex *z, const Vec& v)
{
  *(Vec *) z = v;
}

static inline Vec LOAD(const double *z)
{
  return *(const Vec *) z;
}

static inline void STORE(double *z, const Vec& v)
{
  *(Vec *) z = v;
}

static inline Vec LOAD(double z)
{
  return _mm_load1_pd(&z);
}

}

#endif

#endif
