/* SSE2 complex multiplication routines
   Copyright (C) 2010-2022 John C. Bowman, University of Alberta

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

#include "Complex.h"

namespace fftwpp {

#ifdef __SSE2__

#include <emmintrin.h>

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

// Return (z.x,w.x)
static inline Vec UNPACKL(const Vec& z, const Vec& w)
{
  return _mm_unpacklo_pd(z,w);
}

// Return (z.y,w.y)
static inline Vec UNPACKH(const Vec& z, const Vec& w)
{
  return _mm_unpackhi_pd(z,w);
}

// Return (z.y,z.x)
static inline Vec FLIP(const Vec& z)
{
  return _mm_shuffle_pd(z,z,1);
}

static inline Vec LOAD2(double x)
{
  return _mm_load1_pd(&x);
}

static inline Vec LOAD(double x)
{
  return _mm_load_sd(&x);
}

static inline Vec LOADFLIP(const Complex *z)
{
  return _mm_loadr_pd((double *) z);
}

// Return (z.x,-z.y)
static inline Vec CONJ(const Vec& z)
{
  return _mm_xor_pd(sse2_pm.v,z);
}

// Return I*z.
static inline Vec ZMULTI(const Vec& z)
{
  return _mm_shuffle_pd(-z,z,1);
}

static inline Vec SQRT(const Vec& z)
{
  return _mm_sqrt_pd(z);
}

#else

class Vec {
public:
  double x;
  double y;

  Vec() {};
  Vec(double x, double y) : x(x), y(y) {};
  Vec(const Vec &v) : x(v.x), y(v.y) {};
  Vec(const Complex &z) : x(z.re), y(z.im) {};

  const Vec& operator += (const Vec& v) {
    x += v.x;
    y += v.y;
    return *this;
  }

  const Vec& operator -= (const Vec& v) {
    x -= v.x;
    y -= v.y;
    return *this;
  }

  const Vec& operator *= (const Vec& v) {
    x *= v.x;
    y *= v.y;
    return *this;
  }
};

static inline Vec operator -(const Vec& a)
{
  return Vec(-a.x,-a.y);
}

static inline Vec operator +(const Vec& a, const Vec& b)
{
  return Vec(a.x+b.x,a.y+b.y);
}

static inline Vec operator -(const Vec& a, const Vec& b)
{
  return Vec(a.x-b.x,a.y-b.y);
}

static inline Vec operator *(const Vec& a, const Vec& b)
{
  return Vec(a.x*b.x,a.y*b.y);
}

static inline Vec UNPACKL(const Vec& z, const Vec& w)
{
  return Vec(z.x,w.x);
}

static inline Vec UNPACKH(const Vec& z, const Vec& w)
{
  return Vec(z.y,w.y);
}

static inline Vec FLIP(const Vec& z)
{
  return Vec(z.y,z.x);
}

static inline Vec CONJ(const Vec& z)
{
  return Vec(z.x,-z.y);
}

static inline Vec REFL(const Vec& z)
{
  return Vec(-z.x,z.y);
}

static inline Vec LOAD2(double x)
{
  return Vec(x,x);
}

static inline Vec LOAD(double x)
{
  return Vec(x,0.0);
}

static inline Vec LOADFLIP(const Complex *z)
{
  return Vec(z->im,z->re);
}

// Return I*z.
static inline Vec ZMULTI(const Vec& z)
{
  return Vec(-z.y,z.x);
}

static inline Vec SQRT(const Vec& z)
{
  return Vec(sqrt(z.x),sqrt(z.y));
}

#endif

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

// Return the complex product of z and w.
static inline Vec ZMULT(const Vec& z, const Vec& w)
{
  return UNPACKL(z,z)*w+UNPACKH(-z,z)*FLIP(w);
}

// Return ZMULT(CONJ(z),w).
static inline Vec ZCMULT(const Vec& z, const Vec& w)
{
  return UNPACKL(z,z)*w+UNPACKH(z,-z)*FLIP(w);
}

// Return ZMULT(z,I*w).
static inline Vec ZMULTI(const Vec& z, const Vec& w)
{
  return UNPACKL(-z,z)*FLIP(w)-UNPACKH(z,z)*w;
}

// Return ZMULT(CONJ(z),I*w).
static inline Vec ZCMULTI(const Vec& z, const Vec& w)
{
  return UNPACKL(-z,z)*FLIP(w)+UNPACKH(z,z)*w;
}

// Return ZMULT(z,w)+ZCMULT(z,v).
static inline Vec ZMULT2(const Vec& z, const Vec& w, const Vec& v)
{
  return UNPACKL(z,z)*(w+v)-UNPACKH(z,-z)*FLIP(w-v);
}

// Return ZMULT(z,w) given x=(z.re,z.re), y=(z.im,-z.im).
static inline Vec ZMULT(const Vec& x, const Vec& y, const Vec& w)
{
  return x*w-y*FLIP(w);
}

// Return ZMULT(z,w)+ZCMULT(z,v) given x=(z.re,z.re), y=(z.im,-z.im).
static inline Vec ZMULT2(const Vec& x, const Vec& y, const Vec& w,
                         const Vec& v)
{
  return x*(w+v)-y*FLIP(w-v);
}

// Return ZMULT(z,I*w) given x=(z.re,-z.re), y=(z.im,z.im).
static inline Vec ZMULTI(const Vec& x, const Vec& y, const Vec& w)
{
  return FLIP(x*w)-y*w;
}

}

#endif
