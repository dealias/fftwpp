/* 
   Copyright (C) 1988 Free Software Foundation
   written by Doug Lea (dl@rocky.oswego.edu)

   This file is part of the GNU C++ Library.  This library is free
   software; you can redistribute it and/or modify it under the terms of
   the GNU Library General Public License as published by the Free
   Software Foundation; either version 2 of the License, or (at your
   option) any later version.   This library is distributed in the hope
   that it will be useful, but WITHOUT ANY WARRANTY; without even the
   implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
   PURPOSE.  See the GNU Library General Public License for more details.
   You should have received a copy of the GNU Library General Public
   License along with this library; if not, write to the Free Software
   Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.
*/

#ifndef __Complex_h__
#define __Complex_h__ 1

#define __ATT_complex__

#include <iostream>
#include <cmath>

using std::istream;
using std::ostream;
using std::ws;

class Complex
{
#ifdef __ATT_complex__
public:
#else
protected:
#endif

  double re;
  double im;

public:

  Complex() {}
  Complex(double r, double i=0) : re(r), im(i) {}
  Complex(const Complex& y) : re(y.re), im(y.im) {}
        
  ~Complex() {}

  double real() const {return re;}
  double imag() const {return im;}

  const Complex& operator = (const Complex& y);
        
  const Complex& operator += (const Complex& y);
  const Complex& operator += (double y);
  const Complex& operator -= (const Complex& y);
  const Complex& operator -= (double y);
  const Complex& operator *= (const Complex& y);
  const Complex& operator *= (double y);
  const Complex& operator /= (const Complex& y); 
  const Complex& operator /= (double y); 
        
  void error(char* msg) const;
};

// inline members

inline const Complex& Complex::operator = (const Complex& y) 

{ 
  re = y.re; im = y.im; return *this; 
} 

inline const Complex& Complex::operator += (const Complex& y)
{ 
  re += y.re;  im += y.im; return *this; 
}

inline const Complex& Complex::operator += (double y)
{ 
  re += y; return *this; 
}

inline const Complex& Complex::operator -= (const Complex& y)
{ 
  re -= y.re;  im -= y.im; return *this; 
}

inline const Complex& Complex::operator -= (double y)
{ 
  re -= y; return *this; 
}

inline const Complex& Complex::operator *= (const Complex& y)
{  
  double r = re * y.re - im * y.im;
  im = re * y.im + im * y.re; 
  re = r; 
  return *this; 
}

inline const Complex& Complex::operator *= (double y)
{  
  re *= y; im *= y; return *this; 
}

inline const Complex& Complex::operator /= (const Complex& y)
{
  double t1,t2,t3;
  t2=1.0/(y.re*y.re+y.im*y.im);
  t1=t2*y.re; t2 *= y.im; t3=re;
  re *= t1; re += im*t2;
  im *= t1; im -= t3*t2;
  return *this;
}

inline const Complex& Complex::operator /= (double y)
{
  re /= y;
  im /= y;
  return *this;
}

//      functions

inline int operator == (const Complex& x, const Complex& y)
{
  return x.re == y.re && x.im == y.im;
}

inline int operator == (const Complex& x, double y)
{
  return x.im == 0.0 && x.re == y;
}

inline int operator != (const Complex& x, const Complex& y)
{
  return x.re != y.re || x.im != y.im;
}

inline int operator != (const Complex& x, double y)
{
  return x.im != 0.0 || x.re != y;
}

inline Complex operator - (const Complex& x)
{
  return Complex(-x.re, -x.im);
}

inline Complex conj(const Complex& x)
{
  return Complex(x.re, -x.im);
}

inline Complex operator + (const Complex& x, const Complex& y)
{
  return Complex(x.re+y.re, x.im+y.im);
}

inline Complex operator + (const Complex& x, double y)
{
  return Complex(x.re+y, x.im);
}

inline Complex operator + (double x, const Complex& y)
{
  return Complex(x+y.re, y.im);
}

inline Complex operator - (const Complex& x, const Complex& y)
{
  return Complex(x.re-y.re, x.im-y.im);
}

inline Complex operator - (const Complex& x, double y)
{
  return Complex(x.re-y, x.im);
}

inline Complex operator - (double x, const Complex& y)
{
  return Complex(x-y.re, -y.im);
}

inline Complex operator * (const Complex& x, const Complex& y)
{
  return Complex(x.re*y.re-x.im*y.im, x.re*y.im+x.im*y.re);
}

inline Complex multconj(const Complex& x, const Complex& y)
{
  return Complex(x.re*y.re+x.im*y.im,x.im*y.re-x.re*y.im);
}

inline Complex operator * (const Complex& x, double y)
{
  return Complex(x.re*y, x.im*y);
}

inline Complex operator * (double x, const Complex& y)
{
  return Complex(x*y.re, x*y.im);
}

inline Complex operator / (const Complex& x, const Complex& y)
{
  double t1,t2;
  t2=1.0/(y.re*y.re+y.im*y.im);
  t1=t2*y.re; t2 *= y.im;
  return Complex(x.im*t2+x.re*t1, x.im*t1-x.re*t2);
}

inline Complex operator / (const Complex& x, double y)
{
  return Complex(x.re/y,x.im/y);
}

inline Complex operator / (double x, const Complex& y)
{
  double factor;
  factor=1.0/(y.re*y.re+y.im*y.im);
  return Complex(x*y.re*factor,-x*y.im*factor);
}

inline double real(const Complex& x)
{
  return x.re;
}

inline double imag(const Complex& x)
{
  return x.im;
}

inline double abs2(const Complex& x)
{
  return x.re*x.re+x.im*x.im;
}

inline double abs(const Complex& x)
{
  return sqrt(abs2(x));
}

inline double arg(const Complex& x)
{
  return x.im != 0.0 ? atan2(x.im, x.re) : 0.0;
}

// Return the principal branch of the square root (non-negative real part).
inline Complex sqrt(const Complex& x)
{
  double mag=abs(x);
  if(mag == 0.0) return Complex(0.0,0.0);
  else if(x.re > 0) {
    double re=sqrt(0.5*(mag+x.re));
    return Complex(re,0.5*x.im/re);
  } else {
    double im=sqrt(0.5*(mag-x.re));
    if(x.im < 0) im=-im;
    return Complex(0.5*x.im/im,im);
  }
}

inline Complex polar(double r, double t)
{
  return Complex(r*cos(t), r*sin(t));
}

// Complex exponentiation
inline Complex pow(const Complex& z, const Complex& w)
{
  double u=w.re;
  double v=w.im;
  if(z == 0.0) return w == 0.0 ? 1.0 : 0.0;
  double logr=0.5*log(abs2(z));
  double th=arg(z);
  double phi=logr*v+th*u;
  return exp(logr*u-th*v)*Complex(cos(phi),sin(phi));
}

inline Complex pow(const Complex& z, double u)
{
  if(z == 0.0) return u == 0.0 ? 1.0 : 0.0;
  double logr=0.5*log(abs2(z));
  double theta=u*arg(z);
  return exp(logr*u)*Complex(cos(theta),sin(theta));
}

inline istream& operator >> (istream& s, Complex& y)
{
  char c;
  s >> ws >> c;
  if(c == '(') {
    s >> y.re >> c;
    if(c == ',') s >> y.im >> c;
    else y.im=0.0;
  } else {
    s.putback(c);
    s >> y.re; y.im=0.0;
  }
  return s;
}

inline ostream& operator << (ostream& s, const Complex& y)
{
  s << "(" << y.re << "," << y.im << ")";
  return s;
}

inline bool isfinite(const Complex& z)
{
#ifdef _WIN32
  return _finite(z.re) && _finite(z.im);
#else  
  return !(std::isinf(z.re) || std::isnan(z.re) || std::isinf(z.im) || std::isnan(z.im));
#endif  
}

#endif
