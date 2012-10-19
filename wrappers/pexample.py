#!/usr/bin/python

import sys
import numpy as np
import fftwpp
import pyfftw

import ctypes
hashlib = ctypes.CDLL("./_chash.so")

print "Example of calling fftw++ convolutions from python:"

def hash(f,N):
    return hashlib.hash(f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),N)

nthreads=2
fftwpp.fftwpp_set_maxthreads(nthreads)

returnflag=0

N = 8

def init(f,g):
    k=0
    while k < len(f) :
        f[k]=np.complex(k,k+1)
        g[k]=np.complex(k,2*k+1)
        k += 1
    return;

f = pyfftw.n_byte_align_empty((N), 16, dtype=np.complex128)
g = pyfftw.n_byte_align_empty((N), 16, dtype=np.complex128)

init(f,g)

print "input f:"
print f
print
print "input g:"
print g

print
print "1d non-centered complex convolution:"
conv = fftwpp.Convolution(f.shape)
conv.convolve(f,g)
print f

if (hash(f,N) != -1208058208 ):
    returnflag += 1
    print "ImplicitConvolution output incorect."

init(f,g)

print
print "1d centered Hermitian-symmetric complex convolution:"
hconv = fftwpp.HConvolution(f.shape)
hconv.convolve(f,g)
print f

if (hash(f,N) != -1208087538 ):
    returnflag += 2
    print "ImplicitHConvolution output incorect."


print
print "2d non-centered complex convolution:"

def init2(f,g):
    a=0
    while a < f.shape[0] :
        b=0
        while b < f[0].shape[0] :
            f[a][b]=np.complex(a,b)
            g[a][b]=np.complex(2*a,b+1)
            b += 1
        a += 1
    return;

mx=4
my=4

x = pyfftw.n_byte_align_empty((mx,my), 16, dtype=np.complex128)
y = pyfftw.n_byte_align_empty((mx,my), 16, dtype=np.complex128)

init2(x,y)

conv = fftwpp.Convolution(x.shape)
conv.convolve(x,y)
print x
if (hash(x,mx*my) != -268695633 ):
    returnflag += 4
    print "ImplicitConvolution2 output incorect."


print
print "2d centered Hermitian-symmetric convolution:"
mx=4
my=4
mz=4
hx = pyfftw.n_byte_align_empty((2*mx-1,my), 16, dtype=np.complex128)
hy = pyfftw.n_byte_align_empty((2*mx-1,my), 16, dtype=np.complex128)

init2(hx,hy)
print hx
hconv2 = fftwpp.HConvolution(hx.shape)
hconv2.convolve(hx,hy)
print hx

if (hash(hx,(2*mx-1)*my) != -947771835 ):
    returnflag += 8
    print "ImplicitHConvolution2 output incorect."

print
print "3d non-centered complex convolution:"

x = pyfftw.n_byte_align_empty((mx,my,mz), 16, dtype=np.complex128)
y = pyfftw.n_byte_align_empty((mx,my,mz), 16, dtype=np.complex128)

def init3(f,g):
    a=0
    while a < f.shape[0] :
        b=0
        while b < f.shape[1] :
            c=0
            while c < f.shape[2] :
                f[a][b][c]=np.complex(a+c,b+c)
                g[a][b][c]=np.complex(2*a+c,b+1+c)
                c += 1
            b += 1
        a += 1
    return;

init3(x,y)

conv = fftwpp.Convolution(x.shape)
conv.convolve(x,y)
print x
if (hash(x,mx*my*mz) != 1073436205 ):
    returnflag += 16
    print "ImplicitConvolution3 output incorect."

print
print "3d centered Hermitian-symmetric convolution:"
mx=4
my=4
mz=4
hx = pyfftw.n_byte_align_empty((2*mx-1,2*my-1,mz), 16, dtype=np.complex128)
hy = pyfftw.n_byte_align_empty((2*mx-1,2*my-1,mz), 16, dtype=np.complex128)

init3(hx,hy)
conv = fftwpp.HConvolution(hx.shape)
conv.convolve(hx,hy)
print hx

if (hash(hx,(2*mx-1)*(2*my-1)*mz) != -472674783 ):
    returnflag += 32
    print "ImplicitConvolution3 output incorect."


sys.exit(returnflag)
