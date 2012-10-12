#!/usr/bin/python

import numpy as np
import fftwpp
import sys

print "Example of calling fftw++ convolutions from python:"

N = 8

returnflag=0

def init(f,g):
    k=0
    while k < len(f) :
        f[k]=np.complex(k,k+1)
        g[k]=np.complex(k,2*k+1)
        k += 1
    return;

f = np.ndarray(shape=(N), dtype=complex)
g = np.ndarray(shape=(N), dtype=complex)

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

init(f,g)

print
print "1d centered Hermitian-symmetric complex convolution:"
hconv = fftwpp.HConvolution(f.shape)
hconv.convolve(f,g)
print f

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

print
print "2d non-centered complex convolution:"
x = np.ndarray(shape=(mx,my), dtype=complex)
y = np.ndarray(shape=(mx,my), dtype=complex)

init2(x,y)

conv = fftwpp.Convolution(x.shape)
conv.convolve(x,y)
print x


print
print "2d non-centered complex convolution:"
hx = np.ndarray(shape=(2*mx-1,my), dtype=complex)
hy = np.ndarray(shape=(2*mx-1,my), dtype=complex)

#init2(hx,hy)
#print hx
#hconv2 = fftwpp.HConvolution(mx,my)
#hconv2.convolve(hx,hy) # FIXME: segfault
#print hx

# FIXME: 2D Hermitian convolution wrapper is broken

#init2(x,y)

#print
#print "centered convolution:"
#conv = fftwpp.HConvolution(x.shape)
#conv.convolve(x,y)
#print x


sys.exit(returnflag)
