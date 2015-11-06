#!/usr/bin/python

import sys
import numpy as np
import fftwpp
import ctypes

print "Example of calling fftw++ convolutions from python:"

nthreads = 2
fftwpp.fftwpp_set_maxthreads(nthreads)

N = 8

def init(f, g):
    k = 0
    while k < len(f) :
        f[k] = np.complex(k, k + 1)
        g[k] = np.complex(k, 2 * k + 1)
        k += 1
    return;

f = fftwpp.complex_align([N])
g = fftwpp.complex_align([N])

print
print "1d non-centered complex convolution:"
init(f, g)
print "input f:"
print f
print
print "input g:"
print g
conv = fftwpp.Convolution(f.shape)
conv.convolve(f, g)
#conv.correlate(f, g)
print f

print
print "1d non-centered complex autoconvolution:"
init(f, g)
conv = fftwpp.AutoConvolution(f.shape)
conv.autoconvolve(f)
print f

print
print "1d non-centered complex autocorrelation:"
init(f, g)
conv.autocorrelate(f)
print f
print


print
print "1d centered Hermitian-symmetric complex convolution:"
hconv = fftwpp.HConvolution(f.shape)
init(f, g)
hconv.convolve(f, g)
print f

print
print "2d non-centered complex convolution:"

def init2(f, g):
    a = 0
    while a < f.shape[0] :
        b = 0
        while b < f[0].shape[0] :
            f[a][b] = np.complex(a, b)
            g[a][b] = np.complex(2 * a, b + 1)
            b += 1
        a += 1
    return;

mx = 4
my = 4

x = fftwpp.complex_align([mx, my])
y = fftwpp.complex_align([mx, my])

init2(x, y)

conv = fftwpp.Convolution(x.shape)
conv.convolve(x, y)
#conv.correlate(x, y)
print x


print
print "2d centered Hermitian-symmetric convolution:"
mx = 4
my = 4

hx = fftwpp.complex_align([2 * mx - 1, my])
hy = fftwpp.complex_align([2 * mx - 1, my])

init2(hx, hy)
print hx
hconv2 = fftwpp.HConvolution(hx.shape)
hconv2.convolve(hx, hy)
print hx

print
print "3d non-centered complex convolution:"

mx = 4
my = 4
mz = 4

x = fftwpp.complex_align([mx, my, mz])
y = fftwpp.complex_align([mx, my, mz])

def init3(f, g):
    a = 0
    while a < f.shape[0] :
        b = 0
        while b < f.shape[1] :
            c = 0
            while c < f.shape[2] :
                f[a][b][c] = np.complex(a + c, b + c)
                g[a][b][c] = np.complex(2 * a + c, b + 1 + c)
                c += 1
            b += 1
        a += 1
    return;

init3(x, y)

conv = fftwpp.Convolution(x.shape)
conv.convolve(x, y)
#conv.correlate(x, y)
print x

print
print "3d centered Hermitian-symmetric convolution:"
mx = 4
my = 4
mz = 4
hx = fftwpp.complex_align([2 * mx - 1, 2 * my - 1, mz])
hy = fftwpp.complex_align([2 * mx - 1, 2 * my - 1, mz])

init3(hx, hy)
conv = fftwpp.HConvolution(hx.shape)
conv.convolve(hx, hy)
print hx
