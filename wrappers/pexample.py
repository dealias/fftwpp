#!/usr/bin/python3
# New python 3 code: Robert Joseph

import fftwpp
print("Example of calling fftw++ convolutions from python:")

nthreads = 2
fftwpp.fftwpp_set_maxthreads(nthreads)

### 1D Code

N = 7
def init(f, g, mode="Default"):
    for k in range(len(f)):
        if k == 0 and mode == "Hermitian":
            f[0] = complex(1, 0)
            g[0] = complex(2, 0)
        else:
            f[k] = complex(k, k + 1)
            g[k] = complex(k, 2 * k + 1)

f = fftwpp.complex_align([N])
g = fftwpp.complex_align([N])

print("Input f:")
print(f)
print()
print("Input g:")
print(g)
print()

init(f, g)
print("1D non-centered complex convolution:")
conv = fftwpp.Convolution(f.shape)
conv.convolve(f, g)
print(f)

init(f, g)
print("1D non-centered complex correlation:")
conv.correlate(f, g)
print(f)

init(f, g)
conv = fftwpp.AutoConvolution(f.shape)
print("1D non-centered complex autoconvolution:")
conv.autoconvolve(f)
print(f)

init(f, g)
print("1D non-centered complex autocorrelation:")
conv.autocorrelate(f)
print(f)

print("1D centered Hermitian-symmetric complex convolution:")
hconv = fftwpp.HConvolution(f.shape)
init(f, g, mode="Hermitian")
hconv.convolve(f, g)
print(f)

### 2D Code
def init2(f, g):
    for a in range(f.shape[0]):
        for b in range(f.shape[1]):
            f[a][b] = complex(a, b)
            g[a][b] = complex(2 * a, b + 1)

mx = 4
my = 4

x = fftwpp.complex_align([mx, my])
y = fftwpp.complex_align([mx, my])

init2(x, y)

print("Input x:")
print(x)
print()
print("Input y:")
print(y)
print()

conv = fftwpp.Convolution(x.shape)
print("2D non-centered complex convolution:")
conv.convolve(x, y)
print(x)
print()

init2(x, y)
conv = fftwpp.Convolution(x.shape)
print("2D non-centered complex correlation:")
conv.correlate(x, y)
print(x)
print()

init2(x, y)
conv = fftwpp.AutoConvolution(x.shape)
print("2D non-centered complex autoconvolution:")
conv.autoconvolve(x)
print(x)
print()

init2(x, y)
conv = fftwpp.AutoConvolution(x.shape)
print("2D non-centered complex autocorrelation:")
conv.autocorrelate(x)
print(x)
print()

mx = 4
my = 4

hx = fftwpp.complex_align([2 * mx - 1, my])
hy = fftwpp.complex_align([2 * mx - 1, my])

init2(hx, hy)
print("2D centered Hermitian-symmetric convolution:")
hconv2 = fftwpp.HConvolution(hx.shape)
hconv2.convolve(hx, hy)
print(hx)


### 3D Code
import fftwpp

print("3D non-centered complex convolution:")

mx = 4
my = 4
mz = 4

x = fftwpp.complex_align([mx, my, mz])
y = fftwpp.complex_align([mx, my, mz])

def init3(f, g):
    for a in range(f.shape[0]):
        for b in range(f.shape[1]):
            for c in range(f.shape[2]):
                f[a][b][c] = complex(a + c, b + c)
                g[a][b][c] = complex(2 * a + c, b + 1 + c)

init3(x, y)

conv = fftwpp.Convolution(x.shape)
conv.convolve(x, y)
print(x)

print()
print("3D non-centered complex autoconvolution:")
init3(x, y)
conv = fftwpp.AutoConvolution(x.shape)
conv.autoconvolve(x)
print(x)

print()
print("3D non-centered complex autocorrelation:")
init3(x[::-1], y)
conv.autocorrelate(x)
print(x)

print()
print("3D centered Hermitian-symmetric convolution:")
mx = 4
my = 4
mz = 4
hx = fftwpp.complex_align([2 * mx - 1, 2 * my - 1, mz])
hy = fftwpp.complex_align([2 * mx - 1, 2 * my - 1, mz])

init3(hx, hy)
conv = fftwpp.HConvolution(hx.shape)
conv.convolve(hx, hy)
print(hx)