#!/usr/bin/python3
# New python 3 code: Robert Joseph

import fftwpp
print("Example of calling fftw++ convolutions from python:")

nthreads = 2
fftwpp.fftwpp_set_maxthreads(nthreads)

### 1D Code

f=[]
g=[]

def init(N=8, mode="Default"):
    global f,g
    f = fftwpp.complex_align([N])
    g = fftwpp.complex_align([N])

    for k in range(len(f)):
        if k == 0 and mode == "Hermitian":
            f[0] = complex(1, 0)
            g[0] = complex(2, 0)
        else:
            f[k] = complex(k, k + 1)
            g[k] = complex(k, 2 * k + 1)

init()
print("Input f:")
print(f)
print()
print("Input g:")
print(g)
print()

print("1D non-centered complex convolution:")
conv = fftwpp.Convolution(f.shape)
conv.convolve(f, g)
print(f)

init()
print("1D non-centered complex correlation:")
conv.correlate(f, g)
print(f)

init()
conv = fftwpp.AutoConvolution(f.shape)
print("1D non-centered complex autoconvolution:")
conv.autoconvolve(f)
print(f)

init()
print("1D non-centered complex autocorrelation:")
conv.autocorrelate(f)
print(f)

print("1D centered Hermitian-symmetric complex convolution:")
init(4, mode="Hermitian")
hconv = fftwpp.HConvolution(f.shape)
hconv.convolve(f,g)
print(f)

### 2D Code
def init2(Nx=4,Ny=4):
    global f,g
    f = fftwpp.complex_align([Nx,Ny])
    g = fftwpp.complex_align([Nx,Ny])
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            f[i][j] = complex(i, j)
            g[i][j] = complex(2*i, j + 1)

init2()
print("Input f:")
print(f)
print()
print("Input g:")
print(g)
print()

conv = fftwpp.Convolution(f.shape)
print("2D non-centered complex convolution:")
conv.convolve(f,g)
print(f)
print()

init2()
conv = fftwpp.Convolution(f.shape)
print("2D non-centered complex correlation:")
conv.correlate(f,g)
print(f)
print()

init2()
conv = fftwpp.AutoConvolution(f.shape)
print("2D non-centered complex autoconvolution:")
conv.autoconvolve(f)
print(f)
print()

init2()
conv = fftwpp.AutoConvolution(f.shape)
print("2D non-centered complex autocorrelation:")
conv.autocorrelate(f)
print(f)
print()

init2(7,4)
print("2D centered Hermitian-symmetric convolution:")
conv = fftwpp.HConvolution(f.shape)
conv.convolve(f,g)
print(f)

### 3D Code
import fftwpp

print("3D non-centered complex convolution:")

def init3(Nx=4,Ny=4,Nz=4):
    global f,g
    f = fftwpp.complex_align([Nx,Ny,Nz])
    g = fftwpp.complex_align([Nx,Ny,Nz])
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            for k in range(f.shape[2]):
                f[i][j][k] = complex(i + k, j + k)
                g[i][j][k] = complex(2 * i + k, j + 1 + k)

init3()
conv = fftwpp.Convolution(f.shape)
conv.convolve(f,g)
print(f)

print()
print("3D non-centered complex autoconvolution:")
init3()
conv = fftwpp.AutoConvolution(f.shape)
conv.autoconvolve(f)
print(f)

print()
print("3D non-centered complex autocorrelation:")
init3()
conv.autocorrelate(f)
print(f)

print()
print("3D centered Hermitian-symmetric convolution:")

init3(7,7,4)
conv = fftwpp.HConvolution(f.shape)
conv.convolve(f,g)
print(f)
