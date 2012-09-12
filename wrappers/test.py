
import numpy as np
import fftwpp

L = 2*np.pi
N = 32

x, y = np.mgrid[0.0:L:L/N, 0.0:L:L/N]

z1 = np.fft.fftn(np.sin(x))
z2 = np.fft.fftn(np.sin(5*y))

conv = fftwpp.Convolution(z1.shape)
conv.convolve(z1, z2)
z1 = z1 / N**2

z3 = np.fft.fftn(np.sin(x)*np.sin(5*y))

print np.allclose(z3, z1)

N = 16
L = 2*np.pi
x, y, z = np.mgrid[0.0:L:L/N, 0.0:L:L/N, 0.0:L:L/N]
f = np.fft.fftn(np.sin(x)*np.sin(z))
g = np.fft.fftn(np.sin(5*y))

assert f.shape == (N, N, N)
assert g.shape == (N, N, N)

c = fftwpp.Convolution(f.shape)
c.convolve(f, g)

print np.allclose(f/N**3, np.fft.fftn(np.sin(x)*np.sin(5*y)*np.sin(z)))




# x = np.arange(0.0, L, L/N)
# f = np.fft.fft(np.sin(x))
# g = np.fft.fft(np.sin(5*x))

# assert f.shape == (N,)

# c = fftwpp.Convolution(f.shape)
# c.convolve(f, g)
# f = f / N

# print np.allclose(f, np.fft.fftn(np.sin(x)*np.sin(5*x)))
# print f
# print np.fft.fft(np.sin(x)*np.sin(5*x))
