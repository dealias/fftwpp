#
# Makefile for the Python wrapper of FFTW++
#

CC=g++

CFLAGS?=-g -Wall -ansi
#CFLAGS+=-O3 -DNDEBUG
#CFLAGS+=-fomit-frame-pointer -fstrict-aliasing -ffast-math
#CFLAGS+=-msse2 -mfpmath=sse -march=native
CFLAGS+=-fopenmp
CFLAGS+=-I../

LDFLAGS+=-lm -lfftw3
LDFLAGS+=-lfftw3_omp

all: _fftwpp.so

_fftwpp.so: ../fftw++.cc cfftw++.cc ../convolution.cc
	$(CC) -fPIC $(CFLAGS) -shared -o $@ $^ $(LDFLAGS)

clean:
	rm -f _fftwpp.so

.PHONY: clean
