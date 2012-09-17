#
# Makefile for the Python wrapper of FFTW++
#

CC=g++
C=gcc

CFLAGS?=-g -Wall -ansi
#CFLAGS+=-O3 -DNDEBUG
#CFLAGS+=-fomit-frame-pointer -fstrict-aliasing -ffast-math
#CFLAGS+=-msse2 -mfpmath=sse -march=native
CFLAGS+=-fopenmp
CFLAGS+=-I../

LDFLAGS+=-lm -lfftw3
LDFLAGS+=-lfftw3_omp

all: _fftwpp.so cexample cfftw++.h.gch

_fftwpp.so: ../fftw++.cc cfftw++.cc ../convolution.cc
	$(CC) -fPIC $(CFLAGS) -shared -o $@ $^ $(LDFLAGS)

cfftw++.o: cfftw++.cc ../convolution.o ../fftw++.o
	$(CC) $(CFLAGS) $(LDFLAGS) -c $^ -o $@

cexample.o: cexample.c
	$(C)  -c -fPIC $^ $(LDFLAGS) -o $@

cexample: cexample.o  cfftw++.o
	$(CC) $^  $(CFLAGS) $(LDFLAGS) -o $@

clean:
	rm -f *.so fftwpp.pyc *.o cexample *.h.gch

.PHONY: clean
