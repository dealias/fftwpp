#
# Makefile for the Python wrapper of FFTW++
#

IDIR =../

CC=g++
CFLAGS=-fopenmp -g -Wall -ansi -O3 -DNDEBUG -fomit-frame-pointer -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native -I$(IDIR)
C=gcc

#CFLAGS+=-O3 -DNDEBUG
#CFLAGS+=-fomit-frame-pointer -fstrict-aliasing -ffast-math
#CFLAGS+=-msse2 -mfpmath=sse -march=native

LDFLAGS=-lfftw3 -lfftw3_omp -lm

FFTW=../fftw++
EXTRA=$(FFTW) ../convolution
ALL=$(EXTRA)

all: _fftwpp.so convolution.o #cfftw++.h.gch

fftw++.o: ../fftw++.cc ../fftw++.h
	$(CC) -c $^ -o $@ 

convolution.o: ../convolution.h ../convolution.cc
	$(CC) $(CFLAGS) -c -o $@ ../convolution.cc $(LDFLAGS)

convolution: convolution convolution.o
	$(CC) $(CFLAGS) convolution.o $(EXTRA:=.o) $(LDFLAGS)

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
