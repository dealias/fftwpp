#
# Makefile for the Python wrapper of FFTW++
#

IDIR =../

CC=g++
CFLAGS=-fopenmp -g -Wall -ansi -O3 -DNDEBUG -fomit-frame-pointer -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native -I$(IDIR)
C=gcc
F=gfortran

FFLAGS=-fopenmp


#CFLAGS+=-O3 -DNDEBUG
#CFLAGS+=-fomit-frame-pointer -fstrict-aliasing -ffast-math
#CFLAGS+=-msse2 -mfpmath=sse -march=native

LDFLAGS=-lfftw3 -lfftw3_omp -lm

FFTW=../fftw++
EXTRA=$(FFTW) ../convolution
ALL=$(EXTRA)

all: _fftwpp.so cexample fexample

fftw++.o: ../fftw++.cc ../fftw++.h
	$(CC) -c  $(CFLAGS) ../fftw++.cc 	-o $@  $(LDFLAGS)

convolution.o: ../convolution.h ../convolution.cc fftw++.o
	$(CC) $(CFLAGS) -c -o $@ ../convolution.cc $(LDFLAGS)

cfftw++.o: cfftw++.cc cfftw++.h convolution.o fftw++.o
	$(CC) $(CFLAGS) $(LDFLAGS) -c cfftw++.cc -o $@

cexample: cexample.c cfftw++.o convolution.o fftw++.o
	$(C) $^  $(CFLAGS) $(LDFLAGS) -o $@  -lstdc++ -std=c99

fexample.o: fexample.f90
	$(F) $^ $(FFLAGS) -c -o $@ -ffree-form $(LDFLAGS)

ffftwpp.o: ffftwpp.f90
	$(F) $^ $(FFLAGS) -c -o $@ -ffree-form $(LDFLAGS)

fexample: fexample.o cfftw++.o convolution.o fftw++.o ffftwpp.o
	$(F) $^ $(FFLAGS) -o $@ -lstdc++ -ffree-form  $(LDFLAGS) 

_fftwpp.so: ../fftw++.cc cfftw++.cc ../convolution.cc
	$(CC) -fPIC $(CFLAGS) -shared -o $@ $^ $(LDFLAGS)

clean:
	rm -f cexample fexample *.so *.pyc *.o  *.h.gch

.PHONY: clean
