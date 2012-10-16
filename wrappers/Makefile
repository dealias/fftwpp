#
# Makefile for FFTW++ wrappers.
#

# compilers

CC = g++
C  = gcc
F  = gfortran

# flags
CFLAGS  = -I.. -I$HOME/opt/include
#CFLAGS += -g -Wall 
CFLAGS += -msse2 -mfpmath=sse -march=native
CFLAGS += -fPIC
CFLAGS += -DNDEBUG

a:=$(shell which icpc 2>&1 | tail -c5)
ifeq ($(a),icpc)
CC = icc
C  = icc
F  = ifort 
#CFLAGS=-O3 -ansi-alias -malign-double -fp-model fast=2 -openmp -I$(IDIR)
CFLAGS += -openmp -Ofast 
else
CFLAGS += -fopenmp
endif


FFTWIDIR=/usr/local/include/

FFLAGS  = -I/usr/include -I$HOME/opt/include
FFLAGS += -fopenmp

LDFLAGS = -L$HOME/opt/lib -lfftw3 -lfftw3_omp -lm -lstdc++

# targets

all: _fftwpp.so _chash.so cexample fexample

# rules

fftw++.o: ../fftw++.cc ../fftw++.h
	$(CC) $(CFLAGS) -c -o $@ $<

convolution.o: ../convolution.cc ../convolution.h fftw++.o
	$(CC) $(CFLAGS) -c -o $@ $<

cfftw++.o: cfftw++.cc cfftw++.h convolution.o fftw++.o
	$(CC) $(CFLAGS) -c -o $@ $<

chash.o: chash.cc chash.h
	$(CC)  $(CFLAGS) -c -o $@ $<

cexample: cexample.c cfftw++.o convolution.o fftw++.o chash.o
	$(C) $(CFLAGS) -std=c99 -o $@ $^ $(LDFLAGS)

fftwpp.o: fftwpp.f90
	$(F) -c -o $@ $<

fhash.o: fhash.f90
	$(F) -c -o $@ $<

fexample: fexample.f90 fftwpp.o fhash.o cfftw++.o convolution.o fftw++.o chash.o
	$(F) $(FFLAGS) -o $@ $^ $(LDFLAGS) -I$(FFTWIDIR)

_fftwpp.so: fftw++.o cfftw++.o convolution.o
	$(CC) $(CFLAGS) -shared -o $@ $^ $(LDFLAGS)

_chash.so: chash.o
	$(CC) $(CFLAGS) -shared -o $@ $^ $(LDFLAGS)

clean:
	rm -f cexample fexample *.so *.pyc *.o *.h.gch *.mod

.PHONY: clean
