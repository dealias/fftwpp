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
CFLAGS += -fPIC -fopenmp
CFLAGS += -msse2 -mfpmath=sse -march=native
CFLAGS += -DNDEBUG -Ofast 

FFTWIDIR=/usr/local/include/

FFLAGS  = -I$HOME/opt/include
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
	rm -f cexample fexample *.so *.pyc *.o  *.h.gch ../*.o ../*.d *.mod

.PHONY: clean
