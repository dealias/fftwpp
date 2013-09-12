#
# Makefile for FFTW++ wrappers.
#

CXXFLAGS=$(DEFS)

CXXFLAGS=-I.. -msse2 -mfpmath=sse -march=native -fPIC -DNDEBUG
CFLAGS=-I.. -msse2 -mfpmath=sse -march=native -fPIC -DNDEBUG

# GNU compiler
ifeq ($(shell $(CXX) -v 2>&1 | tail -n 1 | head -c 3),gcc)
CXX=g++
CC=gcc
FC=gfortran
CXXFLAGS+= -fopenmp -g -Wall -ansi -DNDEBUG -fomit-frame-pointer \
	-fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native
CFLAGS += -fopenmp
FFLAGS = -fopenmp
endif

#Intel compiler
ifeq ($(shell $(CXX) -v 2>&1 | head -c 4),icpc)
CXX=icpc
CC=icc
FC=ifort
CFLAGS += -openmp -Ofast
CXXFLAGS += -openmp -Ofast
FFLAGS = -openmp
endif

#IBM compiler
ifeq ($(shell $(CXX) -qversion 2>&1 | head -c 3),IBM)
CXX=mpixlcxx_r
CC=mpixlc_r
FC=mpixlf90_r
CXXFLAGS= -O3 -qsmp -I..
CFLAGS= -O3 -qsmp 
FFLGAS=-qsmp
endif

ifeq ($(strip $(CPLUS_INCLUDE_PATH)),)
else
CXXFLAGS +=-L$(CPLUS_INCLUDE_PATH)
endif

ifeq ($(strip $(FORTRAN_INCLUDE_PATH)),)
else
FIFLAGS =-I$(FORTRAN_INCLUDE_PATH)
endif

ifeq ($(strip $(LD_LIBRARY_PATH)),)
else
LDFLAGS +=-L$(LD_LIBRARY_PATH)
endif

LDFLAGS += -lfftw3_omp -lfftw3 -lm -lstdc++

all: _fftwpp.so _chash.so cexample fexample

fftw++.o: ../fftw++.cc ../fftw++.h
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(LDFLAGS)

convolution.o: ../convolution.cc ../convolution.h fftw++.o
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(LDFLAGS)

cfftw++.o: cfftw++.cc cfftw++.h convolution.o fftw++.o
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(LDFLAGS)

chash.o: chash.cc chash.h
	$(CXX)  $(CXXFLAGS) -c -o $@ $< $(LDFLAGS)

cexample: cexample.c cfftw++.o convolution.o fftw++.o chash.o
	$(CC) $(CFLAGS) -std=c99 -o $@ $^ $(LDFLAGS)

fftwpp.o: fftwpp.f90
	$(FC)  $(FFLAGS) -c -o $@ $< $(LDFLAGS)

fhash.o: fhash.f90
	$(FC)  $(FFLAGS) -c -o $@ $< $(LDFLAGS)

fexample: fexample.f90 fftwpp.o fhash.o cfftw++.o convolution.o fftw++.o chash.o
	$(FC) $(FFLAGS) -o $@ $^ $(FIFLAGS) $(LDFLAGS)

_fftwpp.so: fftw++.o cfftw++.o convolution.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ $(LDFLAGS)

_chash.so: chash.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ $(LDFLAGS)

clean:
	rm -f cexample fexample *.so *.pyc *.o *.h.gch *.mod

.PHONY: clean
