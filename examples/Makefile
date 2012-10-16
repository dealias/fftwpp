IDIR =../

CC=g++
CFLAGS=-fopenmp -g -Wall -ansi -O3 -DNDEBUG -fomit-frame-pointer -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native -I$(IDIR)

#For valgrind:
#CFLAGS=-fopenmp -g -Wall -ansi -O3 -DNDEBUG -fomit-frame-pointer -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -I$(IDIR)

a:=$(shell which icpc 2>&1 | tail -c5)
ifeq ($(a),icpc)
CC=icpc
CFLAGS=-O3 -ansi-alias -malign-double -fp-model fast=2 -openmp -I$(IDIR)
endif

MAKEDEPEND=$(CFLAGS) -O0 -M -DDEPEND
#LDFLAGS=-lfftw3 -lfftw3_threads -lm
LDFLAGS=-lfftw3 -lfftw3_omp -lm

vpath %.cc ../

FILES=example0 example0r example1 example1r example2 example2r \
example3 example3r exampleconv

FFTW=fftw++
EXTRA=$(FFTW) convolution
ALL=$(FILES) $(EXTRA)

all: $(FILES)

icpc: all
	CC=icpc

convolution: convolution convolution.o
	$(CC) $(CFLAGS) convolution.o $(EXTRA:=.o) $(LDFLAGS)

example0: example0.o $(FFTW:=.o)
	$(CC) $(CFLAGS) example0.o $(FFTW:=.o) $(LDFLAGS) -o example0

example0r: example0r.o $(FFTW:=.o)
	$(CC) $(CFLAGS) example0r.o $(FFTW:=.o) $(LDFLAGS) -o example0r

example1: example1.o $(FFTW:=.o)
	$(CC) $(CFLAGS) example1.o $(FFTW:=.o) $(LDFLAGS) -o example1

example1r: example1r.o $(FFTW:=.o)
	$(CC) $(CFLAGS) example1r.o $(FFTW:=.o) $(LDFLAGS) -o example1r

example2: example2.o $(FFTW:=.o)
	$(CC) $(CFLAGS) example2.o $(FFTW:=.o) $(LDFLAGS) -o example2

example2r: example2r.o $(FFTW:=.o)
	$(CC) $(CFLAGS) example2r.o $(FFTW:=.o) $(LDFLAGS) -o example2r

example3: example3.o $(FFTW:=.o)
	$(CC) $(CFLAGS) example3.o $(FFTW:=.o) $(LDFLAGS) -o example3

example3r: example3r.o $(FFTW:=.o)
	$(CC) $(CFLAGS) example3r.o $(FFTW:=.o) $(LDFLAGS) -o example3r

exampleconv: exampleconv.o $(EXTRA:=.o)
	$(CC) $(CFLAGS) exampleconv.o $(EXTRA:=.o) $(LDFLAGS) -o exampleconv

clean:  FORCE
	rm -rf $(ALL) $(ALL:=.o) $(ALL:=.d)

.SUFFIXES: .c .cc .o .d
.cc.o:
	$(CXX) $(CFLAGS) $(INCL) -o $@ -c $<
.cc.d:
	@echo Creating $@; \
	rm -f $@; \
	${CXX} $(MAKEDEPEND) $(INCL) $< > $@.$$$$ 2>/dev/null && \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

ifeq (,$(findstring clean,${MAKECMDGOALS}))
-include $(ALL:=.d)
endif

FORCE:
