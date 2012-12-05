IDIR =../

CXX=g++
CFLAGS=-fopenmp -g -Wall -ansi -O3 -DNDEBUG -fomit-frame-pointer -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native -I$(IDIR)

#For valgrind:
#CFLAGS=-fopenmp -g -Wall -ansi -O3 -DNDEBUG -fomit-frame-pointer -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -I$(IDIR)

a:=$(shell which icpc 2>&1 | tail -c5)
ifeq ($(a),icpc)
CXX=icpc
CFLAGS=-O3 -ansi-alias -malign-double -fp-model fast=2 -openmp -I$(IDIR)
endif

MAKEDEPEND=$(CFLAGS) -O0 -M -DDEPEND
#LDFLAGS=-lfftw3 -lfftw3_threads -lm
LDFLAGS=-lfftw3 -lfftw3_omp -lm

vpath %.cc ../

FILES=\
example0 example0r example1 example1r example2 example2r example3 example3r \
examplecconv exampleconv examplecconv2 exampleconv2 examplecconv3 exampleconv3

FFTW=fftw++
EXTRA=$(FFTW) convolution
ALL=$(FILES) $(EXTRA)

all: $(FILES)

icpc: all
	CXX=icpc

convolution: convolution convolution.o
	$(CXX) $(CFLAGS) convolution.o $(EXTRA:=.o) $(LDFLAGS)

example0: example0.o $(FFTW:=.o)
	$(CXX) $(CFLAGS) example0.o $(FFTW:=.o) $(LDFLAGS) -o example0

example0r: example0r.o $(FFTW:=.o)
	$(CXX) $(CFLAGS) example0r.o $(FFTW:=.o) $(LDFLAGS) -o example0r

example1: example1.o $(FFTW:=.o)
	$(CXX) $(CFLAGS) example1.o $(FFTW:=.o) $(LDFLAGS) -o example1

example1r: example1r.o $(FFTW:=.o)
	$(CXX) $(CFLAGS) example1r.o $(FFTW:=.o) $(LDFLAGS) -o example1r

example2: example2.o $(FFTW:=.o)
	$(CXX) $(CFLAGS) example2.o $(FFTW:=.o) $(LDFLAGS) -o example2

example2r: example2r.o $(FFTW:=.o)
	$(CXX) $(CFLAGS) example2r.o $(FFTW:=.o) $(LDFLAGS) -o example2r

example3: example3.o $(FFTW:=.o)
	$(CXX) $(CFLAGS) example3.o $(FFTW:=.o) $(LDFLAGS) -o example3

example3r: example3r.o $(FFTW:=.o)
	$(CXX) $(CFLAGS) example3r.o $(FFTW:=.o) $(LDFLAGS) -o example3r

examplecconv: examplecconv.o $(EXTRA:=.o)
	$(CXX) $(CFLAGS) examplecconv.o $(EXTRA:=.o) $(LDFLAGS) -o examplecconv

exampleconv: exampleconv.o $(EXTRA:=.o)
	$(CXX) $(CFLAGS) exampleconv.o $(EXTRA:=.o) $(LDFLAGS) -o exampleconv

examplecconv2: examplecconv2.o $(EXTRA:=.o)
	$(CXX) $(CFLAGS) examplecconv2.o $(EXTRA:=.o) $(LDFLAGS) -o examplecconv2

exampleconv2: exampleconv2.o $(EXTRA:=.o)
	$(CXX) $(CFLAGS) exampleconv2.o $(EXTRA:=.o) $(LDFLAGS) -o exampleconv2

examplecconv3: examplecconv3.o $(EXTRA:=.o)
	$(CXX) $(CFLAGS) examplecconv3.o $(EXTRA:=.o) $(LDFLAGS) -o examplecconv3

exampleconv3: exampleconv3.o $(EXTRA:=.o)
	$(CXX) $(CFLAGS) exampleconv3.o $(EXTRA:=.o) $(LDFLAGS) -o exampleconv3

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
