IDIR =../
UDIR =../tests/

CXX=mpic++
CFLAGS=-fopenmp -g -Wall -ansi -O3 -DNDEBUG -fomit-frame-pointer -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native -I$(IDIR) -I$(UDIR)

#For valgrind:
#CFLAGS=-fopenmp -g -Wall -ansi -fomit-frame-pointer -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -I$(IDIR) -I$(UDIR)

a:=$(shell which icpc 2>&1 | tail -c5)
ifeq ($(a),icpc)
CXX=mpic++
CFLAGS=-openmp -O3 -ansi-alias -malign-double -fp-model fast=2 -I$(IDIR) -I$(UDIR)
endif

MAKEDEPEND=$(CFLAGS) -O0 -M -DDEPEND

#LDFLAGS = -lfftw3_mpi -lfftw3 -lfftw3_threads -lm
LDFLAGS = -lfftw3_mpi -lfftw3 -lfftw3_omp -lm

vpath %.cc ../

FFTW=fftw++
FILES=cconv2 conv2 cconv3 conv3
EXTRA=$(FFTW) convolution mpiconvolution
ALL=$(FILES) $(EXTRA)

all: $(FILES)

icpc: all
	CXX=icpc

convolution: convolution convolution.o
	$(CXX) $(CFLAGS) convolution.o $(EXTRA:=.o) $(LDFLAGS)

mpiconvolution: mpiconvolution mpiconvolution.o
	$(CXX) $(CFLAGS) mpiconvolution.o $(EXTRA:=.o) $(LDFLAGS)

seconds: seconds seconds.o
	$(CXX) $(CFLAGS) seconds.o $(EXTRA:=.o) $(LDFLAGS)

cconv2: cconv2.o $(EXTRA:=.o)
	$(CXX) $(CFLAGS) cconv2.o $(EXTRA:=.o) $(LDFLAGS) -o cconv2

conv2: conv2.o $(EXTRA:=.o)
	$(CXX) $(CFLAGS) conv2.o $(EXTRA:=.o) $(LDFLAGS) -o conv2

cconv3: cconv3.o $(EXTRA:=.o)
	$(CXX) $(CFLAGS) cconv3.o $(EXTRA:=.o) $(LDFLAGS) -o cconv3

conv3: conv3.o $(EXTRA:=.o)
	$(CXX) $(CFLAGS) conv3.o $(EXTRA:=.o) $(LDFLAGS) -o conv3

testmpi: testmpi.cc
	$(CXX) $^ -lfftw3_mpi -lfftw3 -lm -o testmpi

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
