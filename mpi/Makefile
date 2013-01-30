IDIR =../
UDIR =../tests/

SHELL=/bin/sh
CFLAGS = -O3

#ifndef CXX
CXX=mpic++
#endif

# GNU compiler
ifeq ($(shell $(CXX) -v 2>&1 | tail -n 1 | head -c 3),gcc)
CFLAGS += -fopenmp -g -Wall -ansi -DNDEBUG -fomit-frame-pointer \
	-fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native
#For valgrind:
#CFLAGS=-fopenmp -g -Wall -ansi -fomit-frame-pointer -fstrict-aliasing \
#	 -ffast-math -msse2 -mfpmath=sse
endif

#Intel compiler
ifeq ($(shell $(CXX) -v 2>&1 | head -c 4),icpc)
CFLAGS += -openmp -ansi-alias -malign-double -fp-model fast=2
endif

#IBM compiler
ifeq ($(shell $(CXX) -qversion 2>&1 | head -c 3),xlc++_r)
#CFLAGS = -O5 -P -qsmp -qalign -qarch -qtune -qcache -qipa -qarch=qp
CFLAGS = -O2 -qsmp
# FIXME
endif

CFLAGS += -I$(IDIR) -I$(UDIR)

MAKEDEPEND=$(CFLAGS) -O0 -M -DDEPEND

#LDFLAGS = -lfftw3_mpi -lfftw3 -lfftw3_threads -lm
LDFLAGS = -lfftw3_mpi -lfftw3_omp -lfftw3 -lm

vpath %.cc ../

FFTW=fftw++
FILES=cconv2 conv2 cconv3 conv3
EXTRA=$(FFTW) convolution mpiconvolution
ALL=$(FILES) $(EXTRA)

all: $(FILES)

%.o : %.cc
	$(CXX) $(CFLAGS) -o $@ -c $<

cconv2: cconv2.o $(EXTRA:=.o)
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

conv2: conv2.o $(EXTRA:=.o)
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

cconv3: cconv3.o $(EXTRA:=.o)
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

conv3: conv3.o $(EXTRA:=.o)
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

testmpi: testmpi.cc
	$(CXX) $^ -lfftw3_mpi -lfftw3 -lm -o $@

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
