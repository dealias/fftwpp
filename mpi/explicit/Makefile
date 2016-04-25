ifndef MPICXX
MPICXX=mpic++
endif

IDIR=../../
UDIR=../../tests/

# GNU compiler
ifeq ($(shell $(MPICXX) -v 2>&1 | tail -n 1 | head -c 3),gcc)
CXXFLAGS=-O3 -fopenmp -g -Wall -ansi -DNDEBUG -fomit-frame-pointer \
	-fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native
#For valgrind:
#CXXFLAGS=-fopenmp -g -Wall -ansi -fomit-frame-pointer -fstrict-aliasing \
#	 -ffast-math -msse2 -mfpmath=sse
endif

#Intel compiler
ifeq ($(shell $(MPICXX) --version 2>&1 | head -c 4),icpc)
CXXFLAGS=-O3 -openmp -ipo -ansi-alias -malign-double -fp-model fast=2
endif

#IBM compiler
ifeq ($(shell $(MPICXX) -qversion 2>&1 | head -c 3),IBM)
CXXFLAGS= -O5 -qsmp=speculative -qtune=qp -qipa -qarch=qp
endif

CXXFLAGS += $(DEFS) -I$(IDIR) -I$(UDIR)

ifneq ($(strip $(FFTW_INCLUDE_PATH)),)
CXXFLAGS += -I$(FFTW_INCLUDE_PATH)
endif

ifneq ($(strip $(MPI_INCLUDE_PATH)),)
CXXFLAGS+=-I$(MPI_INCLUDE_PATH)
endif

LDFLAGS=

ifneq ($(strip $(FFTW_LIB_PATH)),)
LDFLAGS+=-L$(FFTW_LIB_PATH)
endif

ifneq ($(strip $(MPI_LIB_PATH)),)
LDFLAGS+=-L$(MPI_LIB_PATH)
endif

LDFLAGS+= -lfftw3_mpi -lfftw3_omp -lfftw3 -lm

MAKEDEPEND=$(CXXFLAGS) -O0 -M -DDEPEND

vpath %.cc ../../


FILES=fft2 fft3 fft2r fft3r cconv2 cconv3 transpose
EXTRA=exmpiutils
ALL=$(FILES) $(EXTRA)

all: $(FILES)

%.o : %.cc
	$(MPICXX) $(CXXFLAGS) $(OPTS) -o $@ -c $<

fft2: fft2.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

fft3: fft3.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

fft2r: fft2r.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

fft3r: fft3r.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

cconv2: cconv2.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

cconv3: cconv3.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

transpose: transpose.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:  FORCE
	rm -rf $(ALL) $(ALL:=.o) $(ALL:=.d)

.SUFFIXES: .c .cc .o .d

%.o : %.cc %.h
	$(MPICXX) $(CXXFLAGS) $(OPTS) -o $@ -c $<

%.o : %.cc
	$(MPICXX) $(CXXFLAGS) $(OPTS) -o $@ -c $<

.cc.d:
	@echo Creating $@; \
	rm -f $@; \
	${MPICXX} $(MAKEDEPEND) $(INCL) $< > $@.$$$$ 2>/dev/null && \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

ifeq (,$(findstring clean,${MAKECMDGOALS}))
-include $(ALL:=.d)
endif

FORCE:
