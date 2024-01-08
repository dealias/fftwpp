ifndef MPICXX
MPICXX=mpic++
endif

IDIR=../
UDIR=../../tests/

# GNU compiler
ifeq ($(shell $(MPICXX) -v 2>&1 | tail -n 1 | head -c 3),gcc)
CXXFLAGS += -Ofast -g -Wall -DNDEBUG
CXXFLAGS += -fopenmp
CXXFLAGS += -fomit-frame-pointer -fstrict-aliasing -ffast-math
CXXFLAGS += -march=native
#For valgrind:
#CXXFLAGS=-fopenmp -g -Wall -fomit-frame-pointer -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse
endif

#Intel compiler
ifeq ($(shell $(MPICXX) --version 2>&1 | head -c 4),icpc)
CXXFLAGS += -O3 -qopenmp -ipo -ansi-alias -malign-double -fp-model fast=2
LDFLAGS += -lmpi -lmpi_cxx
endif

#IBM compiler
ifeq ($(shell $(MPICXX) -qversion 2>&1 | head -c 3),IBM)
CXXFLAGS += -O5 -qsmp=speculative -qtune=qp -qipa -qarch=qp
endif

CXXFLAGS += $(DEFS) -I$(IDIR) -I$(UDIR) -I../../

ifneq ($(strip $(FFTW_INCLUDE_PATH)),)
CXXFLAGS += -I$(FFTW_INCLUDE_PATH)
endif

ifneq ($(strip $(MPI_INCLUDE_PATH)),)
CXXFLAGS += -I$(MPI_INCLUDE_PATH)
endif

ifneq ($(strip $(FFTW_LIB_PATH)),)
LDFLAGS += -L$(FFTW_LIB_PATH)
endif

ifneq ($(strip $(MPI_LIB_PATH)),)
LDFLAGS += -L$(MPI_LIB_PATH)
endif

LDFLAGS += -lfftw3_omp -lfftw3 -lm

MAKEDEPEND=$(CXXFLAGS) -O0 -M -DDEPEND

vpath %.cc ../:../../:../../tests

FFTW=fftw++ parallel
FILES=gather gatheryz gatherxy transpose fft2 fft3 fft2r fft3r  \
	hybridconv2 hybridconv3 \
	hybridconvh2 hybridconvh3
MPITRANSPOSE=mpitranspose
MPIFFT=$(FFTW) $(MPITRANSPOSE) mpifftw++
MPICONVOLVE=$(MPIFFT) convolve mpiconvolve options

ALL=$(FILES) $(MPICONVOLVE)

all: $(FILES)

%.o : %.cc
	$(MPICXX) $(CXXFLAGS) $(OPTS) -o $@ -c $<

transpose: transpose.o $(MPIFFT:=.o)
	$(MPICXX) $(CXXFLAGS) $(OPTS) $^ $(LDFLAGS) -o $@

gather: gather.o $(MPIFFT:=.o)
	$(MPICXX) $(CXXFLAGS) $(OPTS) $^ $(LDFLAGS) -o $@

gatheryz: gatheryz.o $(MPIFFT:=.o)
	$(MPICXX) $(CXXFLAGS) $(OPTS) $^ $(LDFLAGS) -o $@

gatherxy: gatherxy.o $(MPIFFT:=.o)
	$(MPICXX) $(CXXFLAGS) $(OPTS) $^ $(LDFLAGS) -o $@

fft2: fft2.o $(MPIFFT:=.o)
	$(MPICXX) $(CXXFLAGS) $(OPTS) $^ $(LDFLAGS) -o $@

fft2r: fft2r.o $(MPIFFT:=.o)
	$(MPICXX) $(CXXFLAGS) $(OPTS) $^ $(LDFLAGS) -o $@

fft3: fft3.o $(MPIFFT:=.o)
	$(MPICXX) $(CXXFLAGS) $(OPTS) $^ $(LDFLAGS) -o $@

fft3r: fft3r.o $(MPIFFT:=.o)
	$(MPICXX) $(CXXFLAGS) $(OPTS) $^ $(LDFLAGS) -o $@

hybridconv2: hybridconv2.o $(MPICONVOLVE:=.o)
	$(MPICXX) $(CXXFLAGS) $(OPTS) $^ $(LDFLAGS) -o $@

hybridconvh2: hybridconvh2.o $(MPICONVOLVE:=.o)
	$(MPICXX) $(CXXFLAGS) $(OPTS) $^ $(LDFLAGS) -o $@

hybridconv3: hybridconv3.o $(MPICONVOLVE:=.o)
	$(MPICXX) $(CXXFLAGS) $(OPTS) $^ $(LDFLAGS) -o $@

hybridconvh3: hybridconvh3.o $(MPICONVOLVE:=.o)
	$(MPICXX) $(CXXFLAGS) $(OPTS) $^ $(LDFLAGS) -o $@

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
