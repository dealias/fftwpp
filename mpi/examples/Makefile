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
endif

#Intel compiler
ifeq ($(shell $(MPICXX) --version 2>&1 | head -c 4),icpc)
CXXFLAGS=-O3 -openmp -ipo -ansi-alias -malign-double -fp-model fast=2
endif

#IBM compiler
ifeq ($(shell $(MPICXX) -qversion 2>&1 | head -c 3),IBM)
CXXFLAGS= -O5 -qsmp=speculative -qtune=qp -qipa -qarch=qp
endif

CXXFLAGS += $(DEFS) -I$(IDIR) -I$(UDIR) -I../../

ifneq ($(strip $(FFTW_INCLUDE_PATH)),)
CXXFLAGS += -I$(FFTW_INCLUDE_PATH)
endif

LDFLAGS=

ifneq ($(strip $(FFTW_LIB_PATH)),)
LDFLAGS+=-L$(FFTW_LIB_PATH)
endif

LDFLAGS+= -lfftw3_omp -lfftw3 -lm

MAKEDEPEND=$(CXXFLAGS) -O0 -M -DDEPEND

vpath %.cc ../:../../
vpath %.cc ../../

FFTW=fftw++ parallel
FILES=fft2rconv fft3rconv
UTILS=$(FFTW)
MPITRANSPOSE=$(UTILS) mpitranspose
MPIFFT=$(MPITRANSPOSE) mpifftw++
MPICONVOLUTION=$(MPIFFT) convolution mpiconvolution

ALL=$(FILES) $(MPICONVOLUTION)

all: $(FILES)

%.o : %.cc
	$(MPICXX) $(CXXFLAGS) $(OPTS) -o $@ -c $<

fft2rconv: fft2rconv.o $(MPIFFT:=.o) $(MPICONVOLUTION:=.o)
	$(MPICXX) $(CXXFLAGS) $(OPTS) $^ $(LDFLAGS) -o $@

fft3rconv: fft3rconv.o $(MPIFFT:=.o) $(MPICONVOLUTION:=.o)
	$(MPICXX) $(CXXFLAGS) $(OPTS) $^ $(LDFLAGS) -o $@

clean:  FORCE
	rm -rf $(ALL) $(ALL:=.o) $(ALL:=.d)

.SUFFIXES: .c .cc .o .d
.cc.o:
	$(MPICXX) $(CXXFLAGS) $(OPTS) $(INCL) -o $@ -c $<
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
