ifndef MPICXX
MPICXX=mpic++
endif

MPIDIR=../
IDIR=../../
UDIR=../../tests/

# GNU compiler
ifeq ($(shell $(MPICXX) -v 2>&1 | tail -n 1 | head -c 3),gcc)
 CXXFLAGS=-O3 -fopenmp -g -Wall -ansi -DNDEBUG -fomit-frame-pointer \
	-fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native
#For valgrind:
# CXXFLAGS=-fopenmp -g -Wall -ansi -fomit-frame-pointer -fstrict-aliasing \
# 	 -ffast-math -msse2 -mfpmath=sse
endif

#Intel compiler
ifeq ($(shell $(MPICXX) --version 2>&1 | head -c 4),icpc)
CXXFLAGS=-O3 -openmp -ipo -ansi-alias -malign-double -fp-model fast=2
# note that the -fast option forces static linkage
endif

#IBM compiler
ifeq ($(shell $(MPICXX) -qversion 2>&1 | head -c 3),IBM)
CXXFLAGS= -O5 -qsmp=speculative -qtune=qp -qipa -qarch=qp
#CXXFLAGS= -O5 -P -qsmp -qalign -qarch -qtune -qcache -qipa -qarch=qp
endif

CXXFLAGS += $(DEFS)  -I$(MPIDIR) -I$(IDIR) -I$(UDIR)

ifneq ($(strip $(FFTW_INCLUDE_PATH)),)
CXXFLAGS+=-I$(FFTW_INCLUDE_PATH)
endif

LDFLAGS=
ifneq ($(strip $(FFTW_LIB_PATH)),)
LDFLAGS+=-L$(FFTW_LIB_PATH)
endif
#LDFLAGS= -lfftw3_mpi -lfftw3 -lfftw3_threads -lm
LDFLAGS+= -lfftw3_omp -lfftw3 -lm

MAKEDEPEND=$(CXXFLAGS) -O0 -M -DDEPEND

vpath %.cc ../
vpath %.cc ../../

FFTW=fftw++
FILES= fft2rconv fft3rconv
UTILS=$(FFTW)
MPITRANSPOSE=$(UTILS) mpitranspose
MPIFFT=$(MPITRANSPOSE) mpifftw++
MPICONVOLUTION=$(MPIFFT) convolution mpiconvolution

ALL=$(FILES) $(MPICONVOLUTION)

all: $(FILES)

%.o : %.cc %.h
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
