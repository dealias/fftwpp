ifndef MPICXX
MPICXX=mpic++
endif

IDIR=../
UDIR=../tests/

CXXFLAGS=$(DEFS)

# GNU compiler
ifeq ($(shell $(MPICXX) -v 2>&1 | tail -n 1 | head -c 3),gcc)
CXXFLAGS += -O3 -g -Wall -ansi -DNDEBUG -fomit-frame-pointer \
	-fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native
#For valgrind:
#CXXFLAGS=-fopenmp -g -Wall -ansi -fomit-frame-pointer -fstrict-aliasing \
#	 -ffast-math -msse2 -mfpmath=sse
ifndef FFTWPP_SINGLE_THREAD
CXXFLAGS += -fopenmp
endif
endif

#Intel compiler
ifeq ($(shell $(MPICXX) --version 2>&1 | head -c 4),icpc)
CXXFLAGS += -O3 -ipo -ansi-alias -malign-double -fp-model fast=2
# note that the -fast option forces static linkage
ifndef FFTWPP_SINGLE_THREAD
CXXFLAGS += -openmp
endif
endif

#IBM compiler
ifeq ($(shell $(MPICXX) -qversion 2>&1 | head -c 3),IBM)
#CXXFLAGS= -O5 -P -qsmp -qalign -qarch -qtune -qcache -qipa -qarch=qp
CXXFLAGS += -O3
ifndef FFTWPP_SINGLE_THREAD
CXXFLAGS += -qsmp
endif
endif

CXXFLAGS += -I$(IDIR) -I$(UDIR) -I$(CPLUS_INCLUDE_PATH) -L$(LIBRARY_PATH)

MAKEDEPEND=$(CXXFLAGS) -O0 -M -DDEPEND

#LDFLAGS = -lfftw3_mpi -lfftw3 -lfftw3_threads -lm
LDFLAGS = -lfftw3_mpi -lfftw3_omp -lfftw3 -lm

vpath %.cc ../

FFTW=fftw++
FILES=cconv2 conv2 cconv3 conv3 cfft2 cfft3
UTILS=$(FFTW) mpiutils mpitranspose
EXTRA=$(UTILS) convolution mpiconvolution
ALL=$(FILES) $(EXTRA)

all: $(FILES)

%.o : %.cc %.h
	$(MPICXX) $(CXXFLAGS) $(OPTS) -o $@ -c $<

cconv2: cconv2.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $(OPTS) $^ $(LDFLAGS) -o $@

conv2: conv2.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $(OPTS) $^ $(LDFLAGS) -o $@

cconv3: cconv3.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $(OPTS) $^ $(LDFLAGS) -o $@

conv3: conv3.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $(OPTS) $^ $(LDFLAGS) -o $@

cfft2: cfft2.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $(OPTS) $^ $(LDFLAGS) -o $@

cfft3: cfft3.o $(EXTRA:=.o) mpiutils.o
	$(MPICXX) $(CXXFLAGS) $(OPTS) $^ $(LDFLAGS) -o $@

testmpi: testmpi.cc $(UTILS:=.o)
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
