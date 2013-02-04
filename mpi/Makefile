ifndef MPICXX
MPICXX=mpic++
endif

IDIR =../
UDIR =../tests/

CXXFLAGS= -O3

# GNU compiler
ifeq ($(shell $(MPICXX) -v 2>&1 | tail -n 1 | head -c 3),gcc)
CXXFLAGS += -fopenmp -g -Wall -ansi -DNDEBUG -fomit-frame-pointer \
	-fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native
#For valgrind:
#CXXFLAGS=-fopenmp -g -Wall -ansi -fomit-frame-pointer -fstrict-aliasing \
#	 -ffast-math -msse2 -mfpmath=sse
endif

#Intel compiler
ifeq ($(shell $(MPICXX) -v 2>&1 | head -c 4),icpc)
CXXFLAGS += -openmp -ansi-alias -malign-double -fp-model fast=2
endif

#IBM compiler
ifeq ($(shell $(MPICXX) -qversion 2>&1 | head -c 3),IBM)
#CXXFLAGS= -O5 -P -qsmp -qalign -qarch -qtune -qcache -qipa -qarch=qp
CXXFLAGS= -O3 -qsmp
endif

CXXFLAGS += -I$(IDIR) -I$(UDIR) -I$(CPLUS_INCLUDE_PATH) -L$(LIBRARY_PATH)

MAKEDEPEND=$(CXXFLAGS) -O0 -M -DDEPEND

#LDFLAGS = -lfftw3_mpi -lfftw3 -lfftw3_threads -lm
LDFLAGS = -lfftw3_mpi -lfftw3_omp -lfftw3 -lm

vpath %.cc ../

FFTW=fftw++
FILES=cconv2 conv2 cconv3 conv3
EXTRA=$(FFTW) convolution mpiconvolution
ALL=$(FILES) $(EXTRA)

all: $(FILES)

%.o : %.cc
	$(MPICXX) $(CXXFLAGS) -o $@ -c $<

cconv2: cconv2.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

conv2: conv2.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

cconv3: cconv3.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

conv3: conv3.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

testmpi: testmpi.cc
	$(MPICXX) $^ -lfftw3_mpi -lfftw3 -lm -o $@

clean:  FORCE
	rm -rf $(ALL) $(ALL:=.o) $(ALL:=.d)

.SUFFIXES: .c .cc .o .d
.cc.o:
	$(MPICXX) $(CXXFLAGS) $(INCL) -o $@ -c $<
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
