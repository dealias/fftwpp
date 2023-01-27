MPICXX?=mpic++
MPICC?=mpicc

MPIDIR=../
IDIR=../../
WDIR=../../wrappers
TESTDIR=../../tests


# GNU compiler
ifeq ($(shell $(MPICXX) -v 2>&1 | tail -n 1 | head -c 3),gcc)
 CXXFLAGS=-O3 -fopenmp -g -Wall -ansi -DNDEBUG -fomit-frame-pointer \
	-std=c++14
 CFLAGS=-O3 -fopenmp -g -Wall -ansi -DNDEBUG -fomit-frame-pointer \
	-std=c99	
#-fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native \

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

CXXFLAGS += $(DEFS) -I$(MPIDIR) -I$(IDIR) -I$(WDIR) -I$(TESTDIR) -fPIC
CFLAGS += $(DEFS) -I$(MPIDIR) -I$(IDIR) -I$(WDIR) -I$(TESTDIR) -fPIC
ifneq ($(strip $(FFTW_INCLUDE_PATH)),)
CXXFLAGS+=-I$(FFTW_INCLUDE_PATH)
endif

ifneq ($(strip $(MPI_INCLUDE_PATH)),)
CXXFLAGS+=-I$(MPI_INCLUDE_PATH)
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
vpath %.cc ../../wrappers


MPITRANSPOSE=mpitranspose
MPIFFT= $(MPITRANSPOSE) mpifftw++
FILES=cmpifftw++.so _fftwpp.so cexample

all: $(FILES)

%.o : %.cc %.h
	$(MPICXX) -c $(CXXFLAGS) $(OPTS) -o $@ $<

%.o : %.cc
	$(MPICXX) -c $(CXXFLAGS) $(OPTS) -o $@ $<

%.so : %.cc %.h
	$(MPICXX) -shared $(CXXFLAGS) $(OPTS) -o $@ $^

%.so : %.cc
	$(MPICXX) -shared $(CXXFLAGS) $(OPTS) -o $@ $<

cfftw++.so: cfftw++.cc ../../wrappers/cfftw++.h convolution.o fftw++.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ $(LDFLAGS)

# cfftw++.o: cfftw++.cc ../../wrappers/cfftw++.h convolution.o fftw++.o
# 	$(CXX) $(CXXFLAGS) -c -o $@ $^ $(LDFLAGS)

_fftwpp.so: fftw++.o convolution.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ $(LDFLAGS)

mpitranspose.so: ../mpitranspose.cc ../mpitranspose.h
	$(MPICXX) $(CXXFLAGS) -shared -o $@ $^ $(LDFLAGS)

mpifftwpp.so: ../mpifftw++.cc ../mpifftw++.h 
	$(MPICXX) $(CXXFLAGS) -shared -o $@ $^ $(LDFLAGS)

cmpifftwpp.so: cmpifftw++.cc cmpifftw++.h
	$(MPICXX) $(CXXFLAGS) -shared -o $@ $^ $(LDFLAGS)

cexample.so: cexample.c
	$(MPICC) $(CFLAGS) -shared -std=c99 -lstdc++ -o $@ $< $(LDFLAGS)

cexample: cexample.so cmpifftwpp.so mpifftwpp.so mpitranspose.so cfftw++.so _fftwpp.so 
	$(MPICC) $(CFLAGS) -std=c99 -lstdc++ -o $@ $^ $(LDFLAGS)

cmpitranspose.o: cmpitranspose.c cmpitranspose.h
	$(CXX) $(CXXFLAGS) -c -o $@ $^ $(LDFLAGS)


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

clean:
	rm -f ${FILES} *.o *.so

.PHONY: clean

FORCE:
