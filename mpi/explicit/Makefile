ifndef MPICXX
MPICXX=mpic++
endif

IDIR =../../
UDIR =../../tests/

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
CXXFLAGS= -O5 -qsmp
endif

CXXFLAGS += -I$(IDIR) -I$(UDIR)

ifneq ($(strip $(FFTW_INCLUDE_PATH)),)
CXXFLAGS += -I$(FFTW_INCLUDE_PATH)
endif

ifneq ($(strip $(MPI_INCLUDE_PATH)),)
CXXFLAGS+=-I$(MPI_INCLUDE_PATH)
endif

#-L$(LIBRARY_PATH)

MAKEDEPEND=$(CXXFLAGS) -O0 -M -DDEPEND

LDFLAGS=

ifneq ($(strip $(FFTW_LIB_PATH)),)
LDFLAGS+=-L$(FFTW_LIB_PATH)
endif

ifneq ($(strip $(MPI_LIB_PATH)),)
LDFLAGS+=-L$(MPI_LIB_PATH)
endif

LDFLAGS+= -lfftw3_mpi -lfftw3_omp -lfftw3 -lm

vpath %.cc ../../


FILES=cconv2 cconv3 cfft2 cfft3
EXTRA=exmpiutils
ALL=$(FILES) $(EXTRA)

all: $(FILES)

cconv2: cconv2.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

cconv3: cconv3.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

cfft2: cfft2.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

cfft3: cfft3.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

.SUFFIXES: .c .cc .o .d

%.o : %.cc %.h
	$(MPICXX) $(CXXFLAGS) -o $@ -c $<

.cc.d:
	@echo Creating $@; \
	rm -f $@; \
	${MPICXX} $(MAKEDEPEND) $(INCL) $< > $@.$$$$ 2>/dev/null && \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$


clean:
	rm -rf $(FILES) *.o *.d

ifeq (,$(findstring clean,${MAKECMDGOALS}))
-include $(FILES:=.d)
endif
