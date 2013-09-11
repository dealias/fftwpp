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

CXXFLAGS += -I$(IDIR) -I$(UDIR) -I$(CPLUS_INCLUDE_PATH) -L$(LIBRARY_PATH)

LDFLAGS = -lfftw3_mpi -lfftw3_omp -lfftw3 -lm

vpath %.cc ../../

EXTRA=exmpiutils

FILES=cconv2 cconv3 cfft2

all: $(FILES)

%.o : %.cc %.h
	$(MPICXX) $(CXXFLAGS) $(OPTS) -o $@ -c $<

cconv2: cconv2.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

cconv3: cconv3.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

conv2: conv2.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

cfft2: cfft2.o $(EXTRA:=.o)
	$(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -rf $(FILES) *.o *.d


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
