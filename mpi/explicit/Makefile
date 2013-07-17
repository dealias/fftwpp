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

FILES=cconv2 cconv3 conv2

all: $(FILES)

cconv2: cconv2.cc
	$(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

cconv3: cconv3.cc
	$(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

conv2: conv2.cc
	$(MPICXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -rf cconv2 cconv3

