IDIR =../

# GNU compiler
ifeq ($(shell $(CXX) -v 2>&1 | tail -n 1 | head -c 3),gcc)
CXXFLAGS=-O3 -fopenmp -g -Wall -ansi -DNDEBUG -fomit-frame-pointer \
	-fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native
#For valgrind:
#CXXFLAGS=-fopenmp -g -Wall -ansi -fomit-frame-pointer -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse
endif

# clang compiler
ifeq ($(shell $(CXX) -v 2>&1 | head -c 10),"Apple LLVM")
CXXFLAGS=-O3 -fopenmp -g -Wall -ansi -DNDEBUG -fomit-frame-pointer \
	-fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native
#For valgrind:
#CXXFLAGS=-fopenmp -g -Wall -ansi -fomit-frame-pointer -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse
endif

#Intel compiler
ifeq ($(shell $(CXX) -v 2>&1 | head -c 4),icpc)
CXXFLAGS=-O3 -openmp -ansi-alias -malign-double -fp-model fast=2
endif

#IBM compiler
ifeq ($(shell $(CXX) -qversion 2>&1 | head -c 3),IBM)
CXXFLAGS=-O5 -P -qsmp -qalign -qarch -qtune -qcache -qipa -qarch=qp
endif

CXXFLAGS += $(DEFS) -I$(IDIR) -std=c++11

ifneq ($(strip $(FFTW_INCLUDE_PATH)),)
CXXFLAGS+=-I$(FFTW_INCLUDE_PATH)
endif

ifneq ($(strip $(FFTW_LIB_PATH)),)
LDFLAGS+=-L$(FFTW_LIB_PATH)
endif

#LDFLAGS=-lfftw3_threads -lfftw3 -lm
LDFLAGS+=-lfftw3_omp -lfftw3 -lm

MAKEDEPEND=$(CXXFLAGS) -O0 -M -DDEPEND

vpath %.cc ../

FILES=example0 example0r example0m example0mr \
	example1 example1r example2 example2r example3 example3r \
	exampleconv exampleconv2 exampleconv3 \
	exampleconvh exampleconvh2 exampleconvh3 \
	exampleconvr exampleconvr2 exampleconvr3 \
	exampletranspose optimal

FFTW=fftw++ parallel
HYBRID=$(FFTW) convolve
ALL=$(FILES) $(HYBRID)

all: $(FILES)

%.o : %.cc %.h
	$(CXX) $(CXXFLAGS) -o $@ -c $<

example0: example0.o $(FFTW:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

example0r: example0r.o $(FFTW:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

example0m: example0m.o $(FFTW:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

example0mr: example0mr.o $(FFTW:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

example1: example1.o $(FFTW:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

example1r: example1r.o $(FFTW:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

example2: example2.o $(FFTW:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

example2r: example2r.o $(FFTW:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

example3: example3.o $(FFTW:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

example3r: example3r.o $(FFTW:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

exampletranspose: exampletranspose.o $(FFTW:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

exampleconv: exampleconv.o $(HYBRID:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

exampleconvh: exampleconvh.o $(HYBRID:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

exampleconvr: exampleconvr.o $(HYBRID:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

exampleconv2: exampleconv2.o $(HYBRID:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

exampleconvh2: exampleconvh2.o $(HYBRID:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

exampleconvr2: exampleconvr2.o $(HYBRID:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

exampleconv3: exampleconv3.o $(HYBRID:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

exampleconvh3: exampleconvh3.o $(HYBRID:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

exampleconvr3: exampleconvr3.o $(HYBRID:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

optimal: optimal.o $(FFTW:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:  FORCE
	rm -rf $(ALL) $(ALL:=.o) $(ALL:=.d)

.SUFFIXES: .c .cc .o .d

.cc.o:
	$(CXX) $(CXXFLAGS) $(INCL) -o $@ -c $<
.cc.d:
	@echo Creating $@; \
	rm -f $@; \
	${CXX} $(MAKEDEPEND) $(INCL) $< > $@.$$$$ 2>/dev/null && \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

ifeq (,$(findstring clean,${MAKECMDGOALS}))
-include $(ALL:=.d)
endif

FORCE:
