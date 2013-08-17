IDIR =../

CXXFLAGS= -O3

# GNU compiler
ifeq ($(shell $(CXX) -v 2>&1 | tail -n 1 | head -c 3),gcc)
CXXFLAGS+= -fopenmp -g -Wall -ansi -DNDEBUG -fomit-frame-pointer \
	-fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native
#For valgrind:
#CXXFLAGS=-fopenmp -g -Wall -ansi -fomit-frame-pointer -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse
endif

#Intel compiler
ifeq ($(shell $(CXX) -v 2>&1 | head -c 4),icpc)
CXXFLAGS+= -openmp -ansi-alias -malign-double -fp-model fast=2
endif

#IBM compiler
ifeq ($(shell $(CXX) -qversion 2>&1 | head -c 3),IBM)
#CXXFLAGS = -O5 -qsmp -qalign -qarch -qtune -qcache -qipa -qarch=qp
CXXFLAGS= -O3 -qsmp
endif

CXXFLAGS+= -I$(IDIR) -I$(CPLUS_INCLUDE_PATH) -L$(LIBRARY_PATH)

MAKEDEPEND=$(CXXFLAGS) -O0 -M -DDEPEND
#LDFLAGS= -lfftw3_threads -lfftw3 -lm
LDFLAGS= -lfftw3_omp -lfftw3 -lm

vpath %.cc ../

FILES=\
example0 example0r example0m example1 example1r example2 example2r example3 example3r examplecconv exampleconv examplecconv2 exampleconv2 examplecconv3 exampleconv3 \
examplepcconv examplepcconv2 examplepcconv3

FFTW=fftw++
EXTRA=$(FFTW) convolution
ALL=$(FILES) $(EXTRA)

all: $(FILES)

%.o : %.cc %.h
	$(CXX) $(CXXFLAGS) -o $@ -c $<

example0: example0.o $(FFTW:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

example0r: example0r.o $(FFTW:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

example0m: example0m.o $(FFTW:=.o)
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

examplecconv: examplecconv.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

exampleconv: exampleconv.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

examplecconv2: examplecconv2.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

exampleconv2: exampleconv2.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

examplecconv3: examplecconv3.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

exampleconv3: exampleconv3.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

examplepcconv: examplepcconv.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

examplepcconv2: examplepcconv2.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

examplepcconv3: examplepcconv3.o $(EXTRA:=.o)
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
