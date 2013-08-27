IDIR =../

CXXFLAGS= -O3 $(DEFS)

# GNU compiler
ifeq ($(shell $(CXX) -v 2>&1 | tail -n 1 | head -c 3),gcc)
CXXFLAGS += -fopenmp -g -Wall -ansi -DNDEBUG -fomit-frame-pointer \
	-fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native
#For valgrind:
#CXXFLAGS=-fopenmp -g -Wall -ansi -fomit-frame-pointer -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse
endif

#Intel compiler
ifeq ($(shell $(CXX) -v 2>&1 | head -c 4),icpc)
CXXFLAGS += -openmp -ansi-alias -malign-double -fp-model fast=2
endif

#IBM compiler
ifeq ($(shell $(CXX) -qversion 2>&1 | head -c 3),IBM)
#CXXFLAGS= -O5 -P -qsmp -qalign -qarch -qtune -qcache -qipa -qarch=qp
CXXFLAGS= -O3 -qsmp
endif

CXXFLAGS += -I$(IDIR) -I$(CPLUS_INCLUDE_PATH) -L$(LIBRARY_PATH)

MAKEDEPEND=$(CXXFLAGS) -O0 -M -DDEPEND

#LDFLAGS = -lfftw3 -lfftw3_threads -lm
LDFLAGS = -lfftw3_omp -lfftw3 -lm

vpath %.cc ../
vpath %.h ../

FFTW=fftw++
FILES=conv cconv conv2 cconv2 conv3 cconv3 tconv tconv2 pcconv pcconv2
EXTRA=$(FFTW) convolution explicit direct
ALL=$(FILES) $(EXTRA)

all: $(FILES)

%.o : %.cc %.h
	$(CXX) $(CXXFLAGS) -o $@ -c $<

conv: conv.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

cconv: cconv.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

conv2: conv2.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

cconv2: cconv2.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

conv3: conv3.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

cconv3: cconv3.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

tconv: tconv.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

tconv2: tconv2.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

pcconv: pcconv.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

pcconv2: pcconv2.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

pcconv3: pcconv3.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:  FORCE
	rm -rf $(ALL) $(ALL:=.o) $(ALL:=.d)

.SUFFIXES: .c .cc .o .d

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
