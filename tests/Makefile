IDIR =../

# GNU compiler
ifeq ($(shell $(CXX) -v 2>&1 | tail -n 1 | head -c 3),gcc)
CXXFLAGS+=-Ofast -g -Wall -ansi -DNDEBUG
CXXFLAGS+=-fopenmp
CXXFLAGS+=-fomit-frame-pointer -fstrict-aliasing -ffast-math
CXXFLAGS+=-msse2 -mfpmath=sse -march=native
#CXXFLAGS+= -flto 
#CXXFLAGS+= -fprofile-generate 
#CXXFLAGS+= -fprofile-use -fprofile-dir=.
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

CXXFLAGS += $(DEFS) -I$(IDIR)

ifneq ($(strip $(FFTW_INCLUDE_PATH)),)
CXXFLAGS+=-I$(FFTW_INCLUDE_PATH)
endif

LDFLAGS=
ifneq ($(strip $(FFTW_LIB_PATH)),)
LDFLAGS+=-L$(FFTW_LIB_PATH)
endif
#LDFLAGS+=-lfftw3_threads -lfftw3 -lm
LDFLAGS+=-lfftw3_omp -lfftw3 -lm

MAKEDEPEND=$(CXXFLAGS) -O0 -M -DDEPEND

vpath %.cc ../

FILES=conv cconv conv2 cconv2 conv3 cconv3 tconv tconv2 \
	fft1 fft2 fft3 fft1r fft2r fft3r mfft1 mfft1r transpose

FFTW=fftw++
EXTRA=$(FFTW) convolution explicit direct
ALL=$(FILES) $(EXTRA)

all: $(FILES)

%.o : %.cc
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

fft1: fft1.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

fft2: fft2.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

fft3: fft3.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

fft1r: fft1r.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

fft2r: fft2r.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

fft3r: fft3r.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

mfft1: mfft1.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

mfft1r: mfft1r.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

transpose: transpose.o $(EXTRA:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@


.PHONY: clean
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
