IDIR =../
LDFLAGS=
# GNU compiler
ifeq ($(shell $(CXX) -v 2>&1 | tail -n 1 | head -c 3),gcc)
CXXFLAGS += -Ofast -g -Wall -DNDEBUG
CXXFLAGS += -fopenmp
CXXFLAGS += -fomit-frame-pointer -fstrict-aliasing -ffast-math
CXXFLAGS += -march=native
#CXXFLAGS += -flto
#CXXFLAGS += -fprofile-generate
#CXXFLAGS += -fprofile-use -fprofile-dir=.
#For valgrind:
#CXXFLAGS=-fopenmp -g -Wall -fomit-frame-pointer -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse
endif

#Clang compiler
ifeq ($(shell $(CXX) -v 2>&1 | head -n 1 | head -c 5),clang)
CXXFLAGS += -Ofast -g -Wall -DNDEBUG
CXXFLAGS += -fomit-frame-pointer -fstrict-aliasing -ffast-math
CXXFLAGS += -msse2 -mfpmath=sse -march=native
#Thread nonsense
#CXXFLAGS += -I/opt/local/include/libomp -Xpreprocessor -fopenmp
endif

#Apple Clang compiler
ifeq ($(shell $(CXX) -v 2>&1 | head -n 1 | head -c 11),Apple clang)
CXXFLAGS += -Ofast -g -Wall -DNDEBUG
CXXFLAGS += -fomit-frame-pointer -fstrict-aliasing -ffast-math
#CXXFLAGS += -msse2 -mfpmath=sse -march=native
CXXFLAGS +=-I/opt/local/include/libomp -Xpreprocessor -fopenmp
LDFLAGS +=-L/opt/local/lib -L/opt/local/lib/libomp -lomp
endif

#Intel compiler
ifeq ($(shell $(CXX) --version 2>&1 | head -c 4),icpc)
CXXFLAGS += -O3 -qopenmp -ansi-alias -malign-double -fp-model fast=2
LDFLAGS += -lmpi -lmpi_cxx
endif

#IBM compiler
ifeq ($(shell $(CXX) -qversion 2>&1 | head -c 3),IBM)
CXXFLAGS += -O5 -P -qsmp -qalign -qarch -qtune -qcache -qipa -qarch=qp
endif

CXXFLAGS += $(DEFS) -I$(IDIR) -std=c++11

ifneq ($(strip $(FFTW_INCLUDE_PATH)),)
CXXFLAGS+=-I$(FFTW_INCLUDE_PATH)
endif


ifneq ($(strip $(FFTW_LIB_PATH)),)
LDFLAGS += -L$(FFTW_LIB_PATH)
endif
#LDFLAGS += -lfftw3_threads -lfftw3 -lm
LDFLAGS += -lfftw3_omp -lfftw3 -lm

MAKEDEPEND=$(CXXFLAGS) -O0 -M -DDEPEND

vpath %.cc ../

FILES=fft1 fft2 fft3 fft1r fft2r fft3r mfft1 mfft1r transpose directTest\
	hybrid hybridh hybridr hybridconv hybridconvh hybridconvr \
	hybridconv2 hybridconvh2 hybridconvr2 \
	hybridconv3 hybridconvh3 hybridconvr3

FFTW=fftw++ parallel
HYBRID=$(FFTW) convolve direct options
ALL=$(FILES) $(HYBRID)

all: $(FILES)

%.o : %.cc
	$(CXX) $(CXXFLAGS) -o $@ -c $<

hybrid: hybrid.o $(HYBRID:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

hybridh: hybridh.o $(HYBRID:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

hybridr: hybridr.o $(HYBRID:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

hybridconv: hybridconv.o $(HYBRID:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

hybridconvh: hybridconvh.o $(HYBRID:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

hybridconvr: hybridconvr.o $(HYBRID:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

hybridconv2: hybridconv2.o $(HYBRID:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

hybridconvh2: hybridconvh2.o $(HYBRID:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

hybridconvr2: hybridconvr2.o $(HYBRID:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

hybridconv3: hybridconv3.o $(HYBRID:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

hybridconvh3: hybridconvh3.o $(HYBRID:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

hybridconvr3: hybridconvr3.o $(HYBRID:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

directTest: directTest.o $(HYBRID:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

fft1: fft1.o $(FFTW:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

fft2: fft2.o $(FFTW:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

fft3: fft3.o $(FFTW:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

fft1r: fft1r.o $(FFTW:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

fft2r: fft2r.o $(FFTW:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

fft3r: fft3r.o $(FFTW:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

mfft1: mfft1.o $(FFTW:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

mfft1r: mfft1r.o $(FFTW:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

transpose: transpose.o $(FFTW:=.o)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

.PHONY: clean
clean:  FORCE
	rm -rf $(ALL) $(FFTW) $(HYBRID) $(ALL:=.o) $(ALL:=.d)

.SUFFIXES: .c .cc .o .d
%.o: %.cc
	$(CXX) $(CXXFLAGS) -o $@ -c $<

%.d: %.cc
	@echo Creating $@; \
	rm -f $@; \
	${CXX} $(MAKEDEPEND) $(INCL) $< > $@.$$$$ 2>/dev/null && \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

ifeq (,$(findstring clean,${MAKECMDGOALS}))
-include $(ALL:=.d)
endif

FORCE:
