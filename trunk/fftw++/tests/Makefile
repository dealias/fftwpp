IDIR =../

SHELL=/bin/sh

CFLAGS = -O3

# GNU compiler
ifeq ($(shell $(CXX) -v 2>&1 | tail -n 1 | head -c 3),gcc)
CFLAGS += -fopenmp -g -Wall -ansi -DNDEBUG -fomit-frame-pointer \
	-fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native
#For valgrind:
#CFLAGS=-fopenmp -g -Wall -ansi -fomit-frame-pointer -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse
endif

#Intel compiler
ifeq ($(shell $(CXX) -v 2>&1 | head -c 4),icpc)
CFLAGS += -openmp -ansi-alias -malign-double -fp-model fast=2
endif

#IBM compiler
ifeq ($(shell $(CXX) -qversion 2>&1 | head -c 3),xlc++_r)
#CFLAGS = -O5 -P -qsmp -qalign -qarch -qtune -qcache -qipa -qarch=qp
CFLAGS = -O2 -qsmp
endif

CFLAGS += -I$(IDIR)

MAKEDEPEND=$(CFLAGS) -O0 -M -DDEPEND

#LDFLAGS = -lfftw3 -lfftw3_threads -lm
LDFLAGS = -lfftw3_omp -lfftw3 -lm

vpath %.cc ../

FFTW=fftw++
FILES=conv cconv conv2 cconv2 conv3 cconv3 tconv tconv2
EXTRA=$(FFTW) convolution explicit direct
ALL=$(FILES) $(EXTRA)

all: $(FILES)

%.o : %.cc
	$(CXX) $(CFLAGS) -o $@ -c $<

conv: conv.o $(EXTRA:=.o)
	$(CXX) $(CFLAGS) conv.o $(EXTRA:=.o) $(LDFLAGS) -o conv

cconv: cconv.o $(EXTRA:=.o)
	$(CXX) $(CFLAGS) cconv.o $(EXTRA:=.o) $(LDFLAGS) -o cconv

conv2: conv2.o $(EXTRA:=.o)
	$(CXX) $(CFLAGS) conv2.o $(EXTRA:=.o) $(LDFLAGS) -o conv2

cconv2: cconv2.o $(EXTRA:=.o)
	$(CXX) $(CFLAGS) cconv2.o $(EXTRA:=.o) $(LDFLAGS) -o cconv2

conv3: conv3.o $(EXTRA:=.o)
	$(CXX) $(CFLAGS) conv3.o $(EXTRA:=.o) $(LDFLAGS) -o conv3

cconv3: cconv3.o $(EXTRA:=.o)
	$(CXX) $(CFLAGS) cconv3.o $(EXTRA:=.o) $(LDFLAGS) -o cconv3

tconv: tconv.o $(EXTRA:=.o)
	$(CXX) $(CFLAGS) tconv.o $(EXTRA:=.o) $(LDFLAGS) -o tconv

tconv2: tconv2.o $(EXTRA:=.o)
	$(CXX) $(CFLAGS) tconv2.o $(EXTRA:=.o) $(LDFLAGS) -o tconv2

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
