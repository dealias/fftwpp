IDIR =../

CC=g++
CFLAGS=-fopenmp -g -Wall -ansi -O3 -DNDEBUG -fomit-frame-pointer -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native -I$(IDIR)

#For valgrind:
#CFLAGS=-fopenmp -g -Wall -ansi -O3 -DNDEBUG -fomit-frame-pointer -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -I$(IDIR)

a:=$(shell which icpc 2>&1 | tail -c5)
ifeq ($(a),icpc)
CC=icpc
CFLAGS=-O3 -ansi-alias -malign-double -fp-model fast=2 -openmp -I$(IDIR)
endif

MAKEDEPEND=$(CFLAGS) -O0 -M -DDEPEND
#LDFLAGS=-lfftw3 -lfftw3_threads -lm
LDFLAGS=-lfftw3 -lfftw3_omp -lm

vpath %.cc ../

FFTW=fftw++
FILES=conv cconv conv2 cconv2 conv3 cconv3 tconv tconv2
EXTRA=$(FFTW) convolution explicit direct
ALL=$(FILES) $(EXTRA)

all: $(FILES)

icpc: all
	CC=icpc

convolution: convolution convolution.o
	$(CC) $(CFLAGS) convolution.o $(EXTRA:=.o) $(LDFLAGS)

seconds: seconds seconds.o
	$(CC) $(CFLAGS) seconds.o $(EXTRA:=.o) $(LDFLAGS)

conv: conv.o $(EXTRA:=.o)
	$(CC) $(CFLAGS) conv.o $(EXTRA:=.o) $(LDFLAGS) -o conv

cconv: cconv.o $(EXTRA:=.o)
	$(CC) $(CFLAGS) cconv.o $(EXTRA:=.o) $(LDFLAGS) -o cconv

conv2: conv2.o $(EXTRA:=.o)
	$(CC) $(CFLAGS) conv2.o $(EXTRA:=.o) $(LDFLAGS) -o conv2

cconv2: cconv2.o $(EXTRA:=.o)
	$(CC) $(CFLAGS) cconv2.o $(EXTRA:=.o) $(LDFLAGS) -o cconv2

conv3: conv3.o $(EXTRA:=.o)
	$(CC) $(CFLAGS) conv3.o $(EXTRA:=.o) $(LDFLAGS) -o conv3

cconv3: cconv3.o $(EXTRA:=.o)
	$(CC) $(CFLAGS) cconv3.o $(EXTRA:=.o) $(LDFLAGS) -o cconv3

tconv: tconv.o $(EXTRA:=.o)
	$(CC) $(CFLAGS) tconv.o $(EXTRA:=.o) $(LDFLAGS) -o tconv

tconv2: tconv2.o $(EXTRA:=.o)
	$(CC) $(CFLAGS) tconv2.o $(EXTRA:=.o) $(LDFLAGS) -o tconv2

clean:  FORCE
	rm -rf $(ALL) $(ALL:=.o) $(ALL:=.d)

.SUFFIXES: .c .cc .o .d
.cc.o:
	$(CXX) $(CFLAGS) $(INCL) -o $@ -c $<
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
