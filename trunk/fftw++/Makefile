CC=g++
CFLAGS=-g -Wall -ansi -O3 -DNDEBUG -fomit-frame-pointer -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native
#For valgrind:
#CFLAGS=-g -Wall -ansi -O3 -DNDEBUG -fomit-frame-pointer -fstrict-aliasing -msse2 -mfpmath=sse -march=native

a:=$(shell which icpc 2>&1 | tail -c5)
ifeq ($(a),icpc)
CC=icpc
CFLAGS=-O3 -ansi-alias -malign-double -fp-model fast=2 
endif

MAKEDEPEND=$(CFLAGS) -O0 -M -DDEPEND
LDFLAGS=-lfftw3

FILES=example0 example0r example1 example1r example2 example2r \
example3 example3r conv cconv conv2 cconv2 conv3 cconv3 tconv tconv2
FFTW=fftw++
EXTRA=$(FFTW) convolution
ALL=$(FILES) $(EXTRA)

all: $(FILES)

icpc: all
	CC=icpc

example0: example0.o $(FFTW:=.o)
	$(CC) $(CFLAGS) example0.o $(FFTW:=.o) $(LDFLAGS) -o example0

example0r: example0r.o $(FFTW:=.o)
	$(CC) $(CFLAGS) example0r.o $(FFTW:=.o) $(LDFLAGS) -o example0r

example1: example1.o $(FFTW:=.o)
	$(CC) $(CFLAGS) example1.o $(FFTW:=.o) $(LDFLAGS) -o example1

example1r: example1r.o $(FFTW:=.o)
	$(CC) $(CFLAGS) example1r.o $(FFTW:=.o) $(LDFLAGS) -o example1r

example2: example2.o $(FFTW:=.o)
	$(CC) $(CFLAGS) example2.o $(FFTW:=.o) $(LDFLAGS) -o example2

example2r: example2r.o $(FFTW:=.o)
	$(CC) $(CFLAGS) example2r.o $(FFTW:=.o) $(LDFLAGS) -o example2r

example3: example3.o $(FFTW:=.o)
	$(CC) $(CFLAGS) example3.o $(FFTW:=.o) $(LDFLAGS) -o example3

example3r: example3r.o $(FFTW:=.o)
	$(CC) $(CFLAGS) example3r.o $(FFTW:=.o) $(LDFLAGS) -o example3r

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
	rm -rf $(ALL) $(ALL:=.o) $(ALL:=.d) wisdom3.txt

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
