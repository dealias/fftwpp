IDIR =../../
UDIR =../../tests/

CFLAGS=-O3
LDFLAGS=-lfftw3_mpi -lfftw3 -lm

all: cconv2 cconv3

cconv2:	cconv2.cc
	 mpicxx $(CFLAGS) -o $@ -I$(IDIR) -I$(UDIR) cconv2.cc $(LDFLAGS)

cconv3:	cconv3.cc
	 mpicxx $(CFLAGS) -o $@ -I$(IDIR) -I$(UDIR) cconv3.cc $(LDFLAGS)

clean:
	rm -rf cconv2 cconv3
