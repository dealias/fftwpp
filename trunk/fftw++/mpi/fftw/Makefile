IDIR =../../
UDIR =../../tests/

CFLAGS=-O3
LDFLAGS=-lfftw3_mpi -lfftw3 -lm


cconv2:
	 mpicxx $(CFLAGS) -o $@ -I$(IDIR) -I$(UDIR) cconv2.cc $(LDFLAGS)

clean:
	rm -rf cconv2
