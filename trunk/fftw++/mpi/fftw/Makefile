IDIR =../../
UDIR =../../tests/

LDFLAGS=-lfftw3_mpi -lfftw3 -lm

cconv2fftw:
	 mpicxx -o $@ -I$(IDIR) -I$(UDIR) cconv2.cc $(LDFLAGS)

clean:
	rm -rf cconv2fftw
