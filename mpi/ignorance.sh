#!/bin/bash

# remove lines relevant to FFTW's MPI transpose routine to work around
# a bug in FFTW's transpose routine causing segfaults when tranposing
# arrays with two points in at least one dimension.

sed -i '/fftw_mpi_transpose/d' wisdom3.txt 
