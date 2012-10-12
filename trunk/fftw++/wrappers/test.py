#!/usr/bin/python

# script for unit testing fftw++ convolution wrappers

import os
import subprocess
import sys

flag=0

command = ["./cexample"]
with open(os.devnull, "w") as fnull:
    cresult = subprocess.call(command, stdout = fnull, stderr = fnull)

if cresult == 0:
    print "cexample succesful"
else:
    print "cexample FAILED"
    flag += 1

command = ["python", "fftwpp.py"]
with open(os.devnull, "w") as fnull:
    presult = subprocess.call(command)

if presult == 0:
    print "fftwpp.py succesful"
else:
    print "fftwpp.py FAILED"
    flag += 1

sys.exit(flag)
