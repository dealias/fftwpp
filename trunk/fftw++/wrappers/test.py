#!/usr/bin/python

# script for unit testing fftw++ convolution wrappers

import os
import subprocess
import sys

returnflag=0

command = ["./cexample"]
with open(os.devnull, "w") as fnull:
    result = subprocess.call(command, stdout = fnull, stderr = fnull)

if result == 0:
    print "cexample\tok"
else:
    print "cexample\tFAILED"
    returnflag += 1

command = ["python", "fftwpp.py"]
with open(os.devnull, "w") as fnull:
    result = subprocess.call(command)

if result == 0:
    print "fftwpp.py\tok"
else:
    print "fftwpp.py\tFAILED"
    returnflag += 2

command = ["python", "pexample.py"]
with open(os.devnull, "w") as fnull:
    result = subprocess.call(command, stdout = fnull, stderr = fnull)

if result == 0:
    print "pexample.py\tok"
else:
    print "pexample.py\tFAILED"
    returnflag += 4

command = ["./fexample"]
with open(os.devnull, "w") as fnull:
    result = subprocess.call(command, stdout = fnull, stderr = fnull)

if result == 0:
    print "fexample\tok"
else:
    print "fexample\tFAILED"
    returnflag += 8

sys.exit(returnflag)
