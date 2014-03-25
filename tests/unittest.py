#!/usr/bin/python -u

# unit-testing script for fftw++ (fftwpp.sf.net).
# usage: ./unittest.py
# runs tests for error propagation in the tests directory

from subprocess import *
import os
import sys

retval=0

print "Checking error vs direct routine...",
p=Popen(['./directerror'],stdout=PIPE,stderr=PIPE)
out, err = p.communicate() # capture output
print "...done."
if not (p.returncode == 0):
    retval += 1
    print out
    print
    print err
    print
    print "\tDIRECTERROR FAILED"

print "Checking error scaling...",
error1=["cconv","conv"]
for prog in error1:
    print(prog),
    p=Popen(['./error',prog],stdout=PIPE,stderr=PIPE)
    p.returncode==0
    out, err = p.communicate() # capture output
    if not (p.returncode == 0):
        print out
        print
        print err
        print
        retval += 1
        print "\tNUMERICAL ERROR SCALING FAILED FOR "+prog
print "...done."

print

if(retval == 0):
    print "OK:\tAll tests passed."
else:
    print "ERROR:\t AT LEAST ONE TEST FAILED"

sys.exit(retval)
