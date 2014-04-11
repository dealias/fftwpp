#!/usr/bin/python -u

# Unit-testing script for numerical accuracy of convolutions implicit
# and explicit convolutions.

# usage: ./errorunittest.py

from subprocess import *
import os
import sys

retval=0

print "Checking error vs direct routine...",
p=Popen(['./directerror.py'],stdout=PIPE,stderr=PIPE)
p.returncode=0
out, err = p.communicate() # capture output
print "...done."
if not (p.returncode == 0):
    retval += 1
    print out
    print
    print err
    print
    print "\tDIRECTERROR FAILED"


print "Checking error vs test case...",
p=Popen(['./error.py'],stdout=PIPE,stderr=PIPE)
p.returncode=0
out, err = p.communicate() # capture output
print "...done."
if not (p.returncode == 0):
    retval += 1
    print out
    print
    print err
    print
    print "\tTEST CASE ERROR FAILED"


print "Running error scaling...",
error1=["cconv","conv"]
for prog in error1:
    print(prog),
    p=Popen(['./error',prog],stdout=PIPE,stderr=PIPE)
    p.returncode=0
    out, err = p.communicate() # capture output
    if not (p.returncode == 0):
        print out
        print
        print err
        print
        retval += 1
        print "\tNUMERICAL ERROR SCALING FAILED FOR "+prog
print "...done."


print "Checking error scaling:",
p=Popen(['./checkerror.py'],stdout=PIPE,stderr=PIPE)
p.returncode=0
out, err = p.communicate() # capture output
if not (p.returncode == 0):
    print out
    print
    print err
    print
    retval += 1
    print "\tSCRIPT checkerror.py FAILED"
else:
    print "OK"


print

if(retval == 0):
    print "OK:\tAll tests passed."
else:
    print "ERROR:\t AT LEAST ONE TEST FAILED"

sys.exit(retval)
