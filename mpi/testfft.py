#!/usr/bin/python -u

import sys # so that we can return a value at the end.
import random # for randum number generators

import os.path

from testutils import *

retval = 0

print "MPI transpose unit test"

progname = "fft2"

if not os.path.isfile(progname):
    print "Error: executable", progname, "not present!"
    retval += 1
else:
    
    logfile = 'testfft.log' 
    print "Log in " + logfile + "\n"
    log = open(logfile, 'w')
    log.close()
    
    Xlist = [1,2,3,4,5,random.randint(6,64)]
    Ylist = [1,2,3,4,5,random.randint(6,64)]
    Plist = [1,2,3,4]
    timeout = 5
    
    ntests = 0
    nfails = 0
    for X in Xlist:
        for Y in Ylist:
            for P in Plist:
                ntests += 1
                rtest = runtest(progname, X, Y, P, ["-N0","-qt"], logfile, timeout)
                if not rtest == 0:
                    nfails += 1
                
    print "\n", nfails, "failures out of", ntests, "tests." 

sys.exit(retval)
