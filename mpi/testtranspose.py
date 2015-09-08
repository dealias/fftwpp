#!/usr/bin/python -u

import sys # so that we can return a value at the end.
import random # for randum number generators

import os.path

from testutils import *

retval = 0

print "MPI transpose unit test"

if not os.path.isfile("transpose"):
    print "Error: transpose executable not present!"
    retval += 1
else:
    
    logfile = 'testtranspose.log' 
    print "Log in " + logfile + "\n"
    log = open(logfile, 'w')
    log.close()
    
    Xlist = [1,2,3,4,5,random.randint(6,64)]
    Ylist = [1,2,3,4,random.randint(5,64)]
    Plist = [1,2,3,4,random.randint(5,10)]

    print "Performing", len(Xlist) * len(Ylist) * len(Plist), "tests....\n" 
    
    timeout = 0
    ntests = 0
    nfails = 0
    for X in Xlist:
        for Y in Ylist:
            for P in Plist:
                ntests += 1
                rtest = runtest("transpose", X, Y, P, ["-N0"], logfile, timeout)
                if not rtest == 0:
                    nfails += 1
                
    print "\n", nfails, "failures out of", ntests, "tests." 

sys.exit(retval)
