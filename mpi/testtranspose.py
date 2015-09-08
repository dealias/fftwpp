#!/usr/bin/python -u

import sys # so that we can return a value at the end.
import random # for randum number generators

import os.path

from testutils import *
import time



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

    
    #Xlist = [1,2,3,4,5,random.randint(6,64)]
    #Ylist = [1,2,3,4,random.randint(5,64)]
    #Ylist = [1,2,random.randint(3,64)]
    #Plist = [1,2,3,4,random.randint(5,10)]

    Xlist = [1,2,3,random.randint(6,64)]
    Ylist = [1,2,3,random.randint(5,64)]
    Zlist = [1,random.randint(3,64)]
    Plist = [1,2,3,random.randint(5,10)]

    ntests = len(Xlist) * len(Ylist) * len(Zlist) * len(Plist)
    
    print "Performing", ntests, "tests....\n" 
    start = time.time()
    
    timeout = 0
    ntest = 0
    nfails = 0
    for X in Xlist:
        for Y in Ylist:
            for Z in Zlist:
                for P in Plist:
                    ntest += 1
                    print "Test", ntest, "of", ntests
                    rtest = runtest("transpose", X, Y, P, ["-N0", "-Z"+str(Z), "-t"], logfile, timeout)
                    if not rtest == 0:
                        nfails += 1
                
    end = time.time()
    print "\nElapsed time (s):", end - start
    print "\n", nfails, "failures out of", ntests, "tests." 

sys.exit(retval)
