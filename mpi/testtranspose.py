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

    start=30
    stop=40
    Xlist = [1,2,3,4,5,6,7,8,9,10,random.randint(start,stop)]
    Ylist = [1,2,3,4,5,6,7,8,9,10,random.randint(start,stop)]
    Zlist = [2,3,10,random.randint(start,stop)]
    Plist = [1,2,3,4,5,6,7,8,9]

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
                    for a in range(1,P):
                        if(P % a) :
                            continue
                        ntest += 1
                        for A in range(0,2):
#                            print "Test", ntest, "of", ntests
                            rtest = runtest("transpose", X, Y, P, ["-tq", "-Z"+str(Z), "-a"+str(a), "-A"+str(A)], logfile, timeout)
                            if not rtest == 0:
                                nfails += 1
                
                                end = time.time()
                                print "\nElapsed time (s):", end - start
                                print "\n", nfails, "failures out of", ntests, "tests." 

sys.exit(retval)
