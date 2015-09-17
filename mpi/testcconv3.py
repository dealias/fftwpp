#!/usr/bin/python -u

import sys # so that we can return a value at the end.
import random # for randum number generators
import time

import os.path

from testutils import *

retval = 0

print "MPI cconv3 unit test"

pname = "cconv3"

logfile = 'testcconv3.log' 
print "Log in " + logfile + "\n"
log = open(logfile, 'w')
log.close()
        
if not os.path.isfile(pname):
    print "Error: executable", pname, "not present!"
    retval += 1
else:
    Xlist = [1,2,3,4,5,random.randint(6,64)]
    Ylist = [1,2,3,4,5,random.randint(6,64)]
    Xlist = [1,2,3,4,5,random.randint(6,64)]
    Zlist = [1,2,3,4,5,random.randint(6,64)]
    Plist = [1,2,3,4]
    
    timeout = 10

    ntests = 0
    ntests = len(Xlist) * len(Ylist) * len(Zlist) * len(Plist)
    print "Running", ntests, "tests."
    tstart = time.time()
    
    ntest = 0
    nfails = 0
    for P in Plist:
        for X in Xlist:
            for Y in Ylist:
                for Z in Zlist:
                    ntest += 1
                    args = ["-q", "-t","-z" + str(Z)]
                    rtest = runtest(pname, X, Y, P, args, logfile, timeout)
                    if not rtest == 0:
                        nfails += 1
                            
    print "\n", nfails, "failures out of", ntests, "tests." 

    tend = time.time()
    print "\nElapsed time (s):", tend - tstart
        
sys.exit(retval)
