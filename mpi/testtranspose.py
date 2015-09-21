#!/usr/bin/python -u

import sys # so that we can return a value at the end.
import random # for randum number generators
import os.path
import time
import getopt

from testutils import *
from math import sqrt

def main(argv):
    retval = 0
    
    print "MPI transpose unit test"
    usage = "Usage:\n"\
            "./testtranspose.py\n"\
            "\t-s\t\tSpecify a short run\n"\
            "\t-h\t\tShow usage"

    shortrun = False
    try:
        opts, args = getopt.getopt(argv,"sh")
    except getopt.GetoptError:
        print "Error in arguments"
        print usage
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-s"):
            shortrun = True
        if opt in ("-h"):
            print usage
            sys.exit(0)

    
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
        
        if(shortrun):
            Xlist = [1,2,random.randint(start,stop)]
            Ylist = [1,2,random.randint(start,stop)]
            Zlist = [1,2,random.randint(start,stop)]
            Plist = [1,2,3]
        else:
            Xlist = [10,9,8,7,6,5,4,3,2,1,random.randint(start,stop)]
            Ylist = [9,10,8,7,6,5,4,3,2,1,random.randint(start,stop)]
            Zlist = [1,2,3,10,random.randint(start,stop)]
            Plist = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1]

        tstart = time.time()
        
        failcases = ""

        # timeout cutoff in seconds (0 disables timeout)
        timeout = 0 
        ntests = 0
        nfails = 0
        for X in Xlist:
            for Y in Ylist:
                for Z in Zlist:
                    for P in Plist:
                        for a in range(1,int(sqrt(P)+1.5)):
                            for A in range(0,1):
                                ntests += 1
                                args = []
                                args.append("-x" + str(X))
                                args.append("-y" + str(Y))
                                args.append("-z" + str(Z))
                                args.append("-A" + str(A))
                                args.append("-a" + str(a))
                                args.append("-q")
                                args.append("-t")
                                #print "Test", ntest, "of", ntests
                                rtest, cmd = runtest("transpose", P, args,\
                                                logfile, timeout)
                                if not rtest == 0:
                                    nfails += 1
                                    failcases += " ".join(cmd)
                                    failcases += "\t(code " + str(rtest) + ")"
                                    failcases += "\n"
                                    
        tend = time.time()
        print "\nElapsed time (s):", tend - tstart
        
        if nfails > 0:
            print "Failure cases:"
            print failcases
            retval += 1
        print "\n", nfails, "failures out of", ntests, "tests." 

    sys.exit(retval)

if __name__ == "__main__":
    main(sys.argv[1:])
