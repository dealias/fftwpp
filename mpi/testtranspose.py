#!/usr/bin/python -u

import sys # so that we can return a value at the end.
import random # for randum number generators
import time
import getopt
import os.path
from testutils import *
from math import sqrt

def main(argv):
    retval = 0
    
    Print("MPI transpose unit test")
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
        Print("Log in " + logfile + "\n")
        log = open(logfile, 'w')
        log.close()

        start=30
        stop=40
        
        if(shortrun):
            Xlist = [2,1,random.randint(start,stop)]
            Ylist = [2,1,random.randint(start,stop)]
            Zlist = [2,1,random.randint(start,stop)]
            Plist = [2,1,3,4]
        else:
            Xlist = [10,9,8,7,6,5,4,3,2,1,random.randint(start,stop)]
            Ylist = [9,10,8,7,6,5,4,3,2,1,random.randint(start,stop)]
            Zlist = [2,1,3,10,random.randint(start,stop)]
            Plist = [2,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16]


        argslist = []
        for X in Xlist:
            for Y in Ylist:
                for Z in Zlist:
                    for P in Plist:
                        for a in range(1,int(sqrt(P)+1.5)):
                            for s in range(0,3):
                                args = []
                                args.append("-x" + str(X))
                                args.append("-y" + str(Y))
                                args.append("-z" + str(Z))
                                args.append("-s" + str(s))
                                args.append("-a" + str(a))
                                args.append("-tq")
                                argslist.append(args)


        Print("Running " + str(len(argslist)) + " tests:")
        tstart = time.time()

        failcases = ""

        # timeout cutoff in seconds (0 disables timeout)
        timeout = 300 
        nfails = 0
                                
        itest = 0
        for args in argslist:
            print "test", itest, "of", len(argslist), ":",
            itest += 1

            rtest, cmd = runtest("transpose", P, args, logfile, timeout)
            if not rtest == 0:
                nfails += 1
                failcases += " ".join(cmd)
                failcases += "\t(code " + str(rtest) + ")"
                failcases += "\n"
                                    
        try:                            
            if nfails > 0:
                print "Failure cases:"
                print failcases
                retval += 1
                print "\n", nfails, "failures out of", len(argslist), "tests." 

                tend = time.time()
                print "\nElapsed time (s):", tend - tstart
        except:
            pass
    
        
    sys.exit(retval)

if __name__ == "__main__":
    main(sys.argv[1:])
