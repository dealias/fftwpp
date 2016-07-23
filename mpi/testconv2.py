#!/usr/bin/python

import sys # so that we can return a value at the end.
import random # for randum number generators
import time
import getopt
import os.path
from testutils import *

pname = "conv2"
timeout = 300 # cutoff time in seconds

def main(argv):
    print "MPI conv2 unit test"
    retval = 0
    usage = "Usage:\n"\
            "./testconv2.py\n"\
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

   
    logfile = 'testconv2.log' 
    print "Log in " + logfile + "\n"
    log = open(logfile, 'w')
    log.close()

    if not os.path.isfile(pname):
        print "Error: executable", pname, "not present!"
        retval += 1
    else:
        start=30
        stop=40

        Alist = [2,4]
        
        xlist = [2,1,3,4,5,6,7,8,9,10,random.randint(start,stop)]
        ylist = [2,1,3,4,5,6,7,8,9,10,random.randint(start,stop)]
        Plist = [8,4,3,2,random.randint(9,12),1]
        Tlist = [1,2,random.randint(3,5)]

        if(shortrun):
            print "Short run."
            xlist = [2,3,random.randint(6,64)]
            ylist = [2,3,random.randint(6,64)]
            Plist = [1,2]
            Tlist = [1,2]
            
        testcases = []
        for x in xlist:
            for y in ylist:
                for X in range(0,2):
                    for Y in range(0,2):
                        for A in Alist:
                            for T in Tlist:
                                args = []
                                args.append("-x" + str(x))
                                args.append("-y" + str(y))
                                args.append("-N1")
                                args.append("-s1")
                                args.append("-a1")
                                args.append("-X"+str(X))
                                args.append("-Y"+str(Y))
                                args.append("-A"+str(A))
                                args.append("-T" + str(T))
                                args.append("-tq")
                                testcases.append(args)

        tstart = time.time()
        ntest = len(testcases)*len(Plist)
        print "Running", ntest, "tests."

        failcases = ""
        nfails = 0

        itest = 0
        
        for P in Plist:
            for args in testcases:
                print "test", itest, "of", ntest, ":",
                itest += 1
                rtest, cmd = runtest(pname, P, args, logfile, timeout)
                if not rtest == 0:
                    nfails += 1
                    failcases += " ".join(cmd)
                    failcases += "\t(code " + str(rtest) + ")"
                    failcases += "\n"

        if nfails > 0:
            print "Failure cases:"
            print failcases
            retval += 1
        print "\n", nfails, "failures out of", ntest, "tests." 

        tend = time.time()
        print "\nElapsed time (s):", tend - tstart

    sys.exit(retval)

if __name__ == "__main__":
    main(sys.argv[1:])
