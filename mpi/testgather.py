#!/usr/bin/python

import sys # so that we can return a value at the end.
import random # for randum number generators
import getopt
import os.path
import sys

from testutils import *

timeout = 300

def main(argv):

    usage = "Usage:\n"\
            "./testfft2.py\n"\
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
            
    retval = 0

    proglist = []
    proglist.append("gather")
    proglist.append("gatheryz")
    proglist.append("gatherxy")

    logfile = 'testgather.log' 
    Print("MPI gather unit test")
    Print("Log in " + logfile + "\n")
    log = open(logfile, 'w')
    log.close()

    Xlist = [1,2,3,4,5,random.randint(10,64)]
    Ylist = [1,2,3,4,5,random.randint(10,64)]
    Zlist = [1,2,random.randint(3,64)]
    Plist = [2,3,4,5,1]
    
    if shortrun:
        Xlist = [2,random.randint(10,64)]
        Ylist = [2,5,random.randint(10,64)]
        Zlist = [1,2,random.randint(3,64)]
        Plist = [2,3,1]
    
    ntests = 0
    nfails = 0
    failcases = ""

    for progname in proglist:
        if not os.path.isfile(progname):
            msg = "Error: executable " + str(progname) + " not present!"
            print msg
            log = open(logfile, 'a')
            log.write(msg)
            log.close()
            retval += 1
        else:
            msg = "Running " + str(progname)
            Print(msg)
            log = open(logfile, 'a')
            log.write(msg)
            log.close()

            testcases = []
            for X in Xlist:
                for Y in Ylist:
                    for Z in Zlist:
                        args = []
                        args.append("-x" + str(X))
                        args.append("-y" + str(Y))
                        args.append("-z" + str(Z))
                        args.append("-q")
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
                rtest, cmd = runtest(progname, P, args, logfile, timeout)
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
            print "\n", nfails, "failures out of", ntest, "tests." 
    except:
        pass

    tend = time.time()
    print "\nElapsed time (s):", tend - tstart

    
if __name__ == "__main__":
    main(sys.argv[1:])

    
