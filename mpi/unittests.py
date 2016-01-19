#!/usr/bin/python

import sys # so that we can return a value at the end.
import time
from subprocess import * # so that we can run commands
import os.path
import getopt
import sys
from testutils import Print

def main(argv):

    msg = "MPI fft unit test"
    logfile = 'unittests.log'
    Print(msg)

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

    if(shortrun):
        Print("Running short tests")
    else:
        Print("Running long tests")

    testlist = []
    testlist.append("testgather.py")
    testlist.append("testtranspose.py")
    testlist.append("testfft.py")
    testlist.append("testconvolution.py")
    
    Print("Log in " + logfile + "\n")
    log = open(logfile, 'w')
    log.write(msg)
    log.write("\n")
    log.close()

    ntests = 0
    nfails = 0
    tstart = time.time()
    for test in testlist:
        ntests += 1
        if not os.path.isfile(test):
            msg = "Error: "+ pname + "not present!"
            Print(msg)
            log = open(logfile, 'a')
            log.write(msg + "\n")
            log.close()
            nfails += 1
        else:
            cmd = []
            cmd.append("./" + test)
            if(shortrun):
                cmd.append("-s")

            msg = "Running " + " ".join(cmd)
            try:
                print(msg),
            except:
                pass
            log = open(logfile, 'a')
            log.write(msg)
            log.close()
            
            DEVNULL = open(os.devnull, 'wb')
            proc=Popen(cmd,stdout=DEVNULL,stderr=PIPE,stdin=DEVNULL)
            proc.wait() # sets the return code

            prc = proc.returncode
            if (prc == 0): # did the process succeed?
                msg = "\t\tpass"
                Print(msg)
                log = open(logfile, 'a')
                log.write(msg + "\n")
                log.close()
            else:
                msg = "\t\tFAILED!" 
                Print(msg)
                log = open(logfile, 'a')
                log.write(msg + "\n")
                log.write("stderr:\n" + err + "\n")
                log.close()
                nfails += 1

    print "\n", nfails, "failures out of", ntests, "tests." 

    tend = time.time()
    print "\nElapsed time (s):", tend - tstart

    sys.exit(nfails)

if __name__ == "__main__":
    main(sys.argv[1:])
