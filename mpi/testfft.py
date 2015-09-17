#!/usr/bin/python -u

import sys # so that we can return a value at the end.
import time
from subprocess import * # so that we can run commands

import os.path

retval = 0

msg = "MPI fft unit test"
print msg

ffttestlist = []
ffttestlist.append("testfft2.py")

logfile = 'testfft.log'
print "Log in " + logfile + "\n"
log = open(logfile, 'w')
log.write(msg)
log.write("\n")
log.close()

ntests = 0
nfails = 0
tstart = time.time()
for test in ffttestlist:
    ntests += 1
    if not os.path.isfile(test):
        msg = "Error: "+ pname + "not present!"
        print msg
        log = open(logfile, 'a')
        log.write(msg + "\n")
        log.close()
        retval += 1
    else:
        msg = "Running " + test + ": "
        print(msg),
        log = open(logfile, 'a')
        log.write(msg)
        log.close()
        cmd = []
        cmd.append("./" + test)
        #print cmd
        proc = Popen(cmd, stdout = PIPE, stderr = PIPE)
        proc.wait() # sets the return code
        
        prc = proc.returncode
        out, err = proc.communicate() # capture output
        if (prc == 0): # did the process succeed?
            msg = "\tpass"
            print msg
            log = open(logfile, 'a')
            log.write(msg + "\n")
            log.close()
        else:
            msg = "\tFAILED!"
            print msg
            log = open(logfile, 'a')
            log.write(msg + "\n")
            log.write("stdout:\n" + out + "\n")
            log.write("stderr:\n" + err + "\n")
            log.close()
            retval += 1

print "\n", nfails, "failures out of", ntests, "tests." 

tend = time.time()
print "\nElapsed time (s):", tend - tstart
        
sys.exit(retval)
