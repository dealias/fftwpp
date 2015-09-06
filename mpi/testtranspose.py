#!/usr/bin/python -u

import sys # so that we can return a value at the end.
from subprocess import * # so that we can run commands
import random # for randum number generators

import time
import os.path

retval = 0

print "MPI transpose unit test"

def waitandkill(proc, timeout):
    # Check every dt seconds to see if it's done.
    dt = timeout / 1000.0
    t = 0
    while t < timeout:
        time.sleep(dt)
        retcode = proc.poll()
        if(retcode != None):
            return 0
        t += dt
    retcode = proc.poll()
    if(retcode == None):
        proc.kill()
        return 1
    return 0

if not os.path.isfile("transpose"):
    print "Error: transpose executable not present!"
    retval += 1
else:
    
    logfilename = 'testtranspose.log' 
    print "Log in " + logfilename + "\n"
    log = open(logfilename, 'w')
    
    Xlist = [1,2,3,4,5,random.randint(6,128)]
    Ylist = [1,2,3,4,5,random.randint(6,128)]
    Plist = [1,2,3,4]

    ntests = 0
    nfails = 0
    for X in Xlist:
        for Y in Ylist:
            for P in Plist:
                cmd = []
                cmd.append("mpirun")
                cmd.append("-n")
                cmd.append(str(P))
                cmd.append("./transpose")
                cmd.append("-N0")
                cmd.append("-X" + str(X))
                cmd.append("-Y" + str(Y))
                print " ".join(cmd),
                log.write(" ".join(cmd) + '\n')
                ntests += 1

                proc = Popen(cmd, stdout = PIPE, stderr = PIPE)
                timeout = 1
                if(waitandkill(proc, timeout)):
                    msg = "\tProcess killed after" + str(timeout) +" second(s)!"
                    #print msg
                    log.write(msg + "\n")
                proc.wait() # sets the return code
                
                prc = proc.returncode
                out, err = proc.communicate() # capture output
                if (prc == 0): # did the process succeed?
                    msg = "\tSuccess"
                    print msg
                    log.write(msg + "\n")
                else:
                    msg = "\tFAILED!"
                    print msg
                    log.write(msg + "\n")
                    log.write("stdout:\n" + out + "\n")
                    log.write("stderr:\n" + err + "\n")
                    #print out
                    #print err
                    nfails += 1
                    
                    retval += 1
    log.close()
    print "\n", nfails, "failures out of", ntests, "tests." 

sys.exit(retval)
