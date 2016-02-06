#!/usr/bin/python

import time
from subprocess import * # so that we can run commands
import os
import sys

def Print(msg) :
    try:
        print msg
    except:
        pass
    return

def waitandkill(proc, timeout):
    if(timeout == 0):
        return
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

def runtest(filename, P, extraargs, logfilename, timeout):
    retval = 0
    log = open(logfilename, 'a')

    cmd = []
    cmd.append("mpirun")
    cmd.append("-n")
    cmd.append(str(P))
    cmd.append("./" + filename)
    for arg in  extraargs:
        cmd.append(arg)
    try:
        print " ".join(cmd),
    except:
        pass
    log.write(" ".join(cmd)),
    
    DEVNULL = open(os.devnull, 'wb')
    proc = Popen(cmd, stdout = DEVNULL, stderr = PIPE, stdin = DEVNULL)
    if(waitandkill(proc, timeout)):
        msg = "\tFAIL: Process killed after " + str(timeout) +"s!"
        #print msg
        log.write(msg + "\n")
    proc.wait() # sets the return code
                
    prc = proc.returncode
    out, err = proc.communicate() # capture output
    if (prc == 0): # did the process succeed?
        msg = "pass"
        Print(msg)
        log.write(" "+msg + "\n")
    else:
        msg = " FAIL with code " + str(prc) + "!"
        Print(msg)
        log.write(" "+msg + "\n")
        log.write("stderr:\n" + str(err) + "\n")
        #print out
        #print err
        retval += 1

    log.close()
    return retval, cmd
