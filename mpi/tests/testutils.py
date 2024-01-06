#!/usr/bin/python3

import time
import subprocess 
import os
import sys

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
        print(" ".join(cmd),)
    except:
        pass
    log.write(" ".join(cmd)),
    
    DEVNULL = open(os.devnull, 'wb')
    proc = subprocess.Popen(cmd,
                            stdout = DEVNULL,
                            stderr = subprocess.PIPE,
                            stdin = DEVNULL)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        msg = "\tFAIL: Process killed after " + str(timeout) +"s!"
        log.write(msg + "\n")
        proc.kill()
    
    prc = proc.returncode
    out, err = proc.communicate() # capture output
    if (prc == 0): # did the process succeed?
        msg = "pass"
        print(msg)
        log.write(" "+msg + "\n")
    else:
        msg = " FAIL with code " + str(prc) + "!"
        print(msg)
        log.write(" "+msg + "\n")
        log.write("stderr:\n" + err.decode() + "\n")
        #print out
        #print err
        retval += 1

    log.close()
    return retval, cmd
