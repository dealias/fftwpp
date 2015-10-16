import time
from subprocess import * # so that we can run commands
import os

os.close(0)

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
    
    proc = Popen(cmd, stdout = PIPE, stderr = PIPE, stdin = None)
    if(waitandkill(proc, timeout)):
        msg = "\tFAIL: Process killed after " + str(timeout) +"s!"
        #print msg
        log.write(msg + "\n")
    proc.wait() # sets the return code
                
    prc = proc.returncode
    out, err = proc.communicate() # capture output
    if (prc == 0): # did the process succeed?
        msg = "pass"
        try:
            print msg
        except:
            pass
        log.write(msg + "\n")
    else:
        msg = "\tFAIL with code " + str(prc) + "!"
        try:
            print msg
        except:
            pass
        log.write(msg + "\n")
        log.write("stdout:\n" + out + "\n")
        log.write("stderr:\n" + err + "\n")
        #print out
        #print err
        retval += 1

    log.close()
    return retval, cmd
