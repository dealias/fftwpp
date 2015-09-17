import time
from subprocess import * # so that we can run commands

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

def runtest(filename, X, Y, P, extraargs, logfilename, timeout):
    retval = 0
    log = open(logfilename, 'a')

    cmd = []
    cmd.append("mpirun")
    cmd.append("-n")
    cmd.append(str(P))
    cmd.append("./" + filename)
    for arg in  extraargs:
        cmd.append(arg)
    cmd.append("-x" + str(X))
    cmd.append("-y" + str(Y))
    print " ".join(cmd),
    log.write(" ".join(cmd) + '\n')
    
    proc = Popen(cmd, stdout = PIPE, stderr = PIPE)
    if(waitandkill(proc, timeout)):
        msg = "\tProcess killed after" + str(timeout) +" second(s)!"
        #print msg
        log.write(msg + "\n")
    proc.wait() # sets the return code
                
    prc = proc.returncode
    out, err = proc.communicate() # capture output
    if (prc == 0): # did the process succeed?
        msg = "\tpass"
        print msg
        log.write(msg + "\n")
    else:
        msg = "\tFAILED with code " + str(prc) + "!"
        print msg
        log.write(msg + "\n")
        log.write("stdout:\n" + out + "\n")
        log.write("stderr:\n" + err + "\n")
        #print out
        #print err
        retval += 1

    log.close()
    return retval

