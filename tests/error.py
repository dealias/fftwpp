#!/usr/bin/python -u

# Error comparison for a variety of implicit and explicit convolutions
# as compared to a known solution convolution.  

import sys # so that we can return a value at the end.
from subprocess import * # 
import re # regexp package
import random
import os.path #for file-existence checking

retval=0

print "Comparison of routine versus test case:"
print

# problem sizes used in test:
mlist = [1, 2, 12, 13, 14, 15, 17, 1024, random.randint(10, 2048)] 

Alist = [2, 4]
Blist = [1, 2, 4]

# Implicit convolution tests.
plist = ["conv", "cconv"]
convtypes = ["implicit", "explicit"]

# Create the list of commands to be run:
cmds = []
for prog in plist:
    if (os.path.isfile(prog)):
        # Explicit version: A=2, B=1 only:
        for mval in mlist:
            if not mval == 1:
                cmd = []
                cmd.append("./" + prog)
                cmd.append("-N1")
                cmd.append("-e")
                cmd.append("-t")
                cmd.append("-m" + str(mval))
                cmds.append(cmd)

        # Implicit version:
        for mval in mlist:
            for A in Alist:
                for B in Blist:
                    cmd = []
                    cmd.append("./" + prog)
                    cmd.append("-N1")
                    cmd.append("-i")
                    cmd.append("-t")
                    cmd.append("-m" + str(mval))
                    cmd.append("-A" + str(A))
                    cmd.append("-B" + str(B))
                    cmds.append(cmd)
    else:
        print(prog + " does not exist; please compile.")
        retval += 1

# Find the length of the text output for pretty formatting
maxcmdlen = 0
for cmd in cmds:
    thislen = len(" ".join(cmd))
    if thislen > maxcmdlen:
        maxcmdlen = thislen

# Run the commands and store the failed cases
print "command".ljust(maxcmdlen) + "\terror"
print
failures = []
for cmd in cmds:
    p = Popen(cmd, stdout = PIPE, stderr = PIPE)
    p.wait()
    out, err = p.communicate() # capture output
    if (p.returncode == 0): # did the process succeed?
        # Find the text after "error=":
        m = re.search("(?<=error=)(.*)", out) 
        errval = float(m.group(0))
        print " ".join(cmd).ljust(maxcmdlen) + "\t" +  str(errval),
        
        if(errval > 1e-10):
            print "\tERROR TOO LARGE"
            failures.append([cmd, "error: " + str(errval)])
            retval += 1
        else:
            print
    else:
        print " ".join(cmd) + "\tFAILED"
        failures.append([cmd, "return code: " + str(p.returncode)])
        retval += 1

print

if retval == 0:
    print "OK: all tests passed."
else:
    print "Failed cases:"
    for fail in failures:
        print " ".join(fail[0]).ljust(maxcmdlen) + "\t" + fail[1] 
    print
    print "ERROR: " + str(len(failures)) + " TEST(S) FAILED"

sys.exit(retval)
