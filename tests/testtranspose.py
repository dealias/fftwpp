#!/usr/bin/python

# Check whether the local-memory transpose routine produces the
# correct results for a variety of problem sizes and number of
# threads.

import random # for randum number generators
from subprocess import * # so that we can run commands
import sys # so that we can return a value at the end.

failures = 0

prog = "transpose"


maxsize = 32
xlist = [1,2,3,4,5]
xlist.append(random.randint(1, maxsize))
ylist = [1,2,3,4,5]
ylist.append(random.randint(1, maxsize))
zlist = [1,3]

tlist = [1,2,3,4,5,6,7]

print xlist
print ylist
print tlist

ntests = 0
nfailures = 0

for mx in xlist:
    for my in ylist:
        for mz in zlist:
            for t in tlist:
                for inplace in [0,1]:
                    ntests += 1
                    command = ["./" + prog];
                    command.append("-x"+str(mx))
                    command.append("-y"+str(my))
                    command.append("-z"+str(mz))
                    command.append("-T"+str(t))
                    command.append("-i"+str(inplace))
                    print " ".join(command),
                    p = Popen(command, stdout = PIPE, stderr = PIPE)
                    p.wait() # sets the return code
                    prc = p.returncode
                    out, err = p.communicate()
                    #print prc
                    if(prc == 0):
                        print "\tpass"
                    else:
                        nfailures += 1
                        print "\tERROR!"
print

if(nfailures == 0):
    print "OK:\tall", ntests, "tests passed"
else:
    print "*" * 80
    print "ERROR:\t", str(nfailures) + " FAILURES in", ntests, "tests!"

sys.exit(failures)
