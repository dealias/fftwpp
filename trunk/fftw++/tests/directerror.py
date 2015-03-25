#!/usr/bin/python -u

# Error comparison for a variety of implicit and explicit convolutions
# as compared to a direct convolution.

import sys # so that we can return a value at the end.
from subprocess import * # so that we can run commands
import re # regexp package
import random # for randum number generators
import os.path #for file-existence checking
import math # for isnan

failures = 0

print "Comparison of routine versus direct routine:"
print
print "program\ttype\t\tsize\tA\terror:"
print


mlist=[8, 9, 0] # problem sizes (0 gives a random value)
# 2D and 3D direct convolutions can take forever, so we add limits:
maxm1d = 64;
maxm2d = 16;
maxm3d = 10;

autolist = ["cconv"]
convlist = ["conv", "conv2", "conv3", "cconv", "cconv2", "cconv3"]
compactlist = ["conv2", "conv3"]
tconvlist = ["tconv", "tconv2"]
elist = ["cconv", "cconv2", "cconv3", "conv"] 

Alist = [2]

lists = [convlist, compactlist, tconvlist, elist, autolist]

ntests=0

for list in lists:
    typearg = "-i"
    name = "implicit"
    if list == compactlist:
        name = "non-compact"
    if list == elist:
        typearg = "-e"
        name = "explicit"
        Alist = [2] # values of A to be tested
    else:
        Alist = [2, 4] # values of A to be tested
    if list == autolist:
        Alist = [1]
    for prog in list:
        if (os.path.isfile(prog)): # check that file exists
            for mval in mlist:
                mstr = str(mval)
                mcom =  ["-m" + mstr]

                dimension = 1
                if "2" in prog:
                    dimension = 2
                    if mval > maxm2d:
                        # TODO: do non-square case
                        mval = random.randint(1, maxm2d)
                if "3" in prog:
                    dimension = 3
                    if mval > maxm3d:
                        # TODO: do non-cube case
                        mval = random.randint(1, maxm3d)

                if(dimension == 2):
                    mstr = str(mval) + "x" + str(mval)
                if(dimension == 3):
                    mstr = str(mval) + "x" + str(mval) + "x" + str(mval)
                    
                if(mval == 0): # the random case
                    if(dimension == 1):
                        mstr = str(random.randint(1, maxm1d))
                        mcom = ["-m" + mstr]
                    if(dimension == 2):
                        x = str(random.randint(1, maxm2d))
                        y = str(random.randint(1, maxm2d))
                        mcom = ["-x" + str(x), "-y" + str(y)]
                        mstr = str(x) + "x" + str(y)
                    if(dimension == 3):
                        x = str(random.randint(1, maxm3d))
                        y = str(random.randint(1, maxm3d))
                        z = str(random.randint(1, maxm3d))
                        mcom = ["-x" + str(x), "-y" + str(y), "-z" + str(z)]
                        mstr = str(x) + "x" + str(y) + "x" + str(z)

                for A in Alist:
                    command = ["./" + prog, "-N1", typearg, "-d",\
                               "-A" + str(A), "-T1"];
                    command += mcom

                    ntests += 1
                    if(list == compactlist):
                        clist = ["-c0"]
                        command += clist
                    #print command
                    print prog + "\t" + name + "\t" + mstr \
                                                      + "\tA="+str(A),
                    p = Popen(command, stdout = PIPE, stderr = PIPE)
                    p.wait() # sets the return code
                    prc = p.returncode
                    out, err = p.communicate() # capture output
                    if (prc == 0): # did the process succeed?
                        # find the text after "error="
                        m = re.search("(?<=error=)(.*)", out) 
                        print "\t" + m.group(0),
                        if float(m.group(0)) > 1e-10 :
                            print "\tERROR TOO LARGE!!!",
                            failures += 1
                        else:
                            if math.isnan(float(m.group(0))) :
                                print "\tERROR IS NaN!!!",
                                failures += 1
                        print

                    else:
                        print "\tFAILURE:"
                        print command
                        print "with, return code:"
                        print prc
                        print "stdout:"
                        print out
                        print "stderr:"
                        print err
                        failures += 1
        else:
            print(prog+" does not exist; please compile.")
            failures+=1

print

print str(ntests)+" tests were performed with " + str(failures) + " failure(s)."

print 

if(failures == 0):
    print "OK:\tall tests passed"
else:
    print "*" * 80
    print "ERROR:\t" + str(failures) + " TEST(S) FAILED"

sys.exit(failures)
