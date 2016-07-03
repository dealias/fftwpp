#!/usr/bin/python

# Error comparison for a variety of implicit and explicit convolutions
# as compared to a direct convolution.

import sys # so that we can return a value at the end.
from subprocess import * # so that we can run commands
import re # regexp package
import random # for randum number generators
import os.path #for file-existence checking
import math # for isnan
import copy

def runcommand(command, tolerance):
    p = Popen(command, stdout = PIPE, stderr = PIPE)
    p.wait() # sets the return code
    prc = p.returncode
    
    # capture output:
    out, err = p.communicate() 

    # did the process succeed?
    if (prc == 0): 
        # find the text after "error="
        m = re.search("(?<=error=)(.*)", out) 
        if not m == None:
            print "\t" + m.group(0),
            if float(m.group(0)) > tolerance :
                print "\tERROR TOO LARGE!!!",
                return 1
            else:
                if math.isnan(float(m.group(0))) :
                    print "\tERROR IS NaN!!!",
                    return 1
            print
        else:
            print "\tFAILURE: output not found"
            print " ".join(command)
            return 1
    else:
        print "\tFAILURE:"
        print " ".join(command)
        print command
        print "with, return code:"
        print prc
        print "stdout:"
        print out
        print "stderr:"
        print err
        return 1
    return 0

def progdim(prog):
    dimension = 1
    if "2" in prog:
        dimension = 2
    if "3" in prog:
        dimension = 3
    return dimension

# 2D and 3D direct convolutions can take forever, so we add limits:
maxm1d = 64;
maxm2d = 16;
maxm3d = 10;

# Run tests with one dimension
# preprint: output text before result
# cmdbase: array of command arguments
# xlist: array of arguments to go with -m<int>
def run1d(preprint, cmdbase, xlist):
    nfails = 0
    ntests = 0
    for mval in xlist:
        ntests += 1
        command = copy.deepcopy(cmdbase)
        if(mval == 0):
            m = random.randint(1, maxm1d)
        else:
            m = mval
        print(preprint + "\t" + str(m)),
        command.append("-m" + str(m))
        prc = runcommand(command, 1e-10)
        if not prc == 0:
            nfails += 1
    return ntests, nfails

# Run tests with two dimensions
# preprint: output text before result
# cmdbase: array of command arguments
# xlist, ylist: array of arguments to go with -x<int> -y<int>
def run2d(preprint, cmdbase, xlist, ylist):
    nfails = 0
    ntests = 0
    for xval in xlist:
        x = xval
        if(x == 0):
            x = random.randint(1, maxm2d)
        for yval in ylist:
            ntests += 1
            command = copy.deepcopy(cmdbase)
            y = yval
            if(y == 0):
                y = random.randint(1, maxm2d)
            print(preprint + "\t" + str(x) + "x" + str(y)),
            command.append("-x" + str(x))
            command.append("-y" + str(y))
            prc = runcommand(command, 1e-10)
            if not prc == 0:
                nfails += 1
    return ntests, nfails

# Run tests with three dimensions
# preprint: output text before result
# cmdbase: array of command arguments
# xlist, ylist, zlist: array of arguments to go with -x<int> -y<int> -z<int>
def run3d(preprint, cmdbase, xlist, ylist, zlist):
    nfails = 0
    ntests = 0
    for xval in xlist:
        x = xval
        if(x == 0):
            x = random.randint(1, maxm3d)
        for yval in ylist:
            ntests += 1
            command = copy.deepcopy(cmdbase)
            y = yval
            if(y == 0):
                y = random.randint(1, maxm3d)
            for zval in zlist:
                ntests += 1
                command = copy.deepcopy(cmdbase)
                z = zval
                if(z == 0):
                    z = random.randint(1, maxm3d)
                print(preprint + "\t" + str(x) + "x" + str(y) + "x" + str(z)),
                command.append("-x" + str(x))
                command.append("-y" + str(y))
                command.append("-z" + str(z))
                prc = runcommand(command, 1e-10)
                if not prc == 0:
                    nfails += 1
    return ntests, nfails

# Run tests for autoconvolutions
def check_auto(proglist):
    ntests = 0
    nfails = 0
    xlist = [0,8,9,10]
    Alist = [1]
    typearg = "-i"
    for prog in proglist:
        preprint = prog + "\tauto\t\tA=1"
        command = []
        command.append("./" + prog)
        command.append("-N1")
        command.append(typearg)
        command.append("-d")
        command.append("-A1")
        if os.path.isfile(prog):
            dimension = progdim(prog)
            if dimension == 1:
                ntests1, nfails1 = run1d(preprint, command, xlist)
                ntests += ntests1
                nfails += nfails1
        else:
            print(prog + " does not exist; please compile.")
            fails += 1
    return ntests, nfails

# Run tests for ternary convolutions
def check_ternary(proglist):
    ntests = 0
    nfails = 0
    xlist = [0,8,9,10]
    ylist = [0,8,9,10]
    Alist = [2,4]
    typearg = "-i"
    for prog in proglist:
        for A in Alist:
            preprint = prog + "\tternary\t\tA=" + str(A)
            command = []
            command.append("./" + prog)
            command.append("-N1")
            command.append(typearg)
            command.append("-d")
            command.append("-A" + str(A))
            command.append("-T1")
            if os.path.isfile(prog):
                dimension = progdim(prog)
                if dimension == 1:
                    ntests1, nfails1 = run1d(preprint, command, xlist)
                    ntests += ntests1
                    nfails += nfails1
                if dimension == 2:
                    ntests2, nfails2 = run2d(preprint, command, xlist, ylist)
                    ntests += ntests2
                    nfails += nfails2
            else:
                print(prog + " does not exist; please compile.")
                fails += 1
    return ntests, nfails

# Run tests for implicit convolutions 
def check_conv(proglist):
    ntests = 0
    nfails = 0
    xlist = [0,8,9,10]
    ylist = [0,8,9,10]
    zlist = [0,8,9,10]
    Alist = [2,4]
    typearg = "-i"
    for prog in proglist:
        for A in Alist:
            preprint = prog + "\timplicit\tA=" + str(A)
            command = []
            command.append("./" + prog)
            command.append("-N1")
            command.append(typearg)
            command.append("-d")
            command.append("-A" + str(A))
            command.append("-T1")
            if os.path.isfile(prog):
                dimension = progdim(prog)
                if dimension == 1:
                    ntests1, nfails1 = run1d(preprint, command, xlist)
                    ntests += ntests1
                    nfails += nfails1
                if dimension == 2:
                    ntests2, nfails2 = run2d(preprint, command, xlist, ylist)
                    ntests += ntests2
                    nfails += nfails2
                if dimension == 3:
                    ntests3, nfails3 = run3d(preprint, command, \
                                             xlist, ylist, zlist)
                    ntests += ntests3
                    nfails += nfails3
            else:
                print(prog + " does not exist; please compile.")
                fails += 1
    return ntests, nfails

# Run tests for compact/non-compact Hermitian-symmetric convolutions
def check_compact(proglist):
    ntests = 0
    nfails = 0
    xlist = [0,8,9,10]
    ylist = [0,8,9,10]
    zlist = [0,8,9,10]
    Alist = [2,4]
    typearg = "-i"
    for prog in proglist:
        for A in Alist:
            command = []
            if os.path.isfile(prog):
                dimension = progdim(prog)
                if dimension == 2:
                    for X in [0, 1]:
                        for Y in [0, 1]:
                            command.append("./" + prog)
                            command.append("-N1")
                            command.append(typearg)
                            command.append("-d")
                            command.append("-A" + str(A))
                            command.append("-T1")
                            command.append("-X" + str(X))
                            command.append("-Y" + str(Y))
                            preprint = prog + "\tXY="\
                                           + str(X) + str(Y) \
                                           + "\tA=" + str(A)
                            ntests2, nfails2 = run2d(preprint, command, \
                                                     xlist, ylist)
                            ntests += ntests2
                            nfails += nfails2
                if dimension == 3:
                    for X in [0, 1]:
                        for Y in [0, 1]:
                            for Z in [0, 1]:
                                command.append("./" + prog)
                                command.append("-N1")
                                command.append(typearg)
                                command.append("-d")
                                command.append("-A" + str(A))
                                command.append("-T1")
                                command.append("-X" + str(X))
                                command.append("-Y" + str(Y))
                                command.append("-Z" + str(Z))
                                preprint = prog + "\tXYZ="\
                                           + str(X) + str(Y) + str(Z) \
                                           + "\tA=" + str(A)
                                ntests3, nfails3 = run3d(preprint, command, \
                                                         xlist, ylist, zlist)
                    ntests += ntests3
                    nfails += nfails3
            else:
                print(prog + " does not exist; please compile.")
                fails += 1
    return ntests, nfails

# Run test for epliclty dealiased convolutions
def check_explicit(proglist):
    ntests = 0
    nfails = 0
    xlist = [0,8,9,10]
    ylist = [0,8,9,10]
    zlist = [0,8,9,10]
    Alist = [2]
    typearg = "-e"
    for prog in proglist:
        for A in Alist:
            preprint = prog + "\texplicit\tA=" + str(A)
            command = []
            command.append("./" + prog)
            command.append("-N1")
            command.append(typearg)
            command.append("-d")
            command.append("-A" + str(A))
            command.append("-T1")
            if os.path.isfile(prog):
                dimension = progdim(prog)
                if dimension == 1:
                    ntests1, nfails1 = run1d(preprint, command, xlist)
                    ntests += ntests1
                    nfails += nfails1
                if dimension == 2:
                    ntests2, nfails2 = run2d(preprint, command, xlist, ylist)
                    ntests += ntests2
                    nfails += nfails2
                if dimension == 3:
                    ntests3, nfails3 = run3d(preprint, command, \
                                             xlist, ylist, zlist)
                    ntests += ntests3
                    nfails += nfails3
            else:
                print(prog + " does not exist; please compile.")
                fails += 1
    return ntests, nfails


print "Comparison of routine versus direct routine:"
print
print "program\ttype\t\tA\tsize\terror:"
print


ntests = 0
nfails = 0

autolist = ["cconv"]
atests, afails = check_auto(autolist)
ntests += atests
nfails += afails

tconvlist = ["tconv", "tconv2"]
ttests, tfails = check_ternary(tconvlist)
ntests += ttests
nfails += tfails

convlist = ["conv", "conv2", "conv3", "cconv", "cconv2", "cconv3"]
ctests, cfails = check_conv(convlist)
ntests += ctests
nfails += cfails

compactlist = ["conv2", "conv3"]
cotests, cofails = check_compact(compactlist)
ntests += cotests
nfails += cofails

elist = ["cconv", "cconv2", "cconv3", "conv", "conv2"] 
etests, efails = check_explicit(elist)
ntests += etests
nfails += efails

print

print str(ntests)+" tests were performed with " + str(nfails) + " failure(s)."

print

if(nfails == 0):
    print "OK:\tall tests passed"
else:
    print "*" * 80
    print "ERROR:\t" + str(nfails) + " TEST(S) FAILED"

sys.exit(nfails)
