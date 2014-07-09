#!/usr/bin/python -u

# Error comparison for a variety of implicit and explicit convolutions
# as compared to a direct convolution.  One problem size per program
# (scaling of numerical error vs problem size is dominated by direct
# convolution error).

import sys # so that we can return a value at the end.
from subprocess import * # 
import re # regexp package
import random # for randum number generators
import os.path #for file-existence checking

retval=0

print "Comparison of routine versus direct routine:"
print
print "program\ttype\t\tsize\tA\terror:"
print


mlist=[8,9,random.randint(1,64)] # problem sizes
# 2D and 3D direct convolutions can take forever, so we add limits:
maxm2d=16;
maxm3d=10;

convlist=["conv", "conv2", "conv3", "cconv", "cconv2", "cconv3"]
compactlist=["conv2", "conv3"]
tconvlist=["tconv", "tconv2"]
elist=["cconv", "cconv2", "cconv3", "conv"] 

Alist=[2]

lists=[convlist,compactlist,tconvlist,elist]

for list in lists:
    typearg="-i"
    name="implicit"
    if list==compactlist:
        name="non-compact"
    if list==elist:
        typearg="-e"
        name="explicit"
        Alist=[2] # values of A to be tested
    else:
        Alist=[2,4] # values of A to be tested
    for prog in list:
        if (os.path.isfile(prog)): # check that file exists
            for mval in mlist:
                if "2" in prog:
                    if mval > maxm2d:
                        mval=random.randint(1,maxm2d)
                if "3" in prog:
                    if mval > maxm3d:
                        mval=random.randint(1,maxm3d)

                for A in Alist:
                    command=["./"+prog,"-N1",typearg,"-d","-A"+str(A),"-m"+str(mval),"-T1"];
                    if(list == compactlist):
                        clist=["-c0"]
                        command += clist
                    #print command
                    print prog +"\t"+name+"\tm="+str(mval)+"\tA="+str(A),
                    p=Popen(command,stdout=PIPE,stderr=PIPE)
                    p.wait() # sets the return code
                    prc=p.returncode
                    out, err = p.communicate() # capture output
                    if (prc == 0): # did the process succeed?
                        m=re.search("(?<=error=)(.*)",out) # find the text after "error="
                        print "\t"+m.group(0),
                        if(float(m.group(0)) > 1e-10):
                            print "\tERROR TOO LARGE"
                            retval+=1
                        else:
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
                        retval+=1
        else:
            print(prog+" does not exist; please compile.")
            retval+=1

print
if(retval == 0):
    print "OK\tall tests passed"
else:
    print "ERROR\tAT LEAST ONE TEST FAILED"

sys.exit(retval)
