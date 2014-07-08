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
print "program\ttype\t\tsize\terror:"
print


mlist=[8,9,random.randint(1,32)] # problem sizes


convlist=["conv", "conv2", "conv3", "cconv", "cconv2", "cconv3"]
compactlist=["conv2", "conv3"]
tconvlist=["tconv", "tconv2"]
elist=["cconv", "cconv2", "cconv3", "conv"] 

lists=[convlist,compactlist,tconvlist,elist]

for list in lists:
    typearg="-i"
    name="implicit"
    if list==compactlist:
        name="non-compact"
    if list==elist:
        typearg="-e"
        name="explicit"
    for prog in list:
        if (os.path.isfile(prog)): # check that file exists
            for mval in mlist:
                command=["./"+prog,"-N1",typearg,"-d","-m"+str(mval)];
                if(list == compactlist):
                    clist=["-c0"]
                    command += clist
                #print command
                p=Popen(command,stdout=PIPE,stderr=PIPE)
                p.wait() # sets the return code
                prc=p.returncode
                out, err = p.communicate() # capture output
                if (prc == 0): # did the process succeed?
                    m=re.search("(?<=error=)(.*)",out) # find the text after "error="
                    print prog +"\t"+name+"\t"+str(mval)+"\t"+m.group(0)
                    if(float(m.group(0)) > 1e-10):
                        print "ERROR TOO LARGE"
                        retval+=1
                else:
                    print "FAILURE:"
                    print command
                    print "with, return code:"
                    print prc
                    print "output:"
                    print out
                    print "error:"
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
