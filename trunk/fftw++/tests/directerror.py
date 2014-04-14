#!/usr/bin/python -u

# Error comparison for a variety of implicit and explicit convolutions
# as compared to a direct convolution.  One problem size per program
# (scaling of numerical error vs problem size is dominated by direct
# convolution error).

import sys # so that we can return a value at the end.
from subprocess import * # 
import re # regexp package

retval=0

print "Comparison of routine versus direct routine:"
print
print "program\ttype\t\tsize\terror:"
print

mlist=[8,9] # problem sizes

dlist=["cconv", "cconv2", "cconv3", "conv", "conv2", "conv3", "tconv", "tconv2"]
for prog in dlist:
    for mval in mlist:
        p=Popen(["./"+prog,"-N1","-i","-d","-m"+str(mval)],stdout=PIPE,stderr=PIPE)
        p.wait() # sets the return code
        prc=p.returncode
        out, err = p.communicate() # capture output
        if (prc == 0): # did the process succeed?
            m=re.search("(?<=error=)(.*)",out) # find the text after "error="
            print prog +"\t"+"implicit\t"+str(mval)+"\t"+m.group(0)
            if(float(m.group(0)) > 1e-10):
                print "ERROR TOO LARGE"
                retval+=1
        else:
            print "./"+prog+" -N1"+" -i "+"-d "+"-m"+str(mval)+" FAILED with code "+str(prc)
            retval+=1


elist=["cconv", "cconv2", "cconv3", "conv"] 
for prog in elist:
    for mval in mlist:
        p=Popen(["./"+prog,"-N1","-e","-d","-m"+str(mval)],stdout=PIPE,stderr=PIPE)
        p.wait() # sets the return code
        if (p.returncode == 0): # did the process succeed?
            out, err = p.communicate() # capture output
            m=re.search("(?<=error=)(.*)",out) # find the text after "error="
            print prog +"\t"+"explicit\t"+str(mval)+"\t"+m.group(0)
            if(float(m.group(0)) > 1e-10):
                print "ERROR TOO LARGE"
                retval+=1
        else:
            print "./"+prog+" -N1"+" -e "+"-d "+"-m"+str(mval)+" FAILED with code "+str(prc)
            retval+=1

print

if(retval == 0):
    print "OK\tall tests passed"
else:
    print "ERROR\tAT LEAST ONE TEST FAILED"

sys.exit(retval)
