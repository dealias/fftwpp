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
print "program\ttype    \tlength\t\terror:"
print

mlist=[1,2,8,9,1024,random.randint(10,2048)] # problem sizes used in test.
ilist=["cconv", "conv"]
for prog in ilist:
    if (os.path.isfile(prog)):
        for mval in mlist:
            p=Popen(["./"+prog,"-N1","-i","-t","-m"+str(mval)],stdout=PIPE,stderr=PIPE)
            p.wait() # sets the return code
            out, err = p.communicate() # capture output
            if (p.returncode == 0): # did the process succeed?
                m=re.search("(?<=error=)(.*)",out) # find the text after "error="
                print prog +"\t"+"implicit\t"+str(mval)+"\t"+m.group(0)
                if(float(m.group(0)) > 1e-10):
                    print "ERROR TOO LARGE"
                    retval+=1
            else:
                print prog+" FAILED"
                retval+=1
    else:
        print(prog+" does not exist; please compile.")
        retval+=1


elist=["cconv","conv"] 
for prog in elist:
    if (os.path.isfile(prog)):
        for mval in mlist:
            p=Popen(["./"+prog,"-N1","-e","-t"],stdout=PIPE,stderr=PIPE)
            p.wait() # sets the return code
            out, err = p.communicate() # capture output
            if (p.returncode == 0): # did the process succeed?
                m=re.search("(?<=error=)(.*)",out) # find the text after "error="
                print prog +"\t"+"explicit\t"+str(mval)+"\t"+m.group(0)
                if(float(m.group(0)) > 1e-10):
                    print "ERROR TOO LARGE"
                    retval+=1
            else:
                print prog+" FAILED"
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
