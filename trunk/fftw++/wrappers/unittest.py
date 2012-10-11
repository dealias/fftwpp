#!/usr/bin/python

import os
import subprocess
import sys

flag=0

ccommand = ["./cexample"]
with open(os.devnull, "w") as fnull:
    cresult = subprocess.call(ccommand, stdout = fnull, stderr = fnull)

if cresult == 0:
    print "cexample succesful"
else:
    print "cexample FAILED"
    flag += 1

sys.exit(flag)
