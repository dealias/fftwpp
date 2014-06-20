#!/usr/bin/python

# a timing script for running multiplit timing tests using timin.py

import sys, getopt
import numpy as np
from math import *
import os


def main(argv):
    usage='Usage: FIXME' 

    dryrun=False
    r=0.0
    threadset=False
    nthreads=1
    r="implicit"
    out=""

    try:
        opts, args = getopt.getopt(argv,"dp:T:r:R:o:D:g:")
    except getopt.GetoptError:
        print usage
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-T"):
            threadset=True
            ntrheads=int(arg)
        elif opt in ("-R"):
            r=float(arg)
        elif opt in ("-d"):
            dryrun=True
        elif opt in ("-o"):
            out=str(arg)
        elif opt in ("-D"):
            outdir=str(arg)
        elif opt in ("-g"):
            rname=str(arg)

    progs=[["cconv" , "conv", "tconv"], ["cconv2","conv2","tconv2"],["cconv3","conv3"]]
    ab=[[6,20],[6,10],[2,6]] # problem size limits

    i=0
    while(i<3):
        a=ab[i][0]
        b=ab[i][1]
        cmd="./timing.py -a"+str(a)+" -b"+str(b)
        if threadset:
            cmd+=" -T"+str(nthreads)
        if out != "":
            cmd+=" -o"+out
        for p in progs[i]:
            pcmd=cmd+" -p "+p
            print(pcmd)
            if not dryrun:
                os.system(pcmd)
        i += 1 
            
if __name__ == "__main__":
    main(sys.argv[1:])
