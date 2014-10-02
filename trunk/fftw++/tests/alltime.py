#!/usr/bin/python

# a timing script for running multiplit timing tests using timin.py

import sys, getopt
import numpy as np
from math import *
import os


def main(argv):
    usage = '''
    Usage:
    \nalltime.py
    -T<number of threads> 
    -R<ram in gigabytes> 
    -d dry run
    -D<outdir>
    -o<outfile>
    -r<implicit/explicit/pruned/fft>
    -A<quoted arg list for timed program>
    '''

    dryrun=False
    R=0

    nthreads=0
    r="implicit"
    out=""
    A=""
    runtype=""

    try:
        opts, args = getopt.getopt(argv,"dp:T:r:R:o:D:g:")
    except getopt.GetoptError:
        print usage
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-T"):
            nthreads=int(arg)
        elif opt in ("-R"):
            R=float(arg)
        elif opt in ("-d"):
            dryrun=True
        elif opt in ("-o"):
            out=str(arg)
        elif opt in ("-D"):
            outdir=str(arg)
        elif opt in ("-g"):
            rname=str(arg)
        elif opt in ("-A"):
            A+=str(arg)
        elif opt in ("-r"):
            runtype=str(arg)

    progs=[["cconv" , "conv", "tconv"], ["cconv2","conv2","tconv2"],["cconv3","conv3"]]
    ab=[[6,20],[6,10],[2,6]] # problem size limits
    
    if runtype == "explicit":
        progs = [["cconv","conv"], ["cconv2","conv2"],["cconv3"]]
    if runtype == "pruned":
        progs = [["cconv2","conv2"],["cconv3"]]
        ab=[[6,10],[2,6]] # problem size limits
    if out == "":
        out="implicit"
        if runtype == "explicit":
            out="explicit"
        if runtype == "pruned":
            out="pruned"
    i=0
    while(i<len(progs)):
        a=ab[i][0]
        b=ab[i][1]
        cmd = "./timing.py"
        if R == 0.0:
            cmd += " -a " + str(a)+" -b "+str(b)
        else:
            cmd += " -R "+str(R)
        if not nthreads == 0 :
            cmd += " -T"+str(nthreads)
        if out != "":
            cmd += " -o "+out
        if runtype != "":
            cmd += " -r"+runtype
        if runtype != "":
            cmd += " -r"+runtype
        if A != "":
            cmd += " -o"+out
        for p in progs[i]:
            pcmd=cmd+" -p "+p
            print(pcmd)
            if not dryrun:
                os.system(pcmd)
        i += 1 
            
if __name__ == "__main__":
    main(sys.argv[1:])
