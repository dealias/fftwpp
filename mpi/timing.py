#!/usr/bin/python

# a timing script for convolutions using MPI.

import sys, getopt
import numpy as np
from math import *
import os

def main(argv):
    usage='Usage: timings.py -a<start> -b<stop> -p<cconv2,conv2,cconv3,conv3> -T<number of threads per node> -P<number of nodes> -A<quoted arg list for timed program> -M <quoted arg list for mpiexec>' 

    P=1
    T=1
    p=""
    cargs=""
    A=""
    M=""
    a=4
    b=8
    try:
        opts, args = getopt.getopt(argv,"p:T:P:a:b:A:M:")
    except getopt.GetoptError:
        print usage
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-p"):
            p=arg
        elif opt in ("-T"):
            cargs+=" -T"+str(arg)
            T=arg
        elif opt in ("-P"):
            P=int(arg)
        elif opt in ("-a"):
            a=int(arg)
        elif opt in ("-b"):
            b=int(arg)
        elif opt in ("-A"):
            A+=str(arg)
        elif opt in ("-M"):
            M+=str(arg)

    if p == "":
        print "please specify a program with -p"
        print usage
        sys.exit(2)

    outdir=""
   
    if p == "cconv2":
        outdir="timings2c"
    if p == "conv2":
        outdir="timings2r"
    if p == "cconv3":
        outdir="timings3c"
    if p == "conv3":
        outdir="timings3r"

    if outdir == "":
        print "empty outdir"
        print usage
        sys.exit(2)

    outdir=outdir+"/"+str(P)+"x"+str(T)
    command="mpiexec "+M+" -n "+str(P)+" ./"+str(p)

    print "output in "+outdir+"/implicit"

    print "command: "+command+cargs+A
    os.system("mkdir -p "+outdir)
    os.system("rm -f "+outdir+"/implicit")

    for i in range(a,b):
        print i,
        sys.stdout.flush()
        run=command+cargs+" -m"+str(int(pow(2,i)))+A
        grepc=" | grep -A 1 Implicit | tail -n 1"
        cat=" | cat >> "+outdir+"/implicit"
        #print run
        os.system("echo "+str(pow(2,i))+"\t $("+run+grepc+")"+cat)

if __name__ == "__main__":
    main(sys.argv[1:])
