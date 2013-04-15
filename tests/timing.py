#!/usr/bin/python

# a timing script for convolutions using OpenMP

import sys, getopt
import numpy as np
from math import *
import os

P=1

def main(argv):
    usage='Usage: timings.py -a<start> -b<stop> -p<cconv,cconv2,cconv3,conv,conv2,conv3,tconv,tconv2> -T<number of threads> -A<quoted arg list for timed program> -r<implicit/explicit> -l<name of mpi run command> -R<ram in gigabytes>' 
    bset=0
    Tset=0
    T=1
    p=""
    cargs=""
    A=""
    a=6
    b=8
    r="implicit"
    RAM=0
    try:
        opts, args = getopt.getopt(argv,"p:T:a:b:A:r:R:")
    except getopt.GetoptError:
        print usage
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-p"):
            p=arg
        if opt in ("-T"):
            Tset=1
            cargs+=" -T"+str(arg)
            T=arg
        elif opt in ("-a"):
            a=int(arg)
        elif opt in ("-b"):
            b=int(arg)
        elif opt in ("-A"):
            A+=str(arg)
        elif opt in ("-r"):
            r=str(arg)
        elif opt in ("-R"):
            RAM=float(arg)*2**30

    if p == "":
        print "please specify a program with -p"
        print usage
        sys.exit(2)

    outdir=""
   
    if p == "cconv":
        if RAM != 0:
            b=int(floor(log(RAM/4)/log(2)))
            b=min(b,14) # because we aren't crazy
        outdir="timings1c"
    if p == "cconv2":
        if RAM != 0:
            if r != "explicit":
                b=int(floor(0.5*log(RAM/64)/log(2)))
            else:
                b=int(floor(log(RAM/16/2/2**2)/log(2)/2))
        outdir="timings2c"
    if p == "cconv3":
        if RAM != 0:
            if r != "explicit":
                b=int(floor(log(RAM/96)/log(2)/3))
            else:
                b=int(floor(log(RAM/16/2**3)/log(2)/3))
        outdir="timings3c"

    if p == "conv":
        b=int(floor(log(RAM/6)/log(2)))
        b=min(b,14) # because we aren't crazy
        outdir="timings1r"
    if p == "conv2":
        if RAM != 0:
            if r != "explicit":
                b= int(floor(0.5*log(RAM/96)/log(2)))
            else:
                b=int(floor(log(RAM/8/3**2)/log(2)/2))
        outdir="timings2r"
    if p == "conv3":
        if RAM != 0:
            b=int(floor(log(RAM/192)/log(2)/3))
        outdir="timings3r"

    if p == "tconv":
        b=int(floor(log(RAM/6)/log(2)))
        b=min(b,14) # because we aren't crazy
        outdir="timings1t"
    if p == "tconv2":
        if a == 0:
            a=int(ceil(log(P)/(log(2))))
        if RAM != 0:
            if r != "explicit":
                b=int(floor(log(RAM/(8*12))/(2*log(2))))
            else:
                b=int(floor(log(RAM/(8*6))/(2*log(2))))
        outdir="timings2t"

    if RAM != 0:
        print "max problem size is "+str(2**b)

    if outdir == "":
        print "empty outdir: please select a different program!"
        print
        print usage
        sys.exit(2)

    outdir=outdir+"/"
    command="./"+str(p)

    print "output in "+outdir+r

    print "command: "+command+cargs+" "+A
    os.system("mkdir -p "+outdir)
    os.system("rm -f "+outdir+"/"+r)

    
    rname="Implicit"
    if r == "explicit":
        rname="Explicit"

    for i in range(a,b+1):
        print i,
        run=command+cargs+" -m "+str(int(pow(2,i)))+" "+A
        if(r == "explicit"):
            run += " -e"
        else:
            run += " -i"
        grepc=" | grep -A 1 "+rname+" | tail -n 1"
        cat=" | cat >> "+outdir+"/"+r
        print run
        sys.stdout.flush()
        #print("echo "+"$("+run+grepc+")"+cat)
        os.system("echo "+"$("+run+grepc+")"+cat)
        sys.stdout.flush()

    # clean up output
    os.system("sed -i 's/[ \t]*$//' "+outdir+"/"+r)
    os.system("sed -i '/^$/d' "+outdir+"/"+r) # remove empty lines
if __name__ == "__main__":
    main(sys.argv[1:])
