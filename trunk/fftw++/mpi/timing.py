#!/usr/bin/python

# a timing script for convolutions using MPI.

import sys, getopt
import numpy as np
from math import *
import os

def main(argv):
    usage='Usage: timings.py -a<start> -b<stop> -p<cconv2,conv2,cconv3,conv3> -T<number of threads per node> -P<number of nodes> -A<quoted arg list for timed program> -M <quoted arg list for mpi run command> -r<implicit/explicit> -l<name of mpi run command> -R<ram in gigabytes>' 
    
    Tset=0
    P=1
    T=1
    p=""
    cargs=""
    A=""
    M=""
    a=0
    b=8
    r="implicit"
    l="mpiexec"
    RAM=0
    try:
        opts, args = getopt.getopt(argv,"p:T:P:a:b:A:M:r:l:R:")
    except getopt.GetoptError:
        print usage
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-p"):
            p=arg
        elif opt in ("-T"):
            Tset=1
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
        elif opt in ("-r"):
            r=str(arg)
        elif opt in ("-l"):
            l=str(arg)
        elif opt in ("-R"):
            RAM=float(arg)*2**30

    if p == "":
        print "please specify a program with -p"
        print usage
        sys.exit(2)

    outdir=""
   
    if p == "cconv2":
        if a == 0:
            a=int(ceil(log(P)/(log(2))))
        if RAM != 0:
            if r != "explicit":
                b=int(floor(0.5*log(RAM/64)/log(2)))
            else:
                b=int(floor(log(RAM/16/2/2**2)/log(2)/2))
        outdir="timings2c"
    if p == "conv2":
        if a == 0:
            a=int(ceil(log(P)/(log(2))))
        if RAM != 0:
            if r != "explicit":
                b= int(floor(0.5*log(RAM/96)/log(2)))
            else:
                b=int(floor(log(RAM/8/3**2)/log(2)/2))

        outdir="timings2r"
    if p == "cconv3":
        if a == 0:
            a=int(ceil(log(P)/(2*log(2))))
        if RAM != 0:
            if r != "explicit":
                b=int(floor(log(RAM/96)/log(2)/3))
            else:
                b=int(floor(log(RAM/16/2**3)/log(2)/3))

        outdir="timings3c"
    if p == "conv3":
        if a == 0:
            a=int(ceil(log(P)/(2*log(2))))
        if RAM != 0:
            b=int(floor(log(RAM/192)/log(2)/3))
        outdir="timings3r"

    if RAM != 0:
        print "max problem size is "+str(2**b)

    if outdir == "":
        print "empty outdir: please select a different program!"
        print
        print usage
        sys.exit(2)

    outdir=outdir+"/"+str(P)+"x"+str(T)
    command=l+"  --np "+str(P)+" "+M+"  ./"+str(p)

    print "output in "+outdir+"/"+r

    print "command: "+command+cargs+A
    os.system("mkdir -p "+outdir)
    os.system("rm -f "+outdir+"/"+r)

    
    rname="Implicit"
    if r == "explicit":
        rname="Explicit"
        if Tset == 1:
            print "cannot use multiple threads with explicit: try without -T"
            sys.exit(2)

    for i in range(a,b+1):
        print i,
        sys.stdout.flush()
        run=command+cargs+" -m "+str(int(pow(2,i)))+A
        grepc=" | grep -A 1 "+rname+" | tail -n 1"
        cat=" | cat >> "+outdir+"/"+r
        print run
        #print "echo "+"$("+run+grepc+")"+cat
        os.system("echo "+"$("+run+grepc+")"+cat)
        #print run+grepc+" "+cat
        #os.system(run+grepc+" "+cat)
        #print run
        #os.system(run)
        sys.stdout.flush()

    os.system("sed -i 's/[ \t]*$//' "+outdir+"/"+r)
    os.system("sed -i '/^$/d' "+outdir+"/"+r)
if __name__ == "__main__":
    main(sys.argv[1:])
