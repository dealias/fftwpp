#!/usr/bin/python

# a timing script for convolutions using MPI.

import sys, getopt
import numpy as np
from math import *
import os

def main(argv):
    usage='Usage: timings.py -a<start> -b<stop> -p<cconv2,conv2,cconv3,conv3> -T<number of threads per node> -P<number of nodes> -A<quoted arg list for timed program> -M <quoted arg list for mpi run command> -r<implicit/explicit> -l<name of mpi run command> -R<ram in gigabytes> -d' 
    helpnotes=\
        "\n -a <int> speficies the max exponent of the min problem size "\
        "\n -b <int> speficies the max exponent of the max problem size "\
        "\n -p <program name> speficies the program name "\
        "\n -P <int> specifies the number of MPI process "\
        "\n -T <int> specifies the number of OpenMP threads per MPI process"\
        "\n -A <string> allows one to pass extra arguments to the convolution program "\
        "\n -M <string> allows one to pass extra arguments to the mpi exec program "\
        "\n -r <implicit/explicit> specifies the type of convoution  "\
        "\n -l <string> specifies the name of the MPI launch program  "\
        "\n -R <real value> spedifies the rame size "\
        "\n -d specifies a dry run; commands are shown but no convolutions are computed\n "\
        "\nfor example, try the command \n ./timing.py -pcconv2 -a 3 -b 4"

    
    dryrun=False
    Tset=0
    P=1
    T=1
    p=""
    cargs=""
    A=""
    M=""
    a=0
    b=0
    r="implicit"
    l="mpiexec --np"
    RAM=0
    try:
        opts, args = getopt.getopt(argv,"p:T:P:a:b:A:M:r:l:R:d")
    except getopt.GetoptError:
        print usage
        print helpnotes
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
        elif opt in ("-d"):
            dryrun=True

    if p == "":
        print "please specify a program with -p"
        print usage
        print helpnotes
        sys.exit(2)

    outdir=""
   

    # if both the max problem size and the ram are unset, go up to 2^8
    if (b == 0 and RAM == 0):
        b=8
    # if RAM is set, but not the max problem size, RAM determines the
    # problem size on its own.
    if (b == 0 and RAM != 0):
        b=sys.maxint

    if p == "cconv2":
        if a == 0:
            a=int(ceil(log(P)/(log(2))))
        if RAM != 0:
            if r != "explicit":
                b=min(b,int(floor(0.5*log(RAM/64)/log(2))))
            else:
                b=min(b,int(floor(log(RAM/16/2/2**2)/log(2)/2)))
        outdir="timings2c"
    if p == "conv2":
        if a == 0:
            a=int(ceil(log(P)/(log(2))))
        if RAM != 0:
            if r != "explicit":
                b=min(b,int(floor(0.5*log(RAM/96)/log(2))))
            else:
                b=min(b,int(floor(log(RAM/8/3**2)/log(2)/2)))

        outdir="timings2r"
    if p == "cconv3":
        if a == 0:
            a=int(ceil(log(P)/(2*log(2))))
        if RAM != 0:
            if r != "explicit":
                b=min(b,int(floor(log(RAM/96)/log(2)/3)))
            else:
                b=min(b,int(floor(log(RAM/16/2**3)/log(2)/3)))

        outdir="timings3c"
    if p == "conv3":
        if a == 0:
            a=int(ceil(log(P)/(2*log(2))))
        if RAM != 0:
            b=min(b,int(floor(log(RAM/192)/log(2)/3)))
        outdir="timings3r"

    if RAM != 0:
        print "max problem size is "+str(2**b)

    if outdir == "":
        print "empty outdir: please select a different program!"
        print
        print usage
        sys.exit(2)

    outdir=outdir+"/"+str(P)+"x"+str(T)
    command=l+" "+str(P)+" "+M+"  ./"+str(p)

    print "output in "+outdir+"/"+r

    print "command: "+command+cargs+" "+A
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
        run=command+cargs+" -m "+str(int(pow(2,i)))+" "+A
        grepc=" | grep -A 1 "+rname+" | tail -n 1"
        cat=" | cat >> "+outdir+"/"+r
        print run
        sys.stdout.flush()
        #print "echo "+"$("+run+grepc+")"+cat
        if dryrun == False:
            os.system("echo "+"$("+run+grepc+")"+cat)
        else:
            print("echo "+"$("+run+grepc+")"+cat)
        #print run+grepc+" "+cat
        #os.system(run+grepc+" "+cat)
        #print run
        #os.system(run)
        sys.stdout.flush()

    if dryrun == False:
        os.system("sed -i 's/[ \t]*$//' "+outdir+"/"+r)
        os.system("sed -i '/^$/d' "+outdir+"/"+r)
if __name__ == "__main__":
    main(sys.argv[1:])
