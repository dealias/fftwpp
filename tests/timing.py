#!/usr/bin/python

# a timing script for FFTs and convolutions using OpenMP

import sys, getopt
import numpy as np
from math import *
from subprocess import * # for popen, running processes
import os
import re # regexp package

def max_m(p,RAM,runtype):
    b = 0
    if p == "cconv":
        b = int(floor(log(RAM / 4) / log(2)))
        b = min(b, 20) # because we aren't crazy
    if p == "cconv2":
        if runtype == "implicit":
            b = int(floor(0.5 * log(RAM / 64) / log(2)))
        else:
            b = int(floor(log(RAM / 16 / 2 / 2**2) / log(2) / 2))
    if p == "cconv3":
       if runtype == "implicit":
           b = int(floor(log(RAM / 96) / log(2) / 3))
       else:
           b = int(floor(log(RAM / 16 / 2**3) / log(2) / 3))

    if p == "tconv":
        b = int(floor(log(RAM / 6) / log(2)))
        b = min(b, 20) # because we aren't crazy
    if p == "tconv2":
        if runtype == "implicit":
            b = int(floor(log(RAM / (8 * 12)) / (2 * log(2))))
        else:
            b = int(floor(log(RAM / (8 * 6)) / (2 * log(2))))

    if p == "conv":
        b = int(floor(log(RAM / 6) / log(2)))
        b = min(b, 20) # because we aren't crazy
    if p == "conv2":
        if runtype == "implicit":
            b = int(floor(0.5 * log(RAM / 96) / log(2)))
        else:
            b = int(floor(log(RAM / 8 / 3**2)/log(2) / 2))
    if p == "conv3":
        b = int(floor(log(RAM / 192) / log(2) / 3))

    if p == "fft1":
        b = int(floor(0.5 * log(RAM / 64) / log(2)))
    if p == "mft1":
        b = int(floor(0.5 * log(RAM / 64) / log(2)))
    if p == "ft2":
        b = int(floor(0.5 * log(RAM / 64) / log(2)))

    return b

def default_outdir(p):
    outdir=""

    if p == "cconv":
        outdir="timings1c"
    if p == "cconv2":
        outdir="timings2c"
    if p == "cconv3":
        outdir="timings3c"

    if p == "conv":
        outdir="timings1r"
    if p == "conv2":
        outdir="timings2r"
    if p == "conv3":
        outdir="timings3r"

    if p == "tconv":
        outdir="timings1t"
    if p == "tconv2":
        outdir="timings2t"

    if p == "fft1":
        outdir="timingsf1"
    if p == "mfft1":
        outdir="timingsmf1"
    if p == "fft2":
        outdir="timingsf2"
    if p == "transpose":
        outdir="transpose2"

    return outdir


def main(argv):
    usage = '''Usage:
    \ntimings.py
    -a<start>
    -b<stop>
    -p<cconv,cconv2,cconv3,conv,conv2,conv3,tconv,tconv2,pcconv,pcconv2,pcconv3>
    -T<number of threads> 
    -A<quoted arg list for timed program>
    -B<pre-commands (eg srun)>
    -r<implicit/explicit/pruned/fft>
    -R<ram in gigabytes> 
    -d dry run
    -o<output file name>
    -D<outdir>
    -o<outfile>
    -g<grep string>
    -N<int> Number of tests to perform
    '''

    dryrun=False
    bset=0
    dorun=True
    Tset=False
    T=1
    p=""
    cargs=""
    B=[]
    A=[]
    a=6
    b=0
    out=""
    runtype="implicit"
    RAM=0
    outdir=""
    outfile=""
    rname=""
    N=0


    try:
        opts, args = getopt.getopt(argv,"hdp:T:a:b:A:B:r:R:o:D:g:N:")
    except getopt.GetoptError:
        print "error in parsing arguments."
        print usage
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-p"):
            p=arg
        if opt in ("-T"):
            Tset=True
            cargs+=" -T"+str(arg)
            T=arg
        elif opt in ("-a"):
            a=int(arg)
        elif opt in ("-N"):
            N=int(arg)
        elif opt in ("-b"):
            b=int(arg)
        elif opt in ("-A"):
            A+=[str(arg)]
        elif opt in ("-B"):
            B+=[str(arg)]
        elif opt in ("-r"):
            runtype=str(arg)
        elif opt in ("-R"):
            RAM=float(arg)*2**30
        elif opt in ("-d"):
            dryrun=True
        elif opt in ("-o"):
            out=str(arg)
        elif opt in ("-D"):
            outdir=str(arg)
        elif opt in ("-g"):
            rname=str(arg)
        elif opt in ("-h"):
            print usage
            sys.exit(0)
            
    if dryrun:
        print "Dry run!  No output actually created."

    if p == "":
        print "please specify a program with -p"
        print usage
        sys.exit(2)

    # if both the max problem size and the ram are unset, go up to 2^8
    if (b == 0 and RAM == 0):
        b=8
    # if RAM is set and the max problem size is not set, then RAM
    # determines the problem size on its own.
    if (b == 0 and RAM != 0):
        b=sys.maxint

    hermitian=False
    ternary=False

    if p == "cconv":
        if(runtype == "pruned"):
            print p+" has no pruned option"
            dorun=False

    if p == "conv":
        hermitian=True
        if(runtype == "pruned"):
            print p+" has no pruned option"
            dorun=False
    if p == "conv2":
        hermitian=True
    if p == "conv3":
        hermitian=True
        if(runtype != "implicit"):
            print p+" has no "+r+" option"
            dorun=False

    if p == "tconv":
        ternary=True
        if(runtype == "pruned"):
            print p+" has no pruned option"
            dorun=False
    if p == "tconv2":
        ternary=True

    if p == "fft1":
        runtype="fft"
    if p == "mfft1":
        runtype="fft"
    if p == "fft2":
        runtype="fft"
    if p == "transpose":
        runtype="transpose"

    if outdir == "":
        outdir=default_outdir(p)

    if outdir == "":
        print "empty outdir: please select a program or specify an outdir (-D)"
        print
        print usage
        sys.exit(2)

    if RAM != 0:
        b = max_m(p, RAM, runtype)
        print "max value of b with ram provided:", b
        
    if out == "":
        if runtype == "implicit":
            outfile="implicit"
        if runtype == "explicit":
            outfile="explicit"
        if runtype == "pruned":
            outfile="pruned"
        if runtype == "fft":
            outfile="fft"
        if runtype == "transpose":
            outfile="tranpose"
    else:
        outfile=out

    if dorun:
        if RAM != 0:
            print "max problem size is "+str(2**b)

        if rname == "":
            if runtype == "implicit":
                rname="Implicit"
            if runtype == "explicit":
                rname="Explicit"
            if runtype == "pruned":
                rname="rune"
            if runtype == "fft":
                rname="fft"
            if runtype == "transpose":
                rname="transpose"

        print "Search string for timing: "+rname

        print "output in "+outdir+"/"+outfile

        if not dryrun:
            os.system("mkdir -p "+outdir)
            os.system("rm -f "+outdir+"/"+outfile)

        cmd=[]
        i=0
        while i < len(B):
            cmd.append(B[i]);
            i += 1

        cmd+=["./"+str(p)]

        if(N > 0):
            cmd.append("-N" + str(N))

        if(runtype == "explicit"):
            cmd.append("-e")
        if(runtype == "pruned"):
            cmd.append("-p")
        if(runtype == "implicit"):
            cmd.append("-i")
        if(Tset):
            cmd.append("-T"+T)

        # Add the extra arguments to the program being timed.
        i=0
        while i < len(A):
            cmd.append(A[i]);
            i += 1
            
        print cmd

        for i in range(a,b+1):
            if not hermitian or runtype == "implicit": 
                m=str(int(pow(2,i)))
            else:
                if not ternary:
                    m=str(int(floor((pow(2,i+1)+2)/3)))
                else:
                    m=str(int(floor((pow(2,i+2)+3)/4)))
                    
            print str(i)+" m="+str(m)
            

            mcmd=cmd+["-m"+str(m)]

            if dryrun:
                print mcmd
            else:
                p=Popen(mcmd,stdout=PIPE,stderr=PIPE)
                p.wait() # sets the return code
                prc=p.returncode
                out, err = p.communicate() # capture output
                if (prc == 0): # did the process succeed?
                    #print out
                    outlines=out.split('\n')
                    itline=0
                    dataline=""
                    while itline < len(outlines):
                        line=outlines[itline]
                        #print line
                        re.search(rname,line)
                        if re.search(rname, line) is not None:
                            print "\t"+str(outlines[itline])
                            print "\t"+str(outlines[itline+1])
                            dataline=outlines[itline+1]
                            itline=len(outlines)
                        itline += 1
                    if not dataline == "":
                        # append to output file
                        with open(outdir+"/"+outfile, "a") as myfile:
                            myfile.write(dataline+"\n")
                    else:
                        print "ERROR: no timing data found"
                else:
                    print "FAILURE:"
                    print cmd
                    print "with, return code:"
                    print prc
                    print "output:"
                    print out
                    print "error:"
                    print err
                    
            sys.stdout.flush()

if __name__ == "__main__":
    main(sys.argv[1:])
