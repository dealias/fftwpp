#!/usr/bin/python

# a timing script for FFTs and convolutions using OpenMP

import sys, getopt
import numpy as np
from math import *
from subprocess import * # for popen, running processes
import os
import re # regexp package


def main(argv):
    usage='Usage: \ntimings.py\n -a<start>\n -b<stop>\n -p<cconv,cconv2,cconv3,conv,conv2,conv3,tconv,tconv2,pcconv,pcconv2,pcconv3>\n -T<number of threads> -A<quoted arg list for timed program>\n -r<implicit/explicit/pruned/fft>\n -R<ram in gigabytes> -d -o<output file name>\n -D<outdir>\n -o<outfile>\n -<grep string>' 

    dryrun=False
    bset=0
    dorun=1
    Tset=0
    T=1
    p=""
    cargs=""
    A=""
    a=6
    b=0
    out="implicit"
    runtype="implicit"
    RAM=0
    outdir=""
    outfile=""
    rname="Implicit"
    
    try:
        opts, args = getopt.getopt(argv,"dp:T:a:b:A:r:R:o:D:g:")
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

    if out == "":
        if runtype == "implicit":
            outfile="implicit"
        if runtype == "explicit":
            outfile="explicit"
        if runtype == "pruned":
            outfile="pruned"
    else:
        outfile=out

    if p == "cconv":
        if RAM != 0:
            b=min(int(floor(log(RAM/4)/log(2))),b)
            b=min(b,14) # because we aren't crazy
        if outdir == "": outdir="timings1c"
        if(runtype == "pruned"):
            print p+" has no pruned option"
            dorun=0
    if p == "cconv2":
        if RAM != 0:
            if runtype == "implicit":
                b=min(int(floor(0.5*log(RAM/64)/log(2))),b)
            else:
                b=min(int(floor(log(RAM/16/2/2**2)/log(2)/2)),b)
        if outdir == "": outdir="timings2c"
    if p == "cconv3":
        if RAM != 0:
            if runtype == "implicit":
                b=min(int(floor(log(RAM/96)/log(2)/3)),b)
            else:
                b=min(int(floor(log(RAM/16/2**3)/log(2)/3)),b)
        if outdir == "": outdir="timings3c"

    if p == "conv":
        if RAM != 0:
            b=min(int(floor(log(RAM/6)/log(2))),b)
            b=min(b,14) # because we aren't crazy
        if outdir == "": outdir="timings1r"
        if(runtype == "pruned"):
            print p+" has no pruned option"
            dorun=0
    if p == "conv2":
        if RAM != 0:
            if runtype == "implicit":
                b=min(int(floor(0.5*log(RAM/96)/log(2))),b)
            else:
                b=min(int(floor(log(RAM/8/3**2)/log(2)/2)),b)
        if outdir == "": outdir="timings2r"
    if p == "conv3":
        if RAM != 0:
            b=min(int(floor(log(RAM/192)/log(2)/3)),b)
        if outdir == "": outdir="timings3r"
        if(runtype != "implicit"):
            print p+" has no "+r+" option"
            dorun=0

    if p == "tconv":
        if RAM != 0:
            b=int(floor(log(RAM/6)/log(2)))
            b=min(b,14) # because we aren't crazy
        if outdir == "": outdir="timings1t"
        if(runtype == "pruned"):
            print p+" has no pruned option"
            dorun=0
    if p == "tconv2":
        if RAM != 0:
            if runtype == "implicit":
                b=int(floor(log(RAM/(8*12))/(2*log(2))))
            else:
                b=int(floor(log(RAM/(8*6))/(2*log(2))))
        outdir="timings2t"

    if p == "pcconv":
        if RAM != 0:
            b=min(int(floor(log(RAM/4)/log(2))),b)
            b=min(b,14) # because we aren't crazy
        outdir="timings1cp"
        if(runtype == "pruned"):
            print p+" has no pruned option"
            dorun=0
    if p == "pcconv2":
        if RAM != 0:
            if runtype == "implicit":
                b=min(int(floor(0.5*log(RAM/64)/log(2))),b)
            else:
                b=min(int(floor(log(RAM/16/2/2**2)/log(2)/2)),b)
        if outdir == "": outdir="timings2cp"
    if p == "pcconv3":
        if RAM != 0:
            if runtype == "implicit":
                b=min(int(floor(log(RAM/96)/log(2)/3)),b)
            else:
                b=min(int(floor(log(RAM/16/2**3)/log(2)/3)),b)
        if outdir == "": outdir="timings3cp"
    if p == "c2cfft2":
        if RAM != 0:
            b=min(int(floor(0.5*log(RAM/64)/log(2))),b)
        if outdir == "": outdir="timings2c2c"
        runtype="fft"
        rname="fft"
    if outdir == "":
        print "empty outdir: please select a different program!"
        print
        print usage
        sys.exit(2)


    if(dorun == 1):
        if RAM != 0:
            print "max problem size is "+str(2**b)

        if runtype == "implicit":
            rname="Implicit"
        if runtype == "explicit":
            rname="Explicit"
        if runtype == "pruned":
            rname="rune"
        if runtype == "fft":
            rname="fft"

        print "output in "+outdir+"/"+outfile

        if not dryrun:
            os.system("mkdir -p "+outdir)
            os.system("rm -f "+outdir+"/"+outfile)
            
        cmd=["./"+str(p)]
        if(runtype == "explicit"):
            cmd.append("-e")
        if(runtype == "pruned"):
            cmd.append("-p")
        if(runtype == "implicit"):
            cmd.append("-i")
        if(Tset):
            cmd.append("-T"+T)
            

        print cmd


        for i in range(a,b+1):
            m=str(int(pow(2,i)))
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
                            #print outlines[itline+1]
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
