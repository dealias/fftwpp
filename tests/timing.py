#!/usr/bin/python

# a timing script for FFTs and convolutions using OpenMP

import sys, getopt
import numpy as np
from math import *
from subprocess import * # for popen, running processes
import os
import re # regexp package

def max_m(p, RAM, runtype):
    print "runtype:", runtype
    b = 0
    if p == "cconv":
        b = int(floor(log(RAM / 4) / log(2)))
        b = min(b, 20) # because we aren't crazy
        
    if p == "cconv2":
        if runtype == "implicit":
            # A * 2m^2 * 16
            b = int(floor(log(RAM / 64) / ( 2 * log(2)) ))
        else:
            # A * 4m^2 * 16
            b = int(floor(log(RAM / 128) / (2 * log(2)) ))
            
    if p == "cconv3":
       if runtype == "implicit":
           # A * 2m^3 * 16
           b = int(floor( log(RAM / 64) / (3 * log(2)) ))
       else:
           # A * 8m^3 * 16
           b = int(floor( log(RAM / 256) / (3 * log(2)) ))

    if p == "tconv":
        b = int(floor(log(RAM / 6) / log(2)))
        b = min(b, 20) # because we aren't crazy

    if p == "tconv2":
        if runtype == "implicit":
            # A * 6m^2 * 16
            b = int(floor( log(RAM / 192) / (2 * log(2)) ))
        else:
            # A * 12m^2 * 16
            b = int(floor( log(RAM / 768) / (2 * log(2)) ))

    if p == "conv":
        b = int(floor(log(RAM / 6) / log(2)))
        b = min(b, 20) # because we aren't crazy
        
    if p == "conv2":
        if runtype == "implicit":
            # A * 3 m^2 * 16
            b = int(floor(log(RAM / 96) / (2 * log(2)) ))
        else:
            # A * 4.5 m^2 * 16
            b = int(floor(log(RAM / 144) / (2 * log(2)) ))
            
    if p == "conv3":
        # A * 6 m^3 * 16
        b = int(floor(log(RAM / 192) / (3 * log(2)) ))

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
        outdir = "timings1c"
    if p == "cconv2":
        outdir = "timings2c"
    if p == "cconv3":
        outdir = "timings3c"
    if p == "conv":
        outdir = "timings1r"
    if p == "conv2":
        outdir = "timings2r"
    if p == "conv3":
        outdir = "timings3r"
    if p == "tconv":
        outdir = "timings1t"
    if p == "tconv2":
        outdir="timings2t"
    if p == "fft1":
        outdir = "timingsf1"
    if p == "mfft1":
        outdir = "timingsmf1"
    if p == "fft2":
        outdir = "timingsf2"
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

    dryrun = False
    bset = 0
    dorun = True
    T = 0 # number of threads
    p = "" # program name
    B = [] # precommands
    A = [] # postcommands
    E = [] # environment variables (eg: -EGOMP_CPU_AFFINITY -E"0 1 2 3")
    a = 6  # minimum log of problem size
    b = 0  # maximum log of problem size
    runtype = "implicit"  # type of run
    RAM = 0  # ram limit in GB
    outdir = ""  # output directory
    outfile = "" # output filename
    rname = ""   # output grep string
    N = 0        # number of tests
    stats = 0
    
    try:
        opts, args = getopt.getopt(argv,"hdp:T:a:b:A:B:E:r:R:S:o:D:g:N:")
    except getopt.GetoptError:
        print "error in parsing arguments."
        print usage
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-p"):
            p = arg
        if opt in ("-T"):
            T = arg
        elif opt in ("-a"):
            a = int(arg)
        elif opt in ("-N"):
            N = int(arg)
        elif opt in ("-b"):
            b = int(arg)
        elif opt in ("-A"):
            A += [str(arg)]
        elif opt in ("-B"):
            B += [str(arg)]
        elif opt in ("-E"):
            E += [str(arg)]
        elif opt in ("-r"):
            runtype = str(arg)
        elif opt in ("-R"):
            RAM = float(arg)*2**30
        elif opt in ("-S"):
            stats = int(arg)
        elif opt in ("-d"):
            dryrun = True
        elif opt in ("-o"):
            outfile = str(arg)
        elif opt in ("-D"):
            outdir = str(arg)
        elif opt in ("-g"):
            rname = str(arg)
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
        b = 8

    hermitian = False
    ternary = False

    if p == "cconv":
        if(runtype == "pruned"):
            print p + " has no pruned option"
            dorun = False

    if p == "conv":
        hermitian = True
        if(runtype == "pruned"):
            print p + " has no pruned option"
            dorun = False
            
    if p == "conv2":
        hermitian = True
        
    if p == "conv3":
        hermitian = True
        if(runtype != "implicit"):
            print p + " has no " + r + " option"
            dorun = False

    if p == "tconv":
        ternary = True
        if(runtype == "pruned"):
            print p + " has no pruned option"
            dorun = False
            
    if p == "tconv2":
        ternary = True

    if p == "fft1":
        runtype = "fft"
        
    if p == "mfft1":
        runtype = "fft"
        
    if p == "fft2":
        runtype = "fft"
        
    if p == "transpose":
        runtype = "transpose"

    if outdir == "":
        outdir = default_outdir(p)

    if outdir == "":
        print "empty outdir: please select a program or specify an outdir (-D)"
        print
        print usage
        sys.exit(2)

    if RAM != 0:
        b = max_m(p, RAM, runtype)
        print "max value of b with ram provided:", b
        
    if outfile == "":
        outfile = "implicit"
        
    if dorun:
        if RAM != 0:
            print "max problem size is "+str(2**b)

        if rname == "":
            if runtype == "implicit":
                rname = "Implicit"
            if runtype == "explicit":
                rname = "Explicit"
            if runtype == "pruned":
                rname = "rune"
            if runtype == "fft":
                rname = "fft"
            if runtype == "transpose":
                rname = "transpose"

        print "Search string for timing: " + rname

        print "output in " + outdir + "/" + outfile

        print "environment variables:", E
        
        if not dryrun:
            os.system("mkdir -p " + outdir)
            os.system("rm -f " + outdir + "/" + outfile)

        cmd = []
        i = 0
        while i < len(B):
            cmd.append(B[i]);
            i += 1

        cmd += ["./" + str(p)]
        
        if(runtype == "explicit"):
            cmd.append("-e")
            
        if(runtype == "pruned"):
            cmd.append("-p")
            
        if(runtype == "implicit"):
            cmd.append("-i")
        
        cmd.append("-S" + str(stats))
        if(N > 0):
            cmd.append("-N" + str(N))
        if(T > 0):
            cmd.append("-T" + T)

        # Add the extra arguments to the program being timed.
        i = 0
        while i < len(A):
            cmd.append(A[i]);
            i += 1
            
        print " ".join(cmd)

        if(stats == -1):
            try:
                os.remove("timing.dat")
            except OSError:
                pass

        if not dryrun:
            filename = outdir + "/" + outfile
            if(stats == -1):
                filename = "timing.dat"
            with open(filename, "a") as myfile:
                myfile.write("# " + " ".join(cmd) + "\n")
                        
        for i in range(a, b + 1):
            if not hermitian or runtype == "implicit": 
                m = str(int(pow(2, i)))
            else:
                if not ternary:
                    m = str(int(floor((pow(2, i + 1) + 2) / 3)))
                else:
                    m = str(int(floor((pow(2, i + 2) + 3) / 4)))
                    
            print str(i) + " m=" + str(m)
            
            mcmd = cmd + ["-m" + str(m)]

            if dryrun:
                print mcmd
            else:
                denv = dict(os.environ)
                i = 0
                while i < len(E):
                    denv[E[i]] = E[i + 1]
                    i += 2
                    
                p = Popen(mcmd, stdout = PIPE, stderr = PIPE, env = denv)
                p.wait() # sets the return code
                prc = p.returncode
                out, err = p.communicate() # capture output
                if (prc == 0): # did the process succeed?
                    #print out
                    outlines = out.split('\n')
                    itline = 0
                    dataline = ""
                    while itline < len(outlines):
                        line = outlines[itline]
                        #print line
                        re.search(rname, line)
                        if re.search(rname, line) is not None:
                            print "\t" + str(outlines[itline])
                            print "\t" + str(outlines[itline + 1])
                            dataline = outlines[itline + 1]
                            itline = len(outlines)
                        itline += 1
                    if not dataline == "":
                        # append to output file
                        with open(outdir + "/" + outfile, "a") as myfile:
                            myfile.write(dataline + "\n")
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
                    
        if(stats == -1):
            os.rename("timing.dat", outdir + "/" + outfile)
            
            sys.stdout.flush()

if __name__ == "__main__":
    main(sys.argv[1:])
