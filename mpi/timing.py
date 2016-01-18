#!/usr/bin/python

# a timing script for convolutions using MPI.
import subprocess

import sys, getopt
import numpy as np
from math import *
from subprocess import * # for popen, running processes
import os
import re # regexp package

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

def main(argv):
    usage = '''
A python script for launching MPI test runs for fftw++
Usage:
./timing.py:
\t-a <int> specifies the max exponent of the min problem size 
\t-b <int> specifies the max exponent of the max problem size 
\t-d Launch a dry run.'
\t-l Extra MPI command (eg: -l "mpiexec" -l "--np")
\t-o <filename> The output filename.
\t-p <string> The program to be tested (eg: -p cconv2)
\t-r <implicit, explicit, fft, transpose> the type of run
\t-A <string> Additional arguments for the program being tested.
\t-D <directory> Output directory
\t-E <string> Environment options (eg: 
\t-M <string> Additional MPI launch arguments
\t-P <int> Number of MPI processes eg: -E"HYDRA_TOPO_DEBUG" -E"1"
\t-S <int> Choice ofstatistical output (-1: raw data, 0: mean, 1: median)
\t-R <float> Max amount of RAM that can be used (in GB).

For example, try the command:\n./timing.py -pcconv2 -a 3 -b 4
    '''

    dryrun = False

    mpiruncmd = [] # MPI launch arguments
    mpirunextra = []  # Extra MPI launch arguments (after number of procs)
    P = 1   # Number of MPI procs
    T = 1  # number of threads
    p = "" # program name
    runtype = "implicit"  # type of run
    cargs = ""
    A = [] # Additional arguments for the program
    E = [] # Environment options (eg: -E"HYDRA_TOPO_DEBUG" -E"1"
    a = 6
    b = 8
    stats = 0
    outdir = "" # output directory
    outfile = "" #output filename
    RAM = 0 # ram limit in GB
    
    try:
        opts, args = getopt.getopt(argv,"a:b:d:l:o:p:r:A:D:E:M:P:S:T:R:")
    except getopt.GetoptError:
        print usage
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-a"):
            a = int(arg)
        elif opt in ("-b"):
            b = int(arg)
        elif opt in ("-d"):
            dryrun=True
        elif opt in ("-l"):
            mpiruncmd.append(str(arg))
        elif opt in ("-o"):
            outfile = str(arg)
        elif opt in ("-p"):
            p = arg
        elif opt in ("-r"):
            runtype = str(arg)
        elif opt in ("-A"):
            A.append(str(arg))
        elif opt in ("-D"):
            outdir = str(arg)
        elif opt in ("-E"):
            E.append(str(arg))
        elif opt in ("-M"):
            mpirunextra.append(str(arg))
        elif opt in ("-P"):
            P = int(arg)
        elif opt in ("-S"):
            stats = int(arg)
        elif opt in ("-T"):
            T = int(arg)
        elif opt in ("-R"):
            RAM = float(arg)*2**30
    if dryrun:
        print "Dry run!  No output actually created."

    if p == "":
        print "please specify a program with -p"
        print usage
        sys.exit(2)

    if outdir == "":
        outdir = default_outdir(p)

    if outfile == "":
        outfile = runtype
            
    if outdir == "":
        print "empty outdir: please select a different program!"
        print
        print usage
        sys.exit(2)

    if(mpiruncmd == []):
        mpiruncmd.append("mpiexec")
        mpiruncmd.append("--np")
    mpiruncmd.append(str(P))
    i = 0
    while i < len(mpirunextra):
        mpiruncmd.append(mpirunextra[i])
        i += 1
        
    # TODO: the mpi run command doesn't always take the number of
    # processes, for example on Stampede.

    print "runtype:", runtype

    command = []
    if(runtype == "explicit"):
        command.append("explicit/" + p)
    else:
        command.append("./" + p)
        command.append("-T" + str(T))
    command.append("-S" + str(stats))
    i = 0
    while(i < len(A)):
        command.append(A[i])
        i += 1
    
    print "Output in " + outdir + "/" + outfile
    logname = outdir + "/log"
    print "Log of run in " + logname

    print "environment variables:", E
    
    rname = "Implicit"
    if runtype == "explicit":
        rname = "Explicit"
    if runtype == "fft":
        rname = "FFT"
    if runtype == "transpose":
        rname = "transpose"
    if(stats != -1):
        print "Search string for timing: " + rname
        
    if RAM != 0:
        b = max_m(p, RAM, runtype)
        print "max value of b with ram provided:", b
        
    if not dryrun:
        os.system("mkdir -p " + outdir)
        os.system("rm -f " + outdir + "/" + outfile)
        if(stats == -1):
            os.system("rm -f timing.dat")
        
    print "Max size: " + str(b) + ": " +  str(int(pow(2, b)))
    for logm in range(a, b + 1):
        m = int(pow(2, logm))
        mcommand = []
        i = 0
        while i < len(mpiruncmd):
            mcommand.append(mpiruncmd[i])
            i += 1
        i = 0
        while i < len(command):
            mcommand.append(command[i])
            i += 1
        mcommand.append("-m" + str(m))
        print logm, m, mcommand
        print " ".join(mcommand)
        if(not dryrun):
            
            denv = dict(os.environ)
            i = 0
            while i < len(E):
                denv[E[i]] = E[i + 1]
                i += 2
                    
            p = Popen(mcommand, stdout = PIPE, stderr = PIPE, env = denv)
            p.wait() # sets the return code
            prc = p.returncode
            out, err = p.communicate() # capture output
            os.system("touch "+outdir+"/log")
            with open(logname, "a") as logfile:
                logfile.write(out)
            with open(logname, "a") as logfile:
                logfile.write(err)
            
            if (prc == 0): # did the process succeed?
                with open(logname, "a") as logfile:
                    logfile.write("SUCCESS\n")
                if(stats != -1):
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
                with open(logname, "a") as logfile:
                    logfile.write("FAILURE\n")
                print "FAILURE:"
                print mcommand
                print "with, return code:"
                print prc
    if(stats == -1):
        os.rename("timing.dat", outdir + "/" + outfile)

    print("\ntiming finished.") # Time for a beer!

if __name__ == "__main__":
    main(sys.argv[1:])
