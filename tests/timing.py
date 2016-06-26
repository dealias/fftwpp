#!/usr/bin/python

# a timing script for FFTs and convolutions using OpenMP

import sys, getopt
import numpy as np
from math import *
from subprocess import * # for popen, running processes
import os
import re # regexp package
import shutil


def mvals_from_file(filename):
    mvals = []
    if os.path.isfile(filename):
        with open(filename, 'r') as fin:
            for line in fin:
                if not line.startswith("#"):
                    mvals.append(int(line.split()[0]))
    return mvals

def max_m(p, RAM, runtype):
    print "program:", p
    print "runtype:", runtype
    print "ram:", RAM
    
    b = 0
    if "transpose" in p:
        # NB: assumes Z=1 and out-of-place
        return int(floor(log(RAM / 32) / log(2) / 2))
    
    if "cconv2" in p:
        if runtype == "implicit":
            # A * 2m^2 * 16
            return int(floor(log(RAM / 64) / ( 2 * log(2)) ))
        else:
            # A * 4m^2 * 16
            return int(floor(log(RAM / 128) / (2 * log(2)) ))

    if "cconv3" in p:
       if runtype == "implicit":
           # A * 2m^3 * 16
           return int(floor( log(RAM / 64) / (3 * log(2)) ))
       else:
           # A * 8m^3 * 16
           return int(floor( log(RAM / 256) / (3 * log(2)) ))

    if "cconv" in p:
        b = int(floor(log(RAM / 4) / log(2)))
        b = min(b, 20) # because we aren't crazy
        return b
  
    if "tconv2" in p:
        if runtype == "implicit":
            # A * 6m^2 * 16
            return int(floor( log(RAM / 192) / (2 * log(2)) ))
        else:
            # A * 12m^2 * 16
            return int(floor( log(RAM / 768) / (2 * log(2)) ))
    
    if "tconv" in p:
        b = int(floor(log(RAM / 6) / log(2)))
        b = min(b, 20) # because we aren't crazy
        return b
        
    if "conv2" in p:
        if runtype == "implicit":
            # A * 3 m^2 * 16
            return int(floor(log(RAM / 96) / (2 * log(2)) ))
        else:
            # A * 4.5 m^2 * 16
            return int(floor(log(RAM / 144) / (2 * log(2)) ))
            
    if "conv3" in p:
        # A * 6 m^3 * 16
        return int(floor(log(RAM / 192) / (3 * log(2)) ))
        
    if "conv" in p:
        b = int(floor(log(RAM / 6) / log(2)))
        b = min(b, 20) # because we aren't crazy
        return b

    if "mft1" in p:
        return int(floor(0.5 * log(RAM / 64) / log(2)))
    
    if "fft1" in p:
        return int(floor(0.5 * log(RAM / 64) / log(2)))
        
    if p == "fft2":
        return int(floor(log(RAM / 32) / log(2) / 2))

    if p == "fft2r":
        return int(floor(log(RAM / 32) / log(2) / 2))

    if p == "fft3":
        return int(floor(log(RAM / 32) / log(2) / 3))

    if p == "fft3r":
        return int(floor(log(RAM / 32) / log(2) / 3))

    if p == "transpose":
        return int(floor(log(RAM / 32) / log(2) / 2))

    print "Error! Failed to determine b."
    return 0

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
    -p<cconv,cconv2,cconv3,conv,conv2,conv3,tconv,tconv2>
    -T<number of threads> 
    -A<quoted arg list for timed program>
    -B<pre-commands (eg srun)>
    -r<implicit/explicit/pruned/fft>
    -R<ram in gigabytes> 
    -d dry run
    -o<output file name>
    -D<outdir>
    -o<outfile>
    -P<path to executable>
    -g<grep string>
    -N<int> Number of tests to perform
    -e<0 or 1>: append to the timing data already existent (skipping 
           already-done problem sizes).
    -c<string>: extra commentary for output file.
    -v: verbose output
    '''

    dryrun = False
    #dryrun = True
    

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
    appendtofile = False
    stats = 0
    path = "./"
    verbose = False
    extracomment = ""
    
    try:
        opts, args = getopt.getopt(argv,"hdp:T:a:b:c:A:B:E:e:r:R:S:o:P:D:g:N:v")
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
        elif opt in ("-c"):
            extracomment = arg
        elif opt in ("-A"):
            A += [str(arg)]
        elif opt in ("-B"):
            B += [str(arg)]
        elif opt in ("-E"):
            E += [str(arg)]
        elif opt in ("-e"):
            appendtofile = (int(arg) == 1)
        elif opt in ("-r"):
            runtype = str(arg)
        elif opt in ("-R"):
            print "ram arg:", arg
            RAM = float(arg)*2**30
        elif opt in ("-S"):
            stats = int(arg)
        elif opt in ("-d"):
            dryrun = True
        elif opt in ("-o"):
            outfile = str(arg)
        elif opt in ("-P"):
            path = arg
        elif opt in ("-D"):
            outdir = str(arg)
        elif opt in ("-g"):
            rname = str(arg)
        elif opt in ("-v"):
            verbose = True
        elif opt in ("-h"):
            print usage
            sys.exit(0)
            
    if dryrun:
        print "Dry run!  No output actually created."

    if p == "":
        print "please specify a program with -p"
        print usage
        sys.exit(2)

    print "RAM:", RAM
        
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

    goodruns = []
    badruns = []

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

        filename = outdir + "/" + outfile
        print "output in", filename

        mdone = mvals_from_file(filename)
        print "problem sizes already done:", mdone
        
        print "environment variables:", E
        
        if not dryrun:
            os.system("mkdir -p " + outdir)
            with open(outdir + "/log", "a") as logfile:
                logfile.write(str(sys.argv))
                logfile.write("\n")
                logfile.write("intial exponent: " + str(a) + "\n")
                logfile.write("final exponent: " + str(b) + "\n")
            if not appendtofile:
                os.system("rm -f " + filename)

        cmd = []
        i = 0
        while i < len(B):
            cmd.append(B[i]);
            i += 1

        cmd += [path + str(p)]

        if not os.path.isfile(path + str(p)):
            print path + str(p), "does not exist!"
            sys.exit(1)
                        

        if not "fft" in p:
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

        if not dryrun and stats == -1:
            try:
                os.remove("timing.dat")
            except OSError:
                pass

            
        if not dryrun:
            comment = "#"

            # Add the base run command as a comment
            comment += " " + " ".join(cmd)

            # Add the run date as a comment
            import time
            date = time.strftime("%Y-%m-%d")
            comment += "\t" + date

            if extracomment == "":
                # If we can get the commit and commit date, add as a comment
                vcmd = []
                vcmd.append("git")
                vcmd.append("log")
                vcmd.append("-1")
                vcmd.append("--format=%h")
                vp = Popen(vcmd, stdout = PIPE, stderr = PIPE)
                vp.wait()
                prc = vp.returncode
                if prc == 0:
                    out, err = vp.communicate()
                    comment += "\t" + out.rstrip()

                vcmd = []
                vcmd.append("git")
                vcmd.append("log") 
                vcmd.append("-1")
                vcmd.append("--format=%ci")
                vp = Popen(vcmd, stdout = PIPE, stderr = PIPE)
                vp.wait()
                prc = vp.returncode
                if prc == 0:
                    out, err = vp.communicate()
                    out = out.rstrip()
                    comment += " (" + out[0:10] + ")"
            else:
                comment += "\t" + extracomment
                    
            comment += "\n"
            
            if(appendtofile):
                if os.path.isfile(filename):
                    with open(filename, "a") as myfile:
                        myfile.write(comment)
                else:
                    with open(filename, "a") as myfile:
                        myfile.write(comment)
            else:
                if stats == -1:
                    with open("timing.dat", "w") as myfile:
                        myfile.write(comment)
                else:
                    with open(filename, "w") as myfile:
                        myfile.write(comment)

        for i in range(a, b + 1):
            if not hermitian or runtype == "implicit": 
                m = str(int(pow(2, i)))
            else:
                if not ternary:
                    m = str(int(floor((pow(2, i + 1) + 2) / 3)))
                else:
                    m = str(int(floor((pow(2, i + 2) + 3) / 4)))
                    
            print str(i) + " m=" + str(m)

            dothism = True
            
            if appendtofile and int(m) in mdone:
                print "problem size", m, "is already done; skipping."
                dothism = False
                
            if dothism:
                mcmd = cmd + ["-m" + str(m)]

                if dryrun:
                    #print mcmd
                    print " ".join(mcmd)
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
                    if(verbose):
                        print "Output from timing.py's popen:"
                        #print " ".join(mcmd)
                        print "cwd:" , os.getcwd()
                        print "out:"
                        print out
                        print "err:"
                        print err

                    # copy the output and error to a log file.
                    with open(outdir + "/log", "a") as logfile:
                        logfile.write(" ".join(mcmd))
                        logfile.write("\n")
                        logfile.write(out)
                        logfile.write(err)

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
                        
                        if not stats == -1:
                            if not dataline == "":
                                goodruns.append(mcmd)
                                # append to output file
                                with open(filename, "a") as myfile:
                                    myfile.write(dataline + "\n")
                            else:
                                print "ERROR: no timing data found"
                                badruns.append(mcmd)
                        else:
                            goodruns.append(mcmd)
                    else:
                        print "FAILURE:"
                        print cmd
                        print "with, return code:"
                        print prc
                        print "output:"
                        print out
                        print "error:"
                        print err
                        badruns.append(mcmd)
                        
            if not dryrun and (stats == -1 and os.path.isfile("timing.dat")):
                if(appendtofile):
                    # Append the new data ot the output.
                    with open(filename, "a") as fout:
                        with open("timing.dat") as fin:
                            lines = []
                            for line in fin:
                                lines.append(line)
                            fout.write(lines[len(lines) - 1])
                                
                else:
                    shutil.copyfile("timing.dat", filename)
                    
        if not dryrun and stats == -1:
            try:
                os.remove("timing.dat")
            except OSError:
                pass

    if not dryrun:
         with open(outdir + "/log", "a") as logfile:
            goodbads = ""
            if len(goodruns) > 0:
                goodbads += "Successful runs:\n"
                for mcmd in goodruns:
                    goodbads += " ".join(mcmd) + "\n"
            if len(badruns) > 0:
                goodbads += "Unsuccessful runs:\n"
                for mcmd in badruns:
                    goodbads += " ".join(mcmd) + "\n"
            logfile.write(goodbads)
            print goodbads


            
if __name__ == "__main__":
    main(sys.argv[1:])
