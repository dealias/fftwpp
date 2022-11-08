#!/usr/bin/env python3

# a timing script for FFTs and convolutions using OpenMP

import sys, getopt
from math import *
from subprocess import * # for Popen, running processes
import os
import re # regexp package
import shutil

def getParam(name, comment):
  try:
    param=re.search(r"(?<="+name+"=)\d+",comment).group()
    return param
  except:
    print("Could not find "+name)
    return -1

def collectParams(comment,L,M):
  m=getParam("m",comment)
  p=getParam("p",comment)
  q=getParam("q",comment)
  C=getParam("C",comment)
  S=getParam("S",comment)
  D=getParam("D",comment)
  I=getParam("I",comment)

  params=str(L)+" "+str(M)+" "+m+" "+p+" "+q+" "+C+" "+S+" "+D+" "+I
  return params

def Lvals_from_file(filename):
    Lvals = []
    if os.path.isfile(filename):
        with open(filename, 'r') as fin:
            for line in fin:
                if not line.startswith("#"):
                    Lvals.append(int(line.split()[0]))
    return Lvals

def max_m(p, RAM, runtype):
    print("program:", p)
    print("runtype:", runtype)
    print("ram:", RAM)

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

    if "fft1" in p :
        return int(floor(0.5 * log(RAM / 64) / log(2)))

    if p == "fft2":
        return int(floor(log(RAM / 32) / log(2) / 2))

    if p == "fft2h":
        return int(floor(log(RAM / 32) / log(2) / 2))

    if p == "fft3":
        return int(floor(log(RAM / 32) / log(2) / 3))

    if p == "fft3h":
        return int(floor(log(RAM / 32) / log(2) / 3))

    if p == "transpose":
        return int(floor(log(RAM / 32) / log(2) / 2))

    print("Error! Failed to determine b.")
    return 0

def default_outdir(p):
    outdir=""
    if p == "cconv" or p == "hybridconv":
        outdir = "timings1c"
    if p == "cconv2" or p == "hybridconv2":
        print(p)
        outdir = "timings2c"
    if p == "cconv3" or p == "hybridconv3":
        outdir = "timings3c"
    if p == "conv" or p == "hybridconvh" :
        outdir = "timings1h"
    if p == "conv2" or p == "hybridconvh2":
        outdir = "timings2h"
    if p == "conv3" or p == "hybridconvh3":
        outdir = "timings3h"
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
    -I<increment (if not testing powers of 2)>
    -p<cconv,hybridconv,cconv2,cconv3,conv,hybridconv,conv2,conv3,tconv,tconv2>
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
    -S<int> Type of statistics (default 3=MEDIAN)
    -e: erase existing timing data
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
    I = 0  # optional increment between tests (0 means use powers of 2)
    a = 6  # minimum log2 of problem size if i=0 else minimum problem size
    b = 0  # maximum log2 of problem size if i=0 else maximum problem size
    runtype = "implicit"  # type of run
    RAM = 0  # ram limit in GB
    outdir = ""  # output directory
    outfile = "" # output filename
    rname = ""   # output grep string
    N = 0        # number of tests
    appendtofile = True
    stats = 3
    path = "."
    verbose = False
    extracomment = ""

    try:
        opts, args = getopt.getopt(argv,"dhep:T:a:b:c:I:A:B:E:r:R:S:o:P:D:g:N:v")
    except getopt.GetoptError:
        print("error in parsing arguments.")
        print(usage)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-p"):
            p = arg
        if opt in ("-T"):
            T = int(arg)
        elif opt in ("-a"):
            a = int(arg)
        elif opt in ("-N"):
            N = int(arg)
        elif opt in ("-b"):
            b = int(arg)
        elif opt in ("-I"):
            I = int(arg)
        elif opt in ("-c"):
            extracomment = arg
        elif opt in ("-A"):
            A += [str(arg)]
        elif opt in ("-B"):
            B += [str(arg)]
        elif opt in ("-E"):
            E += [str(arg)]
        elif opt in ("-e"):
            appendtofile = False
        elif opt in ("-r"):
            runtype = str(arg)
        elif opt in ("-R"):
            print("ram arg:", arg)
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
            print(usage)
            sys.exit(0)

    if dryrun:
        print("Dry run!  No output actually created.")

    if p == "":
        print("please specify a program with -p")
        print(usage)
        sys.exit(2)

    print("RAM:", RAM)

    # if both the max problem size and the ram are unset, go up to 2^8
    if (b == 0 and RAM == 0):
        b = 8

    hybrid = False

    # For hybrid convolutions.
    dimension=1

    hermitian = False
    ternary = False

    if p == "hybridconv2" or p == "hybridconvh2":
        dimension=2

    if p == "hybridconv3" or p == "hybridconvh3":
        dimension=3

    if p == "cconv":
        if(runtype == "pruned"):
            print(p + " has no pruned option")
            dorun = False

    if re.search("hybrid",p) is not None:
        hybrid=True
        runtype=None

    if p == "conv":
        hermitian = True
        if(runtype == "pruned"):
            print(p + " has no pruned option")
            dorun = False

    if p == "hybridconvh":
        hermitian = True

    if p == "conv2":
        hermitian = True

    if p == "conv3":
        hermitian = True
        if(runtype != "implicit"):
            print(p + " has no " + r + " option")
            dorun = False

    if p == "tconv":
        ternary = True
        if(runtype == "pruned"):
            print(p + " has no pruned option")
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
        print("empty outdir: please select a program or specify an outdir (-D)")
        print()
        print(usage)
        sys.exit(2)

    if RAM != 0:
        b = max_m(p, RAM, runtype)
        print("max value of b with ram provided:", b)

    if outfile == "":
        if hybrid:
            outfile = "hybrid"
        else:
            outfile = runtype

    if hybrid:
        optFile=outdir+os.sep+"hybridParams"

    goodruns = []
    badruns = []

    if dorun:
        if RAM != 0:
            print("max problem size is "+str(2**b))

        if rname == "":
            if hybrid:
                rname = "Hybrid"
            else:
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
            rname += ':'
        print("Search string for timing: " + rname)

        filename = outdir + os.sep + outfile
        print("output in", filename)

        Ldone = Lvals_from_file(filename)
        if appendtofile:
            print("problem sizes already done:", Ldone)

        print("environment variables:", E)

        if not dryrun:
            try:
                os.makedirs(outdir)
            except:
                pass
            with open(outdir + os.sep+"log", "a") as logfile:
                logfile.write(str(sys.argv))
                logfile.write("\n")
                logfile.write("intial exponent: " + str(a) + "\n")
                logfile.write("final exponent: " + str(b) + "\n")
            if not appendtofile:
                try:
                    os.remove(filename)
                    if hybrid:
                        os.remove(optFile)
                except:
                    pass

        cmd = []
        i = 0
        while i < len(B):
            cmd.append(B[i]);
            i += 1

        path += os.sep
        cmd += [path + str(p)]

        if not os.path.isfile(path + str(p)):
            print(path + str(p), "does not exist!")
            sys.exit(1)


        if not hybrid and not "fft" in p:
            if(runtype == "explicit"):
                cmd.append("-e")

            if(runtype == "pruned"):
                cmd.append("-p")

            if(runtype == "implicit"):
                cmd.append("-i")

        cmd.append("-S" + str(stats))
        if N > 0:
            cmd.append(("-K" if hybrid else "-N") + str(N))
        if T > 0:
            cmd.append("-T" + str(T))
        cmd.append("-u")

        # Add the extra arguments to the program being timed.
        i = 0
        while i < len(A):
            cmd.append(A[i]);
            i += 1

        print(" ".join(cmd))

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
                    comment += "\t" + out.rstrip().decode()

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
                    comment += " (" + out[0:10].decode() + ")"
            else:
                comment += "\t" + extracomment
            comment += "\n"

            if(appendtofile):
                with open(filename, "a") as myfile:
                    myfile.write(comment)
                if hybrid:
                    with open(optFile,"a") as logfile:
                        logfile.write("#\n"+comment)
            else:
                if stats == -1:
                    with open("timing.dat", "w") as myfile:
                        myfile.write(comment)
                else:
                    with open(filename, "w") as myfile:
                        myfile.write(comment)
                    if hybrid:
                        with open(optFile,"w") as logfile:
                            logfile.write("# Optimal values for "+p+"\n")
                            logfile.write("# L M m p q C S D I"+"\n")
                            logfile.write("#\n"+comment)

        for i in range(a,b+1,1 if I == 0 else I):
            if I != 0:
                m=i
            elif not hermitian or runtype == "implicit": #or hybrid:
                m = int(pow(2,i))
            else:
                if not ternary:
                    m = int((pow(2,i+1)+2) // 3)
                else:
                    m = int((pow(2,i+2)+3) // 4)

            print(str(i) + " m=" + str(m))

            dothism = True

            L = 2*m-1 if hermitian else m
            M = 3*m-2 if hermitian else 2*m

            alreadyDone=(dimension == 1 and L in Ldone) or\
                        (dimension == 2 and L**2 in Ldone) or\
                        (dimension == 3 and L**3 in Ldone)

            if appendtofile and alreadyDone:
                if hybrid:
                    print(f"problem size L={L} is already done; skipping.")
                else:
                    print(f"problem size {L} is already done; skipping.")
                dothism = False

            if dothism:
                if hybrid:
                    mcmd=cmd+["-L"+str(L)]+["-M"+str(M)]
                else:
                    mcmd = cmd + ["-m" + str(m)]

                if dryrun:
                    #print mcmd
                    print(" ".join(mcmd))
                else:
                    denv = dict(os.environ)
                    i = 0
                    while i < len(E):
                        denv[E[i]] = E[i + 1]
                        i += 2

                    popen = Popen(mcmd, stdout = PIPE, stderr = PIPE, env = denv)
                    popen.wait() # sets the return code
                    prc = popen.returncode
                    out, err = popen.communicate() # capture output
                    if(verbose):
                        print("Output from timing.py's popen:")
                        #print " ".join(mcmd)
                        print("cwd:" , os.getcwd())
                        print("out:")
                        print(out)
                        print("err:")
                        print(err)

                    # copy the output and error to a log file.
                    with open(outdir + os.sep+"log", "a") as logfile:
                        logfile.write(" ".join(mcmd))
                        logfile.write("\n")
                        logfile.write(out.decode())
                        logfile.write(err.decode())

                    if (prc == 0): # did the process succeed?
                        if hybrid:
                            comment = out.decode()
                            params=collectParams(comment,L,M)
                            with open(optFile, "a") as logfile:
                                logfile.write(params+"\n")

                        outlines = out.decode().split('\n')
                        itline = 0
                        dataline = ""
                        while itline < len(outlines):
                            line = outlines[itline]
                            if re.search(rname, line) is not None:
                                print( "\t" + str(outlines[itline]))
                                print("\t" + str(outlines[itline + 1]))
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
                                print("ERROR: no timing data found")
                                badruns.append(mcmd)
                        else:
                            goodruns.append(mcmd)
                    else:
                        print( "FAILURE:")
                        print(cmd)
                        print("with, return code:")
                        print(prc)
                        print("output:")
                        print(out)
                        print("error:")
                        print(err)
                        badruns.append(mcmd)

            if not dryrun and (stats == -1 and os.path.isfile("timing.dat")):
                if(appendtofile):
                    # Append the new data to the output.
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
         with open(outdir + os.sep+"log", "a") as logfile:
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
            print(goodbads)



if __name__ == "__main__":
    main(sys.argv[1:])
