#!/usr/bin/env python3

# a timing script for FFTs and convolutions using OpenMP

import sys, getopt
from math import *
from subprocess import * # for Popen, running processes
import os
import re # regexp package
import shutil
import time
from utils import *

def forwardBackward(comment):

  try:
    FR=re.findall(r"(?<=Forwards Routine: )\S+",comment)
    BR=re.findall(r"(?<=Backwards Routine: )\S+",comment)
    params="\n"+"\t".join(FR)+"\n"+"\t".join(BR)
    return params
  except:
    print("Could not find routines used.")
    return -1

def getParam(name, comment):
  try:
    param=re.findall(r"(?<="+name+"=)\d+",comment)
    l=len(param)
    xyz=["x","y","z"] if l > 1 else [""]
    for i in range(l):
        param[i]=f"{name}{xyz[i]}={param[i]}"
    return"\t".join(param)
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

  params=f"{m}\n{p}\n{q}\n{C}\n{S}\n{D}\n{I}"
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


def default_outdir(p,T,I):
    outdir=""
    if p == "cconv" or p == "hybridconv" or p == "directTest":
        outdir = "timings1"
    if p == "cconv2" or p == "hybridconv2":
        outdir = "timings2"
    if p == "cconv3" or p == "hybridconv3":
        outdir = "timings3"
    if p == "conv" or p == "hybridconvh" :
        outdir = "timingsh1"
    if p == "conv2" or p == "hybridconvh2":
        outdir = "timingsh2"
    if p == "conv3" or p == "hybridconvh3":
        outdir = "timingsh3"
    if p == "rconv" or p == "hybridconvr":
        outdir = "timingsr1"
    if p == "rconv2" or p == "hybridconvr2":
        outdir = "timingsr2"
    if p == "tconv":
        outdir = "timings1t"
    if p == "tconv2":
        outdir="timingst2"
    if p == "fft1":
        outdir = "timingsf1"
    if p == "mfft1":
        outdir = "timingsmf1"
    if p == "fft2":
        outdir = "timingsf2"
    if p == "transpose":
        outdir="transpose2"
    outdir += "-T"+str(T)
    if I != 0:
      outdir += "I"+str(I)
    return outdir

def main(argv):
    usage = '''Usage:
    \ntimings.py
    -a<start>
    -b<stop>
    -I<increment (if not testing powers of 2)>
    -p<hybridconv,hybridconv2,hybridconv3,hybridconvh,hybridconvh2,hybridconvh3,cconv,cconv2,cconv3,conv,conv2,conv3,tconv,tconv2>
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
    -s<float> minimum time per test (default 1s)
    -S<int> Type of statistics (default 0=MEDIAN)
    -e: erase existing timing data
    -c<string>: extra commentary for output file.
    -v: verbose output
    '''

    dryrun = False

    # File extension for output files
    fileExt=""


    bset = 0
    dorun = True
    T = 1 # number of threads
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
    s = 0
    appendtofile = True
    stats = 0
    path = "."
    verbose = False
    extracomment = ""

    try:
        opts, args = getopt.getopt(argv,"dhep:T:a:b:c:I:A:B:E:r:R:S:o:P:D:g:s:v")
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
        elif opt in ("-s"):
            s = float(arg)
        elif opt in ("-b"):
            b = int(arg)
        elif opt in ("-I"):
            I = int(arg)
        elif opt in ("-c"):
            extracomment = arg
        elif opt in ("-A"):
            A += [str(arg)]
        elif opt in ("-B"):
            B = str(arg).split(" ")
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

    dimension=1
    hybrid = False
    hermitian = False
    real = False
    ternary = False
    direct = False

    dim2routines=["cconv2","conv2","rconv2","hybridconv2","hybridconvh2","hybridconvr2"]
    dim3routines=["cconv3","conv3","hybridconv3","hybridconvh3"]
    if p in dim2routines:
        dimension=2

    if p in dim3routines:
        dimension=3

    if p == "directTest":
        direct = True
        runtype="direct"

    if p == "cconv":
        if(runtype == "pruned"):
            print(p + " has no pruned option")
            dorun = False

    if re.search("hybrid",p) is not None:
        hybrid=True

    if p == "conv" or p == "conv2" or p == "conv3":
        hermitian = True
        if p == "conv" and runtype == "pruned":
            print(p + " has no pruned option")
            dorun = False

    if p == "hybridconvh" or p == "hybridconvh2" or p == "hybridconvh3":
        hermitian = True

    if p == "hybridconvr" or p == "hybridconvr2":# or p == "hybridconvh3":
        real = True

    if p == "rconv" or p == "rconv2":
        real = True



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
        outdir = default_outdir(p,T,I)

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
            outfile = "hybrid"+fileExt
        else:
            outfile = runtype+fileExt
    optFile=""
    if hybrid:
        optFile=outdir+os.sep+"hybridParams"+fileExt

    goodruns = []
    badruns = []

    if dorun:
        if RAM != 0:
            print("max problem size is "+str(2**b))

        if rname == "":
            if hybrid:
                rname = "Hybrid"
                #if runtype == "explicit":
                #    rname="explicit"
            elif direct:
                rname="Direct"
            else:
                if runtype == "implicit":
                    rname = "Implicit"
                if runtype == "explicit":
                    rname = "Explicit"
                if runtype == "explicito":
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
            with open(outdir + os.sep+"log"+fileExt, "a") as logfile:
                #logfile.write(str(sys.argv))
                #logfile.write("\n")
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
            cmd.append(B[i])
            i += 1

        path += os.sep
        cmd += [path + str(p)]

        if not os.path.isfile(path + str(p)):
            print(path + str(p), "does not exist!")
            sys.exit(1)

        if not hybrid and not direct and not "fft" in p:
            if((runtype == "explicit" or runtype == "explicito") and not real):
                cmd.append("-e")

            if(runtype == "pruned"):
                cmd.append("-p")

            if(runtype == "implicit"):
                cmd.append("-i")

        cmd.append("-S" + str(stats))
        cmd.append("-T" + str(T))
        if dimension == 1:
            if runtype == "explicito":
                cmd.append("-I0")
            elif runtype == "explicit":
                cmd.append("-I1")
        if hybrid:
            cmd.append("-R")
        if s > 0:
            cmd.append("-s" + str(s))
        cmd.append("-u")

        # Add the extra arguments to the program being timed.
        i = 0
        while i < len(A):
            cmd.append(A[i])
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
            date = time.strftime("%Y-%m-%d  %H:%M:%S-%Z")
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
                gitcommit=""
                if prc == 0:
                    out, err = vp.communicate()
                    gitcommit += out.rstrip().decode()
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
                    gitcommit += " (" + out[0:10].decode() + ")"
                    comment += "\t" +gitcommit
            else:
                comment += "\t" + extracomment
            comment += "\n"
            filenameMessage=comment

            hybridParamsMessage="#\n#\n# Run on "+date+" using commit "+gitcommit+":\n"


        newdata=False
        for i in range(a,b+1,1 if I == 0 else I):
            if I != 0:
                m=i
                L = 2*m if hermitian and hybrid else 2*m-1 if hermitian else m
                M = 3*m if hermitian and hybrid else 3*m-2 if hermitian else 2*m
            elif not hermitian or runtype == "implicit":
                m = int(pow(2,i))
                L = 2*m if hermitian and hybrid else 2*m-1 if hermitian else m
                M = 3*m if hermitian and hybrid else 3*m-2 if hermitian else 2*m
            else:
                if not ternary:
                    m = int((pow(2,i+1)+2) // 3)
                else:
                    m = int((pow(2,i+2)+3) // 4)
                L = 2*m-1
                M = 3*m-2

            print(str(i) + " m=" + str(m))
            options=[]
            if hybrid:
              if dimension == 2:
                if not real:
                    Sx=(ceilquotient(L,2) if hermitian else L)+2
                    options.append(f'-Sx={Sx}')
              if dimension == 3:
                Sy=ceilquotient(L,2) if hermitian else L
                if T == 1:
                  Sy += 2
                options.append(f'-Sy={Sy}')
                Sx=Sy*L+2
                if T > 1:
                  Sx += 2
                options.append(f'-Sx={Sx}')
            elif hermitian:
              if dimension == 2:
                options.append('-X1')
                options.append('-Y1')
              if dimension == 3 :
                options.append('-X1')
                options.append('-Y1')
                options.append('-Z1')

            dothism = True

            alreadyDone=(dimension == 1 and L in Ldone) or\
                        (dimension == 2 and L**2 in Ldone) or\
                        (dimension == 3 and L**3 in Ldone)

            if appendtofile and alreadyDone:
                if hybrid:
                    print(f"problem size L={L} is already done; skipping.")
                else:
                    print(f"problem size {L} is already done; skipping.")
                dothism = False

            if dothism and not dryrun:
                if newdata == False:
                    newdata=True
                    if(appendtofile):
                        with open(filename, "a") as myfile:
                            myfile.write(comment)
                        if hybrid:
                            with open(optFile,"a") as logfile:
                                logfile.write(hybridParamsMessage)
                    else:
                        if stats == -1:
                            with open("timing.dat", "w") as myfile:
                                myfile.write(filenameMessage)
                        else:
                            with open(filename, "w") as myfile:
                                myfile.write(filenameMessage)
                            if hybrid:
                                with open(optFile,"w") as logfile:
                                    logfile.write("# Optimal values for "+p+"\n")
                                    logfile.write(hybridParamsMessage)
                if hybrid:
                    mcmd=cmd+["-L"+str(L)]+["-M"+str(M)]
                    if runtype == "explicit":
                        mcmd += ["-m"+str(M),"-I1"]
                    elif runtype == "explicito":
                        mcmd += ["-m"+str(M),"-I0"]
                elif direct:
                    mcmd=cmd+["-L"+str(L)]
                else:
                    mcmd=cmd+["-m" + str(m)]

                mcmd += options
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
                    with open(outdir + os.sep+"log"+fileExt, "a") as logfile:
                        logfile.write(" ".join(mcmd))
                        logfile.write("\n")
                        logfile.write(out.decode())
                        logfile.write(err.decode())

                    if (prc == 0): # did the process succeed?
                        if hybrid:
                            results = out.decode()
                            params=collectParams(results,L,M)
                            FB=forwardBackward(results)
                            with open(optFile, "a") as logfile:
                                logfile.write("#\n# "+" ".join(mcmd)+"\n")
                                logfile.write(params)
                                logfile.write(FB+"\n")

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
         with open(outdir + os.sep+"log"+fileExt, "a") as logfile:
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
            logfile.write("\n")
            print(goodbads)



if __name__ == "__main__":
    main(sys.argv[1:])
