#!/usr/bin/python

# a timing script for running multiple timing tests using timing.py

import sys, getopt
import numpy as np
from math import *
import os
from subprocess import * # for popen, running processes

def main(argv):
    usage = '''
    Usage:
    \nalltime.py
    -T <number of threads> 
    -R <ram in gigabytes> 
    -S <int> statistical choice (-1 keeps raw data). 
    -d dry run
    -D<outdir>
    -r<implicit/explicit/pruned/fft>
    -A<quoted arg list for timed program>
    -B<quoted arg list inserted before timed program>
    '''

    dryrun=False
    R=0

    outdir = "timings"
    nthreads = 0
    #out = ""
    A = []
    B = []
    E = []
    runtype = "implicit"
    stats = 0

    try:
        opts, args = getopt.getopt(argv,"dp:T:r:R:S:D:g:A:B:E:")
    except getopt.GetoptError:
        print usage
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-T"):
            nthreads=int(arg)
        elif opt in ("-R"):
            R=float(arg)
        if opt in ("-S"):
            stats = int(arg)
        elif opt in ("-d"):
            dryrun=True
        elif opt in ("-D"):
            outdir = str(arg)
        elif opt in ("-g"):
            rname = str(arg)
        elif opt in ("-A"):
            A.append(str(arg))
        elif opt in ("-B"):
            B.append(str(arg))
        elif opt in ("-E"):
            E += [str(arg)]
        elif opt in ("-r"):
            runtype = str(arg)

    progs = [["cconv" , "conv", "tconv"], ["cconv2", "conv2", "tconv2"], \
             ["cconv3","conv3"]]
    
    ab = [[6,20],[4,10],[2,6]] # problem size limits
    
    if runtype == "explicit":
        progs = [["cconv", "conv", "tconv"], ["cconv2", "conv2", "tconv2"], \
                 ["cconv3"]]
    if runtype == "pruned":
        progs = [[], ["cconv2", "conv2", "tconv2"], ["cconv3"]]

    print "extra args:", A
    print "environment args:", E

    i = 0
    while(i < len(progs)):
        a = ab[i][0]
        b = ab[i][1]
        for p in progs[i]:
            cmd = []
            cmd.append("./timing.py")
            cmd.append("-p" + p)
            cmd.append("-S" + str(stats))
            cmd.append("-a" + str(a))
            if R == 0.0:
                cmd.append("-b" + str(b))
            else:
                cmd.append("-R" + str(R))
            if not nthreads == 0 :
                cmd.append("-T" + str(nthreads))
            if runtype != "":
                cmd.append("-r" + runtype)
            cmd.append("-D" + outdir)
            cmd.append("-o" + p + "_" + runtype)
            iA = 0
            while iA < len(A):
                cmd.append("-A" + A[iA])
                iA += 1
            iB = 0
            while iB < len(B):
                cmd.append("-B" + B[iB])
                iB += 1
            iE = 0
            while iE < len(E):
                cmd.append("-E" + E[iE])
                iE += 1


            if(runtype == "implicit" and (p == "conv" or p == "conv2" or p == "conv3")):
                if(p == "conv"):
                    for X in [0, 1]:
                        cmd0 = []
                        ci =  0
                        while ci < len(cmd):
                            cmd0.append(cmd[ci])
                            ci += 1
                        cmd0.append("-A-X" + str(X))
                        cmd0.append("-oconv_implicit" + "X" + str(X))
                        print " ".join(cmd0)
                        if not dryrun:
                            p = Popen(cmd0)
                            p.wait()
                            prc = p.returncode
                if(p == "conv2"):
                    for X in [0, 1]:
                        for Y in [0, 1]:
                            cmd0 = []
                            ci =  0
                            while ci < len(cmd):
                                cmd0.append(cmd[ci])
                                ci += 1
                            cmd0.append("-A-X" + str(X))
                            cmd0.append("-A-Y" + str(Y))
                            cmd0.append("-oconv2_implicitX"\
                                        + str(X) + "Y" + str(Y))
                            print " ".join(cmd0)
                            if not dryrun:
                                p = Popen(cmd0)
                                p.wait()
                                prc = p.returncode
                if(p == "conv3"):
                    for X in [0, 1]:
                        for Y in [0, 1]:
                            for Z in [0, 1]:
                                cmd0 = []
                                ci =  0
                                while ci < len(cmd):
                                    cmd0.append(cmd[ci])
                                    ci += 1
                                cmd0.append("-A-X" + str(X))
                                cmd0.append("-A-Y" + str(Y))
                                cmd0.append("-A-Z" + str(Z))
                                cmd0.append("-oconv3_implicitX"\
                                            + str(X) + "Y" + str(Y)\
                                            + "Z" + str(Z))
                                print " ".join(cmd0)
                                if not dryrun:
                                    p = Popen(cmd0)
                                    p.wait()
                                    prc = p.returncode

            else:
                print " ".join(cmd)
                if not dryrun:
                    p = Popen(cmd)
                    p.wait() # sets the return code
                    prc = p.returncode

            # pcmd=cmd+" -p "+p
            # print(pcmd)
            # if not dryrun:
            #     os.system(pcmd)
        i += 1 
            
if __name__ == "__main__":
    main(sys.argv[1:])
