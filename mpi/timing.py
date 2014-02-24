#!/usr/bin/python

# a timing script for convolutions using MPI.
import subprocess

import sys, getopt
import numpy as np
from math import *
import os

def main(argv):
    usage='Usage: timings.py -a<start> -b<stop> -p<cconv2,conv2,cconv3,conv3> -T<number of threads per node> -P<number of nodes> -A<quoted arg list for timed program> -M <quoted arg list for mpi run command> -r<implicit/explicit> -l<name of mpi run command> -R<ram in gigabytes> -d' 
    helpnotes=\
        "\n -a <int> specifies the max exponent of the min problem size "\
        "\n -b <int> specifies the max exponent of the max problem size "\
        "\n -p <program name> specifies the program name "\
        "\n -P <int> specifies the number of MPI process "\
        "\n -T <int> specifies the number of OpenMP threads per MPI process"\
        "\n -A <string> allows one to pass extra arguments to the convolution program "\
        "\n -M <string> allows one to pass extra arguments to the mpi exec program "\
        "\n -r <implicit/explicit> specifies the type of convolution  "\
        "\n -l <string> specifies the name of the MPI launch program  "\
        "\n -s indicates that the MPI launcher does not take the number of processes as an argument  "\
        "\n -R <real value> specifies the ram size, which is used in determining the maximum problem size. "\
        "\n -d specifies a dry run; commands are shown but no convolutions are computed\n "\
        "\nfor example, try the command\n ./timing.py -pcconv2 -a 3 -b 4" \
        "\nor, to see the command run, try\n ./timing.py -pcconv2 -a 3 -b 4 -d"\

    
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
    skipnum=False
    try:
        opts, args = getopt.getopt(argv,"p:T:P:a:b:A:M:r:l:R:ds")
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
            if(r != "explicit"):
                if(r != "implicit"):
                    print r
                    print "is not a valid entry for -r"
                    print "please specify -r \"implicit \" or -r \"explicit \""
                    sys.exit(2)
                
        elif opt in ("-l"):
            l=str(arg)
        elif opt in ("-R"):
            RAM=float(arg)*2**30
        elif opt in ("-d"):
            dryrun=True
        elif opt in ("-s"):
            skipnum=True

    if dryrun:
        print "Dry run!  No output actually created."

    if p == "":
        print "please specify a program with -p"
        print usage
        print helpnotes
        sys.exit(2)

    outdir=""


    # if both the max problem size and the ram are unset, go up to 2^8
    if (b == 0 and RAM == 0):
        b=8
    # if RAM is set and the max problem size is not set, then RAM
    # determines the problem size on its own.
    if (b == 0 and RAM != 0):
        b=sys.maxint

    if p == "cconv2":
        if a == 0:
            a=int(floor(log(P)/(log(2))))
        if RAM != 0:
            if r != "explicit":
                b=min(b,int(floor(0.5*log(RAM/64)/log(2))))
            else:
                b=min(b,int(floor(log(RAM/16/2/2**2)/log(2)/2)))
        outdir="timings2c"
    if p == "conv2":
        if a == 0:
            a=int(floor(log(P)/(log(2))))
        if RAM != 0:
            if r != "explicit":
                b=min(b,int(floor(0.5*log(RAM/96)/log(2))))
            else:
                b=min(b,int(floor(log(RAM/8/3**2)/log(2)/2)))

        outdir="timings2r"
    if p == "cconv3":
        if a == 0:
            if r != "explicit":
                a=int(floor(log(P)/(2*log(2))))
            else:
                a=int(floor(log(P)/(log(2))))
        if RAM != 0:
            if r != "explicit":
                b=min(b,int(floor(log(RAM/96)/log(2)/3)))
            else:
                b=min(b,int(floor(log(RAM/16/2**3)/log(2)/3)))
        outdir="timings3c"
    if p == "conv3":
        if a == 0:
            a=int(floor(log(P)/(2*log(2))))
        if RAM != 0:
            b=min(b,int(floor(log(RAM/192)/log(2)/3)))
        outdir="timings3r"
    if p == "cfft2":
        if a == 0:
            a=int(floor(log(P)/(log(2))))
        if RAM != 0:
            b=min(b,int(floor(log(RAM/16/2/2**2)/log(2)/2)))
        outdir="timings2c"
        r="fft"
    if p == "cfft3":
        if a == 0:
            a=int(floor(log(P)/(2*log(2))))
        if RAM != 0:
            b=min(b,int(floor(log(RAM/96)/log(2)/3)))
        outdir="timings3c"
        r="fft"
    if p == "transpose":
        if a == 0:
            a=int(floor(log(P)/log(2)))
        if b == 0:
            b=10
        r="transpose"
        outdir="tran"

    if p == "otranspose":
        if a == 0:
            a=int(floor(log(P)/log(2)))
        if b == 0:
            b=10
        r="transpose"
        outdir="otran"

    if RAM != 0:
        print "max problem size is "+str(2**b)

    if outdir == "":
        print "empty outdir: please select a different program!"
        print
        print usage
        sys.exit(2)

    outdir=outdir+"/"+str(P)+"x"+str(T)
    command=l+" "
    if not skipnum:
        command+=str(P)+" "
    command+=M+"  ./"+str(p)
#    command=l+" "+str(P)+" "+M+"  ./"+str(p)

    print "Output in "+outdir+"/"+r

    if not dryrun:
        print "command: "+command+cargs+" "+A
        os.system("mkdir -p "+outdir)
        #os.system("rm -f "+outdir+"/"+r)
    
    rname="Implicit"
    if r == "explicit":
        rname="Explicit"
        if Tset == 1:
            print "cannot use multiple threads with explicit: try without -T"
            sys.exit(2)
    if r == "fft":
        rname="FFT"
    if r == "transpose":
        rname="transpose"
    if a == 0:
        a=1

    if r == "transpose":
        rlist=["Tininit","Tinwait0","Tinwait1","Tin",
               "Toutinit","Toutwait0","Toutwait1","Tout"]
        rnamelist=rlist
    else:
        rlist=[r]
        rnamelist=[rname]

    # remove old output files
    #for r in rlist:
        #if dryrun == False:
            #os.system("rm -f "+outdir+"/"+r)
            #print("rm -f "+outdir+"/"+r)

    for i in range(a,b+1):
        print i,
        m=str(int(pow(2,i))) # problem size
        run=command+cargs+" -m "+m+" "+A
        print run
        runfile=outdir+"/run"
        
        # run the program, collect the output in out.
        if not dryrun:
            # run the run "run", record the output to out (and the err to err)
            proc = subprocess.Popen([run], stdout=subprocess.PIPE, shell=True)
            (out, err) = proc.communicate()
        
            #print "program output:", out
            #print "program cerr:", err
            
            # append output to log file:
            os.system("touch "+outdir+"/log")
            with open(outdir+"/log", "a") as logfile:
                logfile.write(out)
            
            # put the output of this particular run in the runfile

            #os.system("rm -f "+runfile)
            os.system("touch "+runfile)
            with open(outdir+"/run","a") as runout:
                runout.write(out)
            
        # loop over all cases
        pos=0
        while(pos < len(rlist)):
            #print(rlist[pos],rnamelist[pos])
            r=rlist[pos] # output file
            rname=rnamelist[pos] # string for which we grep
            pos += 1

            grepc=" | grep -A 1 "+rname+" | tail -n 1"
            cat=" | cat >> "+outdir+"/"+r
            sys.stdout.flush()

            if not dryrun:
                    # grep for the string.
                gproc = subprocess.Popen(["cat "+runfile+grepc], stdout=subprocess.PIPE, shell=True)
                (out, err) = gproc.communicate()
                    
                # put the output of the grep in the output file:
                outfile=outdir+"/"+r
                os.system("touch "+outfile)
                # remove lines starting with the current problem size:
                os.system("sed -i '/^"+m+"[ \\t]/d' " +outfile)
                with open(outfile,"a") as fileout:
                    fileout.write(out)
            else:
                print("cat "+runfile+grepc)
        
        # clean up the run file (which is just temporary anyway)
        if not dryrun:
            os.system("rm -f "+runfile)

    # the run might have produced errors or been killed: remove empty
    # lines so asy doesn't choke.  Also sort output.
    if not dryrun:
        for r in rlist:
            outfile=outdir+"/"+r
            os.system("sed -i 's/[ \t]*$//' "+outfile)
            os.system("sed -i '/^$/d' "+outfile)
            os.system("sort -g  "+outfile+ " -o "+outfile)

    print("\ntiming finished.") # Definitely time for a beer.
if __name__ == "__main__":
    main(sys.argv[1:])