#!/usr/bin/env python3

from math import *
from subprocess import *
import os
import re
import argparse

def ceilquotient(a,b):
  return -(a//-b)

def main():
  args=getArgs()
  programs=getPrograms(args)
  test(programs, args)

class Program:
  def __init__(self, name, centered, extraArgs=""):
    self.name=name
    self.centered=centered
    self.extraArgs=extraArgs
    self.total=0
    self.failed=0
    self.failedCases=[]

  def passed(self):
    return self.total-self.failed

def getArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument("-S", help="Test Standard convolutions. Not specifying\
  										S or C or H is the same as specifying all of them",
  										action="store_true")
  parser.add_argument("-C", help="Test Centered convolutions. Not specifying\
  										S or C or H is the same as specifying all of them",
  										action="store_true")
  parser.add_argument("-H", help="Test Hermitian convolutions. Not specifying\
  										S or C or H is the same as specifying all of them",
  										action="store_true")
  parser.add_argument("-X", help="Test 1D Convolutions. Not specifying\
  										X or Y or Z is the same as specifying all of them",
  										action="store_true")
  parser.add_argument("-Y", help="Test 2D Convolutions. Not specifying\
  										Z or Y or Z is the same as specifying all of them",
  										action="store_true")
  parser.add_argument("-Z", help="Test 3D Convolutions. Not specifying\
  										X or Y or Z is the same as specifying all of them",
  										action="store_true")
  parser.add_argument("-T", help="Number of threads to use in timing. If set to\
                      0, iterates over 1, 2, and 4 threads. Default is 1.",
                      default=1)
  parser.add_argument("-L", help="L value. Default is 8.", default=8)
  parser.add_argument("-t",help="Error tolerance. Default is 1e-12.",
                      default=1e-12)
  parser.add_argument("-l",help="Show log of failed cases",
                      action="store_true")
  parser.add_argument("-v",help="Show the results of every test.",
                      action="store_true")
  return parser.parse_args()

def getPrograms(args):
  programs=[]

  SCH=(not (args.S or args.C or args.H))
  SH=(not (args.S or args.C or args.H))
  XYZ=(not (args.X or args.Y or args.Z))

  SorSCH=(args.S or SCH)
  CorSCH=(args.C or SCH)
  HorSCH=(args.H or SCH)

  if args.X or XYZ:
    if SorSCH:
      programs.append(Program("hybridconv",False))
    if CorSCH:
      programs.append(Program("hybridconv",True,"-c"))
    if HorSCH:
      programs.append(Program("hybridconvh",True))

  if args.Y or XYZ:
    if SorSCH:
      programs.append(Program("hybridconv2",False))
    if HorSCH:
      programs.append(Program("hybridconvh2",True))

  if args.Z or XYZ:
    if SorSCH:
      programs.append(Program("hybridconv3",False))
    if HorSCH:
      programs.append(Program("hybridconvh3",True))

  return programs

def test(programs, args):
  lenP=len(programs)
  T=int(args.T)
  print("\n***************\n")
  if lenP == 1:
    p=programs[0]
    name=p.name
    if p.extraArgs:
      name+=" "+p.extraArgs
    if T == 0:
      print("Testing "+name+" with 1, 2, and 4 threads.\n")
    elif T == 1:
      print("Testing "+name+" with "+args.T+" thread.\n")
    elif T > 1:
      print("Testing "+name+" with "+args.T+" threads.\n")
    else:
      raise ValueError(str(T)+" is an invalid number of threads.")
    iterate(p,int(args.L),T,float(args.t),args.v)
    print("Finished testing "+name+".")
    print("\n***************\n")
    print("Finished testing 1 program.")
    print("Out of "+str(p.total)+" tests, "+str(p.passed())+" passed, "+str(p.failed)+" failed.\n")
    if args.l and len(p.failedCases) > 0:
      print("Failed Cases:\n")
      for case in p.failedCases:
        print(case)
      print()

  elif lenP > 1:
    total=0
    passed=0
    failed=0
    failedCases=[]
    for p in programs:
      name=p.name
      if p.extraArgs:
        name+=" "+p.extraArgs
      if T == 0:
        print("Testing "+name+" with 1, 2, and 4 threads.\n")
      elif T == 1:
        print("Testing "+name+" with "+str(args.T)+" thread.\n")
      elif T > 1:
        print("Testing "+name+" with "+str(args.T)+" threads.\n")
      else:
        raise ValueError(str(T)+" is an invalid number of threads.")
      iterate(p,int(args.L),T,float(args.t),args.v)
      ptotal=p.total
      pfailed=p.failed
      ppassed=p.passed()
      print("Finished testing "+name+".")
      print("Out of "+str(ptotal)+" tests, "+str(ppassed)+" passed, "+str(pfailed)+" failed.")
      print("\n***************\n")
      total+=ptotal
      passed+=ppassed
      failed+=pfailed
      if args.l:
        failedCases+=p.failedCases

    print("Finished testing "+str(lenP)+" programs.")
    print("Out of "+str(total)+" tests, "+str(passed)+" passed, "+str(failed)+" failed.\n")
    if args.l and len(failedCases) > 0:
      print("Failed Cases:\n")
      for case in failedCases:
        print(case)
      print()
  else:
    print("\nNo programs to test.\n")

def iterate(program, L, thr, tol, verbose):
  centered=program.centered

  Dstart=2 if centered else 1
  if thr ==  0:
    threads=[1,2,4]
  else:
    threads=[thr]

  M0=3*L//2 if centered else 2*L-1
  M02=M0//2
  for T in threads:
    for M in [M0,3*M02,2*M0,5*M02]:
      for m in [L//4,L//2,L,L+2,M,M+2]:
        p=ceilquotient(L,m)
        q=ceilquotient(M,m) if p <= 2 else ceilquotient(M,m*p)*p
        n=q//p
        Istart=0 if q > 1 else 1
        for I in range(Istart,2):
          D=Dstart
          while(D < n):
            check(program,L,M,m,D,I,T,tol,verbose)
            D*=2
          check(program,L,M,m,n,I,T,tol,verbose)

def check(program, L, M, m, D, I, T, tol, verbose):
  program.total+=1
  name=program.name
  cmd = [name,"-L"+str(L),"-M"+str(M),"-m"+str(m),"-D"+str(D),"-I"+str(I),"-T"+str(T),"-E"]
  if program.extraArgs != "":
    cmd.append(program.extraArgs)

  vp = Popen(cmd, stdout = PIPE, stderr = PIPE)
  vp.wait()
  prc = vp.returncode

  boldPassedTest="\033[1mPassed Test:\033[0m"
  boldFailedTest="\033[1mFailed Test:\033[0m"

  if prc == 0:
    out, err = vp.communicate()
    comment = out.rstrip().decode()

  try:
    error=re.search(r"(?<=Error: )(\w|\d|\.|e|-|\+)*",comment).group()
    if float(error) > tol or error == "nan" or error == "inf":
      program.failed+=1
      print("\t"+boldFailedTest+" Error="+error)
      case=" ".join(cmd)
      print("\t"+case)
      program.failedCases.append(case)
      print()
    elif verbose:
      print("\t"+boldPassedTest+" Error="+error)
      case=" ".join(cmd)
      print("\t"+case)
      print()
  except:
    program.failed+=1
    print("\t"+boldFailedTest+" Error not found.")
    case=" ".join(cmd)
    print("\t"+case)
    program.failedCases.append(case)
    print()

if __name__ == "__main__":
  main()
