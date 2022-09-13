#!/usr/bin/env python3

from math import *
from subprocess import *
import os
import re
import argparse
from OptimalValues import *

def ceilquotient(a,b):
  return -(a//-b)

def main():
  args=getArgs()
  programs=getPrograms(args)
  test(programs, args)

class Program:
  def __init__(self, name, centered, dim=1, extraArgs=""):
    self.name=name
    self.centered=centered
    self.dim=dim
    self.extraArgs=extraArgs
    self.total=0
    self.failed=0
    self.failedCases=[]

  def passed(self):
    return self.total-self.failed

def getArgs():
  parser = argparse.ArgumentParser(description="Perform Unit Tests on convolutions with hybrid dealiasing.")
  parser.add_argument("-S", help="Test Standard convolutions. Not specifying\
  										S or C or H is the same as specifying all of them",
  										action="store_true")
  parser.add_argument("-C", help="Test Centered convolutions. Not specifying\
  										S or C or H is the same as specifying all of them",
  										action="store_true")
  parser.add_argument("-H", help="Test Hermitian convolutions. Not specifying\
  										S or C or H is the same as specifying all of them",
  										action="store_true")
  parser.add_argument("-1","--one", help="Test 1D Convolutions. Not specifying\
  										1 or 2 or 3 is the same as specifying all of them",
  										action="store_true")
  parser.add_argument("-2","--two", help="Test 2D Convolutions. Not specifying\
  										1 or 2 or 3 is the same as specifying all of them",
  										action="store_true")
  parser.add_argument("-3","--three", help="Test 3D Convolutions. Not specifying\
  										1 or 2 or 3 is the same as specifying all of them",
  										action="store_true")
  parser.add_argument("-T", help="Number of threads to use in timing. If set to\
                      0, iterates over 1, 2, and 4 threads. Default is 1.",
                      default=1)
  parser.add_argument("-t",help="Error tolerance. Default is 1e-12.",
                      default=1e-12)
  parser.add_argument("-l",help="Show log of failed cases",
                      action="store_true")
  parser.add_argument("-v",help="Show the results of every test.",
                      action="store_true")
  return parser.parse_args()

def getPrograms(args):
  programs=[]
  S=args.S
  C=args.C
  H=args.H
  X=args.one
  Y=args.two
  Z=args.three

  notSCH=(not (S or C or H))
  notSH=(not (S or C or H))
  notXYZ=(not (X or Y or Z))

  SorNotSCH=(S or notSCH)
  CorNotSCH=(C or notSCH)
  HorNotSCH=(args.H or notSCH)

  if X or notXYZ:
    if SorNotSCH:
      programs.append(Program("hybridconv",False))
    if CorNotSCH:
      programs.append(Program("hybridconv",True,extraArgs="-c"))
    if HorNotSCH:
      programs.append(Program("hybridconvh",True))

  if Y or notXYZ:
    if SorNotSCH:
      programs.append(Program("hybridconv2",False,dim=2))
    if HorNotSCH:
      programs.append(Program("hybridconvh2",True,dim=2))

  if Z or notXYZ:
    if SorNotSCH:
      programs.append(Program("hybridconv3",False,dim=3))
    if HorNotSCH:
      programs.append(Program("hybridconvh3",True,dim=3))

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
      print("Testing "+name+" with "+str(T)+" thread.\n")
    elif T > 1:
      print("Testing "+name+" with "+str(T)+" threads.\n")
    else:
      raise ValueError(str(T)+" is an invalid number of threads.")

    iterate(p,T,float(args.t),args.v)

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
        print("Testing "+name+" with "+str(T)+" thread.\n")
      elif T > 1:
        print("Testing "+name+" with "+str(T)+" threads.\n")
      else:
        raise ValueError(str(T)+" is an invalid number of threads.")

      iterate(p,T,float(args.t),args.v)

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

def iterate(program, thr, tol, verbose):

  dim=program.dim

  if thr ==  0:
    threads=[1,2,4]
  else:
    threads=[thr]

  if dim == 1:
    vals=OptimalValues(program,fillOptValues).vals
    for x in vals:
      for T in threads:
        check(program,[x],T,tol,verbose)
  elif dim == 2:
    vals=OptimalValues(program,fillOptValues).vals
    for x in vals:
      for y in vals:
        for T in threads:
          check(program,[x,y],T,tol,verbose)
  elif dim == 3:
    vals=OptimalValues(program,fillOptValues).vals
    for x in vals:
      for y in vals:
        for z in vals:
          for T in threads:
            check(program,[x,y,z],T,tol,verbose)
  else:
    exit("Dimension must be 1 2 or 3.")    

def fillOptValues(program):
  C=1
  S=1

  centered=program.centered
  vals=[]

  # Because of symmetry concerns in the centered cases (compact/noncompact),
  # we check even and odd L values
  if centered:
    Ls=[7,8]
  else:
    Ls=[8]

  Dstart=2 if centered else 1
  for L in Ls:
    L4=ceilquotient(L,4)
    L2=ceilquotient(L,2)
    Ms=[]
    if centered:
      Ms.append(3*L2-2*(L%2))
    Ms+=[2*L,5*L2]
    for M in Ms:
      ms=[L4,L2]
      if not centered:
        ms+=[L,L+1]
      ms+=[M,M+1]
      for m in ms:
        p=ceilquotient(L,m)
        q=ceilquotient(M,m) if p <= 2 else ceilquotient(M,m*p)*p
        n=q//p
        Istart=0 if q > 1 else 1
        for I in range(Istart,2):
          D=Dstart
          while(D < n):
            vals.append(OptValue(L,M,m,p,q,C,S,D,I))
            D*=2
          vals.append(OptValue(L,M,m,p,q,C,S,D,I))

  return vals

def check(program, ovals, T, tol, verbose):
  program.total+=1
  name=program.name
  directions=["x","y","z"]
  cmd=[]
  for i in range(len(ovals)):
    o=ovals[i]
    d=directions[i]
    cmd+=[name,"-L"+d+"="+str(o.L),"-M"+d+"="+str(o.M),"-m"+d+"="+str(o.m),"-D"+d+"="+str(o.D),"-I"+d+"="+str(o.I)]

  cmd+=["-T="+str(T),"-E"]
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
