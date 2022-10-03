#!/usr/bin/env python3

from math import *
from subprocess import *
import os
import re
import argparse
from HybridParameters import *

def main():
  args=getArgs()
  programs=getPrograms(args)
  test(programs, args)

class Program:
  def __init__(self, name, centered, dim=1, extraArgs="",mult=True):
    self.name=name
    self.centered=centered
    self.dim=dim
    self.extraArgs=extraArgs
    self.mult=mult
    self.failed=0
    self.passed=0
    self.total=0
    self.failedCases=[]

  def passTest(self):
    self.passed+=1
    self.total+=1

  def failTest(self):
    self.failed+=1
    self.total+=1

def getArgs():
  parser = argparse.ArgumentParser(description="Perform Unit Tests on convolutions with hybrid dealiasing.")
  parser.add_argument("-s", help="Test Standard convolutions. Not specifying\
  										s or c or H is the same as specifying all of them",
  										action="store_true")
  parser.add_argument("-c", help="Test Centered convolutions. Not specifying\
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
  parser.add_argument("-S", help="Test different strides.",
                      action="store_true")
  parser.add_argument("-t",help="Error tolerance. Default is 1e-12.",
                      default=1e-12)
  parser.add_argument("-l",help="Show log of failed cases",
                      action="store_true")
  parser.add_argument("-v",help="Show the results of every test.",
                      action="store_true")
  return parser.parse_args()

def getPrograms(args):
  programs=[]
  S=args.s
  C=args.c
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
      programs.append(Program("hybrid",False,mult=False))
      programs.append(Program("hybridconv",False))
    if CorNotSCH:
      programs.append(Program("hybrid",True,extraArgs="-c",mult=False))
      programs.append(Program("hybridconv",True,extraArgs="-c"))
    if HorNotSCH:
      programs.append(Program("hybridh",True,mult=False))
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

  if lenP >= 1:
    passed=0
    failed=0
    total=0
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

      iterate(p,T,float(args.t),args.v,args.S)

      ppassed=p.passed
      pfailed=p.failed
      ptotal=p.total

      print("Finished testing "+name+".")
      print("Out of "+str(ptotal)+" tests, "+str(ppassed)+" passed, "+str(pfailed)+" failed.")
      print("\n***************\n")
      total+=ptotal
      passed+=ppassed
      failed+=pfailed
      if args.l:
        failedCases+=p.failedCases

    if lenP > 1:
      print("Finished testing "+str(lenP)+" programs.")
      print("Out of "+str(total)+" tests, "+str(passed)+" passed, "+str(failed)+" failed.\n")

    if args.l and len(failedCases) > 0:
      print("Failed Cases:\n")
      for case in failedCases:
        print(case+";")
      print()
  else:
    print("\nNo programs to test.\n")

def iterate(program, thr, tol, verbose, testS):

  dim=program.dim

  if thr ==  0:
    threads=[1,2,4]
  else:
    threads=[thr]

  vals=ParameterCollection(fillValues,program,1,testS and not program.mult).vals
  if dim == 1:
    for x in vals:
      for T in threads:
        check(program,[x],T,tol,verbose)
    if not program.mult:
      vals=ParameterCollection(fillValues,program,8,testS and not program.mult).vals
      for x in vals:
        for T in threads:
          check(program,[x],T,tol,verbose)
  else:
    if dim == 2:
      for y in vals:
        xvals=ParameterCollection(fillValues,program,y.L,testS).vals
        for x in xvals:
          for T in threads:
            check(program,[x,y],T,tol,verbose)

    elif dim == 3:
      for z in vals:
        yvals=ParameterCollection(fillValues,program,z.L,testS).vals
        for y in yvals:
          xvals=ParameterCollection(fillValues,program,y.L*y.S,testS).vals
          for x in xvals:
            for T in threads:
              check(program,[x,y,z],T,tol,verbose)
    else:
      exit("Dimension must be 1 2 or 3.")

def fillValues(program, minS, testS):

  # This doesn't quite correspond to C in the main code but it has the property
  # that C == 1 in 1D and C > 1 in higher dimensions
  # It also works for tesing hybrid.cc and hybridh.cc.
  C=minS

  centered=program.centered
  hermitian = centered and program.extraArgs != "-c"

  dim=program.dim
  vals=[]

  # Because of symmetry concerns in the centered cases (compact/noncompact),
  # we check even and odd L values
  if centered and dim == 1:
    Ls=[7,8]
  else:
    Ls=[8]

  Dstart=2 if hermitian and C == 1 else 1

  Ss=[minS]
  if testS:
    Ss+=[minS+1]

  for S in Ss:
    for L in Ls:
      L4=ceilquotient(L,4)
      L2=ceilquotient(L,2)
      Ms=[]
      if centered:
        Ms.append(3*L2-2*(L%2))

      if dim != 1:
        Ms+= [2*L]
      else:
        Ms+=[2*L,5*L2]
      for M in Ms:
        ms=[L4,L2]
        if not centered:
          ms+=[L,L+1]
        ms+=[M]
        for m in ms:
          p=ceilquotient(L,m)
          q=ceilquotient(M,m) if p <= 2 else ceilquotient(M,m*p)*p
          n=q//p
          Istart=0 if q > 1 else 1
          for I in range(Istart,2):
            if C == 1:
              D=Dstart
              while(D < n):
                vals.append(Parameters(L,M,m,p,q,C,S,D,I))
                D*=2
              vals.append(Parameters(L,M,m,p,q,C,S,D,I))
            else:
              D=1
              vals.append(Parameters(L,M,m,p,q,C,S,D,I))
  return vals

def check(program, vals, T, tol, verbose):

  cmd=getcmd(program,vals,T)

  vp = Popen(cmd, stdout = PIPE, stderr = PIPE)
  vp.wait()
  prc = vp.returncode

  if prc == 0:
    out, err = vp.communicate()
    comment = out.rstrip().decode()

  if program.mult:
    checkError(program, comment, cmd, tol, verbose, r"Error")
  else:
    checkError(program, comment, cmd, tol, verbose, r"Forward Error")
    checkError(program, comment, cmd, tol, verbose, r"Backward Error")

def checkError(program, comment, cmd, tol, verbose, message):
  boldPassedTest="\033[1mPassed Test:\033[0m"
  boldFailedTest="\033[1mFailed Test:\033[0m"
  try:
    error=re.search(r"(?<="+message+": )(\w|\d|\.|e|-|\+)*",comment).group()
    if float(error) > tol or error == "nan" or error == "inf":
      program.failTest()
      print("\t"+boldFailedTest+" "+message+": "+error)
      case=" ".join(cmd)
      print("\t"+case)
      program.failedCases.append(case)
      print()
    else:
      program.passTest()
      if verbose:
        print("\t"+boldPassedTest+" "+message+": "+error)
        case=" ".join(cmd)
        print("\t"+case)
        print()
  except:
    program.failTest()
    print("\t"+boldFailedTest+" "+message+" not found.")
    case=" ".join(cmd)
    print("\t"+case)
    program.failedCases.append(case)
    print()


def getcmd(program, vals, T):
  name=program.name
  lenvals=len(vals)

  cmd=[name]

  if lenvals == 3:
    x,y,z = vals
    cmd+=addParams3D(x.L,y.L,z.L,"L")
    cmd+=addParams3D(x.M,y.M,z.M,"M")
    cmd+=addParams3D(x.m,y.m,z.m,"m")
    cmd+=addParams3D(x.I,y.I,z.I,"I")
    cmd+=addParams3D(x.D,y.D,z.D,"D")
    cmd+=["-Sx="+str(x.S),"-Sy="+str(y.S)]

  elif lenvals == 2:
    x,y = vals
    cmd+=addParams2D(x.L,y.L,"L")
    cmd+=addParams2D(x.M,y.M,"M")
    cmd+=addParams2D(x.m,y.m,"m")
    cmd+=addParams2D(x.I,y.I,"I")
    cmd+=addParams2D(x.D,y.D,"D")
    cmd+=["-Sx="+str(x.S)]

  else:
    x=vals[0]
    cmd+=["-L="+str(x.L),"-M="+str(x.M),"-m="+str(x.m),"-D="+str(x.D),"-I="+str(x.I)]
    if not program.mult:
      cmd+=["-S"+str(x.S), "-C"+str(x.C)]
  cmd+=["-T="+str(T),"-E"]

  if program.extraArgs != "":
    cmd.append(program.extraArgs)

  return cmd

def addParams3D(px,py,pz,pname):
  # This code avoids reduntant arguments in the 3D case for readability
  # e.g. if Lx == Ly == Lz then we can just use L
  if px == py == pz:
    cmd=["-"+pname+"="+str(px)]
  elif  px == py:
    cmd=["-"+pname+"="+str(px), "-"+pname+"z="+str(pz)]
  elif  px == pz:
    cmd=["-"+pname+"="+str(px), "-"+pname+"y="+str(py)]
  elif  py == pz:
    cmd=["-"+pname+"="+str(py), "-"+pname+"x="+str(px)]
  else:
    cmd=["-"+pname+"x="+str(px), "-"+pname+"y="+str(py),"-"+pname+"z="+str(pz)]
  return cmd

def addParams2D(px,py,pname):
  # This code avoids reduntant arguments in the 2D case for readability
  # e.g. if Lx == Ly then we can just use L
  if px == py:
    cmd=["-"+pname+"="+str(px)]
  else:
    cmd=["-"+pname+"x="+str(px), "-"+pname+"y="+str(py)]
  return cmd

if __name__ == "__main__":
  main()
