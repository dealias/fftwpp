#!/usr/bin/env python3

import subprocess
import re
import argparse
import sys
from HybridParameters import *

def main():
  args=getArgs()
  programs=getPrograms(args)
  test(programs, args)

class Program:
  def __init__(self, name, dim=1, mult=True, centered=False, hermitian=False, real=False):
    self.name=name
    self.dim=dim
    self.mult=mult
    self.centered=centered or hermitian
    self.hermitian=hermitian
    self.real=real

    self.extraArgs="-c" if (centered and not hermitian) else ""

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
  										s or c or H or r is the same as specifying all of them",
  										action="store_true")
  parser.add_argument("-c", help="Test Centered convolutions. Not specifying\
  										s or c or H or r is the same as specifying all of them",
  										action="store_true")
  parser.add_argument("-H", help="Test Hermitian convolutions. Not specifying\
  										s or c or H or r is the same as specifying all of them",
  										action="store_true")
  parser.add_argument("-r", help="Test Real convolutions. Not specifying\
                      s or c or H or r is the same as specifying all of them",
                      action="store_true")
  parser.add_argument("-i","--identity", help="Test forward backward routines (hybrid.cc\
                       and/or hybridh.cc). Only in 1D.",
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
  parser.add_argument("-T",metavar='threads',help="Number of threads to use in timing. If set to\
                      0, tests over 1 and OMP_NUM_THREADS (which must be set as an environment variable). Default is 1.",
                      default=1)
  parser.add_argument("-R", help="Find routines used in output.",
                      action="store_true")
  parser.add_argument("-S", help="Test different strides.",
                      action="store_true")
  parser.add_argument("-A","--All", help="Perform all tests. If the dimension\
                      (1, 2 or 3) and/or convolution type (s, c or H) is\
                      specified then perform all tests in that dimension and/or\
                      type.",
                      action="store_true")
  parser.add_argument("-t",metavar='tolerance',help="Error tolerance. Default is 1e-12.",
                      default=1e-12)
  parser.add_argument("-p",help="Print out everything.",action="store_true")
  parser.add_argument("-l",help="Show log of failed cases",
                      action="store_true")
  parser.add_argument("-v",help="Show the results of every test.",
                      action="store_true")
  return parser.parse_args()

def getPrograms(args):
  programs=[]
  A=args.All
  S=args.s
  C=args.c
  H=args.H
  R=args.r
  i=args.identity or A
  X=args.one
  Y=args.two
  Z=args.three

  notSCHR=(not (S or C or H or R))
  notXYZ=(not (X or Y or Z))

  SorNotSCHR=(S or notSCHR)
  CorNotSCHR=(C or notSCHR)
  HorNotSCHR=(H or notSCHR)
  rorNotSCHR=(R or notSCHR)

  if X or notXYZ:
    if SorNotSCHR:
      programs.append(Program("hybridconv"))
      if i:
        programs.append(Program("hybrid",mult=False))
    if CorNotSCHR:
      programs.append(Program("hybridconv",centered=True))
      if i:
        programs.append(Program("hybrid",centered=True,mult=False))
    if HorNotSCHR:
      programs.append(Program("hybridconvh",hermitian=True))
      if i:
        programs.append(Program("hybridh",hermitian=True,mult=False))
    if rorNotSCHR:
      programs.append(Program("hybridconvr",real=True))

  if Y or notXYZ:
    if SorNotSCHR:
      programs.append(Program("hybridconv2",dim=2))
    if HorNotSCHR:
      programs.append(Program("hybridconvh2",dim=2,hermitian=True))

  if Z or notXYZ:
    if SorNotSCHR:
      programs.append(Program("hybridconv3",dim=3))
    if HorNotSCHR:
      programs.append(Program("hybridconvh3",dim=3,hermitian=True))

  return programs

def test(programs, args):
  lenP=len(programs)
  T=0 if args.All else int(args.T)

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
        cmd = 'echo $OMP_NUM_THREADS'
        OMP_NUM_THREADS=str(subprocess.check_output(cmd, shell=True))
        Tnum=re.search(r"\d+",OMP_NUM_THREADS)
        if Tnum is not None:
          Tnum=int(Tnum.group(0))
        else:
          sys.exit("Could not find OMP_NUM_THREADS environment variable for T=0 option.")
        Ts=[1,Tnum]
        print(f"Testing {name} with 1 and {Tnum} threads.\n")
      elif T == 1:
        print(f"Testing {name} with 1 thread.\n")
        Ts=[1]
      elif T > 1:
        print(f"Testing {name} with {T} threads.\n")
        Ts=[T]
      else:
        raise ValueError(f"{T} is an invalid number of threads.")

      iterate(p,Ts,float(args.t),args.v,args.R,args.S or args.All,args.p)

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
      print("Out of "+str(total)+" tests, "+str(passed)+" passed, "+str(failed)+" failed.")

    if args.l and len(failedCases) > 0:
      print("\nFailed Cases:\n")
      for case in failedCases:
        print(case+";")
      print()
  else:
    print("No programs to test.\n")

def iterate(program, threads, tol, verbose, R, testS, printEverything):

  dim=program.dim

  vals=ParameterCollection(fillValues(program,1,testS and not program.mult)).vals
  if dim == 1:
    for T in threads:
      checkOptimizer(program,vals[0].L,vals[0].M,T,tol,R,verbose,printEverything)
      for x in vals:
        check(program,[x],T,tol,R,verbose,printEverything)
    if not program.mult:
      vals=ParameterCollection(fillValues(program,8,testS and not program.mult)).vals
      for x in vals:
        for T in threads:
          check(program,[x],T,tol,R,verbose,printEverything)
  else:
    if dim == 2:
      for y in vals:
        minS=(ceilquotient(y.L,2) if program.hermitian else y.L)
        xvals=ParameterCollection(fillValues(program,minS,testS)).vals
        for x in xvals:
          for T in threads:
            check(program,[x,y],T,tol,R,verbose,printEverything)

    elif dim == 3:
      for z in vals:
        minSy=ceilquotient(z.L,2) if program.hermitian else z.L
        yvals=ParameterCollection(fillValues(program,minSy,testS)).vals
        for y in yvals:
          minSy=y.S*y.L
          xvals=ParameterCollection(fillValues(program,y.L*y.S,testS)).vals
          for x in xvals:
            for T in threads:
              check(program,[x,y,z],T,tol,R,verbose,printEverything)
    else:
      exit("Dimension must be 1 2 or 3.")

def fillValues(program, minS, testS):
  # This doesn't quite correspond to C in the main code but it has the property
  # that C == 1 in 1D and C > 1 in higher dimensions
  # It also works for testing hybrid.cc and hybridh.cc.
  C=minS

  centered=program.centered
  hermitian=program.hermitian
  real=program.real

  dim=program.dim
  vals=[]

  # Because of symmetry concerns in the centered cases (compact/noncompact),
  # we check even and odd L values
  if centered and dim == 1 and program.mult:
    Ls=[7,8]
  else:
    Ls=[8]

  Dstart=2 if hermitian and C == 1 else 1

  Ss=[minS]
  if testS:
    if dim == 2:
      Ss+=[minS+2]
    if dim == 3:
      Ss+=[minS+2]

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
        if real:
          ms=[]
        else:
          ms=[L4,L2]
        if not centered:
          ms+=[L,L+1]
        elif not hermitian:
          ms+=[ceilquotient((L2+L),2)]
        ms+=[M]
        for m in ms:
          p=ceilquotient(L,m)
          q=ceilquotient(M,m) if p <= 2 else ceilquotient(M,m*p)*p
          n=q//p
          Istart=0
          Dstop=n
          if q == 1 or real:
            Dstop=1
          elif hermitian:
            Dstop=2
          for I in range(Istart,2):
            if C == 1:
              D=Dstart
              while(D < Dstop):
                vals.append(Parameters(L,M,m,p,q,C,S,D,I))
                D*=2
              vals.append(Parameters(L,M,m,p,q,C,S,Dstop,I))
            else:
              D=1
              vals.append(Parameters(L,M,m,p,q,C,S,D,I))
  return vals

def checkOptimizer(program, L, M, T, tol, R, verbose, printEverything):

  cmd=[program.name,f"-L={L}",f"-M={M}",f"-T={T}","-E","-t"]

  if R:
    cmd.append("-R")

  if program.extraArgs != "":
    cmd.append(program.extraArgs)

  vp = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  vp.wait()
  prc = vp.returncode
  comment = ""

  if prc == 0:
    out, _ = vp.communicate()
    comment = out.rstrip().decode()

  if printEverything:
    print(f"{' '.join(cmd)}\n{comment}\n")

  if program.mult:
    checkError(program, comment, cmd, tol, verbose, R, r"Error")
  else:
    checkError(program, comment, cmd, tol, verbose, R, r"Forward Error")
    checkError(program, comment, cmd, tol, verbose, R, r"Backward Error")

def check(program, vals, T, tol, R, verbose, printEverything):

  cmd=getcmd(program,vals,T,R)

  vp = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  vp.wait()
  prc = vp.returncode
  comment = ""

  if prc == 0:
    out, _ = vp.communicate()
    comment = out.rstrip().decode()

  if printEverything:
    print(f"{' '.join(cmd)}\n{comment}\n")

  if program.mult:
    checkError(program, comment, cmd, tol, verbose, R, r"Error")
  else:
    checkError(program, comment, cmd, tol, verbose, R, r"Forward Error")
    checkError(program, comment, cmd, tol, verbose, R, r"Backward Error")

def checkError(program, comment, cmd, tol, verbose, R, message):
  boldPassedTest="\033[1mPassed Test:\033[0m"
  boldFailedTest="\033[1mFailed Test:\033[0m"
  boldWarning="\033[1mWARNING:\033[0m"
  try:
    error=re.search(r"(?<="+message+r": )(\w|\d|\.|e|-|\+)*",comment)
    if error is not None:
      error=error.group()
      if float(error) > tol or error == "nan" or error == "-nan" or error == "inf":
        program.failTest()
        print("\t"+boldFailedTest+" "+message+": "+error)
        case=" ".join(cmd)
        print("\t"+case)
        if R:
          findRoutines(comment)
        program.failedCases.append(case)
        print()
      else:
        warning=re.search(r"(?<=WARNING: )(\S| )*",comment)
        if warning is not None:
          warning=warning.group()
          program.failTest()
          print("\t"+boldWarning+" "+warning)
          case=" ".join(cmd)
          print("\t"+case)
          if R:
            findRoutines(comment)
          program.failedCases.append(case)
          print()
        else:
          program.passTest()
          if verbose:
            print("\t"+boldPassedTest+" "+message+": "+error)
            case=" ".join(cmd)
            print("\t"+case)
            if R:
              findRoutines(comment)
            print()
    else:
      program.failTest()
      print("\t"+boldFailedTest+" "+message+" not found.")
      case=" ".join(cmd)
      print("\t"+case)
      if R:
        findRoutines(comment)
      program.failedCases.append(case)
      print()
  except Exception as e:
    program.failTest()
    print("\t"+boldFailedTest+f" Exception raised.")
    print(f"\t{e}.")
    case=" ".join(cmd)
    print("\t"+case)
    if R:
      findRoutines(comment)
    program.failedCases.append(case)
    print()

def findRoutines(comment):
  try:
    FR=re.findall(r"(?<=Forwards Routine: )\S+",comment)
    BR=re.findall(r"(?<=Backwards Routine: )\S+",comment)
    params="\t"+"\t\t".join(FR)+"\n\t"+"\t\t".join(BR)
    print(params)
  except:
    print("Could not find routines used.")

def getcmd(program, vals, T, R):
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
  cmd+=["-T="+str(T)]
  if T > 1:
    cmd+=["-threshold","0"]
  cmd+=["-E"]
  if R:
    cmd+=["-R"]

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
