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
    self.warnings=0
    self.total=0
    self.warningCases=[]
    self.failedCases=[]

  def passTest(self):
    self.passed+=1
    self.total+=1

  def warningTest(self, case):
    self.passed+=1
    self.warnings+=1
    self.total+=1
    self.warningCases.append(case)

  def failTest(self,case):
    self.failed+=1
    self.total+=1
    self.failedCases.append(case)

class Options:
  def __init__(self, tol, verbose, R, testS, printEverything, valid):
    self.tol=tol
    self.verbose=verbose
    self.R=R
    self.testS=testS
    self.printEverything=printEverything
    self.valid=valid

class Command:
  def __init__(self, program, T, R, vals=None, L=None, M=None):
    self.name=[program.name]
    self.extraArgs=[]
    if L is not None and M is not None:
      self.L=["-L="+str(L)]
      self.M=["-M="+str(M)]
      self.m=[]
      self.I=[]
      self.D=[]
      self.S=[]
      self.C=[]
      self.extraArgs.append("-t")
    elif vals is not None:
      lenvals=len(vals)
      if lenvals == 3:
        x,y,z = vals
        self.L=self.addParams("L",x.L,y.L,z.L)
        self.M=self.addParams("M",x.M,y.M,z.M)
        self.m=self.addParams("m",x.m,y.m,z.m)
        self.I=self.addParams("I",x.I,y.I,z.I)
        self.D=self.addParams("D",x.D,y.D,z.D)
        self.S=["-Sx="+str(x.S),"-Sy="+str(y.S)]
        self.C=[]
      elif lenvals == 2:
        x,y = vals
        self.L=self.addParams("L",x.L,y.L)
        self.M=self.addParams("M",x.M,y.M)
        self.m=self.addParams("m",x.m,y.m)
        self.I=self.addParams("I",x.I,y.I)
        self.D=self.addParams("D",x.D,y.D)
        self.S=["-Sx="+str(x.S)]
        self.C=[]
      else:
        x=vals[0]
        self.L=["-L="+str(x.L)]
        self.M=["-M="+str(x.M)]
        self.m=["-m="+str(x.m)]
        self.D=["-D="+str(x.D)]
        self.I=["-I="+str(x.I)]
        if not program.mult:
          self.S=["-S"+str(x.S)]
          self.C=["-C"+str(x.C)]
        else:
          self.S=[]
          self.C=[]
    else:
      sys.exit("Either vals or both L and M must be specified to initialize a Command.")
    self.T=["-T="+str(T)]
    if T > 1:
      self.extraArgs.append("-threshold")
      self.extraArgs.append("0")
    self.extraArgs.append("-E")
    if R:
      self.extraArgs.append("-R")

    if program.extraArgs != "":
      self.extraArgs.append(program.extraArgs)

    self.list=self.name+self.L+self.M+self.m+self.D+self.I+self.S+self.C+self.T+self.extraArgs

    self.case=" ".join(self.list)

  def addParams(self, pname, px, py, pz=None):
    # This code avoids reduntant arguments in the 2D and 3D cases for readability
    # e.g. if Lx == Ly == Lz then we can just use L
    if pz is not None:
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
    else:
      if px == py:
        cmd=["-"+pname+"="+str(px)]
      else:
        cmd=["-"+pname+"x="+str(px), "-"+pname+"y="+str(py)]
    return cmd

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
  parser.add_argument("-V", "--valid", help="Check tests to see if they're valid.",
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
    warnings=0
    total=0
    failedCases=[]

    for p in programs:
      name=p.name
      if p.extraArgs:
        name+=" "+p.extraArgs
      if T == 0:
        omp = 'echo $OMP_NUM_THREADS'
        OMP_NUM_THREADS=str(subprocess.check_output(omp, shell=True))
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

      options=Options(float(args.t),args.v,args.R,args.S or args.All,args.p,args.valid)

      iterate(p,Ts,options)

      ppassed=p.passed
      pfailed=p.failed
      pwarnings=p.warnings
      ptotal=p.total

      s="s" if pwarnings > 1 else ""
      warningText=f" with {pwarnings} warning{s}," if pwarnings > 0 else ","

      print(f"Finished testing {name}.")
      print(f"Out of {ptotal} tests, {ppassed} passed{warningText} {pfailed} failed.\n")
      if lenP > 1:
        print("***************\n")
      total+=ptotal
      passed+=ppassed
      failed+=pfailed
      warnings+=pwarnings
      if args.l:
        failedCases+=p.failedCases

    if lenP > 1:
      s="s" if warnings > 1 else ""
      warningText=f" with {warnings} warning{s}," if warnings > 0 else ","
      print(f"Finished testing {lenP} programs.")
      print(f"Out of {total} tests, {passed} passed{warningText} {failed} failed.\n")
    if args.l and len(failedCases) > 0:
      print("\nFailed Cases:\n")
      for case in failedCases:
        print(case+";")
      print()
  else:
    print("No programs to test.\n")

def iterate(program, threads, options):
  dim=program.dim
  testS=options.testS
  vals=ParameterCollection(findTests(program,1,testS and not program.mult)).vals
  if dim == 1:
    for T in threads:
      checkOptimizer(program,vals[0].L,vals[0].M,T,options)
      for x in vals:
        check(program,[x],T,options)
    if not program.mult:
      vals=ParameterCollection(findTests(program,8,testS and not program.mult)).vals
      for x in vals:
        for T in threads:
          check(program,[x],T,options)
  else:
    if dim == 2:
      for y in vals:
        minS=(ceilquotient(y.L,2) if program.hermitian else y.L)
        xvals=ParameterCollection(findTests(program,minS,testS)).vals
        for x in xvals:
          for T in threads:
            check(program,[x,y],T,options)

    elif dim == 3:
      for z in vals:
        minSy=ceilquotient(z.L,2) if program.hermitian else z.L
        yvals=ParameterCollection(findTests(program,minSy,testS)).vals
        for y in yvals:
          minSy=y.S*y.L
          xvals=ParameterCollection(findTests(program,y.L*y.S,testS)).vals
          for x in xvals:
            for T in threads:
              check(program,[x,y,z],T,options)
    else:
      exit("Dimension must be 1 2 or 3.")

def collectTests(program, L, M, m, minS, testS, Dmin=0, Dmax=0, I0=True, I1=True):
  vals=[]

  C=minS
  dim=program.dim
  centered=program.centered
  hermitian=program.hermitian
  real=program.real

  p,q,n=getpqn(centered, hermitian, real, L, M, m)

  if Dmax == 0:
    Dmax=n

  if q == 1:
    Dstart=1
    Dstop=Dstart
  elif C > 1:
    Dstart=2 if hermitian and not program.mult else 1
    Dstop=Dstart
  elif hermitian:
    Dstart=2
    Dstop=Dstart
  elif real:
    Dstart=max(Dmin,1)
    Dstop=min(max(Dmax,1),(n-1)//2)
  else:
    Dstart=max(Dmin,1)
    Dstop=min(max(Dmax,1),n)

  Ds=getDs(Dstart,Dstop)

  Ss=[minS]
  if testS:
    if dim == 2:
      Ss+=[minS+2]
    if dim == 3:
      Ss+=[minS+2]

  Istart=0
  Istop=2
  if not I0:
    Istart=1
  if not I1:
    Istop=1

  for S in Ss:
    for I in range(Istart,Istop):
      for D in Ds:
        vals.append(Parameters(L,M,m,p,q,C,S,D,I))

  return vals

def getpqn(centered, hermitian, real, L, M, m):
  p=ceilquotient(L,m)
  if ((centered or hermitian) and p%2==0) or p == 2:
    P=p//2
  else:
    P=p
  n=ceilquotient(M,m*P)
  q=ceilquotient(M,m) if p <= 2 else n*p
  return p, q, n

def getDs(start, stop, pow2=False):
  result=[start]
  if pow2:
    result+=[x for x in range(start+2-start%2,stop,2) if ceilpow2(x)==x]
  else:
    result+=list(range(start+2-start%2, stop,2))
  if stop > start:
    result.append(stop)
  return result

def findTests(program, minS, testS):
  if program.real:
    return realTests(program, minS, testS)
  elif program.hermitian:
    return hermitianTests(program, minS, testS)
  elif program.centered:
    return centeredTests(program, minS, testS)
  else:
    return complexTests(program, minS, testS)

def complexTests(program, minS, testS):
  vals=[]
  L=8
  Mvalues=[2*L,ceilquotient(5*L,2)]
  for M in Mvalues:
    mvalues=[ceilquotient(L,4),ceilquotient(L,2),L,L+1,M]
    for m in mvalues:
      vals+=collectTests(program, L=L, M=M, m=m, minS=minS, testS=testS)
  return vals

def centeredTests(program, minS, testS):
  assert program.centered
  vals=[]
  Lvalues=[]
  if program.dim == 1 and program.mult:
    Lvalues=[7,8]
  else:
    Lvalues=[8]
  for L in Lvalues:
    L2=ceilquotient(L,2)
    Mvalues=[3*L2-2*(L%2),2*L,5*L2]
    for M in Mvalues:
      mvalues=[ceilquotient(L,4),L2,ceilquotient(L2+L,2),M]
      for m in mvalues:
        vals+=collectTests(program, L=L, M=M, m=m, minS=minS, testS=testS)
  return vals

def hermitianTests(program, minS, testS):
  assert program.hermitian
  vals=[]
  Lvalues=[]
  if program.dim == 1 and program.mult:
    Lvalues=[7,8]
  else:
    Lvalues=[8]
  for L in Lvalues:
    L2=ceilquotient(L,2)
    Mvalues=[3*L2-2*(L%2),2*L,5*L2]
    for M in Mvalues:
      mvalues=[L2,M]
      if minS == 1:
        mvalues=[ceilquotient(L,4)]+mvalues
      for m in mvalues:
        vals+=collectTests(program, L=L, M=M, m=m, minS=minS, testS=testS)
  return vals

def realTests(program, minS, testS):
  assert program.real
  vals=[]
  mvalues=[8,10,12]
  for m in mvalues:
    vals+=collectTests(program, L=16, M=40, m=m, minS=minS, testS=testS)
  mvalues=[4,8,16,17]
  for m in mvalues:
    vals+=collectTests(program, L=8, M=48, m=m, minS=minS, testS=testS)
    vals+=collectTests(program, L=8, M=40, m=m, minS=minS, testS=testS)
    vals+=collectTests(program, L=4, M=16, m=m, minS=minS, testS=testS)
    vals+=collectTests(program, L=3, M=16, m=m, minS=minS, testS=testS)
  mvalues=[4,8,16]
  for m in mvalues:
    vals+=collectTests(program, L=8, M=64, m=m, minS=minS, testS=testS)
    vals+=collectTests(program, L=8, M=56, m=m, minS=minS, testS=testS)
    vals+=collectTests(program, L=8, M=32, m=m, minS=minS, testS=testS)
  return vals

def checkOptimizer(program, L, M, T, options):
  R=options.R
  cmd=Command(program,T,R,L=L,M=M)

  vp = subprocess.Popen(cmd.list, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  vp.wait()
  prc = vp.returncode
  output = ""

  if prc == 0:
    out, _ = vp.communicate()
    output = out.rstrip().decode()

  if options.printEverything:
    print(f"{cmd.case}\n{output}\n")

  if R:
    routines=findRoutines(output)
  else:
    routines=None

  if program.mult:
    errorSearch(program,output,cmd,options,routines,r"Error")
  else:
    errorSearch(program,output,cmd,options,routines,r"Forward Error")
    errorSearch(program,output,cmd,options,routines,r"Backward Error")

def check(program, vals, T, options):
  R=options.R
  cmd=Command(program,T,R,vals)

  vp = subprocess.Popen(cmd.list, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  vp.wait()
  prc = vp.returncode
  output = ""

  if prc == 0:
    out, _ = vp.communicate()
    output = out.rstrip().decode()

  if options.printEverything:
    print(f"{cmd.case}\n{output}\n")

  if R:
    routines=findRoutines(output)
  else:
    routines=None

  testPassed=True
  if options.valid:
    testPassed=invalidSearch(program,output,cmd,routines)

  if testPassed:
    if program.mult:
      errorSearch(program,output,cmd,options,routines,r"Error")
    else:
      testPassed=errorSearch(program,output,cmd,options,routines,r"Forward Error")
      if testPassed:
        errorSearch(program,output,cmd,options,routines,r"Backward Error")

def errorSearch(program, output, cmd, options, routines, msg):
  try:
    error=re.search(r"(?<="+msg+r": )(\w|\d|\.|e|-|\+)*",output)
    if error is not None:
      error=error.group()
      if float(error) > options.tol or error == "nan" or error == "-nan" or error == "inf":
        return evaluate(program,"f",msg+": "+error,cmd.case,routines)
      else:
        warning=re.search(r"(?<=WARNING: )(\S| )*",output)
        if warning is not None:
          return evaluate(program,"w",warning.group(),cmd.case,routines)
        else:
          return evaluate(program,"p",msg+": "+error,cmd.case,routines,options.verbose)
    else:
      return evaluate(program,"f",msg+" not found.",cmd.case,routines)
  except Exception as e:
    return evaluate(program,"f",f" Exception raised.\n\t{e}",cmd.case,routines)

def invalidSearch(program, output, cmd, routines):
  msg="Optimizer found no valid cases with specified parameters."
  try:
    invalidTest=re.search(msg,output)
    if invalidTest is not None:
      return evaluate(program,"w",msg,cmd.case,routines)
  except Exception as e:
    return evaluate(program,"f",f" Exception raised.\n\t{e}",cmd.case,routines)

  if program.dim == 1:
    # optimization parameters
    op=cmd.m+cmd.D+cmd.I
    for p in op:
      msg=p[1:]
      try:
        invalidTest=re.search(msg,output)
        if invalidTest is None:
          evaluate(program,"w",msg+" not found.",cmd.case,routines)
          break
      except Exception as e:
        evaluate(program,"f",f" Exception raised.\n\t{e}",cmd.case,routines)
  return True

def evaluate(program, result, message, case, routines, verbose=True):
  boldPassedTest="\033[1mPassed Test:\033[0m"
  boldFailedTest="\033[1mFailed Test:\033[0m"
  boldWarning="\033[1mWARNING:\033[0m"
  if result=="p":
    program.passTest()
    printResults(boldPassedTest, message, case, routines, verbose)
    return True
  elif result=="w":
    program.warningTest(case)
    printResults(boldWarning, message, case, routines, verbose)
    return False
  elif result=="f":
    program.failTest(case)
    printResults(boldFailedTest, message, case, routines, verbose)
    return False
  else:
    sys.exit("In evaluate, result must either be 'p', 'w', or 'f'.")

def printResults(resultMessage, message, case, routines, verbose):
  if verbose:
    print("\t"+resultMessage+" "+message)
    print("\t"+case)
    if routines is not None:
      print(routines)
    print()

def findRoutines(output):
  try:
    FR=re.findall(r"(?<=Forwards Routine: )\S+",output)
    BR=re.findall(r"(?<=Backwards Routine: )\S+",output)
    params="\t"+"\t\t".join(FR)+"\n\t"+"\t\t".join(BR)
    return params
  except:
    return "Could not find routines used."

def ceilpow2(n):
  n-=1
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  return n+1;

if __name__ == "__main__":
  main()
