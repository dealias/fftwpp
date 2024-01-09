#!/usr/bin/env python3

import subprocess
import re
import argparse
import sys
from utils import *
import copy

def main(mpi=False):
  args=getArgs()
  args.mpi=mpi
  programs=getPrograms(args)
  test(programs, args)

class Program:
  def __init__(self, name, dim=1, mult=True, centered=False, hermitian=False, real=False):
    self.name="./"+name
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
  def __init__(self, args):
    self.tol=args.t
    self.verbose=args.v
    self.R=args.R
    self.testS=(args.S or args.All)
    self.printEverything=args.p
    self.valid=args.valid
    self.vg=args.valgrind
    self.d=args.d
    self.mpi=args.mpi

class Command:
  def __init__(self, program, T, options, vals=None, L=None, M=None, nodes=0):
    self.name=[program.name]
    self.extraArgs=[]
    self.mpi=[]

    if options.R:
      self.extraArgs.append("-R")

    if options.mpi and nodes !=0:
        self.mpi=["mpiexec","-n",str(nodes)]

    self.name=[program.name]
    self.vg=["valgrind"] if options.vg else []


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
        self.S=[]
        if program.hermitian:
          if y.S != ceilquotient(z.L,2):
            self.S+=["-Sy="+str(y.S)]
        else:
          if y.S != z.L:
            self.S+=["-Sy="+str(y.S)]
        if x.S != y.L*y.S:
          self.S+=["-Sx="+str(x.S)]
        self.C=[]
      elif lenvals == 2:
        x,y = vals
        self.L=self.addParams("L",x.L,y.L)
        self.M=self.addParams("M",x.M,y.M)
        self.m=self.addParams("m",x.m,y.m)
        self.I=self.addParams("I",x.I,y.I)
        self.D=self.addParams("D",x.D,y.D)
        self.S=[]
        if program.hermitian and x.S != ceilquotient(y.L,2):
          self.S+=["-Sx="+str(x.S)]
        elif not program.hermitian and x.S != y.L:
          self.S+=["-Sx="+str(x.S)]
        self.C=[]
      else:
        x=vals[0]
        self.L=["-L="+str(x.L)]
        self.M=["-M="+str(x.M)]
        self.m=["-m="+str(x.m)]
        self.D=["-D="+str(x.D)]
        self.I=["-I="+str(x.I)]
        if not program.mult:
          self.S=[]
          if x.S != x.C:
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


    if program.extraArgs != "":
      self.extraArgs.append(program.extraArgs)

    self.list=self.vg+self.mpi+self.name+self.L+self.M+self.m+self.D+self.I+self.S+self.C+self.T+self.extraArgs

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
                       and/or hybridh.cc and/or hybridr.cc). Only in 1D.",
                      action="store_true")
  parser.add_argument("-I","--Identity", help="Test forward backward routines and nothing else. Only in 1D.",
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
  parser.add_argument("-d",help="Check all 1D cases inside multidimensional convolutions. Note: using this flag results in much slower testing.",
                      action="store_true")
  parser.add_argument("--valgrind",help="Run tests under valgrind. Requires valgrind to be installed. Note: using this flag results in much slower testing.",
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
  I=args.Identity
  X=args.one
  Y=args.two
  Z=args.three

  iorI=(i or I)

  notSCHR=(not (S or C or H or R))
  notXYZ=(not (X or Y or Z))

  SorNotSCHR=(S or notSCHR)
  CorNotSCHR=(C or notSCHR)
  HorNotSCHR=(H or notSCHR)
  rorNotSCHR=(R or notSCHR)

  if not args.mpi:
    if X or notXYZ:
      if SorNotSCHR:
        if iorI:
          programs.append(Program("hybrid",mult=False))
        if not I:
          programs.append(Program("hybridconv"))

      if CorNotSCHR:
        if iorI:
          programs.append(Program("hybrid",centered=True,mult=False))
        if not I:
          programs.append(Program("hybridconv",centered=True))

      if HorNotSCHR:
        if iorI:
          programs.append(Program("hybridh",hermitian=True,mult=False))
        if not I:
          programs.append(Program("hybridconvh",hermitian=True))

      if rorNotSCHR:
        if iorI:
          programs.append(Program("hybridr",real=True,mult=False))
        if not I:
          programs.append(Program("hybridconvr",real=True))

  if Y or notXYZ:
    if SorNotSCHR:
      programs.append(Program("hybridconv2",dim=2))
    if HorNotSCHR:
      programs.append(Program("hybridconvh2",dim=2,hermitian=True))
    if rorNotSCHR:
      programs.append(Program("hybridconvr2",dim=2,real=True))

  if Z or notXYZ:
    if SorNotSCHR:
      programs.append(Program("hybridconv3",dim=3))
    if HorNotSCHR:
      programs.append(Program("hybridconvh3",dim=3,hermitian=True))
    if rorNotSCHR:
      programs.append(Program("hybridconvr3",dim=3,real=True))

  return programs

def test(programs, args):
  lenP=len(programs)
  T=0 if args.All else int(args.T)
  mpi=args.mpi

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

      iterate(p,Ts,Options(args))

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
  mpi=options.mpi
  if dim == 1:
    vals=ParameterCollection(findTests(program,1,options)).vals
    for T in threads:
      checkOptimizer(program,vals[0].L,vals[0].M,T,options)
      for x in vals:
        checkCase(program,[x],T,options)
    if not program.mult:
      cols=[ParameterCollection(findTests(program,2,options,outer=True))]
      if testS:
        cols+=[copy.deepcopy(cols[0])]
        updateStride(cols[1],3)
      for S in range(len(cols)):
        vals=cols[S].vals
        for x in vals:
          for T in threads:
            checkCase(program,[x],T,options)
  else:
    vals=ParameterCollection(findTests(program,1,options,inner=True)).vals
    if dim == 2:
      nodevals=[0]
      if mpi:
        nodevals=[1,2,4]
      xcols=[ParameterCollection(findTests(program,2,options,outer=True))]
      if testS:
        xcols+=[copy.deepcopy(xcols[0])]
      lenxcols=len(xcols)
      for y in vals:
        minS=(ceilquotient(y.L,2) if program.hermitian else y.L)
        for S in range(lenxcols):
          updateStride(xcols[S],minS+S)
          xvals=xcols[S].vals
          for x in xvals:
            for T in threads:
              for nodes in nodevals:
                checkCase(program,[x,y],T,options,nodes)

    elif dim == 3:
      nodevals=[0]
      if mpi:
        nodevals=[1,2,4]
      ycol=ParameterCollection(findTests(program,2,options,outer=True))
      ycols=[ycol]
      xcols=[copy.deepcopy(ycol)]
      if testS:
        ycols+=[copy.deepcopy(ycol)]
        xcols+=[copy.deepcopy(ycol)]
      lenycols=len(ycols)
      lenxcols=len(xcols)
      for z in vals:
        minSy=ceilquotient(z.L,2) if program.hermitian else z.L
        for Sy in range(lenycols):
          updateStride(ycols[Sy],minSy+Sy)
          yvals=ycols[Sy].vals
          for y in yvals:
            minSx=y.S*y.L
            for Sx in range(lenxcols):
              updateStride(xcols[Sx],minSx+Sx)
              xvals=xcols[Sx].vals
              for x in xvals:
                for T in threads:
                  for nodes in nodevals:
                    checkCase(program,[x,y,z],T,options,nodes)
    else:
      exit("Dimension must be 1 2 or 3.")

# Replace all strides in collection with value newS
def updateStride(collection, newS):
  for v in collection.vals:
    v.S=newS

def collectTests(program, L, M, m, minS, Dmin=0, Dmax=0, I0=True, I1=True):
  vals=[]

  C=minS
  notmult=not program.mult
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
    Dstart=2 if hermitian and notmult else 1
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

  Istart=0
  Istop=2
  if not I0:
    Istart=1
  if not I1:
    Istop=1

  for I in range(Istart,Istop):
    for D in Ds:
      vals.append(Parameters(L,M,m,p,q,C,minS,D,I))

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

def transformType(program, outer=False, inner=False):
  dim=program.dim
  if program.real:
    if dim == 1 or outer:
      return "r"
    else:
      return "s"
  elif program.hermitian:
    if dim == 1 or inner:
      return "H"
    else:
      return "c"
  elif program.centered:
    return "c"
  else:
    return "s"

def findTests(program, minS, options,outer=False, inner=False):
  ttype=transformType(program, outer, inner)
  if ttype == "r":
    return realTests(program, minS, options)
  elif ttype == "H":
    return hermitianTests(program, minS, options)
  elif ttype == "c":
    return centeredTests(program, minS, options)
  else:
    return complexTests(program, minS, options)

def details(C,dim,d):
  return ((C == 1 and dim > 1) or (dim == 1)) or d

def complexTests(program, minS, options):
  det=details(minS,program.dim,options.d)
  vals=[]
  L=8
  Ms=[2*L]
  if details:
    Ms+=[ceilquotient(5*L,2)]
  for M in Ms:
    ms=[L]
    if det:
      ms=[M,L+1]+ms+[ceilquotient(L,2),ceilquotient(L,4)]
    for m in ms:
      vals+=collectTests(program, L=L, M=M, m=m, minS=minS)
  return vals

def centeredTests(program, minS, options):
  assert program.centered
  vals=[]
  det=details(minS,program.dim,options.d)

  Ls=[8]
  if det:
    Ls+=[7]
  for L in Ls:
    L2=ceilquotient(L,2)
    Ms=[3*L2-2*(L%2)]
    if det:
      Ms+=[2*L,5*L2]
    for M in Ms:
      ms=[L2]
      if det:
        ms=[ceilquotient(L,4),L2,ceilquotient(L2+L,2),M]
      for m in ms:
        vals+=collectTests(program, L=L, M=M, m=m, minS=minS)
  return vals

def hermitianTests(program, minS, options):
  assert program.hermitian
  vals=[]
  det=details(minS,program.dim,options.d)

  Ls=[8]
  if det:
    Ls+=[7]
  for L in Ls:
    L2=ceilquotient(L,2)
    Ms=[3*L2-2*(L%2)]
    if det:
      Ms+=[2*L,5*L2]
    for M in Ms:
      ms=[L2]
      if det and minS == 1:
        ms=[M]+ms+[ceilquotient(L,4)]
      for m in ms:
        vals+=collectTests(program, L=L, M=M, m=m, minS=minS)
  return vals

def realTests(program, minS, options):
  assert program.real
  vals=[]
  det=details(minS,program.dim,options.d)

  # p = 1
  Ls=[8]
  Ms=[16]
  if det:
    Ls+=[3]
    Ms+=[24,64]
  for M in Ms:
    for L in Ls:
      vals+=collectTests(program, L=L, M=M, m=8, minS=minS)

  if det:
    # p = 2
    Ls=[8,5]
    Ms=[16,24,32]
    for M in Ms:
      for L in Ls:
        vals+=collectTests(program, L=L, M=M, m=4, minS=minS)
    # p > 2
    Ls=[8,7]
    M=16
    for L in Ls:
      vals+=collectTests(program, L=L, M=M, m=2, minS=minS)
    Ls=[9,7]
    M=63
    for L in Ls:
      vals+=collectTests(program, L=L, M=M, m=3, minS=minS)
    Ls=[24,21]
    M=96
    for L in Ls:
      vals+=collectTests(program, L=L, M=M, m=4, minS=minS)

  return vals

def checkOptimizer(program, L, M, T, options):
  cmd=Command(program,T,options,L=L,M=M)
  check(program, cmd, T, options)

def checkCase(program, vals, T, options, nodes=0):
  cmd=Command(program,T,options,vals,nodes=nodes)
  check(program, cmd, T, options)

def check(program, cmd, T, options):
  output=usecmd(cmd.list)

  if options.printEverything:
    print(f"{cmd.case}\n{output}\n")

  if options.R:
    routines=findRoutines(output)
  else:
    routines=None

  testPassed=True
  if options.valid:
    testPassed=invalidSearch(program,output,cmd,routines)

  if options.vg:
    testPassed=segFaultSearch(program,output,cmd,routines)

  if testPassed:
    if program.mult:
      errorSearch(program,output,cmd,options,routines,r"Error")
    else:
      testPassed=errorSearch(program,output,cmd,options,routines,r"Backward Error")
      if testPassed and not program.real:
        # Forward error is not yet supported in real case
        errorSearch(program,output,cmd,options,routines,r"Forward Error")

def usecmd(cmd):
  vp = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  vp.wait()
  prc = vp.returncode
  output = ""

  if prc == 0:
    out, _ = vp.communicate()
    output = out.rstrip().decode()
  return output

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

def segFaultSearch(program, output, cmd, routines):
  errSum="ERROR SUMMARY"
  invRead="Invalid read"
  invWrite="Invalid write"
  try:
    error=re.search(r"(?<="+errSum+r": )(\d|\.|e|-|\+)*",output)
    iR=re.search(invRead,output)
    iW=re.search(invWrite,output)
    if iR is not None:
      if iW is not None:
        return evaluate(program,"f",f"{invRead} and {invWrite.lower()}",cmd.case,routines)
      return evaluate(program,"f",invRead,cmd.case,routines)
    if iW is not None:
      return evaluate(program,"f",invWrite,cmd.case,routines)
    if error is not None:
      error=int(error.group())
      if error == 0:
        return True
      elif error > 0:
        s="s" if error > 1 else ""
        return evaluate(program,"f",f"Valgrind found {error} error{s}",cmd.case,routines)
  except Exception as e:
    return evaluate(program,"f",f" Exception raised.\n\t{e}",cmd.case,routines)
  return evaluate(program,"f","tests.py does not recognize valgrind error.",cmd.case,routines)

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
    sys.stdout.flush()

def findRoutines(output):
  try:
    FR=re.findall(r"(?<=Forwards Routine: )\S+",output)
    BR=re.findall(r"(?<=Backwards Routine: )\S+",output)
    params="\t"+"\t\t".join(FR)+"\n\t"+"\t\t".join(BR)
    return params
  except:
    return "Could not find routines used."

if __name__ == "__main__":
  main()
