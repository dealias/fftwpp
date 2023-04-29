#!/usr/bin/env python3

import subprocess
import re
import argparse
import sys
from HybridParameters import *

boldPassedTest="\033[1mPassed Test:\033[0m"
boldFailedTest="\033[1mFailed Test:\033[0m"
boldWarning="\033[1mWARNING:\033[0m"

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
  def __init__(self, tol, verbose, R, testS, printEverything, checkTests):
    self.tol=tol
    self.verbose=verbose
    self.R=R
    self.testS=testS
    self.printEverything=printEverything
    self.checkTests=checkTests

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
  parser.add_argument("--checkTests", help="Check tests to see if they're valid.\
                      This is only used for testing tests.py",
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

      options=Options(float(args.t),args.v,args.R,args.S or args.All,args.p,args.checkTests)

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
  # threads, tol, verbose, R, testS, printEverything,checkTests
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
  hermitian=program.hermitian

  p=ceilquotient(L,m)
  q=ceilquotient(M,m) if p <= 2 else ceilquotient(M,m*p)*p
  n=q//p

  if Dmax == 0:
    Dmax=n

  if C > 1:
    Dstart=1
    Dstop=1
  elif hermitian:
    Dstart=max(Dmin,2)
    Dstop=min(max(Dmax,2),n)
  else:
    Dstart=max(Dmin,1)
    Dstop=min(max(Dmax,1),n)

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
      D=Dstart
      while(D < Dstop):
        vals.append(Parameters(L,M,m,p,q,C,S,D,I))
        D*=2
      vals.append(Parameters(L,M,m,p,q,C,S,Dstop,I))

  return vals

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
  mvalues=[4,8,16]
  vals+=collectTests(program, L=8, M=32, m=8, Dmin=1, Dmax=1, minS=minS, testS=testS)
  vals+=collectTests(program, L=8, M=32, m=16, Dmin=1, Dmax=1, minS=minS, testS=testS)
  for m in mvalues:
    vals+=collectTests(program, L=4, M=16, m=m, Dmin=1, Dmax=1, minS=minS, testS=testS)
    vals+=collectTests(program, L=3, M=16, m=m, Dmin=1, Dmax=1, minS=minS, testS=testS)
  return vals

def checkOptimizer(program, L, M, T, options):

  cmd=[program.name,f"-L={L}",f"-M={M}",f"-T={T}","-E","-t"]

  if options.R:
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

  if options.printEverything:
    print(f"{' '.join(cmd)}\n{comment}\n")

  if program.mult:
    checkError(program, comment, cmd, options, r"Error")
  else:
    checkError(program, comment, cmd, options, r"Forward Error")
    checkError(program, comment, cmd, options, r"Backward Error")

def check(program, vals, T, options):
  # T, tol, R, verbose, printEverything, checkTests
  R=options.R
  cmd=getcmd(program,vals,T,R)

  vp = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  vp.wait()
  prc = vp.returncode
  comment = ""

  if prc == 0:
    out, _ = vp.communicate()
    comment = out.rstrip().decode()

  if options.printEverything:
    print(f"{' '.join(cmd)}\n{comment}\n")

  if options.checkTests:
    checkInvalidTests(program,comment,cmd,R)

  if program.mult:
    checkError(program, comment, cmd, options, r"Error")
  else:
    checkError(program, comment, cmd, options, r"Forward Error")
    checkError(program, comment, cmd, options, r"Backward Error")

def checkError(program, comment, cmd, options, message):
  R=options.R
  try:
    error=re.search(r"(?<="+message+r": )(\w|\d|\.|e|-|\+)*",comment)
    if error is not None:
      error=error.group()
      if float(error) > options.tol or error == "nan" or error == "-nan" or error == "inf":
        print("\t"+boldFailedTest+" "+message+": "+error)
        case=" ".join(cmd)
        print("\t"+case)
        if R:
          findRoutines(comment)
        program.failTest(case)
        print()
      else:
        warning=re.search(r"(?<=WARNING: )(\S| )*",comment)
        if warning is not None:
          warning=warning.group()

          print("\t"+boldWarning+" "+warning)
          case=" ".join(cmd)
          print("\t"+case)
          if R:
            findRoutines(comment)
          program.warningTest(case)
          print()
        else:
          program.passTest()
          if options.verbose:
            print("\t"+boldPassedTest+" "+message+": "+error)
            case=" ".join(cmd)
            print("\t"+case)
            if R:
              findRoutines(comment)
            print()
    else:
      print("\t"+boldFailedTest+" "+message+" not found.")
      case=" ".join(cmd)
      print("\t"+case)
      if R:
        findRoutines(comment)
      program.failTest(case)
      print()
  except Exception as e:
    testException(program, comment, cmd, R, e)

def checkInvalidTests(program, comment, cmd, R):
  message="Optimizer found no valid cases with specified parameters."
  try:
    invalidTest=re.search(message,comment)
    if invalidTest is not None:
      invalidTest=invalidTest.group()
      print("\t"+boldWarning+" "+message)
      case=" ".join(cmd)
      print("\t"+case)
      program.warningTest(case)
      print()
  except Exception as e:
    testException(program, comment, cmd, R, e)

def testException(program, comment, cmd, R, e):
  print("\t"+boldFailedTest+f" Exception raised.")
  print(f"\t{e}.")
  case=" ".join(cmd)
  print("\t"+case)
  if R:
    findRoutines(comment)
  program.failTest(case)
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
