#!/usr/bin/env python3

import sys, getopt, os
import argparse
from subprocess import *
import re

def main():
  args=getArgs()
  threads=args.T
  dimensions=[]
  S=args.S
  H=args.H
  R=args.R

  one=args.one
  two=args.two
  three=args.three

  not123=not (one or two or three)

  if one or not123:
    dimensions.append(1)
  if two or not123:
    dimensions.append(2)
  if (three or not123) and not R:
    dimensions.append(3)

  if threads == 0:
    cmd = 'echo $OMP_NUM_THREADS'
    OMP_NUM_THREADS=str(check_output(cmd, shell=True))
    T=int(re.search(r"\d+",OMP_NUM_THREADS).group(0))
    Ts=[1,T]
  else:
    Ts=[threads]

  notSorHorR = not (S or H or R)

  for d in dimensions:
    if S or notSorHorR:
      for T in Ts:
        time(args,d,False,False,T)
    if H or notSorHorR:
      for T in Ts:
        time(args,d,True,False,T)
    if R or notSorHorR:
      for T in Ts:
        time(args,d,False,True,T)


def callTiming(args,program, erase,taskset,runtype=None):
  cmd=["timing.py"]
  cmdLine="timing.py"
  if taskset != None and taskset != "":
    cmd+=[f"-B{taskset}"]
    cmdLine+=f" -B\"{taskset}\""
  cmd+=[f"-p{program}"]+args+erase
  cmdLine+=f" -p{program} "+" ".join(args+erase)
  if runtype != None:
    cmd+=[f"-r{runtype}"]
    cmdLine+=f" -r{runtype}"
  print(cmdLine,flush=True)
  output=run(cmd,capture_output=True).stdout.decode();
  print(output,flush=True);

def time(args,dim,Hermitian,real,T):
  a=args.a
  b=args.b
  I=args.I
  e=args.e
  runtype=args.r
  dimString=str(dim) if dim > 1 else ""
  new="hybridconv"
  old="conv"


  if Hermitian:
    new += "h"
  elif real:
    new += "r"
    old = "r"+old
  else:
    old = "c"+old

  new+=dimString
  old+=dimString

  taskset=os.getenv("TASKSET")
  if T == 1 and taskset != None:
    taskset="taskset -c 0"

  args=[f"-a{a}",f"-b{b}",f"-I{I}",f"-T{T}",f"-s{args.s}"]
  erase=[]
  if e:
    erase.append("-e")
  if runtype == "implicit" and not real:
    callTiming(args,old,erase,taskset,runtype)
  elif runtype == "explicit":
    callTiming(args,old,erase,taskset,runtype)
  elif runtype == "explicito":
    if dim != 1:
      print("explicito is only supported for 1 dimensional routines.")
    else:
      callTiming(args,old,erase,taskset,runtype)
  elif runtype == "hybrid":
    callTiming(args,new,erase,taskset)
    if Hermitian and I == 0: # Why?
      callTiming(args,new,erase,taskset,"explicit")
  elif runtype == None:
    if not real:
      callTiming(args,old,erase,taskset,"implicit")
    callTiming(args,old,erase,taskset,"explicit")
    if dim == 1:
      callTiming(args,old,erase,taskset,"explicito")
    callTiming(args,new,erase,taskset)
    if Hermitian and I == 0:
      callTiming(args,new,erase,taskset,"explicit")
  else:
    print(f"runtype=\"{runtype}\" is invalid")


def getArgs():
  parser = argparse.ArgumentParser(description="Call timing.py with correct parameters.")
  parser.add_argument("-1","--one", help="Time 1D Convolutions. Not specifying\
                      1 or 2 or 3 is the same as specifying all of them",
                      action="store_true")
  parser.add_argument("-2","--two", help="Time 2D Convolutions. Not specifying\
                      1 or 2 or 3 is the same as specifying all of them",
                      action="store_true")
  parser.add_argument("-3","--three", help="Time 3D Convolutions. Not specifying\ 1 or 2 or 3 is the same as specifying all of them",
                      action="store_true")
  parser.add_argument("-a",help="Start.", default=1, type=int)
  parser.add_argument("-b",help="End.", default=1, type=int)
  parser.add_argument("-e", help="Erase old data.", action="store_true")
  parser.add_argument("-H", help="Test Hermitian convolutions. Not specifying S or H is the same as specifying both.", action="store_true")

  parser.add_argument("-I",help="Interval. Checks powers of 2 when 0. Default is 0.", default=0, type=int)
  parser.add_argument("-s",help="Minimum time limit (seconds).", default=5, type=float)
  parser.add_argument("-r",help="runtype: implicit, explicit, or hybrid. Not specifying does all of them.", type=str)
  parser.add_argument("-R", help="Test Real convolutions. Not specifying S or H is the same as specifying both.", action="store_true")
  parser.add_argument("-S", help="Test Standard convolutions. Not specifying S or H is the same as specifying both.",
                      action="store_true")
  parser.add_argument("-T",metavar='threads',help="Number of threads. Not specifying, runs T=1 and T=$OMP_NUM_THREADS", default=0, type=int)
  return parser.parse_args()

if __name__ == "__main__":
  main()
