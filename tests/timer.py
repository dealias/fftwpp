#!/usr/bin/env python3

import sys, getopt
import argparse
from subprocess import *
import re

def main():
  args=getArgs()
  threads=args.T
  dimensions=[]
  S=args.S
  H=args.H

  one=args.one
  two=args.two
  three=args.three

  not123=not (one or two or three)

  if one or not123:
    dimensions.append(1)
  if two or not123:
    dimensions.append(2)
  if three or not123:
    dimensions.append(3)

  if threads == 0:
    cmd = 'echo $OMP_NUM_THREADS'
    OMP_NUM_THREADS=str(check_output(cmd, shell=True))
    T=int(re.search(r"\d+",OMP_NUM_THREADS).group(0))
    Ts=[1,T]
  else:
    Ts=[threads]

  notSorH = not (S or H)
  for d in dimensions:
    for T in Ts:
      if S or notSorH:
        time(args,d,False,T)
      if H or notSorH:
        time(args,d,True,T)

def time(args,d,Hermitian, T):
  a=args.a
  b=args.b
  I=args.I
  e=args.e
  dim=str(d) if d > 1 else ""

  new="hybridconv"
  old="conv"

  if Hermitian:
    new += "h"
  else:
    old = "c"+old

  new+=dim
  old+=dim

  cmd1=["timing.py"]
  if not args.t:
    taskset="-Btaskset -c 0"
    if T > 1:
      taskset+="-15"
    cmd1.append(taskset)

  cmd2=[f"-a{a}",f"-b{b}",f"-I{I}",f"-T{T}"]
  erase=[]
  if e:
    erase.append("-e")

  run(cmd1+[f"-p{old}"]+cmd2+["-rimplicit"]+erase)
  run(cmd1+[f"-p{old}"]+cmd2+["-rexplicit"]+erase)
  run(cmd1+[f"-p{new}"]+cmd2+erase)
  if Hermitian and I == 0:
    run(cmd1+[f"-p{new}"]+cmd2+["-rexplicit"])


def getArgs():
  parser = argparse.ArgumentParser(description="Call timing.py with correct parameters.")
  parser.add_argument("-e", help="Erase old data.", action="store_true")
  parser.add_argument("-S", help="Test Standard convolutions. Not specifying S or H is the same as specifying both.",
                      action="store_true")

  parser.add_argument("-H", help="Test Hermitian convolutions. Not specifying S or H is the same as specifying both.",
                      action="store_true")
  parser.add_argument("-T",metavar='threads',help="Number of threads. Not specifying, runs T=1 and T=$OMP_NUM_THREADS", default=0, type=int)

  parser.add_argument("-1","--one", help="Time 1D Convolutions. Not specifying\
                      1 or 2 or 3 is the same as specifying all of them",
                      action="store_true")
  parser.add_argument("-2","--two", help="Time 2D Convolutions. Not specifying\
                      1 or 2 or 3 is the same as specifying all of them",
                      action="store_true")
  parser.add_argument("-3","--three", help="Time 3D Convolutions. Not specifying\
                      1 or 2 or 3 is the same as specifying all of them",
                      action="store_true")
  parser.add_argument("-a",help="Start.",
                      default=1, type=int)
  parser.add_argument("-b",help="End.",
                      default=1, type=int)
  parser.add_argument("-I",help="Interval. Checks powers of 2 when 0. Default is 0.",
                      default=0, type=int)
  parser.add_argument("-t", help="Don't use 'taskset'.",
                      action="store_true")
  return parser.parse_args()

if __name__ == "__main__":
  main()
