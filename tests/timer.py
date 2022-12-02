#!/usr/bin/env python3

import sys, getopt
import argparse
from subprocess import *

def main():
  args=getArgs()
  a=args.a
  b=args.b
  I=args.I
  T=args.T
  Hermitian=args.H
  e=args.e
  d=str(args.d) if args.d > 1 else ""

  new="hybridconv"
  old="conv"

  if Hermitian:
    new += "h"
  else:
    old = "c"+old

  new+=d
  old+=d

  cmd1=["timing.py"]
  if not args.t:
    taskset="-Btaskset -c 0"
    if T > 1:
      taskset+="-15"
    cmd1.append(taskset)

  cmd2=[f"-a{a}",f"-b{b}",f"-I{I}"]
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
  parser.add_argument("-H", help="Test Hermitian convolutions. Not specifying H tests standard convolutions",
                      action="store_true")
  parser.add_argument("-d",metavar='dimension',help="Dimension. Default is 1.",
                      default=1, type=int)
  parser.add_argument("-a",help="Start.",
                      default=1, type=int)
  parser.add_argument("-b",help="End.",
                      default=1, type=int)
  parser.add_argument("-T",metavar='threads',help="Number of threads.",
                      default=1, type=int)
  parser.add_argument("-I",help="Interval. Checks powers of 2 when 0. Default is 0.",
                      default=0, type=int)
  parser.add_argument("-t", help="Don't use 'taskset'.",
                      action="store_true")
  return parser.parse_args()

if __name__ == "__main__":
  main()
