#!/usr/bin/env python3

import sys, getopt
import argparse
from subprocess import *

def main():
  args=getArgs()
  new="hybridconv"
  old="conv"
  Hermitian=args.H
  if Hermitian:
    new += "h"
  else:
    old = "c"+old
  d=str(args.d) if args.d > 1 else ""
  new+=d
  old+=d
  a=args.a
  b=args.b
  I=args.I

  cmd1=["timing.py", "-Btaskset -c 0-15"]
  cmd2=[f"-a{a}",f"-b{b}",f"-I{I}"]
  if args.e:
    cmd2.append("-e")
  run(cmd1+[f"-p{old}"]+cmd2+["-rimplicit"])
  run(cmd1+[f"-p{old}"]+cmd2+["-rexplicit"])
  run(cmd1+[f"-p{new}"]+cmd2)
  if Hermitian and I == 0:
    run(cmd1+[f"-p{new}"]+cmd2+["-rexplicit"])

def getArgs():
  parser = argparse.ArgumentParser(description="Call timing.py with correct parameters.")
  parser.add_argument("-e", help="Erase old data.",
                      action="store_true")
  parser.add_argument("-H", help="Test Hermitian convolutions. Not specifying H tests standard convolutions",
                      action="store_true")
  parser.add_argument("-d",metavar='dimension',help="Dimension. Default is 1.",
                      default=1)
  parser.add_argument("-a",help="Start.",
                      default=1)
  parser.add_argument("-b",help="End.",
                      default=1)
  parser.add_argument("-T",metavar='threads',help="Number of threads.",
                      default=1)
  parser.add_argument("-I",help="Interval. Checks powers of 2 when 0. Default is 0.",
                      default=0)
  return parser.parse_args()

if __name__ == "__main__":
  main()
