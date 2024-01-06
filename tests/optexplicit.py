#!/usr/bin/env python3

# a timing script for FFTs and convolutions using OpenMP

import re
import os
import argparse
from utils import ceilpow2, usecmd, nextfftsize

def main():
  args=getArgs()
  cmd=["./cconv3","-e","-T1", f"-s{args.s}"]

  if os.getenv("TASKSET") != None:
    cmd=["taskset","-c","0"]+cmd

  mstart=args.a
  mstop=args.b
  Nstart=2*mstart-1
  N=nextfftsize(Nstart)
  efficientNs=[N]
  while N < ceilpow2(2*mstop-1):
    N=nextfftsize(N+1)
    efficientNs.append(N)

  data = []
  for m in range(mstart,mstop+1):
    Ns=[N for N in efficientNs if N >= 2*m-1]
    for N in Ns:
      cmdmN=cmd+[f"-m{m}",f"-N{N}"]
      output=usecmd(cmdmN)
      outlines = output.split('\n')
      itline = len(outlines)-1
      while itline >= 0:
          line = outlines[itline]
          if re.search("Explicit", line) is not None:
              time=""
              timeSearch=re.search(r"((\d|\.)+)$",outlines[itline+1])
              if timeSearch is not None:
                time=timeSearch[0]
              dataline=f"{m}\t{N}\t{time}\n"
              data.append(dataline)
              if args.v or True:
                print(dataline)
              break
          itline -= 1
    print(f"Done m={m}")

  if args.e:
    with open("optexplicit.dat","w") as file:
      cmdstring='\t'.join(cmd)
      file.write(f"# {cmdstring}\n")
      file.write("# m\tN\ttime\n")

  with open("optexplicit.dat","a") as file:
    for d in data:
      file.write(d)

def getArgs():
  parser = argparse.ArgumentParser(description="Perform timing tests on cconv3 sizes.")
  parser.add_argument("-e", help="Erase previously stored data.",
                      action="store_true")
  parser.add_argument("-a",help="Start.", default=65, type=int)
  parser.add_argument("-b",help="End.", default=128, type=int)
  parser.add_argument("-s",help="Number of seconds for each timing.",
    default=5.0, type=float)
  parser.add_argument("-v", help="Verbose output.",
                      action="store_true")
  return parser.parse_args()

if __name__ == "__main__":
  main()
