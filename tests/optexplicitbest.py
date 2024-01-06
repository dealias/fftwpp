#!/usr/bin/env python3

# a timing script for FFTs and convolutions using OpenMP

import re
import os
import argparse
from utils import ceilpow2, usecmd, nextfftsize

def main():
  out=""
  with open("optexplicit.dat","r") as file:
    out=file.read()
  lines=out.split('\n')[:-1]
  best=[[eval(i) for i in lines[2].split('\t')]]

  for k in range(3,len(lines)):
    line=lines[k]
    try:
      m,N,t=[eval(i) for i in line.split('\t')]
      if best[-1][0] < m:
        best.append([m,N,t])
      elif t < best[-1][2]:
        best[-1][1]=N
        best[-1][2]=t
    except:
      pass
  try:
    os.makedirs("timings3-T1I1")
  except:
      pass

  with open("timings3-T1I1/explicitbest","w") as file:
    file.write(f"{lines[0]}\n")
    for b in best:
      file.write(f"{b[0]**3}\t{b[2]}\n")

def getArgs():
  parser = argparse.ArgumentParser(description="Find optimal cconv3 FFT sizes.")
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
