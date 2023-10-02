#!/usr/bin/env python3

import subprocess
import argparse
import re
#import sys

def main():
  args=getArgs()
  a=int(args.a)
  b=int(args.b)

  Ls=[2**n for n in range(a,b)]

  scale=1e16
  data=open("accuracy.dat", "w")
  data.write(" ".join(str(L) for L in Ls)+"\n")
  divisors=[2**(a-1-n) for n in range(0,a)]
  data.write(" ".join(str(1/d) for d in divisors))
  data.write(" 2\n")
  for L in Ls:
    for d in divisors:
      m=max(L//d,2)
      err=findError(m,L)
      data.write(f"{scale*err} ")
    m=2*L
    err=findError(m,L)
    data.write(f"{scale*err} \n")
  data.close()


def findError(m,L,T=1,D=1,I=1):
  cmd=["hybridconv",f"-L{L}",f"-M{2*L}",f"-m{m}",f"-T{T}",f"-I{I}",f"-D{D}","-a"]
  vp = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  vp.wait()
  prc = vp.returncode
  output = ""

  if prc == 0:
    out, _ = vp.communicate()
    output = out.rstrip().decode()

  errorSearch=re.search(r"(?<=error: )(\w|\d|\.|e|-|\+)*",output)
  if errorSearch is not None:
    return float(errorSearch.group())
  else:
    print("Error not found.")
    return 0.0

def getArgs():
  parser = argparse.ArgumentParser(description="Test accuracy of 1D complex hybrid dealiased convolutions. Checks powers of 2.")
  parser.add_argument("-a", help="Start (exponent on 2). Default is 6.",default=6)
  parser.add_argument("-b", help="Stop (exponent on 2). Default is 12.",default=12)
  return parser.parse_args()



if __name__ == "__main__":
    main()
