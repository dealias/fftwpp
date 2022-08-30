#!/usr/bin/env python3

from math import *
import os
from subprocess import *
import re # regexp package
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-S", help="Test Standard convolutions. Specifying neither\
											S nor H is the same as specifying both of them",
											action="store_true")
parser.add_argument("-H", help="Test Hermitian convolutions. Specifying neither\
											S nor H is the same as specifying both of them",
											action="store_true")
parser.add_argument("-one", help="Test 1D Convolutions. Not specifying\
											one nor two, nor three is the same as specifying all of them",
											action="store_true")
parser.add_argument("-two", help="Test 2D Convolutions. Not specifying\
											one nor two, nor three is the same as specifying all of them",
											action="store_true")
parser.add_argument("-three", help="Test 3D Convolutions. Not specifying\
											one nor two, nor three is the same as specifying all of them",
											action="store_true")
args=parser.parse_args()

class prog:
	def __init__(self, name, centered):
		self.name=name
		self.centered=centered

def check(prog, L, M, m, D, I, K):
	cmd = [prog,"-L"+str(L),"-M"+str(M),"-m"+str(m),"-D"+str(D),"-I"+str(I),"-K"+str(K)]
	vp = Popen(cmd, stdout = PIPE, stderr = PIPE)
	vp.wait()
	prc = vp.returncode

	comment=""
	if prc == 0:
		out, err = vp.communicate()
		comment += out.rstrip().decode()

def iterate(program, L=8, K=1):
	name=program.name
	centered=program.centered
	Dstart=2 if centered else 1
	for M in [12,16,24,32]:
		for m in range(2,M+2,2):
			p=int(ceil(L/m))
			q=int(ceil(M/m))
			n=q//p
			for I in range(2):
				D=Dstart
				while(D < n):
					check(name,L,M,m,D,I,K)
					D*=2
				check(name,L,M,m,n,I,K)

programs=[]

SH=not (args.S or args.H)

oneTwoThee=not (args.one or args.two or args.three)

if args.one or oneTwoThee:
	if args.S or SH:
		programs.append(prog("hybridconv",False))
	if args.H or SH:
		programs.append(prog("hybridconvh",True))

if args.two or oneTwoThee:
	if args.S or SH:
		programs.append(prog("hybridconv2",False))
	if args.H or SH:
		programs.append(prog("hybridconvh2",True))

if args.three or oneTwoThee:
	if args.S or SH:
		programs.append(prog("hybridconv3",False))
	if args.H or SH:
		programs.append(prog("hybridconvh3",True))

for p in programs:
	iterate(p)
	print("Done "+p.name+".\n")
