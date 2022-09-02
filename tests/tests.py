#!/usr/bin/env python3

from math import *
import os
from subprocess import *
import re # regexp package
import argparse

def main():
	args=getargs()

	SCH=(not (args.S or args.C or args.H))
	oneTwoThee=(not (args.one or args.two or args.three))

	SorSCH=(args.S or SCH)
	CorSCH=(args.C or SCH)
	HorSCH=(args.H or SCH)
	programs=[]

	if args.one or oneTwoThee:
		if SorSCH:
			programs.append(Program("hybridconv",False))
		if CorSCH:
			programs.append(Program("hybridconvc",True))
		if HorSCH:
			programs.append(Program("hybridconvh",True))
'''
	if args.two or oneTwoThee:
		if SorSCH:
			programs.append(Program("hybridconv2",False))
		if HorSCH:
			programs.append(Program("hybridconvh2",True))

	if args.three or oneTwoThee:
		if SorSCH:
			programs.append(Program("hybridconv3",False))
		if HorSCH:
			programs.append(Program("hybridconvh3",True))
'''
	for p in programs:
		iterate(p)
		print("Done "+p.name+".\n")


class Program:
	def __init__(self, name, centered):
		self.name=name
		self.centered=centered

def getargs():
	parser = argparse.ArgumentParser()
	parser.add_argument("-S", help="Test Standard convolutions. Not specifying\
												S nor C nor H is the same as specifying all of them",
												action="store_true")
	parser.add_argument("-C", help="Test Centered convolutions. Not specifying\
												S nor C nor H is the same as specifying all of them",
												action="store_true")
	parser.add_argument("-H", help="Test Hermitian convolutions. Not specifying\
												S nor C nor H is the same as specifying all of them",
												action="store_true")
	parser.add_argument("-one", help="Test 1D Convolutions. Not specifying\
												one nor two nor three is the same as specifying all of them",
												action="store_true")
	parser.add_argument("-two", help="Test 2D Convolutions. Not specifying\
												one nor two nor three is the same as specifying all of them",
												action="store_true")
	parser.add_argument("-three", help="Test 3D Convolutions. Not specifying\
												one nor two nor three is the same as specifying all of them",
												action="store_true")
	return parser.parse_args()

def check(prog, L, M, m, D, I, tol=1e-8):
	cmd = [prog,"-L"+str(L),"-M"+str(M),"-m"+str(m),"-D"+str(D),"-I"+str(I),"-E"]

	vp = Popen(cmd, stdout = PIPE, stderr = PIPE)
	vp.wait()
	prc = vp.returncode

	if prc == 0:
		out, err = vp.communicate()
		comment = out.rstrip().decode()
	try:
		error=float(re.search(r"(?<=Error: )(\d|\.|e|-)+",comment).group())
		if error > tol:
			print("Error too high:")
			print(" ".join(cmd))
			print("Error: "+str(error))
			print()
	except:
		print("Error not found:")
		print(" ".join(cmd))
		print()


def iterate(program, L=8):
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
					check(name,L,M,m,D,I)
					D*=2
				check(name,L,M,m,n,I)

if __name__ == "__main__":
	main()
