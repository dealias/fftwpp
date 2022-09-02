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

	tests=[0,0] # [total,failed]
	for p in programs:
		iterate(p,int(args.L),args.T,tests)
		print("Done "+p.name+".\n")

	total=tests[0]
	failed=tests[1]
	passed=total-failed
	print(str(total)+" tests done: "+str(passed)+" passed, "+str(failed)+" failed.")

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
	parser.add_argument("-T", help="Time over threads.", action="store_true")
	parser.add_argument("-L", help="L value.", default=8)
	return parser.parse_args()

def check(prog, L, M, m, D, I, T,tol,tests):
	tests[0]+=1
	cmd = [prog,"-L"+str(L),"-M"+str(M),"-m"+str(m),"-D"+str(D),"-I"+str(I),\
					"-T"+str(T),"-E"]

	vp = Popen(cmd, stdout = PIPE, stderr = PIPE)
	vp.wait()
	prc = vp.returncode

	if prc == 0:
		out, err = vp.communicate()
		comment = out.rstrip().decode()
	try:
		error=float(re.search(r"(?<=Error: )(\d|\.|e|-)+",comment).group())
		if error > tol:
			tests[1]+=1
			print("Error too high:")
			print(" ".join(cmd))
			print("Error: "+str(error))
			print()
	except:
		tests[1]+=1
		print("Error not found:")
		print(" ".join(cmd))
		print()

def iterate(program, L, timeThreads,tests):
	name=program.name
	centered=program.centered

	Dstart=2 if centered else 1
	threads=[1,2,4] if timeThreads else [1]

	M0=3*L//2 if centered else 2*L-1
	M02=M0//2
	for T in threads:
		for M in [M0,3*M02,2*M0,5*M02]:
			for m in [L//4,L//2,L,L+2,M,M+2]:
				p=int(ceil(L/m))
				q=int(ceil(M/m)) if p <= 2 else int(ceil(M/(p*m))*p)
				n=q//p
				Istart=0 if q > 1 else 1
				for I in range(Istart,2):
					D=Dstart
					while(D < n):
						check(name,L,M,m,D,I,T,1e-8,tests)
						D*=2
					check(name,L,M,m,n,I,T,1e-8,tests)

if __name__ == "__main__":
	main()
