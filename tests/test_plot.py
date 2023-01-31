import sys, getopt
import numpy as np
from math import *
import os
import subprocess
import shlex
import regex as re

d = 0
thread = [[],[],[],[],[],[],[],[]]
t_array = [1,8]
for t in t_array:
	for i in [j for j in range(128, 135)][:]:
		string = "./hybridconv2 -Lx=512 -Ly=" + str(i) + " -Mx=1024 -My=" + str(2*i) + " -t -R -T=" + str(t)
		cmd = subprocess.run(shlex.split(string), capture_output=True, text=True)
		string = cmd.stdout

		pattern = "Optimal padding: "
		matches = re.finditer(pattern, string)
		c = 0
		lines_with_optimal_padding = []
		for match in matches:
			# Get the starting index of the match

			start = match.start()
			if int(start) < 2900:
				# Get the line number by counting the number of newline characters before the match

				line_num = string[:start].count("\n") + 1
				lines_with_optimal_padding.append(line_num)
				line_start = string.rfind("\n", 0, start) + 1
				line_end = string.find("threads=", start)
				line = string[line_start:line_end]
				print("Thread ", t, ": ", end=" ")
				print("Ly=", i, ": ", end=" ")
				print(line)

				if len(line) >=3 and len(line) <= 10:
					if c == 1:
						thread[d].append(float(line[2:]))
				c += 1
	d += 1
	print("-------------------------------------------------------")

#for i in range(len(thread)):
#	print("Thread ", t_array[i], ": ", end=" ")
#	print(thread[i])


# x-axis Ly
#for each Ly 2d plot
# y-axis m
# z-axis time