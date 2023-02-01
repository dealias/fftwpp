#!/usr/bin/env python3

import sys, getopt
import numpy as np
from math import *
import os
import subprocess
import shlex
import regex as re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

%matplotlib widget


def match_pattern(pattern, line):
	m = []
	matches = re.findall(pattern, line)
	for match in matches:
		m.append(float(match[2:]))
	return m

def plot(x,y,z,thread):
	# Create the figure and 3D axis
	fig = plt.figure(figsize = (10,10))
	ax = fig.add_subplot(111, projection='3d')

	# Plot the data
	colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w'][:len(x)]
	for i in range(len(x)):
		ax.scatter(x[i], y[i], z[i], c=colors[i%8], marker='o')
	
	#cb = plt.colorbar(ax1, pad=0.2)

	# Set axis labels
	ax.set_title('Time taken vs Ly and m values')
	ax.set_xticks(x)
	ax.set_xlabel('Ly values')
	ax.set_ylabel('m values')
	ax.set_zlabel('time taken (s)')

	# Show the plot
	plt.show()

thread = [[],[],[],[],[],[],[],[]]
ly_values = [j for j in range(128, 164)]
final_m_values = []
final_time_values = []
t_array = [1,8]

for thread in t_array:
	for i in ly_values:
		string = "./hybridconv2 -Lx=512 -Ly=" + str(i) + " -Mx=1024 -My=" + str(2*i) + " -t -R -T=" + str(thread)
		cmd = subprocess.run(shlex.split(string), capture_output=True, text=True)
		string = cmd.stdout

		pattern = "m="
		matches = re.finditer(pattern, string)
		for match in matches:
			# Get the starting index of the match

			start = match.start()
			if int(start) < 100:
				# Get the line number by counting the number of newline characters before the match

				line_num = string[:start].count("\n") + 1
				line_start = string.rfind("\n", 0, start) + 1
				line_end = string.find("Optimal time: ", start)
				line = string[line_start:line_end]
				print("Thread ", thread, ": ", end=" ")
				print("Ly=", i, end="\n")

				# Extract the values as a float and append to a list
				m_values = match_pattern("m=\d+", line)
				time_values = match_pattern("t=\d+\.\d+", line)
		final_m_values.append(m_values)
		final_time_values.append(time_values)
				
	print("-------------------------------------------------------")
	plot(ly_values,final_m_values,final_time_values,thread)