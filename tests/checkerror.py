#!/usr/bin/python

# Script for reading the results of the directerror script and
# determining the scaling of the numerical error as a function of
# problem size (assuming that the fft's error is dominant and of order
# of sqrt of the problem size).

import csv
import numpy
import sys

import os.path #for file-existence checking

print "Error scaling for convolutions"
print
print "program\ttype\t\tconstant coefficient:"
print


dirs=["timings1c/error.implicit", "timings1c/error.explicit" , "timings1r/error.implicit", "timings1r/error.explicit"]

C=[]
d=0

retval=0

for filename in dirs:
    if (os.path.isfile(filename)):
        a = []
        csvReader = csv.reader(open(filename, 'rb'), delimiter='\t')
        for row in csvReader:
            a.append(row)
    
        C.append(0.0)
        i=0
        while(i < len(a)):
            C[d] += float(a[i][1])/numpy.sqrt(numpy.log(float(a[i][0])))
            i += 1
        C[d] /= float(len(a))
        d += 1
    else:
        print(filename+" does not exist.")
        retval+=1


print("cconv\timplicit\t"+str(C[0]))
print("cconv\texplicit\t"+str(C[1]))
print("conv\timplicit\t"+str(C[2]))
print("conv\texplicit\t"+str(C[3]))

while(i < len(C)):
    if(float(C[i]) > 1e-10):
        retval += 1
    i += 1

print

if(retval == 0):
    print "OK:\tAll tests passed."
else:
    print "ERROR:\t AT LEAST ONE TEST FAILED"

sys.exit(retval)
