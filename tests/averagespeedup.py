#!/usr/bin/python
import csv
import numpy as np


S=[]
basedirs=["timings1c", "timings1r", "timings1t", "timings2c", "timings2r", "timings2t", "timings3c"]
dim=[1.0,1.0,1.0,2.0,2.0,2.0,3.0]
j=0
for d in basedirs:
    filename=d+"/implicit"
    a = [] 
    csvReader = csv.reader(open(filename, 'rb'), delimiter=' ')
    for row in csvReader:
        a.append(row)

    Ci=0.0

    i=0
    while(i < len(a)):
        t=float(a[i][0])
        Ci += float(a[i][1])/(np.power(t,dim[j])*np.log(t))
        i += 1
    Ci /= float(len(a))

    filename=d+"/explicit"
    a = [] 
    csvReader = csv.reader(open(filename, 'rb'), delimiter=' ')
    for row in csvReader:
        a.append(row)
    Ce=0.0

    i=0
    while(i < len(a)):
        t=float(a[i][0])
        Ce += float(a[i][1])/(np.power(t,dim[j])*np.log(t))
        i += 1
    Ce /= float(len(a))

    S.append(Ce/Ci)
    j += 1

print "implicit cconv  speedup:", S[0]
print "implicit  conv  speedup:", S[1]
print "implicit tconv  speedup:", S[2]
print "implicit cconv2 speedup:", S[3]
print "implicit  conv2 speedup:", S[4]
print "implicit tconv2 speedup:", S[5]
print "implicit cconv3 speedup:", S[6]


