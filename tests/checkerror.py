#!/usr/bin/python
import csv
import numpy as np

C=[]
d=0
dirs=["timings1c/error.implicit", "timings1c/error.explicit" , "timings1r/error.implicit", "timings1r/error.explicit"]
for filename in dirs:
    a = []
    csvReader = csv.reader(open(filename, 'rb'), delimiter='\t')
    for row in csvReader:
        a.append(row)
    
    C.append(0.0)
    i=0
    while(i < len(a)):
        C[d] += float(a[i][1])/np.sqrt(np.log(float(a[i][0])))
        i += 1
    C[d] /= float(len(a))
    d += 1
print "implicit cconv error epsilon:", C[0]
print "explicit cconv error epsilon:", C[1]
print "implicit  conv error epsilon:", C[2]
print "explicit  conv error epsilon:", C[3]
