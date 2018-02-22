#calculates LOC of a dirs recursively 
import sys
import glob
import os.path 



def loc(filename):
    with open(filename, "r") as fp:
        return len([ line for line in fp.readlines() 
                              if not line.strip().startswith('#')])
       

def locdir(dir):
    fsum = sum([ loc(f) for f in glob.glob(dir + "/*.py") if os.path.isfile(f) ])
    dsum = sum([ locdir(f) for f in glob.glob(dir + "/*") if os.path.isdir(f) ])
    retunr fsum + dsum
         
print(locdir("."))