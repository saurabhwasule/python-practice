# frequency


str = " I am Ok I am OK I am OK"
import re

pat = re.compile(r"\w+")

alpha_d = dict()
word_d = dict()

for ele in str.lower():
		if ele in alpha_d: alpha_d[ele] += 1
		else: alpha_d[ele] = 1

for ele in pat.findall(str.lower()):
		if ele in word_d: word_d[ele] += 1
		else: word_d[ele] = 1

sorted(alpha_d.items(), key = lambda kv: kv[1], reverse=True )
sorted(word_d.items(), key = lambda kv: kv[1] , reverse=True)

#with functions
def wcount(lst):
	return { w: lst.count(w) for w in set(lst) }

wcount("I am Ok ok ok ok".split())	# word frequency
wcount("I am Ok ok ok ok")    		#alphabet frequency
wcount([al.lower() for al in "I am Ok ok ok ok" ]) 			# ignore case
wcount( map( lambda al: al.lower(), "I am Ok ok ok ok" )) 


#!!
import functools 
s = "hello python"
functools.reduce(lambda d,e: (d.update( dict([[e, list(s).count(e)]])), d)[1], s,{})
#or 
functools.reduce(lambda d,e: (d.update({x:d.get(x,0)+1 for x in e}), d)[1], s,{})

# Removing duplicates

L = [1, 2, 2, 1, 2, 1, 2,3]

uniq_l = []

for el in L:
	if el not in uniq_l:
		uniq_l.append(el)


#OR
uniq_l = L[:]

for el in uniq_l:
	while uniq_l.count(el) > 1:
		uniq_l.remove(el)

#OR

def uniq(lst):
	d = { i:0 for i in lst}
	return list(d.keys())



#.... advanced Math

# Average, std, mean, mode, median
import math
def mean(list):
	return sum(list)/len(list) if list else 0
	
	
def bad_sd(list):	
	return math.sqrt(sum([ (i - mean(list)) ** 2 for i in list ]) /len(list)) if list else 0
	

def sd(list):
	m = mean(list)
	return math.sqrt(sum([ (i - m) ** 2 for i in list ]) /len(list)) if list else 0

	
def frq(lst):
	return { w: lst.count(w) for w in set(lst) }
	

def mode(list):
	return  sorted(list, key = lambda e: list.count(e), reverse=True )[0] if list else 0

def mode_g(lst):
	return  sorted(lst, key = lst.count)[-1] if list else 0
	
	
def avg(*list):
	return mean(list)

def median(lst):
	list = sorted(lst)
	return list[len(list)//2] if len(list) % 2 == 1 else avg( list[len(list)//2], list[len(list)//2 + 1])

	
def sd_g(list):
	return math.sqrt(sum( (i - mean(list)) ** 2 for i in list ) /len(list))
	

def mode_m(list):
	fr =  sorted(frq(list).items(), key = lambda kv: kv[1], reverse=True )
	max = fr[0][1]
	return mean([ e[0] for e in fr if e[1] == max] )

	
	
# adding to List

class MyList(list):
	def mean(self):
		return sum(self)/len(self) if self else 0
	def __add__(self,oth):		
			tmp = super().__add__(oth)
			return MyList(tmp)


		
l = MyList([1,2,3,4])
l.mean()
l + [1,2,3]  
(l + [1,2,3]).mean()

#Binary Search

def bs(lst, item): 
	alist = sorted(lst)
	first = 0 
	last = len(alist)-1 
	found = False 
	index = None	
	while first <= last and not found: 
		midpoint = (first + last)//2 
		if alist[midpoint] == item: 
			found = True 
			index = midpoint
		else: 
			if item < alist[midpoint]: 
				last = midpoint - 1 
			else: 
				first = midpoint + 1 
	return (found, index)
	
#recursive bs

def _binary_search(value, items, low=0, high=None):  #items are sorted		
    high = len(items) if not high  else high
    pos = (high + low) // 2
    #print(low, pos, high)
    if items[pos] == value:
        return pos
    elif pos == len(items) or high == low or pos == low:
        return False
    elif items[pos] < value:
        return _binary_search(value, items, pos + 1, high)
    else:		
        return _binary_search(value, items, low, pos)
	


def binary_search(value, items):
    def _binary_search(value, items, low=0, high=None):  #items are sorted		
        high = len(items) if not high  else high
        pos = (high + low) // 2
        #print(low, pos, high)
        if items[pos] == value:
            return pos
        elif pos == len(items) or high == low or pos == low:
            return False
        elif items[pos] < value:
            return _binary_search(value, items, pos + 1, high)
        else:		
            return _binary_search(value, items, low, pos)
    pd = dict([(e,i)  for i,e in enumerate(items)])
    lst = sorted(items)
    pos = _binary_search(value, lst)
    return pos if not pos else pd[lst[pos]]

for val in range(1,7):
    print(val, binary_search(val, [1, 2, 3, 5]))

	
#Flatten a list

def flatten(lst):
	res = []
	for ele in lst:
		if type(ele) is list:
			res += flatten(ele)
		else:
			res.append(ele)
	return res
	
	
	
l = [1,2,3,[1,2,3,[1,2,3]]]
flatten(l)

#flatten a Dict does not exists, because what would be values

def flatten(d):
	res = {}
	for k in d:
		if type(d[k]) is dict:
			res.update(flatten(d[k]))
		else:
			res[k] = d[k]
	return res


# Union, intersection, difference





def process(in1, in2):
	u_dict = {}
	i_dict ={}
	d_dict = {}
	x_dict = {}
	for k1 in in1:
		u_dict[k1] = in1[k1]
		if k1 in in2:
			i_dict[k1] = in1[k1]
		else:
			d_dict[k1] = in1[k1]
			x_dict[k1] = in1[k1]
	for k2 in in2:
		if k2 not in in1:
			u_dict[k2] = in2[k2]
			x_dict[k2] = in2[k2]
	return (u_dict, i_dict, d_dict, x_dict)
			


>>> in1 = { 'ok' : 1, 'nok': 2}
>>> in2 = { 'nok' : 2, 'new' : 3}
>> process(in1,in2)








			
			
def process(in1, in2):
	u_dict = { k1 : in1[k1] for k1 in in1 }
	i_dict = { k1 : in1[k1] for k1 in in1 if k1 in in2 }
	d_dict = { k1 : in1[k1] for k1 in in1 if k1 not in in2 }
	x_dict = { k1 : in1[k1] for k1 in in1 if k1 not in in2 }	
	u_dict.update( { k2: in2[k2] for k2 in in2  if k2 not in in1 } )
	x_dict.update( { k2: in2[k2] for k2 in in2  if k2 not in in1 } )	
	return (u_dict, i_dict, d_dict, x_dict)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
#merge
in1 = { 'ok' : 1, 'nok': 2}
in2 = { 'nok' : 2, 'new' : 3}

m_dict = {}    
for k1 in in1:
	m_dict[k1] = in1[k1]			   #Initial values
	if k1 in in2:
		m_dict[k1] = in1[k1] + in2[k1] # merge		
for k2 in in2:							# remaining elements
	if k2 not in in1:
		m_dict[k2] = in2[k2]

		
print(m_dict)
	
	
	
	
	
	
	
	
	
	
	
	
	
# merge 

def merge(in1, in2, f):
	m_dict = {}    
	for k1 in in1:
		m_dict[k1] = in1[k1]			   #Initial values
		if k1 in in2:
			m_dict[k1] = f(in1[k1] , in2[k1] )# merge
	for k2 in in2:							# Pending elements
		if k2 not in in1:
			m_dict[k2] = in2[k2]
	return m_dict

	
	
>>> in1 = { 'ok' : 1, 'nok': 2}
>>> in2 = { 'nok' : 2, 'new' : 3}
	
	
>>> merge(in1, in2)
{'new': 3, 'nok': 4, 'ok': 1}
	
	
	
	
	
	
	
	
	
	
	
	

	

def merge( in1, in2, f = lambda x,y: x+y):
	m_dict = {}    
	for k1 in in1:
		m_dict[k1] = in1[k1]      #Initial values
		if k1 in in2:
			m_dict[k1] = f(in1[k1],in2[k1]) 
	for k2 in in2:					# Pending elements
		if k2 not in in1:
			m_dict[k2] = in2[k2]
	return m_dict


in1 = { 'ok' : 1, 'nok': 2}
in2 = { 'nok' : 2, 'new' : 3}



>>> merge( in1, in2, lambda x, y: x+y,)
{'new': 3, 'nok': 4, 'ok': 1}







import operator
>>> merge( in1, in2, operator.add)
{'new': 3, 'nok': 4, 'ok': 1}



def merge( in1, in2, f):
	m_dict = { k1: f(in1[k1],in2[k1]) for k1 in in1 if k1 in in2 }
	m2 = { k1: in1[k1]  for k1 in in1  if k1  not in in2 }
	m3 = { k2: in2[k2] for k2 in in2 if k2 not in in1 }
	m_dict.update(m2, **m3)
	return m_dict
	
	
#with reduce 


def merge(in1 , in2, f = lambda x,y : x+y):
	import functools
	def mg(r, e):
		if e[0] in r:
			r[e[0]] += e[1]
		else:
			r[e[0]] = e[1]
		return r
	return functools.reduce(mg , in2.items(), in1)
			

	


# Recursive get size

import functools
import os.path
import glob

def printsize(root, acc={} ):
	lst = [os.path.normpath(f) for f in glob.glob(os.path.join(root, "*"))]
	acc[root] = sum( os.path.getsize(f) for f in lst if os.path.isfile(f) )
	[printsize(f, acc) for f in lst if os.path.isdir(f)]
	return acc



d = printsize(".")
maxnames = sorted(d.keys(), key = lambda n: d[n], reverse=True )
print([maxnames[0], d[maxnames[0]]])

#another version
def createdirlist(root, acc={} ):
	import os.path
	import glob
	lst = [os.path.normpath(f) for f in glob.glob(os.path.join(root, "*"))]
	acc.update( { f:os.path.getsize(f) for f in lst if os.path.isfile(f) } )
	[createdirlist(f, acc) for f in lst if os.path.isdir(f)]	
	return acc



#Iterator way to return each file starting with root

def ls(root):
	import os.path
	import glob
	lst = [os.path.normpath(f) for f in glob.glob(os.path.join(root, "*"))]
	yield from [f for f in lst if os.path.isfile(f)]
	for dir in lst:
		if os.path.isdir(dir):			
			yield from ls(dir)  
	
	
#version
def ls(root):
	import os.path
	import glob
	lst = [os.path.normpath(f) for f in glob.glob(os.path.join(root, "*"))]
	yield from [f for f in lst if os.path.isfile(f)]
	dirs = [ls(f) for f in lst if os.path.isdir(f) ] #generators
	for d in dirs:     #iterate each generator
		yield from d
	

#with os.walk


def ls(dir):
	import os
	from os.path import join
	for root, dirs, files in os.walk(dir):
		yield from [join(root, curfile)  for curfile in files]

#traverseAndExecute

def traverseAndExecute(dir, code = lambda d : print(d)):
	lst = list(map(lambda f : os.path.normpath(f), glob.glob(os.path.join(dir, "*"))))
	for f in lst:
		if os.path.isdir(f):
			traverseAndExecute(f, code)
		else:
			code(f)

			
traverseAndExecute(".")

#Example to build file tree
import os.path
import glob
dirs = {}

def create(file):
	dir = os.path.dirname(file)
	f = os.path.basename(file)	
	if os.path.isfile(file):
		if dir not in dirs:
			dirs[dir] = [0, 0, { }]
		dirs[dir][0] += 1                        #count of file
		dirs[dir][1] += os.path.getsize(file)    #total size
		dirs[dir][2][f] = os.path.getsize(file)   #filename and it's size
	


traverseAndExecute(".", create)


# OS Walk Examples

import os
from os.path import join, getsize
for root, dirs, files in os.walk(r'.'):     # root= root dir , dirs = subdirs list, files=files list
	print(root, " ", sum(getsize(join(root, name)) for name in files), end= ' ')
	print("bytes in", len(files), "files")
	if '__pycache__' in dirs:
		dirs.remove('__pycache__')  # don't visit __pycache__ directories


		



		
#Optparse Examples

from optparse import OptionParser
p = OptionParser()
p.add_option("-f", "--file", action="store", type="string", dest="filename")
p.add_option("-d", "--dir", action="store", type="string", dest="dir")

(o,r) = p.parse_args()
print(o.filename)
print(o.dir)


#Intel Hex format parsing
s = ":10010000214601360121470136007EFE09D2190140"
#   ":10010000dddddddddddddddddddddddddddddddd40"
import re
p = re.compile("\w\w")
start = ":10010000"
data = [ int(x, 16) for x in p.findall(s[len(start) : len(s) -2]) ]
print("%02x" % (sum(data) % 256, ))
#checking
ints = [ int(x, 16) for x in p.findall(s[1:]) ]
print("%02x" % (sum(data) % 256, ))

#using struct.unpack
s = ":10010000214601360121470136007EFE09D2190140"
import struct
import binascii
bs= binascii.unhexlify(s[1:])
data = struct.unpack(str(len(bs)) + 'B', bs)
print("%02x" % (sum(data) % 256, ))


#Overlapping match

import re 
s = "123456789123456789"
matches = re.finditer(r'(?=(\d{10}))',s)
results = [int(match.group(1)) for match in matches]

#OR
re.findall(r'(?=(\d{10}))',s)


#Module -subprocess


>>> subprocess.check_output(["echo", "Hello World!"])
b'Hello World!\n'
>>> subprocess.check_output(["echo", "Hello World!"]).decode("utf-8")
'Hello World!\n'

#cat regex.py | grep def

p1 = Popen(["cat", "regex.py"], stdout=PIPE)
p2 = Popen(["grep", "def"], stdin=p1.stdout, stdout=PIPE)
p1.stdout.close()
output_str = p2.communicate()[0]  # or p2.stdout.read().decode("utf-8")
print(output_str.decode("utf-8"))
