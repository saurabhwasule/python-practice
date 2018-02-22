#P3.3
def mysum1(L):
	first, *rest = L
	return first if not rest else first + mysum1(rest)


#P2.7
def mysum1(L):
	first, rest = L[0], L[1:]
	return first if not rest else first + mysum1(rest)
	
	
def s(lst):
    return 0 if not lst else lst[0] + s(lst[1:])
	
def mysum(l):
	f, *rest = l + ([0] if not l else [])
	return f if not rest else f+sum(rest)

>>> mysum([1,2,3])
6
>>> mysum([])
0

import sys
sys.getrecursionlimit()


def mysum3(L, csum):
	while True:                     
		if not L: return csum
		L, csum = L[1:], csum + L[0]   

def fact2(n):
	if not n: return 1
	return n*fact2(n-1)


def fact3(n, fact):
	while True:                     
		if not n: return fact
		n, fact = n-1, fact*n  
		
		
#fibs
def fibs(n, s = [0,1]):
  return s if n == 0 else fibs(n-1, s + [ s[-2] + s[-1] ] )

#TCO way
def fibs(n, s = [0,1]):
	while True:
		if not n : return s
		n, s = n-1, s + [ s[-2] + s[-1] ]


#mymap
def mymap(f, lst):
	return [ ] if not lst else [ f(lst[0]) ] + mymap (f, lst[1:])
	
#TCO way

def mymap(f, lst, acc = []):
	while True:
		if not lst : return acc
		lst, acc = lst[1:], acc + [ f(lst[0]) ] 

#flatten

def flatten(array):
	res = []
	for ele in array:
		if type(ele) is list:
			res += flatten(ele)
		else:
			res.append(ele)
	return res
		
		
flatten([1,2,3, [1,2,3,[3,4],2]])


#one example 

l = [[['(0,1,2)','(3,4,5)'],['(5,6,7)','(9,4,2)']],[['(0,1,2)','(3,4,5)'],['(5,6,7)','(9,4,2)']]]
o = [[[[0,1,2],[3,4,5]],[[5,6,7],[9,4,2]]],[[[0,1,2],[3,4,5]],[[5,6,7],[9,4,2]]]

o = [ [ [ [int(i) for i in re.split(r"\(|\)|,", s)[1:-1] ] for s in s1] for s1 in s2 ] for s2 in l ]

def convert(x):
    if isinstance(x, list):
        return [convert(y) for y in x]
    else:
        return [int(y) for y in x.strip('()').split(',')]
		
o = convert(l) 

#Quicksort

#quicksort

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[int(len(arr) / 2)]         #or // in Py3.x or in Py2.x if from __future import division
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
    
print quicksort([3,6,8,10,1,2,1])
# Prints "[1, 1, 2, 3, 6, 8, 10]"


#seive of prime numbers 


#imperative way
res = [] 
lst = list(range(2,98))

while len(lst) != 0 :
	res.append(lst[0])
	lst = list( filter ( lambda i :  i % res[-1] != 0 , lst ))


print(":".join(str(i) for i in res))

#recursive way

def seive(lst , res = []):
	if lst : 		
		return  seive(  list(filter(lambda i:  i % lst[0] != 0 , lst )) , res + [lst[0]])
	else:
		return res
		
#TCO way
def seive(lst , res = []):
	while True:
		if not lst : return res
		lst, res = list(filter(lambda i:  i % lst[0] != 0 , lst )) , res + [lst[0]]
	
		


seive(list(range(2,100)))

#another version

def primes(n):
    """ Returns  a list of primes < n """
    sieve = [True] * n
    for i in range(3,int(n**0.5)+1,2):    #in py3, range and xrange are same
        if sieve[i]:
            sieve[i*i::2*i]=[False]*((n-i*i-1)//(2*i)+1)
    return [2] + [i for i in range(3,n,2) if sieve[i]]

		
#memoization

def mea(f, *args):
	import time
	now = time.time()
	res = f(*args)
	print(time.time() - now)
	return res

def fib1(n):
	if n == 0 : return 0
	if n ==  1 : return 1
	return fib1(n-1) + fib1(n-2)

	
#3.2
import functools
@functools.lru_cache(maxsize=None)   # by default 128
def fib_m(n):
	return n if n<2 else  fib_m(n-1) + fib_m(n-2)
	
#Recurisve
@functools.lru_cache(maxsize=None)
def fib_r(n, prev=0, next=1):
	while True:
		if n == 0 : return prev
		if n ==  1 : return next
		n, prev, next = n-1, next, prev+next


		
#2.7

from repoze.lru import lru_cache

@lru_cache(maxsize=5000)
def fib_r(n, prev, next):
	while True:
		if n == 0 : return prev
		if n ==  1 : return next
		n, prev, next = n-1, next, prev+next
		
fib_r(100001,0,1)


#http://stackoverflow.com/questions/13591970/does-python-optimize-tail-recursion		

#With decorator

import sys

class TailRecurseException(Exception):
	def __init__(self, args, kwargs):
		self.args = args
		self.kwargs = kwargs

	

def tail_call_optimized(g):
	def func(*args, **kwargs):
		f = sys._getframe()
		if f.f_back and f.f_back.f_back and f.f_back.f_back.f_code == f.f_code:
			raise TailRecurseException(args, kwargs)
		else:
			while 1:
				try:
					return g(*args, **kwargs)
				except TailRecurseException as e:
					args = e.args
					kwargs = e.kwargs
		func.__doc__ = g.__doc__
	return func

@tail_call_optimized
def factorial(n, acc=1):
	"calculate a factorial"
	if n == 0:
		return acc
	return factorial(n-1, n*acc)

print factorial(10000)
# prints a big, big number,
# but doesn't hit the recursion limit.

@tail_call_optimized
def fib(i, current = 0, next = 1):
  if i == 0:
    return current
  else:
    return fib(i - 1, next, current + next)

print fib(10000)
# also prints a big number,
# but doesn't hit the recursion limit.



		
#Currying

def f(x,y):
	return x*y 
	

def f1(x):
	def f2(y):
		return x*y
	return f2

f1(2)(3)

from functools import partial
 
f3 = partial(f, 2)
f3(3)

#From toolz

def stem(word):
	""" Stem word to primitive form """
	return word.lower().rstrip(",.!:;'-\"").lstrip("'\"")

from toolz import compose, frequencies, partial
wordcount = compose(frequencies, partial(map, stem), str.split)

sentence = "This cat jumped over this other cat!"
wordcount(sentence)
{'this': 2, 'cat': 2, 'jumped': 1, 'over': 1, 'other': 1}


########################################  ############
#Python precompiled modules: http://www.lfd.uci.edu/~gohlke/pythonlibs/
#######################################

#  http://web.comlab.ox.ac.uk/oucl/work/jeremy.gibbons/publications/spigot.pdf
 
 
def pi_digits():
    """generator for digits of pi"""
    q,r,t,k,n,l = 1,0,1,1,3,3
    while True:
        if 4*q+r-t < n*t:
            yield n
            q,r,t,k,n,l = (10*q, 10*(r-n*t), t, k, (10*(3*q+r))/t-10*n, l)
        else:
            q,r,t,k,n,l = (q*k,(2*q+r)*l,t*l,k+1,(q*(7*k+2)+r*l)/(t*l),l+2)
 
 
>>> import pi
>>> digits = pi.pidigits()
>>> for i in range(30): print digits.next(),
3 1 4 1 5 9 2 6 5 3 5 8 9 7 9 3 2 3 8 4 6 2 6 4 3 3 8 3 2 7
>>>     


#gmpy2: Multiple-precision Integers

>>> import gmpy2
>>> from gmpy2 import mpz
>>> mpz('123') + 1
mpz(124)
>>> 10 - mpz(1)
mpz(9)
>>> gmpy2.is_prime(17)
True


#Example 


import time
import gmpy2

def sieve(limit=1000000):
    '''Returns a generator that yields the prime numbers up to limit.'''

    # Increment by 1 to account for the fact that slices  do not include
    # the last index value but we do want to include the last value for
    # calculating a list of primes.
    sieve_limit = gmpy2.isqrt(limit) + 1
    limit += 1

    # Mark bit positions 0 and 1 as not prime.
    bitmap = gmpy2.xmpz(3)

    # Process 2 separately. This allows us to use p+p for the step size
    # when sieving the remaining primes.
    bitmap[4 : limit : 2] = -1

    # Sieve the remaining primes.
    for p in bitmap.iter_clear(3, sieve_limit):
        bitmap[p*p : limit : p+p] = -1

    return bitmap.iter_clear(2, limit)

#Usage 
start = time.time()
result = list(sieve())
print(time.time() - start)
print(len(result))
	
#gmpy2: Multiple-precision Rationals

>>> import gmpy2
>>> from gmpy2 import mpq
>>> mpq(1,7)
mpq(1,7)
>>> mpq(1,7) * 11
mpq(11,7)
>>> mpq(11,7)/13
mpq(11,91)

#gmpy2: Multiple-precision Reals


>>> import gmpy2
>>> from gmpy2 import mpfr
>>> mpfr('1.2')

>>> gmpy2.const_pi(100000)
>>> gmpy2.get_context().precision=100
>>> gmpy2.atan2(mpfr("+0.0"),mpfr("-0.0"))

>>> gmpy2.sqrt(5)
mpfr('2.2360679774997898')
>>> gmpy2.get_context().precision=100
>>> gmpy2.sqrt(5)
mpfr('2.2360679774997896964091736687316',100)


>>> with gmpy2.local_context(gmpy2.context(), precision=100) as ctx:
	print(gmpy2.sqrt(2))
	ctx.precision += 100
	print(gmpy2.sqrt(2))


#gmpy2: Multiple-precision Complex
>>> import gmpy2
>>> from gmpy2 import mpc
>>> gmpy2.sqrt(mpc("1+2j"))
mpc('1.272019649514069+0.78615137775742328j')
>>> gmpy2.get_context(real_prec=100,imag_prec=200)
>>> gmpy2.sqrt(mpc("1+2j"))
