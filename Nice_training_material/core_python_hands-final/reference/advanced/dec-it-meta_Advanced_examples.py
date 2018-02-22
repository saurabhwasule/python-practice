



# Examples of Iterators
class Squares0: 				
	def __init__(self, start, stop): 	
		self.start = start
		self.stop = stop
		self.c = start
	def __iter__(self):
		return self
	def __next__(self):			# for py3.3 , for py2.7 it is next()
		if self.c >= self.stop : raise StopIteration
		self.c += 1
		return (self.c-1)**2
s = Squares0(1,10)
I1 = iter(s); I2=iter(s)  #Single pass
next(I1), next(I2)  #(1, 4)
next(I1), next(I2) #(9, 16)

#Alternate version
class Squares: 				
	def __init__(self, start, stop): 	
		self.start = start
		self.stop = stop
	def __iter__(self):
		for value in range(self.start, self.stop + 1):
			yield value ** 2

# Another example			
import glob
import os
class OnlyFile: 				
	def __init__(self, pattern): 	
		self.p = pattern
	def __iter__(self):
		for value in [ file for file in glob.glob(self.p) if os.path.isfile(file)] :
			yield value

for f in OnlyFile("*"):
	for f1 in OnlyFile("*"):
		print(f,f1)
			
# only from 3.3			
import glob
import os
class OnlyFile: 				
	def __init__(self, pattern): 	
		self.p = pattern
	def __iter__(self):
		yield from [ file for file in glob.glob(self.p) if os.path.isfile(file)]
		

		
# decorator		Example
def ConvertString(org):
	def _inner(*args, **kargs):
		res = org(*args, **kargs)
		return str(res)
	return _inner

	
@ConvertString
def f(x,y):
	return x+y



def  mockReturnConstant(fun):
	def _mydec(*args, **kargs):		
		res = fun(*args, **kargs)		
		return 10
	return _mydec
	
@ConvertString        #last application's decorator must be at first
@mockReturnConstant
def f(x):
	return x*x

#tracing
def  mydec(fun):
	def _mydec(*args, **kargs):
		print("calling " + fun.__name__)
		res = fun(*args, **kargs)
		print("ending " + fun.__name__)
		return res
	return _mydec

@mydec
def f(x):
	return x*x
	
f(2)

#Another Example
def makeDefault100(f):
	def org(*args):
		val = args[1] if len(args) > 1 else 100
		return f(*args) if len(args) > 1 else f(*args,y=val)
	return org

@makeDefault100
def f1(x,y):
	return x+y

	
f1(2)
f1(2,3)	

# Example with Decorator argument
def makeDefault(d):
	def aDec(f):
		def org(*args):
			val = args[1] if len(args) > 1 else d
			return f(*args) if len(args) > 1 else f(*args,y=val)
		return org
	return aDec

@makeDefault(50)
def f1(x,y):
	return x+y

	
f1(2)
f1(2,3)	

#Other examples

def trace(orginal):
	def _inner(*args, **kargs):
		print("Entering .. ", orginal.__name__)
		res = orginal(*args, **kargs)
		print("Exiting .. ", orginal.__name__)
		return res
	return _inner


def makeConstant(const):
	def inner(orginal):
		def _inner(*args, **kargs):
			res = orginal(*args, **kargs)
			return const
		return _inner
	return inner
	
	
def mock(*o_args):
	def inner(orginal):
		def _inner(*args, **kargs):
			res = orginal(*o_args)
			return res
		return _inner
	return inner
	
@mock(10,20)
def fun2(x,y):   #fun = mock(10,20)(fun2)
	z = x+y
	return z
	

@makeConstant(50)	
@trace	
def fun(x,y):   #fun = makeConstant(50)(trace(fun))
	z = x+y
	return z
	

	
print(fun2(2,3))

#measurement decorator
	
def  mea(fun):
	def _inner(*args, **kargs):
		import time
		now = time.time()
		res = fun(*args,**kargs)
		print(time.time() - now)
		return res
	return _inner
	
	
@mea
def f(x,y):
	return x+y
	
#running a function multiple times
def iter_f(func):
    def newf(*args, **kwargs):
        for i in range(10):
            res = func(*args, **kwargs)
			print(res)
    return newf

@iter_f
def f(x):
	return x*x
	
	
def iter_f(d):
	def inner(f):
		def org(*args, **kwargs):
			for i in range(10):
				res = f(*args, **kwargs)
				print(res)
		return org
	return inner
	
	
@iter_f(10)
def f(x):
	return x*x
#for recursion it calls many times
@mea
def fib(n):
	return n if n<2 else  fib(n-1) + fib(n-2)
	
#to fix

def  mea(fun):
	def _inner(*args, _first = False, **kargs):
		import time
		if _inner.first :
			_inner.first = False
			_first = True
		now = time.time()
		res = fun(*args,**kargs)
		if _first : 
			_inner.first = True;
			print(time.time() - now)
		return res
	_inner.first = True
	return _inner
	
@mea
def fib(n):
	return n if n<2 else  fib(n-1) + fib(n-2)
	
	
    
    
##Using wraps 
#the name of the example function would have been 'wrapper', 
#and the docstring of the original example() would have been lost.
   
    
from functools import wraps
def my_decorator(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        print 'Calling decorated function'
        return f(*args, **kwds)
    return wrapper

@my_decorator
def example():
    """Docstring"""
    print 'Called example function'

>>> example()
Calling decorated function
Called example function
>>> example.__name__
'example'
>>> example.__doc__
'Docstring'
    
    
    
    
    
    
    
    
    
# class decorator
#use below
def __getattr__(self, name): 		# On undefined attribute fetch [obj.name], for old and new style
def __getattribute__(self, name): 		# On all attribute fetch [obj.name], new style
def __setattr__(self, name, value): 	# On all attribute assignment [obj.name=value]
def __delattr__(self, name): 		# On all attribute deletion [del obj.name]




def decorator(cls): 					# On @ decoration
	class Wrapper:
		def __init__(self, *args): 		# On instance creation
			self.wrapped = cls(*args)
		def __getattr__(self, name): 	# On attribute fetch
			print("Getting " + name + "...")
			return getattr(self.wrapped, name)
	return Wrapper

@decorator
class C: 							# C = decorator(C)
	def __init__(self, x, y): 		# Run by Wrapper.__init__
		self.attr = 'spam'

x = C(6, 7) 		# Really calls Wrapper(6, 7)
print(x.attr) 		# Runs Wrapper.__getattr__, prints "spam"


#Note for searching for New style class

#	Attribute Fetch for an instance of class and for class
#1. Instance attribute access – Search below
#a. The __dict__ of the instance I
#b. The __dict__ of all classes on the __mro__ found at I’s __class__, from left to right  

#2. Class attribute access – Search Below
#a. The __dict__ of all classes on the __mro__ found at C itself, from left to right
#b. The __dict__ of all metaclasses on the __mro__ found at C’s __class__, from left to right

#4. In both rule 1 and 2, skip step a for built-in  implicit operations 
#( eg str, len etc which invokes special method don’t check instance’s, but classes’ )

#5. Special case: __getattr__ and __getattribute__  of class or metaclass 
#are skipped for builtins, but not skipped for explicit method invocation



##################################
#generator for both version
def generate_ints(N):
    for i in range(N):
        yield i*i

def counter (maximum):
    i = 0
    while i < maximum:
        val = (yield i)
        # If value provided, change counter
        if val is not None:
            i = val
        else:
            i += 1
			
s = iter(counter(1000))
next(s)
s.send(50)
next(s)


#generator object for both version

a =  (x*x for x in range(100))
next(a)
next(a)


#Upto some n
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

#Infinite
def fib():
    a, b = 0, 1
    while 1:
        yield a
        a, b = b, a + b

a = iter(fib())
next(a)
0

import itertools as it

for x in it.islice(a, 20):
    print(x)


class Fib: 				
    def __init__(self, a=0, b=1): 	
        self.a, self.b = a,b 
    def __iter__(self):
        while 1:
            yield self.a
            self.a, self.b = self.b, self.a + self.b

			
#with functions			
def fib1():
	a,b = 0,1
	def g():
		nonlocal a,b  #else UnboundLocalError: local variable 'b' referenced before assignment
		a,b = b, a+b
		return a
	return g

#Prime
import math
class NextP: 				
	def __init__(self, start=2): 	
		self.st = start
		self.nx = 2 if start <= 2 else self.next_prime(start)
	def __iter__(self):
		while 1:
			yield self.nx
			self.nx = self.next_prime(self.nx)	
	def next_prime(self, n):
		while True:
			n = n + 1
			if self.is_prime(n) : 
				return n
	def is_prime(self, n):
		if n % 2 == 0:
			return False
		sqrt_n = int(math.floor(math.sqrt(n)))
		for i in range(3, sqrt_n + 1, 2):
			if n % i == 0:
				return False
		return True



		



# iterator example
	
def is_prime(n):
    import math	
    if n == 2 : return True 
    if n % 2 == 0:
        return False
    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True

	
[0,1,response]
[0,2,user]
    [1,1,username]
    [1,2,server]
    [1,3,sleep]
        [2,1,actual]
    
    
    
    
    
    
    
    
def nextP():
	np = 2
	yield np
	while True:
		np = np + 1
		if is_prime(np) : 
			yield np


	
	
	
class MyI :
	def __init__(self, st, stop):
		self.st = st
		self.stop = stop
	def __iter__(self):
		for i in range(self.st, self.stop):
			yield i
		
>>> b = MyI(1,10)
>>> list(b)
[1, 2, 3, 4, 5, 6, 7, 8, 9]
			
	
	
	
#Prime generator
	
	
	
	
def primes(n):
    """ Returns  a list of primes < n """
    sieve = [True] * n
    for i in range(3,int(n**0.5)+1,2):    #in py3, range and xrange are same
        if sieve[i]:
            sieve[i*i::2*i]=[False]*((n-i*i-1)//(2*i)+1)
    return [2] + [i for i in range(3,n,2) if sieve[i]]

#Py3.3
import math
class NextP: 				
	def __init__(self, start=2): 	
		self.st = start
		self.nx = start
	def __iter__(self):
		return self
	def __next__(self):
		if self.st == 2:
			return self.nx
		else:
			self.nx = next_prime(self.nx)
			return self.nx
	def create_prime(n):
		while True:
			n = n + 1
			if is_prime(n) : return n
			
s = NextP()
I1 = iter(s)
next(I1)

import itertools
def nextPrime( ):
    '''Yields the sequence of prime numbers via the Sieve of Eratosthenes.'''
    D = dict()  # map each composite integer to its first-found prime factor
    for q in itertools.count(2):     # q gets 2, 3, 4, 5, ... ad infinitum
        p = D.pop(q, None)
        if p is None:
            # q not a key in D, so q is prime, therefore, yield it
            yield q
            # mark q squared as not-prime (with q as first-found prime factor)
            D[q*q] = q
        else:
            # let x <- smallest (N*p)+q which wasn't yet known to be composite
            # we just learned x is composite, with p first-found prime factor,
            # since p is the first-found prime factor of q -- find and mark it
            x = p + q
            while x in D:
                x += p
            D[x] = p

			
a = iter(nextPrime( ))

next(a)
next(a)

b = iter(nextPrime( ))
next(b)
next(b)
next(a)

	
#Py2.7	
class Squares0: 				
	def __init__(self, start, stop): 	
		self.start = start
		self.stop = stop
		self.c = start
	def __iter__(self):
		return self
	def next(self):
		if self.c >= self.stop : raise StopIteration
		self.c += 1
		return (self.c-1)**2		
		
s = Squares0(1,10)
I1 = iter(s)
next(I1)		


import math
class NextP: 				
	def __init__(self, start=2): 	
		self.st = start
		self.nx = start
	def __iter__(self):
		return self
	def next(self):
		if self.st == 2:
			self.st += 1
			return self.nx
		else:
			self.nx = self.next_prime(self.nx)
			return self.nx
	def next_prime(self,n):
		while True:
			n = n + 1
			if self.is_prime(n) : return n
	def is_prime(self,n):
		if n % 2 == 0:
			return False
		sqrt_n = int(math.floor(math.sqrt(n)))
		for i in range(3, sqrt_n + 1, 2):
			if n % i == 0:
				return False
		return True 
			
s = NextP()
I1 = iter(s)
next(I1)
		




#Generator for factors

# operator

# factors generators

def factors(n):
    while n > 1:
        for i in range(2, n + 1):
            if n % i == 0:
                n //= i
                yield i
                break     #break is must as we need start the search

for factor in factors(360):
    print(factor)


from operator import mul
reduce(mul, factors(360))
360

#

def factors(n):    # (cf. http://stackoverflow.com/a/15703327/849891)
    j = 2
    while n > 1:
        for i in xrange(j, int(sqrt(n+0.05)) + 1):
            if n % i == 0:
                n /= i ; j = i
                yield i
                break
        else:
            if n > 1:
                yield n; break
				
				

#yield from examples.. Lazy computations
			
			
#examples of comprehensions- eager - Note lazy is not possible as there should be stop in range else goes infinite
#to find pythogoran triplets
import functools 

[(x,y,z) for x in range(1,100) for y in range(x,100) for z in range(x,100) if x*x + y*y == z*z ]
;compose
l = [ lambda x: x*x, lambda x: x+2, lambda x: x*4 ]
f = functools.reduce(lambda r,e: (lambda x: e(r(x)) ), l, lambda x: x)
l = [ lambda x: x*x ] * 3  # three times 
f = functools.reduce(lambda r,e: (lambda x: e(r(x)) ), l, lambda x: x)
	
	
#https://docs.python.org/3/library/itertools.html

#itertools.count(start,[step])   -> start, start+step...
#itertools.repeat(element, [n])

#dropwhile(predicate, iterable)
#takewhile(predicate, iterable)
#islice(iterable, start, stop[, step]) or islice(iterable, stop)
#tee(iterable, n=2) --> tuple of n independent iterators. (memoized version of it, hence very fast)


# Few lazy definition
from itertools import *

def iterate(x,f):
    yield f(x)
    yield from iterate(f(x),f)
	
#eg #itertools.count(start,[step])  can be implemented as 

def count(st, step=None):
	import functools
	import operator
	f = functools.partial(operator.add, step if step else 1)
	yield st
	yield from iterate(st, f)
	
	
#Few more functions
def tail(iterable):  
	return islice(iterable, 1, None)
	

def take(iterable, n):  
	return list(islice(iterable, 0, n))

def head(iterable):  
	return list(islice(iterable, 1))

#or if islice is not wanted

def tail(it):
	it = iter(it)
	next(it)
	return it


def take(it, n):
	it = iter(it)
	return [ next(it) for i in range(n) ]


def head(it):
	it = iter(it);
	return next(it)
	
#few fibs implementations

def fibfrom(a = 0, b = 1):
	yield a
	yield from fibfrom(b, a+b)
	

>>> take(fibfrom(), 100)

def fib_inf():
	yield from iterate( (0,1) , lambda t: ( t[1], t[0] + t[1]) )

>>> take(map(lambda x: x[0], fib_inf()), 10)

#inefficient as double recursion via fibs() and  tail(fibs()) and no memoization
def fibs():
	import operator
	yield 0
	yield 1
	yield from map(operator.add, fibs(), tail(fibs()))

>>> take(fibs(), 10)

#efficient, using tee to copy instead of double recursion
def fibs():
	import operator
	import itertools
	yield 0
	yield 1
	fibs1, fibs2 = itertools.tee(fibs())
	yield from map(operator.add, fibs1, tail(fibs2))


#all are recursive, hence recursion depth occurs
#Stream implementation, is tuple , (H, T)
#where first element = real element, 2nd element = deferred calculation via function 
#2nd element = put always lambda of direct calculation and get value by ()


null_stream = (None, None)

def head( S ): return S[0]

def tail( S ): return S[1]()     # a Method , hence call it 

def from_inf(N): return (N, lambda: from_inf(N+1))

def continually(N): return (N, lambda: continually(N))

def iterate(x, f): return (f(x), lambda : iterate(f(x), f) )

def tabulate(x, f): return (f(x), lambda : tabulate(x, f) )

def stream(lst):
	return  null_stream if not lst else (lst[0], lambda : stream(lst[1:])) 


def  cons(ele, stream):
	return (ele, lambda : stream)
	
def function_iterate(f):
    return (f, lambda: function_iterate(lambda x: f(f(x))))


def function_tabulate(f):
    return (f, lambda: function_tabulate(f))
	

>>> z = from_inf(1)
>>> head(z)
1
>>> tail(z)
(2, <function from_inf.<locals>.<lambda> at 0x032E5198>)
>>> head(tail(z))
2
>>> head(tail(tail(z)))
3


#few functions implementations , 

def smap(f, stream):
	if stream is null_stream: return null_stream
	return (f(head(stream)), lambda: smap(f, tail(stream)))


def sfilter(pred, stream):
    if stream is null_stream: return null_stream
    if pred(head(stream)):
        return (head(stream), lambda: sfilter(pred, tail(stream)))
    return sfilter(pred, tail(stream))



def sreduce(f, stream, init):
	if stream is null_stream: return init
	return sreduce(f, tail(stream), f(init, head(stream)) )
	
#TCO form
def sreduce(f, stream, init):
	while True:
		if stream is null_stream: return init
		stream, init = tail(stream), f(init, head(stream)) 

#
def to_array(stream):
    return sreduce(lambda a, x: a + [x], stream, [])

	

def take(stream,N): 
	if N <= 0 or stream is null_stream: return null_stream 
	return (head(stream), lambda: take(tail(stream), N-1)) 


	
#TCO form
def sreduce(f, stream, init):
	while True:
		if stream is null_stream: return init
		stream, init = tail(stream), f(init, head(stream)) 

#
def smap(f, stream):
	if stream is null_stream: return null_stream
	return (f(head(stream)), lambda: smap(f, tail(stream)))  #there might be many tail, so recursion depth!!


def sfilter(pred, stream):
	while True:
		if stream is null_stream: return null_stream
		if pred(head(stream)):
			return (head(stream), lambda: sfilter(pred, tail(stream))) #there might be many tail, so recursion depth!!
		stream = tail(stream)


#Sieve implementation


def sieve(stream):
	if stream is null_stream: return null_stream
	h = head(stream)
	return (h, lambda: sieve(sfilter(lambda x: x%h != 0, tail(stream))))


	
>>> primes = sieve(from_inf(2))
>>> to_array(take(primes,10))

#Root finding 


def newton(f, fdash):
	return lambda x: x - f(x)/float(fdash(x))

def newton_solver(iteration, f, fdash):
	def solve(v):
		n = newton(lambda x: f(x) - v, fdash)
		stream = function_iterate(n)
		return to_array(take(stream, iteration))[-1]   #returns function
	return solve


>>> sqrt = newton_solver(1, lambda x: x**2, lambda x: 2*x) # 1 iter
>>> sqrt(64)(4) # Sqrt of 64 with initial guess of 4
10.0
>>> sqrt = newton_solver(3, lambda x: x**2, lambda x: 2*x) # 3 iters
>>> sqrt(64)(4)
8.000000371689179

>>> cuberoot = newton_solver(5, lambda x: x**3, lambda x: 3*x**2)
>>> cuberoot(27)(2)
3.0

				
#  Class Example: User, PreUser, BankAcount
# Amount

class User:
	def __init__(self, name, account):
		self.name = name
		self.account = account
	def transacte(self, amount):
		self.account.transacte(amount)
	def __str__(self):
		return "%s with balance %d" % (self.name, self.account.balance())


	
class PreUser(User):
	type = {"GOLD" : 1.05 , "SILVER" : 1.02}
	def __init__(self, name, account, type="GOLD"):
		User.__init__(self, name, account)
		self.type = type
	def __str__(self):
		return "%s (%s) with balance %d"% (self.name, self.type, self.account.balance())
	def transacte(self, amount):
		try:
			super().transacte( amount)
		except UserException as e:
			print(str(e), "For amount ", amount)
		finally:
			self.account.balance(PreUser.type[self.type] * self.account.balance()) if amount > 0 else None
		

		
class UserException(Exception):
			pass
		

class BankAccount:
	def __init__(self, initAmount=100):
		self.amount = initAmount
	def transacte(self, amount):
		if (self.amount + amount <0):
			raise UserException("Not possible")
		self.amount =  self.amount + amount
	def balance(self, *amount):
		if not amount: return self.amount
		else: self.amount = amount[0]
			
			
me = PreUser("Unknown", BankAccount(100), "SILVER")
for am in [ 100, -200, 300, -400, 400]:
	me.transacte(am)


print(me)


	
# Meta-class Examples

def func4(obj): return obj.value * 4


class Extender(type):
	def __new__(meta, classname, supers, classdict):
		classdict['func4'] = func4					#adding func4 method
		return type.__new__(meta, classname, supers, classdict)

class A(metaclass=Extender):
	def __init__(self, value): self.value = value
	def func2(self): return self.value * 2

# Another example - Tracing, with decorator version	
def trace(f):
	def inner(*args, **kargs):
		print("-->" + f.__name__)
		res = f(*args, **kargs)
		print("<--" + f.__name__)
		return res
	return inner

# Metaclass 

class TracingO(type):
	def __new__(meta, classname, supers, classdict):
		classdict['func2'] = trace(classdict['func2'])
		return type.__new__(meta, classname, supers, classdict)

class B(metaclass=TracingO):
	def __init__(self, value): self.value = value
	def func2(self): return self.value * 2


# adding tracing to all functions
import types

class Tracing(type):
	def __new__(meta, classname, supers, classdict):
		classdict = {k:(trace(v) if type(v) is types.FunctionType and v.__name__.startswith("fun") else v ) for k,v in classdict.items()}
		return type.__new__(meta, classname, supers, classdict)


class B(metaclass=Tracing):
	def __init__(self, value): self.value = value
	def func(self): return self.value * 2


# metaclass execution


class m(type):
	def __new__(meta, c, s, cd):
		print("m.__new__")
		return type.__new__(meta,c,s,cd)
	def __call__(*args, **kargs):
		print("m.__call__")
		return type.__call__(*args, **kargs)
	def __init__ (c, cn, s, cd):
		print("m.__init__")
		return type.__init__(c,cn,s,cd)
		
class A(metaclass=m):pass

...
m.__new__
m.__init__

#creation of metaclass
class mm(type):
	def __new__(meta, c, s, cd):
		print("mm.__new__")
		return type.__new__(meta,c,s,cd)
	def __call__(*args, **kargs):
		print("mm.__call__")
		return type.__call__(*args, **kargs)
	def __init__ (c, cn, s, cd):
		print("mm.__init__")
		return type.__init__(c,cn,s,cd)
		
class m(type, metaclass = mm):
	def __new__(meta, c, s, cd):
		print("m.__new__")
		return type.__new__(meta,c,s,cd)
	def __call__(*args, **kargs):
		print("m.__call__")
		return type.__call__(*args, **kargs)
	def __init__ (c, cn, s, cd):
		print("m.__init__")
		return type.__init__(c,cn,s,cd)
...
mm.__new__
mm.__init__
		
class A(metaclass=m):pass
...
mm.__call__
m.__new__
m.__init__

# attribute accessing

class M(type):pass

class A(metaclass = M):pass

class B(A): pass

b = B()

B.a1 = 100
A.a2 = 200
M.a3 = 300

b.a1, b.a2, b.a3  # 100, 200, Error

B.a1, B.a2, B.a3  #100,200,300

# with instance variable

class M(type):
	def __new__(m,c,s,cd):
		m.m1 = 10
		return type.__new__(m,c,s,cd)
	def __init__(c,cn,s,d):
		c.ma1 = 20
		

class A(metaclass = M):
	def __init__ (self):
		self.ia1 = 30

	

class B(A): 
	def __init__ (self):
		self.ia2 = 40
		super().__init__()
	

b = B()

B.a1 = 100
A.a2 = 200
M.a3 = 300

b.a1, b.a2, b.a3  # 100, 200, Error

b.ia2, b.ia1, b.ma1, b.m1  # 40, 30, 20, Error

B.a1, B.a2, B.a3  #100,200,300

B.ia2, B.ia1, B.ma1, B.m1  # Error, Error, 20, 10