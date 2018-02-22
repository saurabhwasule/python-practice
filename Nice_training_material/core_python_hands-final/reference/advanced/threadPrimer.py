import threading
import time
import random

def  f():
	time.sleep(random.randint(10,15))
	print(threading.current_thread().getName())


th = list();

for ele in range(20):
	t = threading.Thread(target=f, name="Thread " + str(ele))
	t.start()
	th.append(t)

#--------------------Join--------------------------------


import time
import random

def w():
	time.sleep(random.randint(10,15))
	print("Thread finished")
	
t = threading.Thread(target=w)
t.start(); print("Main thread")


t = threading.Thread(target=w)
t.start(); t.join(); print("Main thread")



		
#--------------------------------CONCURRENT EXAMPLES-----------------------
#Version Py3.x
#GIL is not initialised until the threading support is imported, or initialised via the C API, 
#GIL is released for i/o bound eg network or using NumPy modules
#for CPython, alternate approach is
#use  multiprocessing module, migrate from threaded code to multiprocess code, (good for long code)
#use concurrent.futures, Use ThreadPoolExecutor  to dispatch  to multiple threads (for IO bound operations) 
#or Use ProcessPoolExecutot to dispatch to multiple processes (for CPU bound operations), 
#or use the asyncio module in Python 3.4 (which provides full support for explicit asynchronous programming in the standard library) 
#or use async/await syntax for native coroutines in Python 3.5.
#or use event driven eg Twisted library

class concurrent.futures.Executor
    submit(fn, *args, **kwargs)
        returns a Future 
    map(func, *iterables, timeout=None, chunksize=1)
        returns Iterator of results 
    shutdown(wait=True)
    
concurrent.futures.wait(fs, timeout=None, return_when=ALL_COMPLETED)
        returns (done_futures, pending_futures)
        return_when can be FIRST_COMPLETED, FIRST_EXCEPTION,ALL_COMPLETED
concurrent.futures.as_completed(fs, timeout=None)
    Returns an iterator over the Future instances 
    given by fs that yields futures as they complete 
    (finished or were cancelled).
    
#Example 
import threading
import concurrent.futures  #in Py2, must do, pip install futures
import requests
import time

def load(url):
    import requests
    import time
    import threading
    time.sleep(5)
    print("Starting to download ", url, " from thread ", threading.current_thread().getName())
    conn = requests.get(url)    
    return [url, len(conn.text)]

def load(url):    
    import time 
    import random 
    time.sleep(5)
    print("Starting to download ", url, " from thread ", 
            threading.current_thread().getName())
    res = [url, random.randint(2000,3000)]    
    return res
    
    
result = []
#note with blocks 
def exmap(urls):
    global result;
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        result= ex.map(load, urls)

        
urls = ["http://www.google.co.in" for i in range(10) ]
exmap(urls)	 #blocks
result = list(result)

#for non blocking for 'with'
t = threading.Thread(target=exmap, args=(urls,))
t.start()
result = list(result)

#OR 
ex = concurrent.futures.ThreadPoolExecutor(max_workers=2)
result = ex.map(load, urls)
#One by one , blocks if result is not ready 
next(result)

#below blocks for complete result 
#Might raise exception if load fails
#hence use with try block 
list(result)
ex.shutdown()


#2nd version with submit, retures Future 

ex = concurrent.futures.ThreadPoolExecutor(max_workers=2)
result = [ex.submit(load, url) for url in urls ]
#do ur work 

#below blocks , Might raise exception if load fails
#hence use with try block 
output = [res.result() for res in concurrent.futures.as_completed(result)] #with Key, fs[res] has to be before res.result()
len(output)
ex.shutdown()

#could use 
done, not_done = concurrent.futures.wait(fs, timeout=None, return_when=ALL_COMPLETED) #FIRST_COMPLETED,FIRST_EXCEPTION
#done set contains done futures



#Another Example with Prime

import concurrent.futures
import time 

import random 

#time.sleep(random.randint(10,15))
def is_prime(n):
    import math
    if n == 2 : return True 
    if n % 2 == 0:	return False
    sqrt_n = int(math.sqrt(n))
    a = [1 for i in range(3, sqrt_n + 1, 2) if n % i == 0]
    return False if sum(a) > 0 else True
	

#map 
nos = list(range(1000))
ex = concurrent.futures.ThreadPoolExecutor(max_workers=2)
result = zip(nos, ex.map(is_prime, nos))  
#do ur work 

#One by one , blocks if result is not ready 
next(result)

#below blocks for complete result 
list(result)
ex.shutdown() 
 
    
#submit 
nos = list(range(1000))
ex = concurrent.futures.ThreadPoolExecutor(max_workers=2)
fs = { ex.submit(is_prime, e): e for e in nos }
#do ur work 
#below blocks , fs[res] must be before calling res.result()
output = [  (fs[res] , res.result()) 
            for res in concurrent.futures.as_completed(fs)]
ex.shutdown()   
    
    
###Copy example 
import shutil
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:
    e.submit(shutil.copy, 'src1.txt', 'dest1.txt')
    e.submit(shutil.copy, 'src2.txt', 'dest2.txt')
    e.submit(shutil.copy, 'src3.txt', 'dest3.txt')
    e.submit(shutil.copy, 'src3.txt', 'dest4.txt')

##Deadlocks can occur 
#when the callable associated with a Future waits 
#on the results of another Future. 
#Use asyncio for this type of coding 

import concurrent.futures
import time
def wait_on_b():
    time.sleep(5)
    print(b.result()) # b will never complete because it is waiting on a.
    return 5

def wait_on_a():
    time.sleep(5)
    print(a.result()) #comment this, then works # a will never complete because it is waiting on b.
    return 6


executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
a = executor.submit(wait_on_b)
b = executor.submit(wait_on_a)
a.result() ###Deadlock 

#OR 
def wait_on_future():
    f = executor.submit(pow, 5, 2)
    # This will never complete because there is only one worker thread and
    # it is executing this function.
    return f.result()

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
a = executor.submit(wait_on_future)
a.result()  #blocks 

##Important methods of Future, 
# Future instances are created by Executor.submit() 
cancel()
    Attempt to cancel the call. Returns true/false if cancelled 
cancelled()
    Return True if the call was successfully cancelled.
running()
    Return True if the call is currently being executed and cannot be cancelled.
done()
    Return True if the call was successfully cancelled or finished running.
result(timeout=None)
    Return the value returned by the call. Blocks till timeout if not done 
    If the future is cancelled before completing then CancelledError will be raised.
    If the call raised, this method will raise the same exception.
exception(timeout=None)
    Return the exception raised by the call. 
add_done_callback(fn_with_one_arg_of_future)
    calls fn when the future is cancelled or finishes running.
    Added callables are called in the order that they were added 
    and are always called in a thread which added them 
    If the callable raises a Exception subclass, 
    it will be logged and ignored.




#with Process Pool - must be in separate file and 'if' clause is must
#no GIL for cpython
import concurrent.futures
import math
import time 
import random 

def main_p(primes, max_w=10):
	d = dict()
	with concurrent.futures.ProcessPoolExecutor(max_workers=max_w) as ex:
		d = dict(zip(primes, ex.map(is_prime, primes)))
	return d
	
if __name__ == '__main__':    
	print(main_p(list(range(3,98))))

#with submit- must be in separate file and 'if' clause is must

def getWorkDone(lst):		
	with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
		fs= { ex.submit(is_prime, e): e for e in lst }
		return [  (fs[res] , res.result()) for res in concurrent.futures.as_completed(fs.keys())]  #with Key, fs[res] has to be before res.result()

		
if __name__ == '__main__':    
	print(getWorkDone(list(range(3,98))))
	




#######synchronization


#Lock

import threading
import time

def worker(lock):
    with lock:
        print('Acquired by' + threading.current_thread().getName())
        time.sleep(5)
        print('Released by' + threading.current_thread().getName())
		
        

lock = threading.Lock()
for i in range(2):
	w = threading.Thread(target=worker, args=(lock,))
	w.start()
	
	
#RLock

import threading
import time

def worker(lock):
	with lock:
		print('Acquired by' + threading.current_thread().getName())
		with lock:
			time.sleep(5)
            print('Releasing by' + threading.current_thread().getName())
        

lock = threading.RLock()
for i in range(2):
	w = threading.Thread(target=worker, args=(lock,))
	w.start()





#Synchronization with Condition

#follow below pattern
#The while loop checking for the application’s condition is necessary 
#because wait() can return after an arbitrary long time, 
#and the condition which prompted the notify() call may no longer hold true

# Consume one item
with cv:
    while not an_item_is_available():  #Initially returns False, producer makes it true 
        cv.wait()
    get_an_available_item() #make an_item_is_available return false

# Produce one item
with cv:
    make_an_item_available()  #make an_item_is_available return true
    cv.notify()

#OR , use wait_for(predicate, timeout=None)
# Consume an item
with cv:
    cv.wait_for(an_item_is_available)
    get_an_available_item()



#example
import threading
import time
import random

shared_var = 0
available = False 

def consumer(cond):
        global available
        print('Starting ' + threading.current_thread().getName())
        with cond:		
            cond.wait_for(lambda : available)
            print('[Consumer] Got Resource ', threading.current_thread().getName(), shared_var)  #access global shared_var
            available = False


def producer(cond, max):	
    global shared_var                #must as it sets global
    global available
    i = 0
    while i <= max:
        with cond:				
            shared_var = random.randint(20,100);
            print('Notify One Consumer ',  threading.current_thread().getName(), shared_var)
            available = True
            cond.notify()			#for only one consumer, for all -use cond.notifyAll()
        time.sleep(1) 			#some other work, Note, consumer would be awakened here only after 'with scope'
        i += 1
	

condition = threading.Condition()
for i in range(10):
	w = threading.Thread(target=consumer, args=(condition,))
	w.start()

p = threading.Thread(name='Producer', target=producer, args=(condition,10))
p.start()




#Semaphore

import random

def worker(s):
	print('Waiting to join the pool ' + threading.current_thread().getName())
	with s:
		print('Got access ' + threading.current_thread().getName())
		time.sleep(random.randrange(10))


s = threading.Semaphore(2) #at a time only 2 can access
for i in range(4):
	t = threading.Thread(target=worker, name=str(i), args=(s,))
	t.start()


	
#Event - one thread signals an event and other threads wait for it. 
#no 'with block'

import threading
import time
                    
def wait_for_event(e):
	event_is_set = e.wait()
	print('Got access ' + threading.current_thread().getName())

def wait_for_event_timeout(e, t):
	while not e.isSet():
		event_is_set = e.wait(t)
		if event_is_set:
			print('processing event')
		else:
			print('doing other work')


e = threading.Event()
t1 = threading.Thread(name='block', target=wait_for_event, args=(e,))
t1.start()

t2 = threading.Thread(name='non-block', target=wait_for_event_timeout, args=(e, 2))
t2.start()

time.sleep(3)
e.set()
# both threads are awakened


#Timer - triggers a code after certain time


def hello():	
	print("hello, world")

t = threading.Timer(5.0, hello)
t.start() # after 5 seconds, "hello, world" will be printed


#Barrier 
#fixed number of threads that need to wait for each other. 
#Each of the threads tries to pass the barrier by calling the wait() method 
#and will block until all of the threads have made the call. 
#At this points, the threads are released simultanously

b = threading.Barrier(2)
def server():
	time.sleep(5)
	b.wait()
	print("server got access..")

	

def client():
	time.sleep(10)
	b.wait()
	print("client got access..")

threading.Thread(target=server).start()
threading.Thread(target=client).start()



#Queue
import queue
import threading
import time

def worker(q):
	while True:
		item = q.get()
		print(threading.current_thread().getName(), item)
		time.sleep(2)
		q.task_done()

		
que = queue.Queue()
for i in range(2):
	t = threading.Thread(target=worker, args=(que,))
	t.start()

for item in range(10):  #Only one thread get one item 
	que.put(item)

que.join()       # block until all tasks are done


###multiprocessing
#Processing Must be in separate file and run as script - no GIL

from multiprocessing import Process

def f(name):
		print('hello', name, "from ", multiprocessing.current_process().name)


if __name__ == '__main__':
    p = Process(target=f, args=('das',))
    p.start()
    p.join()

#Pool
from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    pool = Pool(processes=4)               # start 4 worker processes
    result = pool.apply_async(f, [10])     # evaluate "f(10)" asynchronously
    print(result.get(timeout=1))           # prints "100" unless your computer is *very* slow
    print(pool.map(f, range(10)))          # prints "[0, 1, 4,..., 81]"


#with Process Pool - must be in separate file and 'if' clause is must
from multiprocessing import *

def is_prime(n):
	import math
	if n % 2 == 0:	return False
	sqrt_n = int(math.sqrt(n))
	a = [1 for i in range(3, sqrt_n + 1, 2) if n % i == 0]
	return False if sum(a) > 0 else True
	
	
def main_p(primes, max_w=10):
	p = Pool(max_w)
	d = dict(zip(primes, p.map(is_prime, primes)))
	return d
	
if __name__ == '__main__':    
	print(main_p(list(range(3,98))))



#Process Pipe

from multiprocessing import Pipe
a, b = Pipe()
a.send([1, 'hello', None])
b.recv()
#[1, 'hello', None]
b.send_bytes(b'thank you')
a.recv_bytes()
#b'thank you'

