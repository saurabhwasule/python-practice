#Pool examples will not work in the interactive interpreter
from __future__ import print_function
from multiprocessing import Pool, current_process, Lock, Event, Process, TimeoutError
import os
import time
    
          
#can not use decorator as it needs to pickle 
def is_prime(n):  
    p = current_process()
    with lock:
        print(p.name, " with pid:" , p.pid, " for ", n)
    import math
    if n % 2 == 0:	return False
    sqrt_n = int(math.sqrt(n))
    a = [1 for i in range(3, sqrt_n + 1, 2) if n % i == 0]
    return False if sum(a) > 0 else True
    
def is_prime_tuple(n):  
    p = current_process()
    with lock:
        print(p.name, " with pid:" , p.pid, " for ", n)
    import math
    if n % 2 == 0:	return (n, False)
    sqrt_n = int(math.sqrt(n))
    a = [1 for i in range(3, sqrt_n + 1, 2) if n % i == 0]
    return (n, False) if sum(a) > 0 else (n, True)
	
#You can't pass normal multiprocessing.Lock objects to Pool methods, 
#because they can't be pickled	
#now each process can access lock 
def init(l, e):
    global lock, event
    lock = l
    event = e

	
if __name__ == '__main__':    
    primes = range(3,99)
    l = Lock()
    e = Event()
    pool = Pool(processes=4, initializer=init, initargs=(l,e))
    d = dict(zip(primes, pool.map(is_prime, primes)))
    with l:
        print(d)
  
    
 
    # evaluate "f(20)" asynchronously
    res = pool.apply_async(is_prime_tuple, (20,))      # runs in *only* one process
    print(res.get(timeout=1))         #don't use with l as res itself blocks   

    # evaluate "os.getpid()" asynchronously
    res = pool.apply_async(os.getpid, ()) # runs in *only* one process
    print(res.get(timeout=1))              # prints the PID of that process

    # launching multiple evaluations asynchronously *may* use more processes
    multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
    print([res.get(timeout=1) for res in multiple_results])

    # make a single worker sleep for 10 secs
    res = pool.apply_async(time.sleep, (10,))
    try:
        print(res.get(timeout=1))  #raise TimeOut
    except TimeoutError:
        print("We lacked patience and got a multiprocessing.TimeoutError")
        
      