#examples will not work in the interactive interpreter
#install in py2, #pip install futures
from __future__ import print_function
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from multiprocessing import    current_process 
          
#can not use decorator as it needs to pickle 
def is_prime(n):      
    import math
    if n % 2 == 0:	return False
    sqrt_n = int(math.sqrt(n))
    a = [1 for i in range(3, sqrt_n + 1, 2) if n % i == 0]
    return False if sum(a) > 0 else True
    
def is_prime_tuple(n):      
    import math
    if n % 2 == 0:	return (n, False)
    sqrt_n = int(math.sqrt(n))
    a = [1 for i in range(3, sqrt_n + 1, 2) if n % i == 0]
    return (n, False) if sum(a) > 0 else (n, True)
    
#Calling Executor or Future methods from (is_prime)
#to a ProcessPoolExecutor will result in deadlock.
#so, don't pass a future into is_prime, call get on that 
if __name__ == '__main__':
    primes = range(3,99)
    executor = ProcessPoolExecutor(max_workers=10)
    for number, prime in zip(primes, executor.map(is_prime, primes)):
        print('%d is prime: %s' % (number, prime))
        
    #submit returns future     
    result = [executor.submit(is_prime_tuple, prime) for prime in primes ]
    #wait for each future 
    for res in as_completed(result):
        print(res.result())
    
    #or using callback 
    result = [executor.submit(is_prime_tuple, prime) for prime in primes ]
    #wait for each future 
    for res in result:
        res.add_done_callback(lambda v: print(v.result()))
        
    executor.shutdown()