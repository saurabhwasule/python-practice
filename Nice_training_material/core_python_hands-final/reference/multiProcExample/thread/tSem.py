from __future__ import print_function
import threading
import time                    
import random

#A semaphore manages an internal counter which is decremented by each acquire() call 
#and incremented by each release() call. 
#The counter can never go below zero; when acquire() finds that it is zero, it blocks, 
#waiting until some other thread calls release().


def worker(s, lck):
    with lck:
        print(threading.current_thread().getName(),' Waiting to join the pool '  )
    time.sleep(random.randrange(10))    
    with s:
        with lck:
            print( threading.current_thread().getName(), ' Got access ')
        time.sleep(random.randrange(5)) #work on shared resource
    #otherwork 
    time.sleep(1)


s = threading.Semaphore(2) #at a time only 2 can access
lck = threading.RLock()
for i in range(4):
	t = threading.Thread(target=worker, name=("Thread "+str(i)), args=(s,lck))
	t.start()
    
#do other work 
time.sleep(10)