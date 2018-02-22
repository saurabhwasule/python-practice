from __future__ import print_function
from threading import Thread, RLock, Timer
import time

def f(lck, i):
    with lck:
        print('hello world ', i)
    time.sleep(1)
   
#Timer    
def hello(lck):	
    with lck:
        print("hello, world")



if __name__ == '__main__':
    lock = RLock()
    
    t = Timer(5.0, hello, args=(lock,))
    t.start()
    
    procs = []
    for num in range(10):
        p = Thread(target=f, args=(lock, num))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()
        
       