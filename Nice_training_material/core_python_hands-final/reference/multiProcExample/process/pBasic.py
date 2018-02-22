from __future__ import print_function
from multiprocessing import Process, RLock
import time

def f(lck, i):
    with lck:
        print('hello world ', i)
    time.sleep(1)

if __name__ == '__main__':
    lock = RLock()
    procs = []
    for num in range(10):
        p = Process(target=f, args=(lock, num))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()
        
       