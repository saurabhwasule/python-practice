from __future__ import print_function
from multiprocessing import Process, Value, Array, RLock, Manager, Event


def sharedf(n, a, ev):
    n.value = 3.1415927
    for i in range(len(a)):
        a[i] = -a[i]
    ev.set()  #wake up main 
    
     
def managerf(d, l, ev): #d = dict, l = list 
    d[1] = '1'
    d['2'] = 2
    d[0.25] = None
    l.reverse()
    ev.set()    #wake up main 
    
  

if __name__ == '__main__':
    event = Event()  #flag is false
    lock = RLock()
    #Data can be stored in a shared memory map using Value or Array
    num = Value('d', 0.0)    #typecode from array module, eg int(i), double(d), long(l),
    arr = Array('i', range(10))
    p1 = Process(target=sharedf, args=(num, arr, event))
    p1.start()
    event.wait()   #or call event.is_set() 
    with lock:
        print(num.value)
        print(arr[:])
  
    #Manager 
    #controls a server process which holds Python objects 
    #and allows other processes to manipulate them using proxies
    #supports list, dict, Namespace,
    #Lock, RLock, Semaphore, BoundedSemaphore, Condition, Event, Queue, 
    #Value and Array     
    manager = Manager()
    event2 = manager.Event()
    d = manager.dict()
    l = manager.list(range(10))
    p2 = Process(target=managerf, args=(d, l, event2))
    p2.start()
    event2.wait()   #or call event.is_set() 
    with lock:
        print(d)
        print(l)        
    p1.join()
    p2.join()
