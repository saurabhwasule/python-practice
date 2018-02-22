from __future__ import print_function
from Queue     import Queue     
import threading
import time

'''
What can be pickled 
-None, True, and False
-integers, long integers, floating point numbers, complex numbers
-normal and Unicode strings
-tuples, lists, sets, and dictionaries containing only picklable objects
-functions defined at the top level of a module
-built-in functions defined at the top level of a module
-classes that are defined at the top level of a module
-instances of such classes whose __dict__ or the result of calling __getstate__() is picklable 
'''

class A(object):
    def __init__(self, v):
        self.value = v 
    def __repr__(self):
        return "A(%d)" % (self.value,)
    def __str__(self):
        return "A(%d)" % (self.value,)



def worker(q, lck, ev):
    while True:
        item = q.get()
        with lck:
            print(threading.current_thread().getName(), item, " working...")            
        time.sleep(2)
        with lck:
            print(threading.current_thread().getName(), "Done..") 
        q.task_done()  #for last call, whole process exists , so this hsould be last line of woker thread
                 

#Main thread 		
que = Queue()
lck = threading.RLock()
ev = threading.Event()
for i in range(2):
    t = threading.Thread(target=worker, args=(que,lck, ev))
    t.daemon = True #make them demon such that it exists without main 
    t.start()

lst = [42, None, 'hello', {'from': 'queue'}, A(20), [1,2,3] ]
for l in lst:
    que.put(l)  #Only one thread get one item 

    
que.join()       # block until all tasks are done


