from __future__ import print_function
from multiprocessing import Process, Queue, Pipe, RLock

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

def queuef(q, obj):
    q.put([42, None, 'hello', {'from': 'queue'}, obj])
    
def pipef(conn, obj):
    conn.send([42, None, 'hello', {'from': 'pipe'}, obj])
    conn.close()
    
if __name__ == '__main__':
    #lck
    lck = RLock()
    #A
    a = A(20)
    #Queue
    q = Queue()
    p = Process(target=queuef, args=(q, a))
    p.start()
    with lck:
        print(q.get())    # prints "[42, None, 'hello']"
    p.join()
    #Pipe
    parent_conn, child_conn = Pipe()
    p = Process(target=pipef, args=(child_conn, a))
    p.start()
    with lck:
        print(parent_conn.recv())   # prints "[42, None, 'hello']"
    p.join()