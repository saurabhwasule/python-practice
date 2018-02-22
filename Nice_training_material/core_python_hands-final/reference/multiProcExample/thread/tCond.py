from __future__ import print_function
import threading
import time
                    
import random

shared_var = 0
available = False #this is required as wait() may return spuriousely


#wait, notify and notifyAll must be called inside with block 
#wait() method releases the lock, and then blocks until it is awakened by a notify() or notifyAll() call
#Once awakened, it re-acquires the lock and returns

def consumer(cond):
    global available
    print('Starting ' + threading.current_thread().getName())
    got_val = None 
    with cond:		
        while not available:       #Py3.x: cond.wait_for(lambda : available)
            cond.wait()
        got_val = shared_var
        print('[Consumer] Got Resource ', threading.current_thread().getName(), shared_var)  #access global shared_var
        available = False
    #work with got_val 
    time.sleep(1)

#notify() and notifyAll() methods donot release the lock;
#lock is released only when with block ends 
def producer(cond, max):	
    global shared_var                #must as it sets global
    global available
    for i in range(20):
        with cond:				
            shared_var = random.randint(20,100);
            print('Notify ',  threading.current_thread().getName(), shared_var)
            available = True
            cond.notify()			#for one consumer waking up, use notifyAll() for all waking up 
            time.sleep(1) 	#Note, consumer would be awakened after 'with scope'	
        time.sleep(1)   #some other work, 


#ctor can take Lock or Rlock instance
#if None, then createa a default one 
#(Passing one in is useful when several condition variables must share the same lock.)
condition = threading.Condition() 
for i in range(10):
	w = threading.Thread(target=consumer, args=(condition,))
	w.start()

p = threading.Thread(name='Producer', target=producer, args=(condition,10))
p.start()
p.join()