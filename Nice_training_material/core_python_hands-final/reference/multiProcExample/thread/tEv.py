from __future__ import print_function
import threading
import time
                    
def wait_for_event(e, lck):
    event_is_set = e.wait()
    with lck:
        print(threading.current_thread().getName() + " processing event")
    #process the event 
    time.sleep(1)
    with lck:
        print( threading.current_thread().getName() + " Done")

def wait_for_event_timeout(e, t, lck):
    while True:
        event_is_set = e.wait(t)  #returns internal flag , returns false when timeout, but not set() called
        if event_is_set:
            with lck:
                print(threading.current_thread().getName() , 'processing event')
            time.sleep(1)  #do 
            #exit
            return
        else:
            with lck:
                print(threading.current_thread().getName(), 'doing other work')
            time.sleep(1) #do
#do some work till event is set 
def while_isset(e, lck):
    while not e.is_set():        
        with lck:
            print(threading.current_thread().getName(), 'doing other work')
        time.sleep(1) #do

e = threading.Event()
lck = threading.RLock()
t1 = threading.Thread(name='block', target=wait_for_event, args=(e,lck))
t1.start()

t2 = threading.Thread(name='non-block', target=wait_for_event_timeout, args=(e, 0.5, lck))
t2.start()

t3 = threading.Thread(name='while_isset', target=while_isset, args=(e, lck))
t3.start()

time.sleep(3)
e.set()
# both threads are awakened
t1.join();t2.join();t3.join()