from __future__ import print_function
from multiprocessing import Process, Event, RLock, Queue
import time


class PPng(Process):
    def __init__(self, type, sq, rq, plck):
        Process.__init__(self)
        self.type = type
        self.sq = sq 
        self.rq = rq
        self.plck = plck 
        self._stop = Event()
        self.daemon=True
    def stop(self):
        self._stop.set()
    def stopped(self):
        return self._stop.is_set()
    def run(self):
        while not self.stopped():
            self.sq.put(self.type)
            got = self.rq.get()
            with self.plck:
                print("From %s got %s" % (self.type, got) )
            time.sleep(1)
            
if __name__ == '__main__':           
    sendQ = Queue()
    rcvQ = Queue()
    printLock = RLock()           
    ping = PPng("Ping", sendQ, rcvQ, printLock)
    pong = PPng("Pong", rcvQ, sendQ, printLock)
    #start ping and Pong 

    ping.start()
    pong.start()
    time.sleep(10)
    ping.stop()
    pong.stop()