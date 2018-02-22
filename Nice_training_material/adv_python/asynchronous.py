###Coroutine 
#Functions which exist together with main function in the same thread 
#primitive implementation 
##Coroutine style - single thread
def search_file(filename):
    print('Searching file %s' % (filename))
    my_file = open(filename, 'r')
    file_content = my_file.read()
    my_file.close()
    while True:
        search_result = 0
        search_text = (yield search_result)        
        search_result = file_content.count(search_text)
        print('Number of matches: %d' % (search_result))


        
search = search_file("first.py")
next(search)  #start it 
search.send('import') #yield would return this 
search.close()  #close the coroutine 



###Asyncio - asynchronous io
#get a event loop and run it, all Tasks would be executed  
#has coroutine, task, future, transport proptocol and subprocess related functionality

import asyncio

##features
1.a pluggable event loop with various system-specific implementations;
2.transport and protocol abstractions (similar to those in Twisted);
3.concrete support for TCP, UDP, SSL, subprocess pipes, delayed calls, and others (some may be system-dependent);
4.a Future class that mimics the one in the concurrent.futures module, 
  but adapted for use with the event loop;
5.coroutines and tasks based on yield from , to help write concurrent code in a sequential fashion;
6.cancellation support for Futures and coroutines;
7.synchronization primitives for use between coroutines in a single thread, 
  mimicking those in the threading module;
8.an interface for passing work off to a threadpool, for times when you absolutely, positively have to use a library that makes blocking I/O calls.


##Most asyncio functions don’t accept keywords based arg passing 
#use functools.partial(). 
#For example, 
loop.call_soon(functools.partial(print, "Hello", flush=True)) 
#will call print("Hello", flush=True).


##Enabling the Debug 
1.Enable the asyncio debug mode globally by setting the environment 
  variable PYTHONASYNCIODEBUG to 1, or by calling AbstractEventLoop.set_debug().
2.Set the log level of the asyncio logger to logging.DEBUG. 
  For example, call 
  logging.basicConfig(level=logging.DEBUG) at startup.
  Default log level for the asyncio module is logging.INFO
  To change 
  logging.getLogger('asyncio').setLevel(logging.WARNING)
3.Configure the warnings module to display ResourceWarning warnings. 
  For example, use the -Wdefault command line option of Python to display them.

#Examples debug checks:
•Log coroutines defined but never “yielded from”
•call_soon() and call_at() methods raise an exception 
 if they are called from the wrong thread.
•Log the execution time of the selector
•Log callbacks taking more than 100 ms to be executed. 
 The AbstractEventLoop.slow_callback_duration attribute is the minimum duration in seconds of “slow” callbacks.
•ResourceWarning warnings are emitted 
 when transports and event loops are not closed explicitly.

 
 
 
##Available event loops
class asyncio.SelectorEventLoop #for Windows, Others 
    Event loop based on the selectors module
    Use the most efficient selector available on the platform.
    On Windows
        •SelectSelector is used which only supports sockets and is limited to 512 sockets.
        •add_reader() and add_writer() only accept file descriptors of sockets
        •Pipes are not supported (ex: connect_read_pipe(), connect_write_pipe())
        •Subprocesses are not supported (ex: subprocess_exec(), subprocess_shell())

class asyncio.ProactorEventLoop #for Windows
    Proactor event loop for Windows using “I/O Completion Ports” aka IOCP
    Supports Subprocesses and pipes
        •create_datagram_endpoint() (UDP) is not supported
        •add_reader() and add_writer() are not supported

import asyncio, sys
if sys.platform == 'win32':
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)

    
    


 




###Asyncio - Get a event loop
loop = asyncio.get_event_loop()


##Run an event loop

loop.run_forever()
    Run until stop() is called. 
    If stop() is called before run_forever() is called, 
    already scheduled callbacks would run if corresponding IO happened

loop.run_until_complete(future_or_coroutine_or_task)
    Run until the Future or coroutine
    Return the Future’s result, or raise its exception.


loop.is_running()
loop.stop()
loop.is_closed()
loop.close()

coroutine loop.shutdown_asyncgens()
    Schedule all currently open asynchronous generator objects to close with an aclose() call
    try:
        loop.run_forever()
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


##Calling a method/callback 
        
loop.call_soon(callback, *args) #args for positional arguments to callback 
    Arrange for a callback to be called as soon as possible(callback Q is FIFO)
    Returns asyncio.Handle (can be cancelled)
    Use functools.partial to pass keywords to the callback.
    loop.call_soon(functools.partial(print, "Hello", flush=True)) 
        
loop.call_soon_threadsafe(callback, *args) #args for positional arguments to callback 
    Like call_soon(), but thread safe.
    Returns asyncio.Handle (can be cancelled)
    Use functools.partial to pass keywords to the callback.
    loop.call_soon_threadsafe(functools.partial(print, "Hello", flush=True)) 

#helper class 
class asyncio.Handle
    cancel()
        Cancel the call
##Delayed calls
loop.call_later(delay, callback, *args) #args for positional arguments to callback 
    Arrange for the callback to be called after the given delay seconds (either an int or float).
    Returns asyncio.Handle (can be cancelled)
    Use functools.partial to pass keywords to the callback.#For example, 
    loop.call_later(60, functools.partial(print, "Hello", flush=True)) 

loop.call_at(when, callback, *args) #args for positional arguments to callback 
    'when' (int/float) must be as per loop.time() returns 
    Returns asyncio.Handle (can be cancelled)
    Use functools.partial to pass keywords to the callback.
    #For example, 
    loop.call_at(60,functools.partial(print, "Hello", flush=True)) 

loop.time()
    Return the current time, as a float value

coroutine asyncio.sleep(sleeptime)
    Sleeps that many int/float time



##UNIX signals(not supported on Windows)
#check signum from https://docs.python.org/3/library/signal.html#module-contents
loop.add_signal_handler(signum, callback, *args)  #args for positional arguments to callback
    Use functools.partial to pass keywords to the callback.
loop.remove_signal_handler(sig)
    Return True if a signal handler was removed, False if not.

    
##Executing external blocking function 
#Use external ThreadPoolExecutor(by default) or ProcessPoolExecutor
#or pass executor=None for default executor

#Note inside coroutine use  yield from <<below method>> or await <<below method>>
#but outside , call loop.run_until_complete(<<below_method>>) as yield from/await works only inside coroutine 


coroutine loop.run_in_executor(executor, func, *args)  #args for positional arguments to callback
    Use functools.partial to pass keywords to the *func
    It's a coroutine, hence use below to get result 
    result=await coroutine 
    #or 
    result=yield from coroutine
    
    
loop.set_default_executor(executor)
    Set the default executor used by run_in_executor()

##Error Handling API
loop.set_exception_handler(handler)  
    handler_function(loop, context)
loop.get_exception_handler()
    Return the exception handler, or None if the default one is in use
loop.default_exception_handler(context) 
    Default exception handler.
loop.call_exception_handler(context)  
    Call the current event loop exception handler

#context is a dict object containing the following keys 
    •‘message’: Error message;
    •‘exception’ (optional): Exception object;
    •‘future’ (optional): asyncio.Future instance;
    •‘handle’ (optional): asyncio.Handle instance;
    •‘protocol’ (optional): Protocol instance;
    •‘transport’ (optional): Transport instance;
    •‘socket’ (optional): socket.socket instance.

##Debug mode
loop.get_debug()
    Get the debug mode (bool) of the event loop.
loop.set_debug(enabled: bool)
    Set the debug mode of the event loop.


###Asyncio - Coroutine 
#Coroutines used with asyncio may be implemented using the async def (Py3.5)
#or by using generators(@asyncio.coroutine)

#Things a coroutine can do:
• result = await future or result = yield from future 
  suspends the coroutine until the future is done, 
  then returns the future’s result, or raises an exception, which will be propagated. 
  (If the future is cancelled, it will raise a CancelledError exception.) 
  Note that tasks are futures, and everything said about futures also applies to tasks.
• result = await coroutine or result = yield from coroutine 
  wait for another coroutine to produce a result 
  (or raise an exception, which will be propagated). 
  The coroutine expression must be a call to another coroutine.
• return expression 
  produce a result to the coroutine that is waiting for this one using await or yield from.
• raise exception 
  raise an exception in the coroutine that is waiting for this one using await or yield from.

#Calling a coroutine does not start its code running 
#the coroutine object returned by the call doesn’t do anything 
#until you schedule its execution. 

##Starting coroutine
#call 
await coroutine 
#or 
yield from coroutine #from another coroutine (assuming the other coroutine is already running!), 
#or schedule its execution using the 
asyncio.ensure_future(coroutine(any_future_arg_for_returning_result)) #returns Task 
#or 
task = loop.create_task() 


##Concurrency and multithreading

#An event loop runs in a thread 
#and executes all callbacks and tasks in the same thread
#While a task is running in the event loop, no other task is running in the same thread. 
#But when the task uses yield from, 
#the task is suspended and the event loop executes the next task

#To schedule a callback from a different thread,
loop.call_soon_threadsafe(callback, *args)

#Most asyncio objects are not thread safe. 
#You should only worry if you access objects outside the event loop
#for example to cancel future in another thread
loop.call_soon_threadsafe(fut.cancel)

#To schedule a coroutine object from a different thread
future = asyncio.run_coroutine_threadsafe(coro_func(), loop) #returns concurrent.futures.Future 
result = future.result(timeout)  # Wait for the result with a timeout

#to execute a callback in different thread to not block the thread of the event loop.
executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
loop.run_in_executor(executor, func, *args) #a coroutine, hence to get result, use await or yield from 



##Handle blocking functions correctly
#Blocking functions should not be called directly

#For networking and subprocesses,use 
class asyncio.Protocol
    The base class for implementing streaming protocols 
    (for use with e.g. TCP and SSL transports).
class asyncio.DatagramProtocol
    The base class for implementing datagram protocols 
    (for use with e.g. UDP transports).
class asyncio.SubprocessProtocol
    The base class for implementing protocols communicating 
    with child processes (through a set of unidirectional pipes).



#An executor can be used to run a task in a different thread 
#or even in a different process, to not block the thread of the event loop
executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
loop.run_in_executor(executor, func, *args) #it's coroutine

#to run a method later use 
loop.call_later(delay, callback, *args)


##Detect coroutine objects never scheduled
#When a coroutine function is called and its result is not passed to ensure_future() 
#or to the loop.create_task() method, 
#the execution of the coroutine object will never be scheduled which is probably a bug


import asyncio

@asyncio.coroutine
def test():
    print("never scheduled")

test()

#Output in debug mode:
Coroutine test() at test.py:3 was never yielded from
Coroutine object created at (most recent call last):
  File "test.py", line 7, in <module>
    test()

    
    
##Detect exceptions never consumed
import asyncio

@asyncio.coroutine
def bug():
    raise Exception("not consumed")

loop = asyncio.get_event_loop()
asyncio.ensure_future(bug())
loop.run_forever()
loop.close()

#Output in debug mode:
Task exception was never retrieved
future: <Task finished coro=<bug() done, defined at test.py:3> exception=Exception('not consumed',) created at test.py:8>
source_traceback: Object created at (most recent call last):
  File "test.py", line 8, in <module>
    asyncio.ensure_future(bug())

#The first option is to chain the coroutine in another coroutine 
#and use classic try/except:
@asyncio.coroutine
def handle_exception():
    try:
        yield from bug()
    except Exception:
        print("exception consumed")

loop = asyncio.get_event_loop()
asyncio.ensure_future(handle_exception())
loop.run_forever()
loop.close()


#Another option is to use the loop.run_until_complete() function:
task = asyncio.ensure_future(bug())
try:
    loop.run_until_complete(task)
except Exception:
    print("exception consumed")

    
    
    
##Chain coroutines correctly
#When a coroutine function calls other coroutine functions and tasks, 
#they should be chained explicitly with yield from. 
#Otherwise, the execution is not guaranteed to be sequential.

import asyncio

@asyncio.coroutine
def create():
    yield from asyncio.sleep(3.0)
    print("(1) create file")

@asyncio.coroutine
def write():
    yield from asyncio.sleep(1.0)
    print("(2) write into file")

@asyncio.coroutine
def close():
    print("(3) close file")

@asyncio.coroutine
def test():
    asyncio.ensure_future(create())
    asyncio.ensure_future(write())
    asyncio.ensure_future(close())
    yield from asyncio.sleep(2.0)
    loop.stop()

loop = asyncio.get_event_loop()
asyncio.ensure_future(test())
loop.run_forever()
print("Pending tasks at exit: %s" % asyncio.Task.all_tasks(loop))
loop.close()


#Expected output:
(1) create file
(2) write into file
(3) close file
Pending tasks at exit: set()


#Actual output:
(3) close file
(2) write into file
Pending tasks at exit: {<Task pending create() at test.py:7 wait_for=<Future pending cb=[Task._wakeup()]>>}
Task was destroyed but it is pending!
task: <Task pending create() done at test.py:5 wait_for=<Future pending 

#To fix the example, tasks must be marked with yield from:
@asyncio.coroutine
def test():
    yield from asyncio.ensure_future(create())
    yield from asyncio.ensure_future(write())
    yield from asyncio.ensure_future(close())
    yield from asyncio.sleep(2.0)
    loop.stop()


#Or without asyncio.ensure_future():

@asyncio.coroutine
def test():
    yield from create()
    yield from write()
    yield from close()
    yield from asyncio.sleep(2.0)
    loop.stop()



##Pending task destroyed
#If a pending task is destroyed, 
#the execution of its wrapped coroutine did not complete. 
#It is probably a bug and so a warning is logged.

#Example of log:
Task was destroyed but it is pending!
task: <Task pending coro=<kill_me() done, defined at test.py:5> wait_for=<Future pending cb=[Task._wakeup()]>>


#Enable the debug mode of asyncio to get the traceback where the task was created.
Task was destroyed but it is pending!
source_traceback: Object created at (most recent call last):
  File "test.py", line 15, in <module>
    task = asyncio.ensure_future(coro, loop=loop)
task: <Task pending coro=<kill_me() done, defined at test.py:5> wait

##Close transports and event loops
#When a transport is no more needed, 
#call its close() method to release resources. 
#Event loops must also be closed explicitly.

#If a transport or an event loop is not closed explicitly, 
#a ResourceWarning warning will be emitted in its destructor in debug mode 



###Asyncio - Future 
#Usecase is synonomous with coroutine, but coroutine can return a result via Future 
#Note we have another Future, concurrent.futures.Future (not suitable for eventloop)
# All Future mentioned here is asyncio.Future 

#This future is not threadsafe as eventloop occurs in only one thread 
#hence for cancelling this future from anothor thread, 
loop.call_soon_threadsafe(fut.cancel)
 
#async.Future is used for result passing from coroutine
#which can be used for waiting in loop.run_until_complete(future)
#or result = await future or result = yield from future inside a coroutine 


#creation
#Note inside coroutine use  yield from <<below method>> or await <<below method>>
#but outside , call loop.run_until_complete(<<below_method>>) as yield from/await works only inside coroutine 


future = loop.create_future()  #returns asyncio.Future 
#or
future = asyncio.Future()
#attach this future to a co-routine which does set_result() or raise exception
async def slow_operation(future):
    await asyncio.sleep(1)
    future.set_result('Future is done!')
    
#schedules coroutine 
asyncio.ensure_future(slow_operation(future)) #returns Task , which can be cancelled etc

#Wait for future result
loop.run_until_complete(future)
print(future.result())
loop.close()
#or 
future.add_done_callback(got_result)
try:
    loop.run_forever()
finally:
    loop.close()
#Or inside a coroutine 
result = await future 
#Or
result = yield from future 
#reference
class asyncio.Future(*, loop=None)
    cancel()
        Cancel the future and schedule callbacks
    cancelled()
        Return True if the future was cancelled.
    done()
        Return True if the future is done.
        Done means either that a result / exception are available, 
        or that the future was cancelled.
    result()
        Return the result this future represents.
    exception()
        Return the exception that was set on this future
    add_done_callback(fn)
        Add a callback to be run when the future becomes done.
        Use functools.partial to pass parameters to the callback
    remove_done_callback(fn)
        Remove all instances of a callback from the “call when done” list.
    set_result(result)
        Mark the future done and set its result.
    set_exception(exception)
        Mark the future done and set an exception.


###Asyncio - Tasks
#A task is a subclass of Future
#A task(Future) is responsible for executing a coroutine object in an event loop
#ie schedules the future or coroutine when loop gets executed by loop.run_forever() etc 

#creation
#Note inside coroutine use  yield from <<below method>> or await <<below method>>
#but outside , call loop.run_until_complete(<<below_method>>) as yield from/await works only inside coroutine 
asyncio.ensure_future(coro_or_future, *, loop=None)
    Return a Task object.(schedules coro_or_future for execution )

asyncio.async(coro_or_future, *, loop=None)
    A deprecated alias to ensure_future().

asyncio.wrap_future(future, *, loop=None)
    Wrap a concurrent.futures.Future object in a Future object.
    
asyncio.gather(*coros_or_futures, loop=None, return_exceptions=False)
    Return a future aggregating list of coroutine objects or futures
    Once completed, that result of future is list of results in original order

loop.create_task(coroutine) #Py3.4.2
    Wrap it in a future. Return a Task object.
    Use the asyncio.async(coroutine)  in older Python versions 
    
#Wait for Task result
loop.run_until_complete(task)
print(task.result())
loop.close()
#or 
task.add_done_callback(got_result)
try:
    loop.run_forever()
finally:
    loop.close()
#Or inside a coroutine 
result = await task 
#Or
result = yield from task 


#Reference 
class asyncio.Task(coro, *, loop=None)
    Subclass of Future 
    Calling cancel() will throw a CancelledError to the wrapped coroutine. 
    cancelled() only returns True if the wrapped coroutine did not catch the CancelledError exception, 
    or raised a CancelledError exception.
    This class is not thread safe.
    #other methods 
    classmethod all_tasks(loop=None)#None means default loop
        Return a set of all tasks for an event loop.
    classmethod current_task(loop=None)
        Return the currently running task in an event loop or None.
    cancel()
        Request that this task cancel itself.
    get_stack(*, limit=None)
        Return the list of stack frames for this task’s coroutine.
    print_stack(*, limit=None, file=None)
        Print the stack or traceback for this task’s coroutine.



##Other Task/Future helper functions
#the optional loop argument allows explicitly setting the event loop object used by the underlying task or coroutine. 
#If it’s not provided, the default event loop is used

#Note inside coroutine use  yield from <<below method>> or await <<below method>>
#but outside , call loop.run_until_complete(<<below_method>>) as yield from/await works only inside coroutine 

asyncio.as_completed(list_futures_or_coroutines, *, loop=None, timeout=None)
    Return an iterator whose values, when waited for, are Future instances
    #Example, in a coroutine 
    for f in as_completed(fs):
        result = yield from f  # The 'yield from' may raise
        # Use result


coroutine asyncio.wait_for(single_future_or_coroutine, timeout, *, loop=None)
    Returns result of the Future or coroutine
    To avoid the task cancellation, wrap it in shield().
    result = yield from asyncio.wait_for(fut, 60.0)

    
coroutine asyncio.wait(list_of_futures_or_coroutines, *, loop=None, timeout=None, return_when=ALL_COMPLETED)
    return_when can be  FIRST_COMPLETED, FIRST_EXCEPTION, ALL_COMPLETED 
    Returns two sets of Future: (done, pending).
    #Example 
    done, pending = yield from asyncio.wait(fs)
    result = yield from asyncio.wait_for(fut, 60.0)


coroutine asyncio.sleep(delay, result=None, *, loop=None)
    Sleeps delay(seconds)
    Result is result from 'yield from'
    yield from asyncio.sleep(2.0)
    
asyncio.shield(coroute_future_task, arg, *, loop=None)  
    Wait for a future, shielding it from cancellation.
    res = yield from shield(something())
    #or handle exception 
    try:
        res = yield from shield(something())
    except CancelledError:
        res = None

    
asyncio.iscoroutine(obj)
    Return True if obj is a coroutine object, 
    which may be based on a generator or an async def coroutine.
    
asyncio.iscoroutinefunction(func)
    Return True if func is determined to be a coroutine function, 
    which may be a decorated generator function or an async def function.
    
asyncio.run_coroutine_threadsafe(coro, loop)
    Submit a coroutine object to a given event loop.
    Requires the loop argument to be passed explicitly.
    Return a concurrent.futures.Future to access the result.

    This function is meant to be called from a different thread 
    than the one where the event loop is running. 
    # Create a coroutine
    coro = asyncio.sleep(1, result=3)
    # Submit the coroutine to a given loop
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    # Wait for the result with an optional timeout argument
    assert future.result(timeout) == 3

    If an exception is raised in the coroutine, 
    the returned future will be notified.     
    It can also be used to cancel the task in the event loop:
    try:
        result = future.result(timeout)
    except asyncio.TimeoutError:
        print('The coroutine took too long, cancelling the task...')
        future.cancel()
    except Exception as exc:
        print('The coroutine raised an exception: {!r}'.format(exc))
    else:
        print('The coroutine returned: {!r}'.format(result))






###Asyncio - Example 

##Basic 
import asyncio

async def hello_world():
    print("Hello World!")

loop = asyncio.get_event_loop()
# Blocking call which returns when the hello_world() coroutine is done
loop.run_until_complete(hello_world())
loop.close()


##Hello World with call_soon()

import asyncio

def hello_world(loop):
	print('Hello World')
	loop.stop()             #must

loop = asyncio.get_event_loop()

# Schedule a call to hello_world()
loop.call_soon(hello_world, loop)   #schedule a call , but actually runs when run_forever is called

# hangs forevere, , only can be interrupted by loop.stop() as inside hello_word
loop.run_forever()
loop.close()


##Coroutine displaying the current date every second during 5 seconds using the sleep() function:

import asyncio
import datetime

async def display_date(loop):
    end_time = loop.time() + 5.0
    while True:
        print(datetime.datetime.now())
        if (loop.time() + 1.0) >= end_time:
            break
        await asyncio.sleep(1)

loop = asyncio.get_event_loop()
# Blocking call which returns when the display_date() coroutine is done
loop.run_until_complete(display_date(loop))
loop.close()



##Display the current date with call_later()
import asyncio
import datetime

def display_date(end_time, loop):
    print(datetime.datetime.now())
    if (loop.time() + 1.0) < end_time:
        loop.call_later(1, display_date, end_time, loop)
    else:
        loop.stop()

loop = asyncio.get_event_loop()

# Schedule the first call to display_date()
end_time = loop.time() + 5.0
loop.call_soon(display_date, end_time, loop)

# Blocking call interrupted by loop.stop()
loop.run_forever()
loop.close()

##Chain coroutines

import asyncio

async def compute(x, y):
    print("Compute %s + %s ..." % (x, y))
    await asyncio.sleep(1.0)
    return x + y
    
@asyncio.coroutine
def print_sum(x, y):
    result = yield from compute(x, y)
    print("%s + %s = %s" % (x, y, result))

loop = asyncio.get_event_loop()
loop.run_until_complete(print_sum(1, 2))
loop.close()

##Future with run_until_complete()
import asyncio

async def slow_operation(future):
    await asyncio.sleep(1)
    future.set_result('Future is done!')

loop = asyncio.get_event_loop()
future = asyncio.Future()
asyncio.ensure_future(slow_operation(future))
loop.run_until_complete(future)
print(future.result())
loop.close()


##Future with run_forever()
import asyncio

async def slow_operation(future):
    await asyncio.sleep(1)
    future.set_result('Future is done!')

def got_result(future):
    print(future.result())
    loop.stop()

loop = asyncio.get_event_loop()
future = asyncio.Future()
asyncio.ensure_future(slow_operation(future))
future.add_done_callback(got_result)
try:
    loop.run_forever()
finally:
    loop.close()

    
##Parallel execution of tasks
import asyncio

async def factorial(name, number):
    f = 1
    for i in range(2, number+1):
        print("Task %s: Compute factorial(%s)..." % (name, i))
        await asyncio.sleep(1)
        f *= i
    print("Task %s: factorial(%s) = %s" % (name, number, f))

loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.gather(
    factorial("A", 2),
    factorial("B", 3),
    factorial("C", 4),
))
loop.close()


###Asyncio - Transports and protocols (callback based API) - Low level 

#Transports are classes to abstract various kinds of communication channels

#protocol_factory must be a callable returning a protocol instance.
coroutine loop..create_connection(protocol_factory, host=None, port=None, *, 
    ssl=None, family=0, proto=0, flags=0, sock=None, local_addr=None, 
    server_hostname=None)
    Returns a (transport, protocol) pair.

coroutine loop.create_server(protocol_factory, host=None, port=None, *, 
    family=socket.AF_UNSPEC, flags=socket.AI_PASSIVE, sock=None, backlog=100, 
    ssl=None, reuse_address=None, reuse_port=None)
    

coroutine loop.create_datagram_endpoint(protocol_factory, local_addr=None, 
    remote_addr=None, *, family=0, proto=0, flags=0, reuse_address=None, 
    reuse_port=None, allow_broadcast=None, sock=None)
    Returns a (transport, protocol) pair.

coroutine loop.connect_accepted_socket(protocol_factory, sock, *, ssl=None)
    Handle an accepted connection.
    This is used by servers that accept connections outside of asyncio 
    but that use asyncio to handle them.
    the coroutine returns a (transport, protocol) pair.

##Extrmely low level routine, Donot use these, Use Protocole or Stream based approach  
coroutine loop.sock_recv(sock, nbytes)
    Receive data from the socke
    With SelectorEventLoop event loop, the socket sock must be non-blocking
    With SelectorEventLoop event loop, the socket sock must be non-blocking
    
coroutine loop.sock_sendall(sock, data)
    Send data to the socket
    The socket must be connected to a remote socket 
    None is returned on success. On error, an exception is raised
    With SelectorEventLoop event loop, the socket sock must be non-blocking


coroutine loop.sock_connect(sock, address)
    Connect to a remote socket at address
    With SelectorEventLoop event loop, the socket sock must be non-blocking


coroutine loop.sock_accept(sock)
    Accept a connection
    The socket must be bound to an address and listening for connections.
    The return value is a pair (conn, address) 
    where conn is a new socket object usable to send and receive data on the connection, 
    and address is the address bound to the socket on the other end of the connection.
    The socket sock must be non-blocking.

coroutine loop.getaddrinfo(host, port, *, family=0, type=0, proto=0, flags=0)
    This method is a coroutine, similar to socket.getaddrinfo() function but non-blocking.

coroutine loop.getnameinfo(sockaddr, flags=0)
    This method is a coroutine, similar to socket.getnameinfo() function but non-blocking.

coroutine loop.connect_read_pipe(protocol_factory, pipe)
    Register read pipe in eventloop.
    protocol_factory should instantiate object with Protocol interface. 
    Return pair (transport, protocol), 
    where transport supports the ReadTransport interface.
    With SelectorEventLoop event loop, the pipe is set to non-blocking mode.
    On Windows Use ProactorEventLoop 
    
coroutine loop.connect_write_pipe(protocol_factory, pipe)
    Register write pipe in eventloop.
    protocol_factory should instantiate object with BaseProtocol interface. 
    Return pair (transport, protocol), 
    where transport supports the WriteTransport interface.
    With SelectorEventLoop event loop, the pipe is set to non-blocking mode.
    On Windows Use ProactorEventLoop 
    
##Protocols 
#asyncio provides base classes that you can subclass to implement your network protocols
#override certain methods

class asyncio.Protocol
    The base class for implementing streaming protocols 
    (for use with e.g. TCP and SSL transports).
class asyncio.DatagramProtocol
    The base class for implementing datagram protocols (for use with e.g. UDP transports).
class asyncio.SubprocessProtocol
    The base class for implementing protocols communicating with child processes 
    (through a set of unidirectional pipes).

##State machine:
start -> connection_made() [-> data_received() *] [-> eof_received() ?] 
            -> connection_lost() -> end

##Connection callbacks - Protocol, DatagramProtocol and SubprocessProtocol
BaseProtocol.connection_made(transport)
    Called when a connection is made.
BaseProtocol.connection_lost(exc)
    Called when the connection is lost or closed.

#SubprocessProtocol instances:
SubprocessProtocol.pipe_data_received(fd, data)
    Called when the child process writes data into its stdout or stderr pipe. 
    fd is the integer file descriptor of the pipe. 
    data is a non-empty bytes object containing the data.
SubprocessProtocol.pipe_connection_lost(fd, exc)
    Called when one of the pipes communicating with the child process is closed. 
    fd is the integer file descriptor that was closed.
SubprocessProtocol.process_exited()
    Called when the child process has exited.


##Streaming protocols - Protocol
Protocol.data_received(data)
    Called when some data is received. 
Protocol.eof_received()
    Called when the other end signals it won’t send any more data 
    (for example by calling write_eof(), if the other end also uses asyncio).




##Datagram protocols - DatagramProtocol

DatagramProtocol.datagram_received(data, addr)
    Called when a datagram is received. 
    data is a bytes object containing the incoming data. 
    addr is the address of the peer sending the data; 
    the exact format depends on the transport.
DatagramProtocol.error_received(exc)
    Called when a previous send or receive operation raises an OSError. 
    exc is the OSError instance.



##Flow control callbacks - Protocol, DatagramProtocol and SubprocessProtocol
BaseProtocol.pause_writing()
    Called when the transport’s buffer goes over the high-water mark.
BaseProtocol.resume_writing()
    Called when the transport’s buffer drains below the low-water mark.
    pause_writing() and resume_writing() calls are paired 


    
##TCP CLient and server 
#Use Stream based protocol (easier)

##UDP echo client protocol

import asyncio

class EchoClientProtocol:
    def __init__(self, message, loop):
        self.message = message
        self.loop = loop
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport
        print('Send:', self.message)
        self.transport.sendto(self.message.encode())

    def datagram_received(self, data, addr):
        print("Received:", data.decode())

        print("Close the socket")
        self.transport.close()

    def error_received(self, exc):
        print('Error received:', exc)

    def connection_lost(self, exc):
        print("Socket closed, stop the event loop")
        loop = asyncio.get_event_loop()
        loop.stop()

loop = asyncio.get_event_loop()
message = "Hello World!"
#Not Supported on ProactorEventLoop
connect = loop.create_datagram_endpoint(
    lambda: EchoClientProtocol(message, loop),
    remote_addr=('127.0.0.1', 9999))
transport, protocol = loop.run_until_complete(connect)
loop.run_forever()
transport.close()
loop.close()



##UDP echo server protocol

import asyncio

class EchoServerProtocol:
    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        message = data.decode()
        print('Received %r from %s' % (message, addr))
        print('Send %r to %s' % (message, addr))
        self.transport.sendto(data, addr)

loop = asyncio.get_event_loop()
print("Starting UDP server")
# One protocol instance will be created to serve all client requests
#Not Supported on ProactorEventLoop
listen = loop.create_datagram_endpoint(EchoServerProtocol, local_addr=('127.0.0.1', 9999))
transport, protocol = loop.run_until_complete(listen)

try:
    loop.run_forever()
except KeyboardInterrupt:
    pass

transport.close()
loop.close()


    
    
    
    
    
    
###Asynio -Asynchronous Reader and Writer - Stream based(coroutine) - High level 

class asyncio.StreamReader(limit=None, loop=None)
    Not Thread safe 
        coroutine read(n=-1)
        coroutine readline()
        coroutine readexactly(n)
        coroutine readuntil(separator=b’\n’)

        feed_eof()  #Acknowledge the EOF.
        at_eof() #Return True if the buffer is empty and feed_eof() was called.
        exception()
        feed_data(data)  #Feed data bytes in the internal buffer. Any operations waiting for the data will be resumed.
        set_exception(exc)
        set_transport(transport)



class asyncio.StreamWriter(transport, protocol, reader, loop)
    Not Thread safe 
        write(data)
        writelines(data)
        write_eof()

        can_write_eof()  #Return True if the transport supports write_eof(), False if not. 

        close()
        coroutine drain()   
            Let the write buffer of the underlying transport a chance to be flushed
            #use as 
            w.write(data)
            yield from w.drain()

##Wrapper function - High level
#calls internally low level create_connection and create_server)
coroutine asyncio.open_connection(host=None, port=None, *, loop=None, limit=None, **kwds)
    returning a (reader, writer) pair
    The reader returned is a StreamReader instance; the writer is a StreamWriter instance.
    
coroutine asyncio.start_server(client_connected_cb, host=None, port=None, *, loop=None, limit=None, **kwds)
    Start a socket server, with a callback for each client connected
    client_connected_cb parameter _function(client_reader, client_writer)
    client_reader is a StreamReader object, 
    while client_writer is a StreamWriter object

    
    
    
##Example -TCP echo client using streams

import asyncio

@asyncio.coroutine
def tcp_echo_client(message, loop):
    reader, writer = yield from asyncio.open_connection('127.0.0.1', 8888, loop=loop)

    print('Send: %r' % message)
    writer.write(message.encode())

    data = yield from reader.read(100)
    print('Received: %r' % data.decode())

    print('Close the socket')
    writer.close()

message = 'Hello World!'
loop = asyncio.get_event_loop()
loop.run_until_complete(tcp_echo_client(message, loop))
loop.close()


##Example - TCP echo server using streams

import asyncio

@asyncio.coroutine
def handle_echo(reader, writer):
    data = yield from reader.read(100)
    message = data.decode()
    addr = writer.get_extra_info('peername')
    print("Received %r from %r" % (message, addr))

    print("Send: %r" % message)
    writer.write(data)
    yield from writer.drain()  #must wait for writing to happen

    print("Close the client socket")
    writer.close()


loop = asyncio.get_event_loop()
coro = asyncio.start_server(handle_echo, '127.0.0.1', 8888, loop=loop)
server = loop.run_until_complete(coro)

# Serve requests until Ctrl+C is pressed
print('Serving on {}'.format(server.sockets[0].getsockname()))
try:
    loop.run_forever()
except KeyboardInterrupt:
    pass

# Close the server
server.close()
loop.run_until_complete(server.wait_closed())
loop.close()





##Example - Get HTTP headers


import asyncio
import urllib.parse
import sys

@asyncio.coroutine
def print_http_headers(url):
    url = urllib.parse.urlsplit(url)
    if url.scheme == 'https':
        connect = asyncio.open_connection(url.hostname, 443, ssl=True)
    else:
        connect = asyncio.open_connection(url.hostname, 80)
	
    reader, writer = yield from connect
	
    query = ('HEAD {path} HTTP/1.0\r\n'
             'Host: {hostname}\r\n'
             '\r\n').format(path=url.path or '/', hostname=url.hostname)
    writer.write(query.encode('latin-1'))
    while True:
        line = yield from reader.readline()
        if not line:
            break
        line = line.decode('latin1').rstrip()
        if line:
            print('HTTP header> %s' % line)

    # Ignore the body, close the socket
    writer.close()

url = sys.argv[1]
loop = asyncio.get_event_loop()
task = asyncio.ensure_future(print_http_headers(url))
loop.run_until_complete(task)
loop.close()

#Execution 
$ python example.py http://example.com/path/page.html





###Asyncio - Subprocess 


##In windows, use ProactorEventLoop to support subprocess
import asyncio, sys

if sys.platform == 'win32':
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)




#Create a subprocess: high-level API using Process
coroutine asyncio.create_subprocess_exec(*args, stdin=None, stdout=None, 
    stderr=None, loop=None, limit=None, **kwds)
    Return a Process instance.
    
coroutine asyncio.create_subprocess_shell(cmd, stdin=None, stdout=None, 
    stderr=None, loop=None, limit=None, **kwds)
    Return a Process instance.
    shlex.quote() function can be used to properly escape whitespace and shell metacharacters in strings 
    import shlex 
    command = 'ls -l {}'.format(shlex.quote(filename))

asyncio.subprocess.PIPE
asyncio.subprocess.STDOUT
asyncio.subprocess.DEVNULL


    
class asyncio.subprocess.Process
    coroutine wait()
        Wait for child process to terminate. Set and return returncode attribute
    coroutine communicate(input=None)  
        Interact with process: Send data(arg input) to stdin. 
        Read data from stdout and stderr, until end-of-file is reached
        returns (stdout_data, stderr_data)
        to send data to the process’s stdin, create the Process object with stdin=PIPE. 
        to get anything other than None in the result tuple, give stdout=PIPE and/or stderr=PIPE too.
    send_signal(signal)
        Sends the signal signal to the child process
        On Windows, SIGTERM is an alias for terminate().
    terminate()
        Stop the child
    kill()
        Kills the child
    stdin
        Standard input stream (StreamWriter), None if the process was created with stdin=None.
    stdout
        Standard output stream (StreamReader), None if the process was created with stdout=None.
    stderr
        Standard error stream (StreamReader), None if the process was created with stderr=None.
    pid
    returncode


##Example - Subprocess using stream 

import asyncio.subprocess
import sys

@asyncio.coroutine
def get_date():
    code = 'import datetime; print(datetime.datetime.now())'  #python code 
    # Create the subprocess, redirect the standard output into a pipe
    create = asyncio.create_subprocess_exec(sys.executable, '-c', code,  stdout=asyncio.subprocess.PIPE)
    proc = yield from create
    # Read one line of output
    data = yield from proc.stdout.readline()
    line = data.decode('ascii').rstrip()
    # Wait for the subprocess exit
    yield from proc.wait()
    return line

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)
else:
    loop = asyncio.get_event_loop()

date = loop.run_until_complete(get_date())
print("Current date: %s" % date)
loop.close()






###Asyncio - Synchronization primitives
#Locks: 
asyncio.Lock(*, loop=None)
    locked()
        Return True if the underlying lock is acquired
    coroutine acquire()
            This method blocks until the lock is unlocked, then sets it to locked and returns True.
    release()
    
    
#Event 
class asyncio.Event(*, loop=None)
    clear()
        Reset the internal flag to false. 
        Subsequently, coroutines calling wait() will block until set() is called to set the internal flag to true again.
    is_set()
        Return True if and only if the internal flag is true.
    set()
        Set the internal flag to true. 
        All coroutines waiting for it to become true are awakened. 
        Coroutine that call wait() once the flag is true will not block at all.
    coroutine wait()
        Block until the internal flag is true.


#Condition
class asyncio.Condition(lock=None, *, loop=None)
    coroutine acquire()
        This method blocks until the lock is unlocked, then sets it to locked and returns True.
    notify(n=1)
    locked()
        Return True if the underlying lock is acquired
    notify_all()
    release()
    coroutine wait()
        Wait until notified.
    coroutine wait_for(predicate)
        Wait until a predicate becomes true.


#Semaphore
class asyncio.Semaphore(value=1, *, loop=None)
    coroutine acquire()
            This method blocks until the lock is unlocked, then sets it to locked and returns True.
    locked()
        Return True if the underlying lock is acquired
    release()


#BoundedSemaphore
class asyncio.BoundedSemaphore(value=1, *, loop=None)
    A bounded semaphore implementation. Inherit from Semaphore.
    This raises ValueError in release() if it would increase the value above the initial value.



#Usgae
lock = Lock()
...
yield from lock  #calls acquire() 
try:
    ...
finally:
    lock.release()


#Context manager usage:
lock = Lock()
...
with (yield from lock):
     ...


#Lock objects can be tested for locking state:
if not lock.locked():
   yield from lock
else:
   # lock is acquired
    ...

    
### async-timeout
$ pip install async-timeout   
    
#The context manager is useful in cases 
#when you want to apply timeout logic around block of code  
#or in cases when asyncio.wait_for() is not suitable

async with timeout(1.5):
    await inner()

#1.If inner() is executed faster than in 1.5 seconds nothing happens.
#2.Otherwise inner() is cancelled internally by sending asyncio.CancelledError into but asyncio.TimeoutError is raised outside of context manager scope.

#Context manager has .expired property for check if timeout happens exactly in context manager:
async with timeout(1.5) as cm:
    await inner()
print(cm.expired)

 
    
    
    
    
###aiohttp 
$ pip install aiohttp

##Supports 
•Supports both Client and HTTP Server.
•Supports both Server WebSockets and Client WebSockets out-of-the-box.
•Web-server has Middlewares, Signals and pluggable routing.



#Example:Using aiohttp

import asyncio
import aiohttp
 
@asyncio.coroutine
def fetch_page(url):
    response = yield from aiohttp.request('GET', url)
    assert response.status == 200
    content = yield from response.read()  #response.read_and_close(decode=True)
    print('URL: {0}:  Content: {1}'.format(url, len(content)))
    return (url, len(content))
 



 
loop = asyncio.get_event_loop()
tasks = [
     asyncio.async(fetch_page('http://google.com')),
     asyncio.async(fetch_page('http://cnn.com')),
     asyncio.async(fetch_page('http://twitter.com'))]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()
 
for task in tasks:
    print(task.result())



#Another examples

import asyncio
import aiohttp
import bs4

@asyncio.coroutine
def get(*args, **kwargs):  
    response = yield from aiohttp.request('GET', *args, **kwargs)
    return (yield from response.read())

	
def first_magnet(page):  
    soup = bs4.BeautifulSoup(page, "html.parser")
    a = soup.find('a', title='Download this torrent using magnet')
    return a['href']


@asyncio.coroutine
def print_magnet(query):  
    url = 'http://thepiratebay.se/search/{}/0/7/0'.format(query)
    page = yield from get(url, compress=True)
    magnet = first_magnet(page)
    print('{}: {}'.format(query, magnet))
	

	
distros = ['archlinux', 'ubuntu', 'debian']  
loop = asyncio.get_event_loop()  
f = asyncio.wait([print_magnet(d) for d in distros])  
loop.run_until_complete(f)  
loop.close()


##Py3.5 version 

import aiohttp
import asyncio
import async_timeout

async def fetch(session, url):
    with async_timeout.timeout(10):
        async with session.get(url) as response:
            return await response.text()

async def main():
    async with aiohttp.ClientSession() as session:
        html = await fetch(session, 'http://python.org')
        print(html)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())


#Server example:


from aiohttp import web

async def handle(request):
    name = request.match_info.get('name', "Anonymous")
    text = "Hello, " + name
    return web.Response(text=text)

app = web.Application()
app.router.add_get('/', handle)
app.router.add_get('/{name}', handle)

web.run_app(app)

##Make a Request
async with aiohttp.ClientSession() as session:
    async with session.get('https://api.github.com/events') as resp:
        print(resp.status)
        print(await resp.text())

#Other HTTP methods
session.post('http://httpbin.org/post', data=b'data')
session.put('http://httpbin.org/put', data=b'data')
session.delete('http://httpbin.org/delete')
session.head('http://httpbin.org/get')
session.options('http://httpbin.org/get')
session.patch('http://httpbin.org/patch', data=b'data'


##JSON Request
async with aiohttp.ClientSession() as session:
    async with session.post(json={'test': 'object})
    
##Passing Parameters In URLs
params = {'key1': 'value1', 'key2': 'value2'}
async with session.get('http://httpbin.org/get', params=params) as resp:
    assert str(resp.url) == 'http://httpbin.org/get?key2=value2&key1=value1'

params = [('key', 'value1'), ('key', 'value2')]
async with session.get('http://httpbin.org/get',
                       params=params) as r:
    assert str(r.url) == 'http://httpbin.org/get?key=value2&key=value1'


##Response Content   
#ClientResponse object contains request_info property
#contains request fields: url and headers.

async with session.get('https://api.github.com/events') as resp:
    print(await resp.text())

    
#aiohttp will automatically decode the content from the server. 
#or specify
await resp.text(encoding='windows-1251')

##Binary Response Content    
#The gzip and deflate transfer-encodings are automatically decoded 
print(await resp.read())

##JSON Response Content
async with session.get('https://api.github.com/events') as resp:
    print(await resp.json())

##Streaming Response Content
#It is not possible to use read(), json() and text() after explicit reading from content
with open(filename, 'wb') as fd:
    while True:
        chunk = await resp.content.read(chunk_size)
        if not chunk:
            break
        fd.write(chunk)

##Custom Headers
import json
url = 'https://api.github.com/some/endpoint'
payload = {'some': 'data'}
headers = {'content-type': 'application/json'}

await session.post(url,
                   data=json.dumps(payload),
                   headers=headers)

##Custom Cookies
url = 'http://httpbin.org/cookies'
cookies = {'cookies_are': 'working'}
async with ClientSession(cookies=cookies) as session:
    async with session.get(url) as resp:
        assert await resp.json() == {
           "cookies": {"cookies_are": "working"}}

##to send some form-encoded data 
payload = {'key1': 'value1', 'key2': 'value2'}
async with session.post('http://httpbin.org/post',
                        data=payload) as resp:
    print(await resp.text())


import json
url = 'https://api.github.com/some/endpoint'
payload = {'some': 'data'}

async with session.post(url, data=json.dumps(payload)) as resp:
    ...

##POST a Multipart-Encoded File

url = 'http://httpbin.org/post'
files = {'file': open('report.xls', 'rb')}
await session.post(url, data=files)

#set the filename, content_type explicitly:
url = 'http://httpbin.org/post'
data = FormData()
data.add_field('file',
               open('report.xls', 'rb'),
               filename='report.xls',
               content_type='application/vnd.ms-excel')

await session.post(url, data=data)


##Streaming uploads
##to send large files without reading them into memory.

with open('massive-body', 'rb') as f:
   await session.post('http://httpbin.org/post', data=f)


#Or  use aiohttp.streamer object:
@aiohttp.streamer
def file_sender(writer, file_name=None):
    with open(file_name, 'rb') as f:
        chunk = f.read(2**16)
        while chunk:
            yield from writer.write(chunk)
            chunk = f.read(2**16)

# Then you can use `file_sender` as a data provider:

async with session.post('http://httpbin.org/post',
                        data=file_sender(file_name='huge_file')) as resp:
    print(await resp.text())


#or use a StreamReader object
#to upload a file from another request and calculate the file SHA1 hash:
async def feed_stream(resp, stream):
    h = hashlib.sha256()

    while True:
        chunk = await resp.content.readany()
        if not chunk:
            break
        h.update(chunk)
        stream.feed_data(chunk)

    return h.hexdigest()

resp = session.get('http://httpbin.org/post')
stream = StreamReader()
loop.create_task(session.post('http://httpbin.org/post', data=stream))

file_hash = await feed_stream(resp, stream)

#And chain get and post requests together:
r = await session.get('http://python.org')
await session.post('http://httpbin.org/post',
                   data=r.content)



##Uploading pre-compressed data
#set  the value of the Content-Encoding header:
async def my_coroutine(session, headers, my_data):
    data = zlib.compress(my_data)
    headers = {'Content-Encoding': 'deflate'}
    async with session.post('http://httpbin.org/post',
                            data=data,
                            headers=headers)
        pass



##Keep-Alive, connection pooling and cookie sharing
#ClientSession may be used for sharing cookies between multiple requests:


async with aiohttp.ClientSession() as session:
    await session.get('http://httpbin.org/cookies/set?my_cookie=my_value')
    filtered = session.cookie_jar.filter_cookies('http://httpbin.org')
    assert filtered['my_cookie'].value == 'my_value'
    async with session.get('http://httpbin.org/cookies') as r:
        json_body = await r.json()
        assert json_body['cookies']['my_cookie'] == 'my_value'


#set default headers for all session requests:
async with aiohttp.ClientSession(
    headers={"Authorization": "Basic bG9naW46cGFzcw=="}) as session:
    async with session.get("http://httpbin.org/headers") as r:
        json_body = await r.json()
        assert json_body['headers']['Authorization'] == \
            'Basic bG9naW46cGFzcw=='


##Resolving using custom nameservers
#aiodns is required:
from aiohttp.resolver import AsyncResolver

resolver = AsyncResolver(nameservers=["8.8.8.8", "8.8.4.4"])
conn = aiohttp.TCPConnector(resolver=resolver)

##SSL control for TCP sockets
#TCPConnector constructor accepts mutually exclusive verify_ssl and ssl_context params.
#Certification checks can be relaxed by passing verify_ssl=False:
conn = aiohttp.TCPConnector(verify_ssl=False)
session = aiohttp.ClientSession(connector=conn)
r = await session.get('https://example.com')


#to setup custom ssl parameters (use own certification files for example) 
sslcontext = ssl.create_default_context(
   cafile='/path/to/ca-bundle.crt')
conn = aiohttp.TCPConnector(ssl_context=sslcontext)
session = aiohttp.ClientSession(connector=conn)
r = await session.get('https://example.com')


#to verify client-side certificates
sslcontext = ssl.create_default_context(
   cafile='/path/to/client-side-ca-bundle.crt')
sslcontext.load_cert_chain('/path/to/client/public/key.pem', '/path/to/client/private/key.pem')
conn = aiohttp.TCPConnector(ssl_context=sslcontext)
session = aiohttp.ClientSession(connector=conn)
r = await session.get('https://server-with-client-side-certificates-validaction.com')


#verify certificates via MD5, SHA1, or SHA256 fingerprint:
# Attempt to connect to https://www.python.org
# with a pin to a bogus certificate:
bad_md5 = b'\xa2\x06G\xad\xaa\xf5\xd8\\J\x99^by;\x06='
conn = aiohttp.TCPConnector(fingerprint=bad_md5)
session = aiohttp.ClientSession(connector=conn)
exc = None
try:
    r = yield from session.get('https://www.python.org')
except FingerprintMismatch as e:
    exc = e
assert exc is not None
assert exc.expected == bad_md5

# www.python.org cert's actual md5
assert exc.got == b'\xca;I\x9cuv\x8es\x138N$?\x15\xca\xcb'


#Note that this is the fingerprint of the DER-encoded certificate. 
#If you have the certificate in PEM format, you can convert it to DER with e.g. 
$ openssl x509 -in crt.pem -inform PEM -outform DER > crt.der.

#to convert from a hexadecimal digest to a binary byte-string, 
#you can use binascii.unhexlify:
md5_hex = 'ca3b499c75768e7313384e243f15cacb'
from binascii import unhexlify
assert unhexlify(md5_hex) == b'\xca;I\x9cuv\x8es\x138N$?\x15\xca\xcb'


##Proxy support
sync with aiohttp.ClientSession() as session:
    async with session.get("http://python.org",
                           proxy="http://some.proxy.com") as resp:
        print(resp.status)


#it won’t read environment variables by default. 
#or set proxy_from_env to True
async with aiohttp.ClientSession() as session:
    async with session.get("http://python.org",
                           proxy_from_env=True) as resp:
        print(resp.status)


#It also supports proxy authorization:
async with aiohttp.ClientSession() as session:
    proxy_auth = aiohttp.BasicAuth('user', 'pass')
    async with session.get("http://python.org",
                           proxy="http://some.proxy.com",
                           proxy_auth=proxy_auth) as resp:
        print(resp.status)


#Authentication credentials can be passed in proxy URL:
session.get("http://python.org",
            proxy="http://user:pass@some.proxy.com")



##Response Status Codes
async with session.get('http://httpbin.org/get') as resp:
    assert resp.status == 200



##Response Headers

>>> resp.headers
{'ACCESS-CONTROL-ALLOW-ORIGIN': '*',
 'CONTENT-TYPE': 'application/json',
 'DATE': 'Tue, 15 Jul 2014 16:49:51 GMT',
 'SERVER': 'gunicorn/18.0',
 'CONTENT-LENGTH': '331',
 'CONNECTION': 'keep-alive'}


>>> resp.headers['Content-Type']
'application/json'

>>> resp.headers.get('content-type')
'application/json'

>>> resp.raw_headers
((b'SERVER', b'nginx'),
 (b'DATE', b'Sat, 09 Jan 2016 20:28:40 GMT'),
 (b'CONTENT-TYPE', b'text/html; charset=utf-8'),
 (b'CONTENT-LENGTH', b'12150'),
 (b'CONNECTION', b'keep-alive'))



##Response Cookies
url = 'http://example.com/some/cookie/setting/url'
async with session.get(url) as resp:
    print(resp.cookies['example_cookie_name'])

##Response History
#If a request was redirected
>>> resp = await session.get('http://example.com/some/redirect/')
>>> resp
<ClientResponse(http://example.com/some/other/url/) [200]>
>>> resp.history
(<ClientResponse(http://example.com/some/redirect/) [301]>,)

##Timeouts
#None or 0 disables timeout check.
async with session.get('https://github.com', timeout=60) as r:
    ...




#or use async_timeout.timeout() 
import async_timeout

with async_timeout.timeout(0.001, loop=session.loop):
    async with session.get('https://github.com') as r:
        await r.text()



###Recursive Asyncio 

import asyncio 

@asyncio.coroutine 
def fib(a=0,b=1):
    print(b)
    yield from fib(b,a+b)
    
loop = asyncio.get_event_loop()
loop.run_until_complete(fib())
#Crashes with stack overflow 

#version-1 
import asyncio 

@asyncio.coroutine 
def fib(a=0,b=1):
    print(b)
    yield from asyncio.async(fib(b,a+b))  #schedules Task 

loop = asyncio.get_event_loop()
#
asyncio.async(fib())
loop.run_forever()
#else 
loop.run_until_complete(fib())


#Version-2 
@asyncio.coroutine
def fib(a=0,b=1):
    fut = asyncio.Future()  # We're going to return this right away to our caller
    def set_result(out):  # This gets called when the next recursive call completes
        fut.set_result(out.result()) # Pull the result from the inner call and return it up the stack.
    print(b)    
    in_fut = asyncio.async(fib(b,a+b))  # This returns an asyncio.Task
    in_fut.add_done_callback(set_result) # schedule set_result when the Task is done.
    return fut

loop = asyncio.get_event_loop()
loop.run_until_complete(fib())

#Version-3 
async def fib(a=0,b=1):
    print(b)
    await asyncio.ensure_future(fib(b,a+b))

loop = asyncio.get_event_loop()
loop.run_until_complete(fib())

##Trampoline 
import asyncio

@asyncio.coroutine
def a(n):
    print("A: {}".format(n))
    if n > 1000: return n
    else: yield from b(n+1)

@asyncio.coroutine
def b(n):
    print("B: {}".format(n))
    yield from a(n+1)

loop = asyncio.get_event_loop()
loop.run_until_complete(a(0))
#crashes with stack overflow 

To keep the stack from growing, 
you have to allow each coroutine to actually exit 
after it schedules the next recursive call, 
which means you have to avoid using yield from. 

Instead, you use asyncio.async 
(or asyncio.ensure_future if using Python 3.4.4+) 
to schedule the next coroutine with the event loop, 
and use Future.add_done_callback to schedule a callback 
to run once the recursive call returns. 
Each coroutine then returns an asyncio.Future object, 
which has its result set inside the callback that's run 
when the recursive call it scheduled completes.

import asyncio

@asyncio.coroutine
def a(n):
    fut = asyncio.Future()  # We're going to return this right away to our caller
    def set_result(out):  # This gets called when the next recursive call completes
        fut.set_result(out.result()) # Pull the result from the inner call and return it up the stack.
    print("A: {}".format(n))
    if n > 1000: 
        return n
    else: 
        in_fut = asyncio.async(b(n+1))  # This returns an asyncio.Task
        in_fut.add_done_callback(set_result) # schedule set_result when the Task is done.
    return fut

@asyncio.coroutine
def b(n):
    fut = asyncio.Future()
    def set_result(out):
        fut.set_result(out.result())
    print("B: {}".format(n))
    in_fut = asyncio.async(a(n+1))
    in_fut.add_done_callback(set_result)
    return fut

loop = asyncio.get_event_loop()
print("Out is {}".format(loop.run_until_complete(a(0))))


Output:
A: 0
B: 1
A: 2
B: 3
A: 4
B: 5
...
A: 994
B: 995
A: 996
B: 997
A: 998
B: 999
A: 1000
B: 1001
A: 1002
Out is 1002

Now, your example code doesn't actually return n all the way back up the stack, 
so you could make something functionally equivalent that's a bit simpler:
import asyncio

@asyncio.coroutine
def a(n):
    print("A: {}".format(n))
    if n > 1000: loop.stop(); return n
    else: asyncio.async(b(n+1))

@asyncio.coroutine
def b(n):
    print("B: {}".format(n))
    asyncio.async(a(n+1))

loop = asyncio.get_event_loop()
asyncio.async(a(0))
loop.run_forever()

#Async, await: 
import asyncio

async def a(n):
    if n > 1000:
        return n
    else:
        ret = await asyncio.ensure_future(b(n + 1))
    return ret

async def b(n):
    ret = await asyncio.ensure_future(a(n + 1))
    return ret

import timeit
print(min(timeit.repeat("""
loop = asyncio.get_event_loop()
loop.run_until_complete(a(0))
""", "from __main__ import a, b, asyncio", number=10)))

Result:
% time  python stack.py
0.45157229300002655
python stack.py  1,42s user 0,02s system 99% cpu 1,451 total







    
