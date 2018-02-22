###Installation 
#https://stankirdey.com/2017/03/09/installing-mxnet-deep-learning-framework-on-windows-10/

#Download - 64bit Py2.7 as Py3. has compatibility issues 
#https://www.continuum.io/downloads
#Create a environment 
#https://conda.io/docs/user-guide/tasks/manage-environments.html#viewing-a-list-of-your-environments
#command details 
$ conda create --name <envname> [list of packages]
$ conda create --name <envname> numpy scipy=1.6 


#note 'anaconda' option would install all default packages found on your installation
$ conda create -n mxnet python=2.7 anaconda  #creates at C:\Anaconda2\envs\mxnet

#Once the environment creation is complete, activate it by executing:
$ activate mxnet  
$ deactivate    #to deactivate 

#check env 
$ conda info --envs
#viewing list of packages 
$ conda list -n mxnet
#installing pip 
$ conda install -n mxnet pip


#Download packages from 
#https://github.com/yajiedesign/mxnet/releases

#Download
#https://github.com/yajiedesign/mxnet/releases/download/weekly_binary_build_v2/prebuildbase_win10_x64_vc14_v2.7z
#Unzip to c:\maxnet
##*** In new Window , Must not be in Conda environment, check PATH contains c:\mxnet
$ cd c:\mxnet 
$ setupenv.cmd

#download 
#https://github.com/yajiedesign/mxnet/releases/download/20171222/20171222_mxnet_x64_vc14_cpu.7z
#unzip to c:\mxnet 
'''
below steps are not required
#Requires OpenBlas dll and mingw64.zip's dll, unzip to c:\mxnet\lib 
#https://sourceforge.net/projects/openblas/files/v0.2.14/
'''
$ cd c:\mxnet\python 
$ activate mxnet
$ python setup.py install
$ deactivate

#usage 
$ activate mxnet
$ python 
>>> import mxnet 
#or 
$ jupyter notebook

#Note GPU version requires NVDIA card, 
#Note GPU version needs cuda_8.0.61_windows.exe(download from CUDA site)
#and Use only Cu80 version cudnn-8.0-windows7-x64-v7.zip

##Note: environment mxnet is created for CPU , rename folder c:\mxnet-cpu to mxnet 
#Environment mxnet_gpu is created for GPU, rename folder c:\mxnet-gpu tp mxnet 
#then activate the environment accordingly


###Parallism 
#https://mxnet.incubator.apache.org/how_to/multi_devices.html




###Gluon
#Gluon is the high-level interface for MXNet
#http://gluon.mxnet.io/


###MXNet's ndarray
#similar to NumPy’s multi-dimensional array
#NDArrays support asynchronous computation on CPU, GPU, and distributed cloud architectures. 
#they provide support for automatic differentiation

import mxnet as mx
from mxnet import nd
mx.random.seed(1)
x = nd.empty((3, 4))
print(x)
x = nd.zeros((3, 5))
x = nd.ones((3, 4))
y = nd.random_normal(0, 1, shape=(3, 4))
y.shape#(3, 4)
y.size # 12
#elementwise m but not in place 
x + y
x * y 
nd.exp(y)
nd.dot(x, y.T)  #matrix product 
# in-place operations 
nd.elemwise_add(x, y, out=y)
#or , x is modified in place 
x += y
#or 
x[:] = x + y
#Slicing
x[1:3] #second and third rows 
x[1,2] = 9.0
x[1:2,1:3]  #multidimensional slicing 
x[1:2,1:3] = 5.0 #update 
#Broadcasting

x = nd.ones(shape=(3,3))
print('x = ', x)
y = nd.arange(3)
print('y = ', y)
print('x + y = ', x + y)
#output - Note y as a (1,3) matrix
x =
[[ 1.  1.  1.]
 [ 1.  1.  1.]
 [ 1.  1.  1.]]
<NDArray 3x3 @cpu(0)>
y =
[ 0.  1.  2.]
<NDArray 3 @cpu(0)>
x + y =
[[ 1.  2.  3.]
 [ 1.  2.  3.]
 [ 1.  2.  3.]]
<NDArray 3x3 @cpu(0)>

#OR 
y = y.reshape((3,1))
print('y = ', y)
print('x + y = ', x+y)
#Output
y =
[[ 0.]
 [ 1.]
 [ 2.]]
<NDArray 3x1 @cpu(0)>
x + y =
[[ 1.  1.  1.]
 [ 2.  2.  2.]
 [ 3.  3.  3.]]
<NDArray 3x3 @cpu(0)>

#Converting from MXNet NDArray to NumPy
a = x.asnumpy()
type(a)
y = nd.array(a)



##In MXNet, every array has a context. 
#One context could be the CPU. 
#Other contexts might be various GPUs. 
#cpu or gpu(dev_id=0) - – the CPU/GPU id.
#cpu() is usually the default context for many operations when no context is specified.
z = nd.ones(shape=(3,3), ctx=mx.cpu(0))  #for gpu mx.gpu(0), GPU requires NVDIA card 


#Given an NDArray on a given context, 
#copy it to another context by using the copyto() method.
x_cpu = x.copyto(mx.cpu(0))
#The result of an operator will have the same context as the inputs.
x_gpu + z
#to check 
print(x_cpu.context)
print(z.context)


#In order to perform an operation on two ndarrays x1 and x2, 
#both must live on the same context. 
#or explicitly copy data from one context to another. 


#To copy only if the variables currently lives on different contexts. 
#use  as_in_context(). 
#If the variable is already the specified context then this is a no-op.
print('id(z):', id(z))
z = z.copyto(mx.cpu(0))
print('id(z):', id(z))
z = z.as_in_context(mx.cpu(0))
print('id(z):', id(z))
print(z)



http://gluon.mxnet.io/chapter01_crashcourse/linear-algebra.html

###Linear algebra
from mxnet import nd

##Scalars
#array of dimension (1,) 

# Instantiate two scalars

x = nd.array([3.0])
y = nd.array([2.0])

# Add them
print('x + y = ', x + y)

# Multiply them
print('x * y = ', x * y)

# Divide x by y
print('x / y = ', x / y)

# Raise x to the power y.
print('x ** y = ', nd.power(x,y))
#convert any NDArray to a Python float 
x.asscalar()

##Vectors
#array of dimension (n,) 
u = nd.arange(4)
print('u = ', u)
u[3]
len(u)
#4
u.shape
#(4,)
a = 2
x = nd.array([1,2,3])
y = nd.array([10,20,30])
print(a * x)
print(a * x + y)




##Matrix 
#array of dimension (m,n) 
A = nd.zeros((5,4))

#from 1D
x = nd.arange(20)
A = x.reshape((5, 4))
A
#Out[12]:
[[  0.   1.   2.   3.]
 [  4.   5.   6.   7.]
 [  8.   9.  10.  11.]
 [ 12.  13.  14.  15.]
 [ 16.  17.  18.  19.]]
<NDArray 5x4 @cpu(0)>



print('A[2, 3] = ', A[2, 3])
#output
A[2, 3] =
[ 11.]
<NDArray 1 @cpu(0)>

print('row 2', A[2, :])
print('column 3', A[:, 3])
#row 2
[  8.   9.  10.  11.]
<NDArray 4 @cpu(0)>
#column 3
[  3.   7.  11.  15.  19.]
<NDArray 5 @cpu(0)>

We can transpose the matrix through T. That is, if B=AT
, then bij=aji for any i and j

.
#transpose
A.T

#Out[15]:
[[  0.   4.   8.  12.  16.]
 [  1.   5.   9.  13.  17.]
 [  2.   6.  10.  14.  18.]
 [  3.   7.  11.  15.  19.]]
<NDArray 4x5 @cpu(0)>







##Tensor 
#array of dimension (m,n,l,....) 
X = nd.arange(24).reshape((2, 3, 4))
print('X.shape =', X.shape)
print('X =', X)


##common standard arithmetic operators (+,-,/,*,**) have all been lifted to element-wise operations 

u = nd.array([1, 2, 4, 8])
v = nd.ones_like(u) * 2
print('v =', v)
print('u + v', u + v)
print('u - v', u - v)
print('u * v', u * v)
print('u / v', u / v)
#output
v =
[ 2.  2.  2.  2.]
<NDArray 4 @cpu(0)>
u + v
[  3.   4.   6.  10.]
<NDArray 4 @cpu(0)>
u - v
[-1.  0.  2.  6.]
<NDArray 4 @cpu(0)>
u * v
[  2.   4.   8.  16.]
<NDArray 4 @cpu(0)>
u / v
[ 0.5  1.   2.   4. ]
<NDArray 4 @cpu(0)>

#element-wise operations on any two tensors of the same shape, including matrices.
B = nd.ones_like(A) * 3
print('B =', B)
print('A + B =', A + B)
print('A * B =', A * B)
#output
B =
[[ 3.  3.  3.  3.]
 [ 3.  3.  3.  3.]
 [ 3.  3.  3.  3.]
 [ 3.  3.  3.  3.]
 [ 3.  3.  3.  3.]]
<NDArray 5x4 @cpu(0)>
A + B =
[[  3.   4.   5.   6.]
 [  7.   8.   9.  10.]
 [ 11.  12.  13.  14.]
 [ 15.  16.  17.  18.]
 [ 19.  20.  21.  22.]]
<NDArray 5x4 @cpu(0)>
A * B =
[[  0.   3.   6.   9.]
 [ 12.  15.  18.  21.]
 [ 24.  27.  30.  33.]
 [ 36.  39.  42.  45.]
 [ 48.  51.  54.  57.]]
<NDArray 5x4 @cpu(0)>

##Basic properties of tensor arithmetic
#multiplication by a scalar produces a tensor of the same shape
a = 2
x = nd.ones(3)
y = nd.zeros(3)
print(x.shape)
print(y.shape)
print((a * x).shape)
print((a * x + y).shape)


##Sums and means, dot product 
nd.sum(u)
nd.sum(A)
print(nd.mean(A))
print(nd.sum(A) / A.size)
#dot product 
nd.dot(u, v)
#equivalent to 
nd.sum(u * v)
#Matrix-vector products
#column dimension of A must be the same as the dimension of u
nd.dot(A, u)
#Matrix-matrix multiplication
#column dimension of A must be the same as the row dimension of B
A = nd.ones(shape=(3, 4))
B = nd.ones(shape=(4, 5))
nd.dot(A, B)
#Norms
#Informally, they tell us how big a vector or matrix is
#In machine learning we’re often trying to solve optimization problems: 

#Maximize the probability assigned to observed data. 
#Minimize the distance between predictions and the ground-truth observations. 
#Assign vector representations to items (like words, products, or news articles) 
#such that the distance between similar items is minimized, 
#and the distance between dissimilar items is maximized. 
#Oftentimes, these objectives are expressed as norms.

#L2 norm = Euclidean distance = sqrt(sum(square of xi))
nd.norm(u)
#L1 norm 
#It has the convenient property of placing less emphasis on outliers
nd.sum(nd.abs(u))


http://gluon.mxnet.io/chapter01_crashcourse/probability.html
###Probability and statistics

import mxnet as mx
from mxnet import nd

#process of drawing examples from probability distributions is called sampling. 
#The distribution which assigns probabilities to a number of discrete choices is called the multinomial distribution

probabilities = nd.ones(6) / 6    #size of 6, each one is 1/6
nd.sample_multinomial(probabilities)  #pick one of above index via probabilities
#Out[2]:
[3]
nd.sample_multinomial(probabilities)  
#Out[2]:
[2]

#drawing multiple samples at once,
print(nd.sample_multinomial(probabilities, shape=(10)))
print(nd.sample_multinomial(probabilities, shape=(5,10)))
#output
[3 4 5 3 5 3 5 2 3 3]
<NDArray 10 @cpu(0)>
[[2 2 1 5 0 5 1 2 2 4]
 [4 3 2 3 2 5 5 0 2 0]
 [3 0 2 4 5 4 0 5 5 5]
 [2 4 4 2 3 4 4 0 4 3]
 [3 0 3 5 4 3 0 2 2 1]]
<NDArray 5x10 @cpu(0)>

#Example - in case of die with 1000 sampling
#how many times each number was rolled.
counts = nd.zeros((6,1000))
totals = nd.zeros(6)
for i, roll in enumerate(rolls):
    totals[int(roll.asscalar())] += 1
    counts[:, i] = totals

totals / 1000
#Theoritical is 1/6 = 1.167..
#Out[6]:
[ 0.167       0.168       0.175       0.15899999  0.15800001  0.17299999]
<NDArray 6 @cpu(0)>


#counts array - For each time step (out of 1000), counts says 
#how many times each of the numbers has shown up. 

#then normalize each j-th column of the counts vector 
#by the number of tosses to give the current estimated probabilities at that time. 
counts
#Out[7]:
[[   0.    0.    0. ...,  165.  166.  167.]
 [   1.    1.    1. ...,  168.  168.  168.]
 [   0.    0.    0. ...,  175.  175.  175.]
 [   0.    0.    0. ...,  159.  159.  159.]
 [   0.    1.    2. ...,  158.  158.  158.]
 [   0.    0.    0. ...,  173.  173.  173.]]
<NDArray 6x1000 @cpu(0)>

#Normalizing by the number of tosses
x = nd.arange(1000).reshape((1,1000)) + 1
estimates = counts / x
print(estimates[:,0])
print(estimates[:,1])
print(estimates[:,100])
#out
[ 0.  1.  0.  0.  0.  0.]
<NDArray 6 @cpu(0)>
[ 0.   0.5  0.   0.   0.5  0. ]
<NDArray 6 @cpu(0)>
[ 0.1980198   0.15841584  0.17821783  0.18811882  0.12871288  0.14851485]
<NDArray 6 @cpu(0)>

#Plot it to show the convergence 
from matplotlib import pyplot as plt
plt.plot(estimates[0, :].asnumpy(), label="Estimated P(die=1)")
plt.plot(estimates[1, :].asnumpy(), label="Estimated P(die=2)")
plt.plot(estimates[2, :].asnumpy(), label="Estimated P(die=3)")
plt.plot(estimates[3, :].asnumpy(), label="Estimated P(die=4)")
plt.plot(estimates[4, :].asnumpy(), label="Estimated P(die=5)")
plt.plot(estimates[5, :].asnumpy(), label="Estimated P(die=6)")
plt.axhline(y=0.16666, color='black', linestyle='dashed')
plt.legend()
plt.show()



#A random variable, 
#which we denote here as X can be pretty much any quantity, is not determistic. 
#Random variables could take one value among a set of possibilites. 

#Note that there is a subtle difference between discrete random variables, 
#like the sides of a dice, and continuous ones, like the weight and the height of a person. 
#For example - no two people on the planet have the exact same height precisely

#Hence  it needs to be whether someone’s height falls into a given interval, 
#say between 1.99 and 2.01 meters. 
#In these cases ,the likelihood is a value, understood as a density. 
#The height of exactly 2.0 meters has no probability, but nonzero density. 
#Between any two different heights we have nonzero probability.

##Dealing with multiple random variables
#One quantities of interest- joint distribution Pr(A,B). 
#Given any elements a and b, the joint distribution  is the probability 
#that A=a and B=b simulataneously?
# for any values a and b, 
Pr(A,B)<=Pr(A=a) or Pr(A,B)<=Pr(B)
#since for A and B to happen, 
#A has to happen and B also has to happen (and vice versa). 
#Thus A,B cannot be more likely than A or B individually. 
0<=Pr(A,B)/Pr(A)<=1. 

#conditional probability 
Pr(B|A)=the probability that B happens, provided that A has happened.
       =Pr(A,B)/Pr(A)

#or 
Pr(A,B)=Pr(B|A)Pr(A)
#or 
Pr(A,B)=Pr(A|B)Pr(B)

#Solving for one of the conditional variables , Bayes’ theorem
Pr(A|B)=Pr(B|A)Pr(A)Pr(B)

##marginalization, 
#i.e., the operation of determining Pr(A) and Pr(B) from Pr(A,B)
#probability of seeing A amounts to accounting for all possible choices of B 
#and aggregating the joint probabilities over all of them
Pr(A)=SUM(Pr(A,B′)), B' is variable ie all possible choices
Pr(B)=SUM(Pr(A′,B)), A' is variable ie all possible choices

#dependence and independence. 
#Independence is when the occurrence of one event does not influence 
#the occurrence of the other. 
Pr(B|A)=Pr(B)
Pr(A|B)=Pr(A)
#Statisticians typically use A⊥⊥B to express this. 
#For instance, two successive rolls of a dice are independent. 


##Example 
#Assume that a doctor administers an AIDS test to a patient. 
#This test is fairly accurate and fails only with 1% probability 
#if the patient is healthy by reporting him as diseased, 
#and that it never fails to detect HIV if the patient actually has it. 
#We use D to indicate the diagnosis and H to denote the HIV status. 
#Written as a table the outcome Pr(D|H)
                Patient is HIV positive 	Patient is HIV negative
Test positive 	1 	                        0.01
Test negative 	0 	                        0.99

#Note that the column sums are all one (but the row sums aren’t), 
#since the conditional probability needs to sum up to 1 , just like the probability. 

#Let us work out the probability of the patient having AIDS 
#if the test comes back positive. 
#Obviously this is going to depend on how common the disease is, 
#since it affects the number of false alarms. 
#Assume that the population is quite healthy, e.g. 
Pr(HIV positive)=0.0015

#To apply Bayes Theorem we need to determine
Pr(Test positive)=Pr(D=1|H=0)Pr(H=0)+Pr(D=1|H=1)Pr(H=1)
=0.01*0.9985+1*0.0015=0.011485

Pr(H=1|D=1)=Pr(D=1|H=1)Pr(H=1)/Pr(D=1)=1*0.00150/.011485=0.131
#there’s only a 13.1% chance that the patient actually has AIDS, 
#despite using a test that is 99% accurate! 

##Conditional independence
# Likely, he/she would ask the physician to administer another test to get clarity.
# The second test has different characteristics
  	Patient is HIV positive 	Patient is HIV negative
Test positive 	0.98 	0.03
Test negative 	0.02 	0.97

#Unfortunately, the second test comes back positive, too
#assuming 
Pr(D1,D2|H)=Pr(D1|H)Pr(D2|H)
Pr(D1=1 and D2=1|H=0)=0.01*0.03=0.0001

Pr(D1=1 and D2=1|H=1)=1*0.98=0.98
Pr(D1=1 and D2=1)=0.0001*0.9985+0.98*0.0015=0.00156985
Pr(H=1|D1=1 and D2=1)=0.98*0.0015/0.00156985=0.936

#That is, the second test allowed us to gain much higher confidence 
#that not all is well. 

#Despite the second test being considerably less accurate than the first one, 
#it still improved our estimate quite a bit. 

#Why couldn’t we just run the first test a second time? 
#After all, the first test was more accurate. 
#The reason is that we needed a second test that confirmed independently of the first test 
#In other words, we made the tacit assumption that 
Pr(D1,D2|H)=Pr(D1|H)Pr(D2|H)
#Statisticians call such random variables conditionally independent. 
#This is expressed as D1⊥⊥D2|H.
 

##Naive Bayes classification

Conditional independence is useful when dealing with data, 
since it simplifies a lot of equations. 
A popular algorithm is the Naive Bayes Classifier. 
The key assumption is that the attributes are all independent of each other, given the labels. 
p(x|y)=PRODUCT(p(xi|y)), i is variable over space

#Using Bayes Theorem this leads to the classifier 
p(y|x)=PRODUCT(p(xi|y))p(y)/p(x)

#Unfortunately, this is still intractable, since we don’t know p(x). 
#Fortunately, we don’t need it, since we know that 
SUM(p(y|x))=1, y is variable over space 
#hence , above classifier is nothing but 
p(y|x) PROPORTIONAL_TO PRODUCT(p(xi|y))p(y)

##Example with  distinguishing digits on the MNIST classification dataset.
#xi = pixels in handwritten image of digit 0 to 9 , y is actual number 0 to 9 
#p(xi|y) means, Given actual 0 what is probability of each pixel of handwritten image of Number 0
#the, Given actual  1, what is probability of each pixel of handwritten image of Number 1

#We need to find, p(y|x)
#ie Given pixels of handwritten image, what is the probability of each Actual number 0 to 9 
#then pick the most probable one 


#The problem is that we don’t actually know p(y) and p(xi|y). 
#So we need to estimate it given some training data first. 
#This is what is called training the model. 

#In the case of 10 possible classes(digit 0 to 9) we  compute ny, 
#i.e. the number of occurrences of class y and then divide it by the total number of occurrences. 
#E.g. if we have a total of 60,000 pictures of digits and digit 4 occurs 5800 times,
#we estimate its probability as 5800/60000. 

#Likewise, to get an idea of p(xi|y) 
#we count how many times pixel i is set for digit y 
#and then divide it by the number of occurrences of digit y
#This is the probability that that very pixel will be switched on.


import numpy as np

# we go over one observation at a time (speed doesn't matter here)
#label is actual number 0 to 9 
#data = Each sample is an image (in 3D NDArray) with shape (28, 28, 1)
#=784 in one dimension 
#Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black). 
def transform(data, label):
    return (nd.floor(data/128)).astype(np.float32), label.astype(np.float32)
mnist_train = mx.gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = mx.gluon.data.vision.MNIST(train=False, transform=transform)

# Initialize the count statistics for p(y) and p(x_i|y)
# We initialize all numbers with a count of 1 to ensure that we don't get a
# division by zero.  Statisticians call this Laplace smoothing.
ycount = nd.ones(shape=(10))
xcount = nd.ones(shape=(784, 10))

# Aggregate count statistics of how frequently a pixel is on (or off) for
# zeros and ones.
for data, label in mnist_train:
    x = data.reshape((784,))
    y = int(label)
    ycount[y] += 1
    xcount[:, y] += x

# normalize the probabilities p(x_i|y) (divide per pixel counts by total
# count)
for i in range(10):
    xcount[:, i] = xcount[:, i]/ycount[i]

# likewise, compute the probability p(y)
py = ycount / nd.sum(ycount)

#Now that we computed per-pixel counts of occurrence for all pixels, 
#it’s time to see how our model behaves. 
#Time to plot it. 
#We show the estimated probabilities of observing a switched-on pixel. 


import matplotlib.pyplot as plt
fig, figarr = plt.subplots(1, 10, figsize=(15, 15))
for i in range(10):
    figarr[i].imshow(xcount[:, i].reshape((28, 28)).asnumpy(), cmap='hot')
    figarr[i].axes.get_xaxis().set_visible(False)
    figarr[i].axes.get_yaxis().set_visible(False)

plt.show()
print(py)


[ 0.09871688  0.11236461  0.09930012  0.10218297  0.09736711  0.09035161
  0.09863356  0.10441593  0.09751708  0.09915014]
<NDArray 10 @cpu(0)>

#Now we can compute the likelihoods of an image, given the model. 
#ie  p(x|y)

#Since this is computationally awkward 
#(we might have to multiply many small numbers if many pixels have a small probability of occurring), 
#we are better off computing its logarithm instead. 

#That is, instead of p(x|y)=PRODUCT(p(xi|y)) we compute 
logp(x|y)=SUM(logp(xi|y))
ly:=SUM(logp(xi|y))=SUM(xi*logp(xi=1|y)+(1−xi)*log(1−p(xi=1|y)))

#To avoid recomputing logarithms all the time, we precompute them for all pixels.

logxcount = nd.log(xcount)
logxcountneg = nd.log(1-xcount)
logpy = nd.log(py)

fig, figarr = plt.subplots(2, 10, figsize=(15, 3))

# show 10 images
ctr = 0
for data, label in mnist_test:
    x = data.reshape((784,))
    y = int(label)

    # we need to incorporate the prior probability p(y) since p(y|x) is
    # proportional to p(x|y) p(y)
    logpx = logpy.copy()
    for i in range(10):
        # compute the log probability for a digit
        logpx[i] += nd.dot(logxcount[:, i], x) + nd.dot(logxcountneg[:, i], 1-x)
    # normalize to prevent overflow or underflow by subtracting the largest
    # value
    logpx -= nd.max(logpx)
    # and compute the softmax using logpx
    px = nd.exp(logpx).asnumpy()
    px /= np.sum(px)  #this is final probability of each number 0 to 9 

    # bar chart and image of digit
    figarr[1, ctr].bar(range(10), px)
    figarr[1, ctr].axes.get_yaxis().set_visible(False)
    figarr[0, ctr].imshow(x.reshape((28, 28)).asnumpy(), cmap='hot')
    figarr[0, ctr].axes.get_xaxis().set_visible(False)
    figarr[0, ctr].axes.get_yaxis().set_visible(False)
    ctr += 1
    if ctr == 10:
        break

#As we can see, 
#this classifier is both incompetent and overly confident of its incorrect estimates. 
#That is, even if it is horribly wrong, it generates probabilities close to 1 or 0. 
#Not a classifier we should use very much nowadays any longer. 
#While Naive Bayes classifiers used to be popular in the 80s and 90s, e.g. 
#for spam filtering, their heydays are over. 
#The poor performance is due to the incorrect statistical assumptions 
#that we made in our model: 
#we assumed that each and every pixel are independently generated, 
#depending only on the label. 




##Sampling
#One of the basic tools needed to generate random numbers is 
#to sample from a distribution. 


In [13]:

import random
for i in range(10):
    print(random.random())

0.970844720223
0.11442244666
0.476145849846
0.154138063676
0.925771401913
0.347466944833
0.288795056587
0.855051122608
0.32666729925
0.932922304219

##Uniform Distribution
#their range is between 0 and 1, and they are evenly distributed. 

for i in range(10):
    print(random.randint(1, 100))

75
23
34
85
99
66
13
42
19
14

#to check that randint is actually really uniform. 
#Intuitively the best strategy would be to run it, say 1 million times, 
#count how many times it generates each one of the values 
#and to ensure that the result is uniform.



import math

counts = np.zeros(100)
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
axes = axes.reshape(6)
# mangle subplots such that we can index them in a linear fashion rather than
# a 2d grid

for i in range(1, 1000001):
    counts[random.randint(0, 99)] += 1
    if i in [10, 100, 1000, 10000, 100000, 1000000]:
        axes[int(math.log10(i))-1].bar(np.arange(1, 101), counts)
plt.show()


##The categorical distribution - non uniform distribution 
#For example - a biased coin which comes up heads with probability 0.35 and tails with probability 0.65. 

# A simple way to sample from that is to generate a uniform random variable over [0,1] 
#and if the number is less than 0.35, we output heads 
#and otherwise we generate tails

# number of samples
n = 1000000
y = np.random.uniform(0, 1, n)
x = np.arange(1, n+1)
# count number of occurrences and divide by the number of total draws
p0 = np.cumsum(y < 0.35) / x
p1 = np.cumsum(y >= 0.35) / x

plt.figure(figsize=(15, 8))
plt.semilogx(x, p0)
plt.semilogx(x, p1)
plt.show()

#In General - Given any probability distribution, e.g. p=[0.1,0.2,0.05,0.3,0.25,0.1]
#we can compute its cumulative distribution (python’s cumsum) F=[0.1,0.3,0.35,0.65,0.9,1]. 

#Once we have this we draw a random variable x from the uniform distribution U[0,1]
#and then find the interval where F[i−1]≤x<F[i]. 
#We then return i as the sample. 
#By construction, the chances of hitting interval [F[i−1],F[i]) has probability p(i)

#Note that there are many more efficient algorithms for sampling than the one above. 
#For instance, binary search over F will run in O(logn) time for n random variables. 
#There are even more clever algorithms, such as the Alias Method to sample in constant time, after O(n) preprocessing.


##The Normal distribution

p(x)=(1/sqrt(2Pi))*exp(−x*x/2)


x = np.arange(-10, 10, 0.01)
p = (1/math.sqrt(2 * math.pi)) * np.exp(-0.5 * x**2)
plt.figure(figsize=(10, 5))
plt.plot(x, p)
plt.show()


#Sampling from this distribution is a lot less trivial. 
#the key idea in all algorithms is to stratify p(x) in such a way as to map it to the uniform distribution U[0,1]
#One way to do this is with the probability integral transform.

#Denote by F(x)=INTEGRAL(-oo,x)(p(z)dz)
#the cumulative distribution function (CDF) of p. 
# we can now define the inverse map F−1(ξ), where ξ is drawn uniformly. 

#In a way all distributions converge to it, 
#if we only average over a sufficiently large number of draws 
#from any other distribution

##expected values, means and variances.
#The expected value E(x~p(x))[f(x)] of a function f under a distribution p is given 
#by the integral ∫p(x)f(x)dx. 
#That is, we average over all possible outcomes, as given by p

#A particularly important expected value is that for the function f(x)=x
#i.e. μ(mean). It provides us with some idea about the typical values of x

#Another important quantity is the variance, 
#i.e. the typical deviation from the mean σ**2:=E(x~p(x))[(x−μ)**2]
#note σ**2=E(x~p(x))[x**2]− (E(x~p(x))[x])**2

    .

#for some random variable x with mean μ,
#the random variable x+c has mean μ+c. 
#Moreover, γx has the variance γ2σ2.

#Central Limit Theorem. 
#It states that for sufficiently well-behaved random variables, 
#in particular random variables with well-defined mean and variance, 
#the sum tends toward a normal distribution. 

# generate 10 random sequences(column) of 10,000(row) random uniform variables (0,1)
tmp = np.random.uniform(size=(10000,10))  #uniform over (0,1), 1 not included 

#random variables with integer values of {0,1,2}
# tmp>0.3 would give True, False depending on tmp's each element > 0.3 or not 
# 1.0 * (tmp > 0.3) = either 1 or 0 based on True or False 
x = 1.0 * (tmp > 0.3) + 1.0 * (tmp > 0.8) #ultimately matrix of 0,1,2 corresponding to >.3 and .8
#           0.1,0.2 => 0 , 0.3,.4,.5,.6,0.7 => 1,  0.8,0.9=> 2
#prob       2/10             5/10                     2/10 
#Expected values = Multiply each value times its respective probability and then SUM 
#mean        
mean = 0* 0.2 + 1 * 0.5 + 2 * 0.2
#Note 
>>> np.mean(x,axis=0) #axis=0 means row varying ie column wise 
array([ 0.83,  0.95,  0.9 ,  0.85,  0.95,  0.94,  0.86,  0.83,  1.01,  0.94])
>>> np.mean(np.mean(x,axis=0))
0.90599999999999992
#E(x~p(x))[x**2]− mean**2
variance = 0*0* 0.2 + 1*1 * 0.5 + 2*2 * 0.2 - mean**2
#
>>> np.var(x,axis=0)
array([ 0.4411,  0.4275,  0.55  ,  0.4675,  0.5475,  0.4964,  0.5404,
        0.4611,  0.5499,  0.4764])
>>> np.mean(np.var(x,axis=0))
0.49578000000000005
#
print('mean {}, variance {}'.format(mean, variance)) #mean 0.9, variance 0.49
# cumulative sum and normalization
y = np.arange(1,10001).reshape(10000,1)
z = np.cumsum(x,axis=0) / y #axis=0 means row varying/columnwise 
                            #cummulative sum columnwise 
                            #then /y means cummulative sample average 

plt.figure(figsize=(10,5))
for i in range(10):
    plt.semilogx(y,z[:,i])  #Make a plot with log scaling on the x axis
                            # y vs column of z

plt.semilogx(y,(variance**0.5) * np.power(y,-0.5) + mean,'r')
plt.semilogx(y,-(variance**0.5) * np.power(y,-0.5) + mean,'r')
plt.show()

#Above shows 
#Denote by mean and variance of a random variable the quantities
μ[p]:=E(x~p(x))[x] and σ**2[p]:=E(x~p(x))[(x−μ[p])**2]

#then for lar number, sample average converges to Gaussian 
lim:n->Inf = 1/Sqrt(n)SUM((xi−μ)/σ), i=1..n  -> N(0,1)
#ie for large number 
sqrt(n)*(sample average - μ) -> N(0, σ**2) -> N(0, variance)
#or ,for large number
sample average  -> N(0,variance)*(1/sqrt(n)) + mean 
                ->(1/σ)*N(0,1)*(1/sqrt(n)) + mean 
#Note the scaling factor 1/σ
#because Gaussian  with mean μ and variance σ**2 has the form 
p(x)=1/sqrt(2*σ**2*π)*exp(−(1/(2*σ**2))*(x−μ)**2)



http://gluon.mxnet.io/chapter01_crashcourse/autograd.html
###Automatic differentiation with autograd

#Model getting better means minimizing a loss function, 
#i.e. a score that answers “how bad is our model?” 

#With neural networks, we choose loss functions to be differentiable 
#with respect to our parameters. 
#this means that for each of the model’s parameters, 
#we can determine how much increasing or decreasing it might affect the loss

import mxnet as mx
from mxnet import nd, autograd
mx.random.seed(1)

x = nd.array([[1, 2], [3, 4]])

#to store a gradient by invoking its attach_grad() method.
x.attach_grad()

#Create f(x) which needs to be differentiated 
with autograd.record():
    y = x * 2
    z = y * x  # z= 2 * x * x, z' = 4*x

#backprop by calling z.backward(). 
#When z has more than one entry, z.backward() is equivalent to mx.nd.sum(z).backward().
z.backward()
print(x.grad)
#output of z'
[[  4.   8.]
 [ 12.  16.]]
<NDArray 2x2 @cpu(0)>

##Head gradients and the chain rule
#d/dx(z(y(x))) :Gradient of z with respect to x, where z is a function of y, 
#which in turn, is a function of x. 
#chain rule d/dx(z(y(x)))=dz(y)/dy* dy(x)/dx. 

#we want x.grad to store dz/dx, we can pass in the head gradient dz/dy 
#as an input to backward(). 
#The default argument is nd.ones_like(y)
with autograd.record():
    y = x * 2
    z = y * x

head_gradient = nd.array([[10, 1.], [.1, .01]])
z.backward(head_gradient)
print(x.grad)
#output
[[ 40.           8.        ]
 [  1.20000005   0.16      ]]
<NDArray 2x2 @cpu(0)>

##Advanced example 

a = nd.random_normal(shape=3)
a.attach_grad()

with autograd.record():
    b = a * 2
    while (nd.norm(b) < 1000).asscalar():
        b = b * 2

    if (mx.nd.sum(b) > 0).asscalar():
        c = b
    else:
        c = 100 * b
head_gradient = nd.array([0.01, 1.0, .1])
c.backward(head_gradient)
print(a.grad)
#output
[   2048.  204800.   20480.]
<NDArray 3 @cpu(0)>















http://gluon.mxnet.io/chapter02_supervised-learning/linear-regression-scratch.html
###Linear regression from scratch 

from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd, gluon
mx.random.seed(1)

#specify the contexts where computation should happen
data_ctx = mx.cpu()
model_ctx = mx.cpu()

#predicting a real valued target y given a data point x.
#prediction can be expressed as a linear combination of the input features 
#(thus giving the name linear regression):
y'=w1 * x1+...+wd* xd+b
#or with matrix product X*w
y'=X*w+b

#Given a collection of data points X, and corresponding target values y, 
#find the weight vector w and bias term b (also called an offset or intercept) 
#that approximately associate data points xi with their corresponding labels y_i

#define a loss function that says how far are our predictions from the correct answers. 
#For the classical case of linear regression, we usually focus on the squared error. 
#loss will be the sum, over all examples, of the squared error (yi−y')**2) on each

#Note that squared loss heavily penalizes outliers.

#to minimize the error,by choosing values of the parameters w and b. 


##Example 
#inputs will each be sampled from a random normal distribution with mean 0 and variance 1. 
#Our features will be independent. 
#Another way of saying this is that they will have diagonal covariance. 
#The labels will be generated accoding to the true labeling function 
y[i] = 2 * X[i][0]- 3.4 * X[i][1] + 4.2 + noise 
#where the noise is drawn from a random gaussian with mean 0 and variance .01


num_inputs = 2
num_outputs = 1
num_examples = 10000

def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2

X = nd.random_normal(shape=(num_examples, num_inputs), ctx=data_ctx)
noise = .1 * nd.random_normal(shape=(num_examples,), ctx=data_ctx)
y = real_fn(X) + noise

#Notice that each row in X consists of a 2-dimensional data point (two features)
#and that each row in Y consists of a 1-dimensional target value.

print(X[0])
print(y[0])

#output
[-1.22338355  2.39233518]
<NDArray 2 @cpu(0)>

[-6.09602737]
<NDArray 1 @cpu(0)>

#visualize the correspondence between  second feature (X[:, 1]) 
#and the target values Y
import matplotlib.pyplot as plt
plt.scatter(X[:, 1].asnumpy(),y.asnumpy())
plt.show()

##Data iterators - DataLoader class, that provides way to use an ArrayDataset for training models.
#to be able to grab batches of k data points at a time, to shuffle our data

#The one requirement is that they have equal lengths along the first axis, 
#i.e., len(X) == len(y)

batch_size = 4  #how many examples we want to grab at a time.
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y),
                                      batch_size=batch_size, shuffle=True) #shuffle the data between iterations through the dataset.

#let’s just grab one batch and break out of the loop for checking train_data 
for i, (data, label) in enumerate(train_data):
    print(data, label)  #<class 'mxnet.ndarray.ndarray.NDArray'> <class 'mxnet.ndarray.ndarray.NDArray'>
    break
#output 
[[-0.14732301 -1.32803488]
 [-0.56128627  0.48301753]
 [ 0.75564283 -0.12659997]
 [-0.96057719 -0.96254188]]
<NDArray 4x2 @cpu(0)>
[ 8.25711536  1.30587864  6.15542459  5.48825312]
<NDArray 4 @cpu(0)>



##Model parameters
w = nd.random_normal(shape=(num_inputs, num_outputs), ctx=model_ctx)
b = nd.random_normal(shape=num_outputs, ctx=model_ctx)
params = [w, b]

#update these parameters to better fit our data. 
#This will involve taking the gradient (a multi-dimensional derivative) of some loss function with respect to the parameters.
#We’ll update each parameter in the direction that reduces the loss.
for param in params:
    param.attach_grad()
    
##Neural networks - Model 
#To calculate the output of the linear model, 
#multiply a given input with the model’s weights (w), and add the offset b.

def net(X):
    return mx.nd.dot(X, w) + b

##Loss function
def square_loss(yhat, y):
    return nd.mean((yhat - y) ** 2)
    
##Optimizer
#linear regression actually has a closed-form solution
#However, other models cannot be solved analytically. 
#Hence solve this problem by stochastic gradient descent. 
#At each step, we’ll estimate the gradient of the loss with respect to our weights, 
#using one batch randomly drawn from our dataset. 
#Then, we’ll update our parameters a small amount in the direction 
#that reduces the loss. 
#The size of the step is determined by the learning rate lr.

In [38]:

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


##Execute training loop
#epochs, the number of passes to make over the dataset. 
#Then for each pass, we’ll iterate through train_data, 
#grabbing batches of examples and their corresponding labels.

#For each batch
    Generate predictions (yhat) and the loss (loss) by executing a forward pass through the network.
    Calculate gradients by making a backwards pass through the network (loss.backward()).
    Update the model parameters by invoking our SGD optimizer.


epochs = 10
learning_rate = .0001
num_batches = num_examples/batch_size

for e in range(epochs):
    cumulative_loss = 0
    # inner loop
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx).reshape((-1, 1))
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        SGD(params, learning_rate)
        cumulative_loss += loss.asscalar()
    print(cumulative_loss / num_batches)
#output
24.6606138554
9.09776815639
3.36058844271
1.24549788469
0.465710770596
0.178157229481
0.0721970594548
0.0331197250206
0.0186954441286
0.0133724625537

##Training loop with ploting 

#    Re-initialize parameters because they
#    were already trained in the first loop
w[:] = nd.random_normal(shape=(num_inputs, num_outputs), ctx=model_ctx)
b[:] = nd.random_normal(shape=num_outputs, ctx=model_ctx)


#    Script to plot the losses over time
def plot(losses, X, sample_size=100):
    xs = list(range(len(losses)))
    f, (fg1, fg2) = plt.subplots(1, 2)
    fg1.set_title('Loss during training')
    fg1.plot(xs, losses, '-r')
    fg2.set_title('Estimated vs real function')
    fg2.plot(X[:sample_size, 1].asnumpy(),
             net(X[:sample_size, :]).asnumpy(), 'or', label='Estimated')
    fg2.plot(X[:sample_size, 1].asnumpy(),
             real_fn(X[:sample_size, :]).asnumpy(), '*g', label='Real')
    fg2.legend()

    plt.show()

learning_rate = .0001
losses = []
plot(losses, X)

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx).reshape((-1, 1))
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        SGD(params, learning_rate)
        cumulative_loss += loss.asscalar()

    print("Epoch %s, batch %s. Mean loss: %s" % (e, i, cumulative_loss/num_batches))
    losses.append(cumulative_loss/num_batches)

plot(losses, X)
#output 
Epoch 0, batch 2499. Mean loss: 16.9325145943
Epoch 1, batch 2499. Mean loss: 6.24987681103
Epoch 2, batch 2499. Mean loss: 2.31109857569
Epoch 3, batch 2499. Mean loss: 0.858666448605
Epoch 4, batch 2499. Mean loss: 0.323071002489
Epoch 5, batch 2499. Mean loss: 0.125603744188
Epoch 6, batch 2499. Mean loss: 0.0527891687471
Epoch 7, batch 2499. Mean loss: 0.0259436405713
Epoch 8, batch 2499. Mean loss: 0.0160523827007
Epoch 9, batch 2499. Mean loss: 0.0124009371101


http://gluon.mxnet.io/chapter02_supervised-learning/linear-regression-gluon.html
###Linear regression with gluon 

from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd, gluon

##Set the context
data_ctx = mx.cpu()
model_ctx = mx.cpu()

##Build the dataset

num_inputs = 2
num_outputs = 1
num_examples = 10000

def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
noise = 0.01 * nd.random_normal(shape=(num_examples,))
y = real_fn(X) + noise

##Load the data iterator
batch_size = 4
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y),
                                      batch_size=batch_size, shuffle=True)

##Define the model
#with gluon, compose a network from predefined layers. 
#For a linear model, the appropriate layer is called Dense. 
#It’s called a dense layer because every node in the input is connected to every node in the subsequent layer. 
#we only have one (non-input) layer here, and that layer only contains one node!


#we have an inputdimension of 2 (no of features) and an output dimension of 1. 
net = gluon.nn.Dense(1, in_units=2)  #outputs,inputs

#this model has a weight matrix and bias vector.
print(net.weight) #instances of the Parameter class
print(net.bias)   #instances of the Parameter class
#Out[37]:
Parameter dense4_weight (shape=(1, 2), dtype=None)
Parameter dense4_bias (shape=(1,), dtype=None)


#In gluon, all neural networks are made out of Blocks (gluon.Block). 
#Blocks are just units that take inputs and generate outputs. 
#Blocks also contain parameters that we can update.
net.collect_params() #The returned object is a gluon.parameter.ParameterDict
#Out[38]:
dense4_ (
  Parameter dense4_weight (shape=(1, 2), dtype=None)
  Parameter dense4_bias (shape=(1,), dtype=None)
)

##Initialize parameters
#If we don't initilize parameters , calling Model 
net(nd.array([[0,1]]))
#ERROR 
RuntimeError: Parameter dense1_weight has not been initialized..

#Solution is to initialize 
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)

#the actual initialization is deferred until we make a first forward pass. 
#In other words, the parameters are only initialized when they’re needed. 
#If we try to call net.weight.data() we’ll get the following error:

DeferredInitializationError: Parameter dense2_weight has not been initialized yet because initialization was deferred. Actual initialization happens during the first forward pass. Please pass one batch of data through the network before accessing Parameters.

##Passing data through a gluon model 
#We just sample a batch of the appropriate shape and call net as  a function. 
#This will invoke net’s forward() method., can be used for prediction 

example_data = nd.array([[4,7]])
net(example_data)

#Out[41]:
[[-1.33219385]]
<NDArray 1x1 @cpu(0)>

#Now that net is initialized, we can access each of its parameters.
print(net.weight.data())
print(net.bias.data())

#output 
[[-0.25217363 -0.04621419]]
<NDArray 1x2 @cpu(0)>

[ 0.]
<NDArray 1 @cpu(0)>

##Shape inference
#Because our parameters never come into action until we pass data through the network, 
#we don’t actually have to declare the input dimension (in_units)
net = gluon.nn.Dense(1)
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)

##Define loss
#Just like layers, and whole networks, a loss in gluon is just a Block.
square_loss = gluon.loss.L2Loss()

##Optimizer
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.0001})


##Execute training loop

epochs = 10
loss_sequence = []
num_batches = num_examples / batch_size

for e in range(epochs):
    cumulative_loss = 0
    # inner loop
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        cumulative_loss += nd.mean(loss).asscalar()
    print("Epoch %s, loss: %s" % (e, cumulative_loss / num_examples))
    loss_sequence.append(cumulative_loss)
#output 
Epoch 0, loss: 3.44980202263
Epoch 1, loss: 2.10364257665
Epoch 2, loss: 1.28279426137
Epoch 3, loss: 0.782256319318
Epoch 4, loss: 0.477034088909
Epoch 5, loss: 0.290909814427
Epoch 6, loss: 0.177411796283
Epoch 7, loss: 0.108197494675
Epoch 8, loss: 0.0659899789031
Epoch 9, loss: 0.040249745576

##Visualizing the learning curve
import matplotlib
import matplotlib.pyplot as plt

plt.figure(num=None,figsize=(8, 6))
plt.plot(loss_sequence)

# Adding some bells and whistles to the plot
plt.grid(True, which="both")
plt.xlabel('epoch',fontsize=14)
plt.ylabel('average loss',fontsize=14)

##Getting the learned model parameters
params = net.collect_params() # this returns a ParameterDict

print('The type of "params" is a ',type(params))

# A ParameterDict is a dictionary of Parameter class objects
# therefore, here is how we can read off the parameters from it.

for param in params.values():
    print(param.name,param.data())
#output 
The type of "params" is a  <class 'mxnet.gluon.parameter.ParameterDict'>
dense5_weight
[[ 1.7913872  -3.10427046]]
<NDArray 1x2 @cpu(0)>
dense5_bias
[ 3.85259581]
<NDArray 1 @cpu(0)>





http://gluon.mxnet.io/chapter02_supervised-learning/logistic-regression-gluon.html
###Binary classification with logistic regression

#there are only two categories
#the positive class yi=1 and the negative class yi=0

##support vector machines 
#choose a line that maximizes the marigin to the closest data points on either side of the decision boundary. 
#In these appraoches, only the points closest to the decision boundary (the support vectors) 
#actually influence the choice of the linear separator.

#With neural networks
#we train a probabilistic classifiers which estimates, for each data point, 
#the conditional probability that it belongs to the positive class


#A regular linear model is a poor choice here 
#because it can output values greater than 1 or less than 0. 

#To coerce reasonable answers from our model, modify it , 
#by running the linear function through a sigmoid activation function σ
y'=σ(w^T * x+b).

#The sigmoid function σ
#called a squashing function or a logistic function - thus the name logistic regression - 
#maps a real-valued input to the range 0 to 1. 
σ(z)=1/(1+e**−z)

#Note that and input of 0 gives a value of .5. 
#predict positive whenever the probability is greater than .5 
#and negative whenever the probability is less than .5

import mxnet as mx
from mxnet import nd, autograd, gluon
import matplotlib.pyplot as plt

def logistic(z):
    return 1. / (1. + nd.exp(-z))

x = nd.arange(-5, 5, .1)
y = logistic(x)

plt.plot(x.asnumpy(),y.asnumpy())
plt.show()

##Binary cross-entropy loss
#choose the weights that give the actual labels in the training data the highest probability.

ℓ(y,y')=−SUM(yi*logy'i + (1−yi)*log(1−y'i)), i=1..n 

#this loss function is commonly called log loss 
#and is also commonly referred to as binary cross entropy. 

#It is a special case of negative log likelihood. 
#And it is a special case of cross-entropy, 
#which can apply to the multi-class (>2) setting.

##Adult dataset taken from the UCI repository.
#In its original form, the dataset contained 14 features, including age, education, occupation, sex, native-country, among others. 

# the data have been re-processed to 123 binary features 
#each representing quantiles among the original features. 

#The label is a binary indicator indicating 
#whether the person corresponding to each row made more (yi=1) or less (yi=0) than $50,000 of income in 1994. 
#The data consists of lines like the following:
-1 4:1 6:1 15:1 21:1 35:1 40:1 57:1 63:1 67:1 73:1 74:1 77:1 80:1 83:1 \n
#The first entry in each row is the value of the label. 
#The following tokens are the indices of the non-zero features

data_ctx = mx.cpu()
# Change this to `mx.gpu(0) if you would like to train on an NVIDIA GPU
model_ctx = mx.cpu()

with open("../data/adult/a1a.train") as f:
    train_raw = f.read()

with open("../data/adult/a1a.test") as f:
    test_raw = f.read()
    
    
def process_data(raw_data):
    train_lines = raw_data.splitlines()
    num_examples = len(train_lines)
    num_features = 123
    X = nd.zeros((num_examples, num_features), ctx=data_ctx)
    Y = nd.zeros((num_examples, 1), ctx=data_ctx)
    for i, line in enumerate(train_lines):
        tokens = line.split()
        label = (int(tokens[0]) + 1) / 2  # Change label from {-1,1} to {0,1}
        Y[i] = label
        for token in tokens[1:]:
            index = int(token[:-2]) - 1
            X[i, index] = 1
    return X, Y

Xtrain, Ytrain = process_data(train_raw)
Xtest, Ytest = process_data(test_raw)

print(Xtrain.shape)
print(Ytrain.shape)
print(Xtest.shape)
print(Ytest.shape)


##Instantiate a dataloader
batch_size = 64

train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(Xtrain, Ytrain),
                                      batch_size=batch_size, shuffle=True)

test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(Xtest, Ytest),
                                      batch_size=batch_size, shuffle=True)

##Define the model
net = gluon.nn.Dense(1)
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)

##Instantiate an optimizer
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})

##Define log loss

def log_loss(output, y):
    yhat = logistic(output)
    return  - nd.sum(  y * nd.log(yhat) + (1-y) * nd.log(1-yhat))


epochs = 30
loss_sequence = []
num_examples = len(Xtrain)

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = log_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        cumulative_loss += nd.sum(loss).asscalar()
    print("Epoch %s, loss: %s" % (e, cumulative_loss ))
    loss_sequence.append(cumulative_loss)

##Visualize the learning curve
# plot the convergence of the estimated loss function
%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt

plt.figure(num=None,figsize=(8, 6))
plt.plot(loss_sequence)

# Adding some bells and whistles to the plot
plt.grid(True, which="both")
plt.xlabel('epoch',fontsize=14)
plt.ylabel('average loss',fontsize=14)

##Calculating accuracy
num_correct = 0.0
num_total = len(Xtest)
for i, (data, label) in enumerate(test_data):
    data = data.as_in_context(model_ctx)
    label = label.as_in_context(model_ctx)
    output = net(data)
    prediction = (nd.sign(output) + 1) / 2
    num_correct += nd.sum(prediction == label)
print("Accuracy: %0.3f (%s/%s)" % (num_correct.asscalar()/num_total, num_correct.asscalar(), num_total))

#A naive classifier would predict that nobody had an income greater than $50k (the majority class). 
#This classifier would achieve an accuracy of roughly 75%. 
#By contrast, our classifier gets an accuracy of .84



http://gluon.mxnet.io/chapter02_supervised-learning/softmax-regression-scratch.html
###Multiclass logistic regression from scratch

#Nearly all neural networks that we’ll build in the real world consist of these same fundamental parts. The main differences will be the type and scale of the data and the complexity of the models. And every year or two, a new hipster optimizer comes around, but at their core they’re all subtle variations of stochastic gradient descent.

#problem where each example could belong to one of k classes

#binary classification - activation function on the final layer was crucial because it forced our outputs to take values in the range [0,1]. 
#That allowed us to interpret these outputs as probabilties.

#Given k classes, 
#the most naive way to solve a multiclass classification problem is to train k different binary classifiers fi(x).
#We could then predict that an example x belongs to the class i 
#for which the probability that the label applies is highest :max_{i} fi

#OR 
#We could force the output layer to be a discrete probability distribution over the k classes. 
#To be valid probabity distribution, we’ll want the output y' to 
    contain only non-negative values, 
    and sum to 1. 
#We accomplish this by using the softmax function. 
softmax(z)=e^z/SUM(e^zi), i =1..k 

#Because now we have k outputs and not 1 
#we’ll need weights connecting each of our inputs to each of our outputs

#We can represent these weights one for each input node, 
#output node pair in a matrix W
y'=softmax(x*W+b)

#This model is sometimes called multiclass logistic regression. 
#Other common names for it include softmax regression and multinomial regression. 

#Assume we have d inputs and k outputs
z   = x *   W +b 
1×k   1×d  d×k 1×k
#one-hot encode the output label, for example y'=5 would be y'one−hot=[0,0,0,0,1,0,0,0,0,0] 
#when one-hot encoded for a 10-class classfication problem

y'one−hot=softmax_one−hot(  z)
1xk                         1xk

#When we input a batch of m training examples,
#we would have matrix X,m×d that is the vertical stacking of individual training examples xi, 
#due to the choice of using row vectors.
Y=softmax(Z)=softmax(X*W + B)
#where matrix B, m×k is formed by having m copies of b 
B = [b
     b 
     ...
     b]

#In actual implementation we can often get away with using b directly instead of B in the equation for Z
#due to broadcasting.

#Each row of matrix Z,m×k corresponds to one traning example. 
#The softmax function operates on each row of matrix Z and returns a matrix Y m×k,
#each row of which corresponds to the one-hot encoded prediction of one training example.
     
     
     
##Example -We’re going to classify images of handwritten digits 

from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
mx.random.seed(1)
data_ctx = mx.cpu()
model_ctx = mx.cpu()

#The MNIST dataset
# each a 28 by 28 centrally cropped black & white photograph of a handwritten digit. 
#Our task will be come up with a model that can associate each image with the digit (0-9) that it depicts.


# cast data and label to floats and normalize data to range [0, 1]:
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)
mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)

#There are two parts of the dataset for training and testing. 
#Each part has N items and each item is a tuple of an image and a label:
image, label = mnist_train[0]
print(image.shape, label)

#Note that each image has been formatted as a 3-tuple (height, width, channel). 
#For color images, the channel would have 3 dimensions (red, green and blue).
 
num_inputs = 784  #28*28
num_outputs = 10
num_examples = 60000

#Machine learning libraries generally expect to find images in (batch, channel, height, width) format. 
#most libraries for visualization prefer (height, width, channel). 
#matplotlib expects either (height, width) or (height, width, channel) with RGB channels, 
#so let’s broadcast our single channel to 3.

im = mx.nd.tile(image, (1,1,3))
print(im.shape)

import matplotlib.pyplot as plt
plt.imshow(im.asnumpy())
plt.show()

##Load the data iterator
batch_size = 64
train_data = mx.gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)

test_data = mx.gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

##Allocate model parameters
#flatten each image into a single 1D vector with 28x28 = 784 components. 
#we want to assign a probability to each of the classes P(Y=c|X) given the input X. 
#In order to do this we’re going to need one vector of 784 weights for each class,
#connecting each feature to the corresponding output. 
#Because there are 10 classes, we can collect these weights together in a 784 by 10 matrix.

#We’ll also want to allocate one offset for each of the outputs. 
#We call these offsets the bias term and collect them in the 10-dimensional array b.



W = nd.random_normal(shape=(num_inputs, num_outputs),ctx=model_ctx)
b = nd.random_normal(shape=num_outputs,ctx=model_ctx)

params = [W, b]
for param in params:
    param.attach_grad()
    
##Multiclass logistic regression
#to assign each input X to one of L classes.
#linearly map our input X onto 10 different real valued outputs y_linear. 
#we’ll want to normalize them so that they are non-negative and sum to 1. 
#This normalization allows us to interpret the output yhat as a valid probability distribution.

def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear, axis=1).reshape((-1,1)))
    norms = nd.sum(exp, axis=1).reshape((-1,1))
    return exp / norms

sample_y_linear = nd.random_normal(shape=(2,10))
sample_yhat = softmax(sample_y_linear)
print(sample_yhat)

#Let’s confirm that indeed all of our rows sum to 1.
print(nd.sum(sample_yhat, axis=1))


##Define the model
def net(X):
    y_linear = nd.dot(X, W) + b
    yhat = softmax(y_linear)
    return yhat

##The cross-entropy loss function
#The basic idea is that we’re going to take a target Y that has been formatted as a one-hot vector, 
#meaning one value corresponding to the correct label is set to 1 
#and the others are set to 0, e.g. [0, 1, 0, 0, 0, 0, 0, 0, 0, 0].

#The basic idea of cross-entropy loss is that 
#we only care about how much probability the prediction assigned to the correct label. 
#In other words, for true label 2, we only care about the component of yhat corresponding to 2. Cross-entropy attempts to maximize the log-likelihood given to the correct labels.


def cross_entropy(yhat, y):
    return - nd.sum(y * nd.log(yhat+1e-6))

##Optimizer
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

##Write evaluation loop to calculate accuracy 
def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1,784))
        label = label.as_in_context(model_ctx)
        label_one_hot = nd.one_hot(label, 10)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()
    
    
##Execute training loop
epochs = 5
learning_rate = .005

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1,784))
        label = label.as_in_context(model_ctx)
        label_one_hot = nd.one_hot(label, 10)
        with autograd.record():
            output = net(data)
            loss = cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)
        cumulative_loss += nd.sum(loss).asscalar()


    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))

##Using the model for prediction
# Define the function to do prediction
def model_predict(net,data):
    output = net(data)
    return nd.argmax(output, axis=1)

# let's sample 10 random data points from the test set
sample_data = mx.gluon.data.DataLoader(mnist_test, 10, shuffle=True)
for i, (data, label) in enumerate(sample_data):
    data = data.as_in_context(model_ctx)
    print(data.shape)
    im = nd.transpose(data,(1,0,2,3))
    im = nd.reshape(im,(28,10*28,1))
    imtiles = nd.tile(im, (1,1,3))

    plt.imshow(imtiles.asnumpy())
    plt.show()
    pred=model_predict(net,data.reshape((-1,784)))
    print('model predictions are:', pred)
    break
    
#We can get nearly 90% accuracy at this task just by training a linear model for a few second
#You might reasonably conclude that this problem is too easy to be taken seriously by experts.

###Multiclass logistic regression with gluon
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np

data_ctx = mx.cpu()
model_ctx = mx.cpu()

batch_size = 64
num_inputs = 784
num_outputs = 10
num_examples = 60000
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)
train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                              batch_size, shuffle=False)
                              
                              
##Model 
net = gluon.nn.Dense(num_outputs)
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
#metric 
def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1,784))
        label = label.as_in_context(model_ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]
    
##Execute training loop
epochs = 10
moving_loss = 0.

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1,784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)
        cumulative_loss += nd.sum(loss).asscalar()

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))
#output 
Epoch 0. Loss: 0.000342435105642, Train_acc 0.793733333333, Test_acc 0.809
Epoch 1. Loss: 0.000266353193919, Train_acc 0.83805, Test_acc 0.8477
Epoch 2. Loss: 0.000140365982056, Train_acc 0.856316666667, Test_acc 0.8648
Epoch 3. Loss: 0.000119470739365, Train_acc 0.86695, Test_acc 0.874
Epoch 4. Loss: 0.000254932610194, Train_acc 0.8731, Test_acc 0.8796
Epoch 5. Loss: 0.000143766593933, Train_acc 0.879266666667, Test_acc 0.8847
Epoch 6. Loss: 0.000247673273087, Train_acc 0.882366666667, Test_acc 0.8863
Epoch 7. Loss: 0.000343579641978, Train_acc 0.88615, Test_acc 0.8896
Epoch 8. Loss: 0.000479016272227, Train_acc 0.88865, Test_acc 0.8911
Epoch 9. Loss: 0.000274674447378, Train_acc 0.8905, Test_acc 0.8919

##Visualize predictions
import matplotlib.pyplot as plt

def model_predict(net,data):
    output = net(data.as_in_context(model_ctx))
    return nd.argmax(output, axis=1)

# let's sample 10 random data points from the test set
sample_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                              10, shuffle=True)
for i, (data, label) in enumerate(sample_data):
    data = data.as_in_context(model_ctx)
    print(data.shape)
    im = nd.transpose(data,(1,0,2,3))
    im = nd.reshape(im,(28,10*28,1))
    imtiles = nd.tile(im, (1,1,3))

    plt.imshow(imtiles.asnumpy())
    plt.show()
    pred=model_predict(net,data.reshape((-1,784)))
    print('model predictions are:', pred)
    break
#output 
(10, 28, 28, 1)
model predictions are:
[ 9.  9.  0.  4.  7.  6.  8.  2.  7.  3.]
<NDArray 10 @cpu(0)>





http://gluon.mxnet.io/chapter02_supervised-learning/regularization-scratch.html
###Overfitting and regularization

The goal of supervised learning is to produce models 
that generalize to previously unseen data. 
When a model achieves low error on training data 
but performs much worse on test data, we say that the model has overfit. 

The quantity we really care about is the test error e. 
Because this quantity reflects the error of our model 
when generalized to previously unseen data, 
we commonly call it the generalization error. 

When we have simple models and abundant data, 
we expect the generalization error to resemble the training error. 

When we work with more complex models and fewer examples, 
we expect the training error to go down but the generalization gap to grow

Many factors govern whether a model will generalize well. 
For example a model with more parameters might be considered more complex. 
A model whose parameters can take a wider range of values might be more complex. 
Often with neural networks, we think of a model that takes more training steps 
as more complex, and one subject to early stopping as less complex.

It can be difficult to compare the complexity among members of very different model classes 
(say decision trees versus neural networks). 

A model that can readily explain arbitrary facts is what statisticians view 
as complex, whereas one that has only a limited expressive power 
but still manages to explain the data well is probably closer to the truth

##Few factors that tend to influence the generalizability of a model class:
    The number of tunable parameters. 
            When the number of tunable parameters, 
            sometimes denoted as the number of degrees of freedom, is large, 
            models tend to be more susceptible to overfitting.
    The values taken by the parameters. 
            When weights can take a wider range of values, 
            models can be more susceptible to over fitting.
    The number of training examples. 
        It’s trivially easy to overfit a dataset containing only one 
        or two examples even if your model is simple. 
        But overfitting a dataset with millions of examples 
        requires an extremely flexible model.

#When classifying handwritten digits before, 
#we didn’t overfit because our 60,000 training examples far out numbered 
#the 784×10=7,840 weights plus 10 bias terms, 
#which gave us far fewer parameters than training examples

##Overfitting example with MINST data 
#Decrease the training examples for showcasing overfitting 

from __future__ import print_function
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import autograd
import numpy as np
ctx = mx.cpu()
mx.random.seed(1)


# for plotting purposes
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

##Load the MNIST dataset
mnist = mx.test_utils.get_mnist()
num_examples = 1000
batch_size = 64
train_data = mx.gluon.data.DataLoader(
    mx.gluon.data.ArrayDataset(mnist["train_data"][:num_examples],
                               mnist["train_label"][:num_examples].astype(np.float32)),
                               batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(
    mx.gluon.data.ArrayDataset(mnist["test_data"][:num_examples],
                               mnist["test_label"][:num_examples].astype(np.float32)),
                               batch_size, shuffle=False)

#Allocate model parameters and define model
W = nd.random_normal(shape=(784,10))
b = nd.random_normal(shape=10)

params = [W, b]

for param in params:
    param.attach_grad()

def net(X):
    y_linear = nd.dot(X, W) + b
    yhat = nd.softmax(y_linear, axis=1)
    return yhat

##Define loss function and optimizer
def cross_entropy(yhat, y):
    return - nd.sum(y * nd.log(yhat), axis=0, exclude=True)

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

##Write evaluation loop to calculate accuracy
def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    loss_avg = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        output = net(data)
        loss = cross_entropy(output, label_one_hot)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
        loss_avg = loss_avg*i/(i+1) + nd.mean(loss).asscalar()/(i+1)
    return (numerator / denominator).asscalar(), loss_avg
    

##Write a utility function to plot the learning curves

def plot_learningcurves(loss_tr,loss_ts, acc_tr,acc_ts):
    xs = list(range(len(loss_tr)))

    f = plt.figure(figsize=(12,6))
    fg1 = f.add_subplot(121)
    fg2 = f.add_subplot(122)

    fg1.set_xlabel('epoch',fontsize=14)
    fg1.set_title('Comparing loss functions')
    fg1.semilogy(xs, loss_tr)
    fg1.semilogy(xs, loss_ts)
    fg1.grid(True,which="both")

    fg1.legend(['training loss', 'testing loss'],fontsize=14)

    fg2.set_title('Comparing accuracy')
    fg1.set_xlabel('epoch',fontsize=14)
    fg2.plot(xs, acc_tr)
    fg2.plot(xs, acc_ts)
    fg2.grid(True,which="both")
    fg2.legend(['training accuracy', 'testing accuracy'],fontsize=14)

##Execute training loop

epochs = 1000
moving_loss = 0.
niter=0

loss_seq_train = []
loss_seq_test = []
acc_seq_train = []
acc_seq_test = []


for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        with autograd.record():
            output = net(data)
            loss = cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, .001)
        #  Keep a moving average of the losses
        niter +=1
        moving_loss = .99 * moving_loss + .01 * nd.mean(loss).asscalar()
        est_loss = moving_loss/(1-0.99**niter)

    test_accuracy, test_loss = evaluate_accuracy(test_data, net)
    train_accuracy, train_loss = evaluate_accuracy(train_data, net)

    # save them for later
    loss_seq_train.append(train_loss)
    loss_seq_test.append(test_loss)
    acc_seq_train.append(train_accuracy)
    acc_seq_test.append(test_accuracy)

    if e % 100 == 99:
        print("Completed epoch %s. Train Loss: %s, Test Loss %s, Train_acc %s, Test_acc %s" %
              (e+1, train_loss, test_loss, train_accuracy, test_accuracy))


## Plotting the learning curves
plot_learningcurves(loss_seq_train,loss_seq_test,acc_seq_train,acc_seq_test)


##Result 
#By the 700th epoch, our model achieves 100% accuracy on the training data. 
#However, it only classifies 75% of the test examples accurately. 
#This is a clear case of overfitting. 

#At a high level, there’s a reason this went wrong. 
#Because we have 7450 parameters and only 1000 data points, 
#there are actually many settings of the parameters 
#that could produce 100% accuracy on training data.




##Solution to Overfitting - regularization.

#when a model is overfitting, its training error is substantially lower 
#than its test error. 

#Regularization techniques attempt to trade off training performance 
#in exchange for lowering our test error.

#There are several straightforward techniques 
    Make model less complex. 
        One way to do this would be to lower the number of free parameters. 
        For example, we could throw away some subset of our input features 
        (and thus the corresponding parameters) that are least informative.
    Another approach is to limit the values that our weights might take. 
        One common approach is to force the weights to take small values.
        We can accomplish this by changing our optimization objective 
        to penalize the value of our weights. 
        The most popular regularizer is the ℓ**2 norm. 
        For linear models, ℓ**2 regularization has the additional benefit 
        that it makes the solution unique, 
        even when  model is overparametrized.
        
##ℓ**2 Regularization
    loss function = ∑(y'−y)**2+λ*∥w∥**2

#∥w∥ is the ℓ**2 norm and λ is a hyper-parameter that determines how aggressively 
#we want to push the weights towards 0. 


def l2_penalty(params):
    penalty = nd.zeros(shape=1)
    for param in params:
        penalty = penalty + nd.sum(param ** 2)
    return penalty

##Re-initializing the parameters
for param in params:
    param[:] = nd.random_normal(shape=param.shape)

##Training L2-regularized logistic regression
epochs = 1000
moving_loss = 0.
l2_strength = .1
niter=0

loss_seq_train = []
loss_seq_test = []
acc_seq_train = []
acc_seq_test = []


for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        with autograd.record():
            output = net(data)
            loss = nd.sum(cross_entropy(output, label_one_hot)) + l2_strength * l2_penalty(params)
        loss.backward()
        SGD(params, .001)
        #  Keep a moving average of the losses
        niter +=1
        moving_loss = .99 * moving_loss + .01 * nd.mean(loss).asscalar()
        est_loss = moving_loss/(1-0.99**niter)


    test_accuracy, test_loss = evaluate_accuracy(test_data, net)
    train_accuracy, train_loss = evaluate_accuracy(train_data, net)

    # save them for later
    loss_seq_train.append(train_loss)
    loss_seq_test.append(test_loss)
    acc_seq_train.append(train_accuracy)
    acc_seq_test.append(test_accuracy)

    if e % 100 == 99:
        print("Completed epoch %s. Train Loss: %s, Test Loss %s, Train_acc %s, Test_acc %s" %
              (e+1, train_loss, test_loss, train_accuracy, test_accuracy))

## Plotting the learning curves
plot_learningcurves(loss_seq_train,loss_seq_test,acc_seq_train,acc_seq_test)

##Analysis
#By adding L2 regularization we were able to increase the performance on test data 
#from 75% accuracy to 83% accuracy. That’s a 32% reduction in error.

#Note that L2 regularization is just one of many ways of controlling capacity. 
#Basically we assumed that small weight values are good. 
#But there are many more ways to constrain the values of the weights:
    We could require that the total sum of the weights is small. 
    That is what L1 regularization does via the penalty SUM(|wi|)
    
    We could require that the largest weight is not too large. 
    This is what L-Inf regularization does via the penalty max(|wi|)

    We could require that the number of nonzero weights is small, 
    i.e. that the weight vectors are sparse. This is what the L0 penalty does, 
    i.e. SUM(I{wi≠0}). 
    This penalty is quite difficult to deal with explicitly since it is nonsmooth.


##All of this raises the question of why regularization is any good. 
#After all, choice is good and giving our model more flexibility ought to be better 
#(e.g. there are plenty of papers which show improvements on ImageNet using deeper networks). 

#What is happening is somewhat more subtle. 
#Allowing for many different parameter values allows our model 
#to cherry pick a combination that is just right for all the training data it sees, 
#without really learning the underlying mechanism.

#a few simple rules of thumb suffice:
    Fewer parameters tend to be better than more parameters.
    Better engineering for a specific problem 
        that takes the actual problem into account will lead to better models, 
        due to the prior knowledge that data scientists have about the problem 
        at hand.
    L2 is easier to optimize for than L1. 
        In particular, many optimizers will not work well out of the box for L1
        Using the latter requires something called proximal operators.
    Dropout and other methods to make the model robust to perturbations 
        in the data often work better than off-the-shelf L2 regularization.





http://gluon.mxnet.io/chapter02_supervised-learning/regularization-gluon.html
###Overfitting and regularization (with gluon)
from __future__ import print_function
import mxnet as mx
from mxnet import autograd
from mxnet import gluon
import mxnet.ndarray as nd
import numpy as np
ctx = mx.cpu()

# for plotting purposes
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

#The MNIST Dataset

mnist = mx.test_utils.get_mnist()
num_examples = 1000
batch_size = 64
train_data = mx.gluon.data.DataLoader(
    mx.gluon.data.ArrayDataset(mnist["train_data"][:num_examples],
                               mnist["train_label"][:num_examples].astype(np.float32)),
                               batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(
    mx.gluon.data.ArrayDataset(mnist["test_data"][:num_examples],
                               mnist["test_label"][:num_examples].astype(np.float32)),
                               batch_size, shuffle=False)

#Multiclass Logistic Regression
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(10))

#Parameter initialization
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

#Softmax Cross Entropy Loss
loss = gluon.loss.SoftmaxCrossEntropyLoss()

#Optimizer
#By default gluon tries to keep the coefficients from diverging 
#by using a weight decay penalty. 
#to get the real overfitting experience we need to switch it off. 
#We do this by passing 'wd': 0.0' when we instantiate the trainer.

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01, 'wd': 0.0})

#Evaluation Metric

def evaluate_accuracy(data_iterator, net, loss_fun):
    acc = mx.metric.Accuracy()
    loss_avg = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        output = net(data)
        loss = loss_fun(output, label)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
        loss_avg = loss_avg*i/(i+1) + nd.mean(loss).asscalar()/(i+1)
    return acc.get()[1], loss_avg

def plot_learningcurves(loss_tr,loss_ts, acc_tr,acc_ts):
    xs = list(range(len(loss_tr)))

    f = plt.figure(figsize=(12,6))
    fg1 = f.add_subplot(121)
    fg2 = f.add_subplot(122)

    fg1.set_xlabel('epoch',fontsize=14)
    fg1.set_title('Comparing loss functions')
    fg1.semilogy(xs, loss_tr)
    fg1.semilogy(xs, loss_ts)
    fg1.grid(True,which="both")

    fg1.legend(['training loss', 'testing loss'],fontsize=14)

    fg2.set_title('Comparing accuracy')
    fg1.set_xlabel('epoch',fontsize=14)
    fg2.plot(xs, acc_tr)
    fg2.plot(xs, acc_ts)
    fg2.grid(True,which="both")
    fg2.legend(['training accuracy', 'testing accuracy'],fontsize=14)

#Execute training loop
epochs = 700
moving_loss = 0.
niter=0

loss_seq_train = []
loss_seq_test = []
acc_seq_train = []
acc_seq_test = []

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            cross_entropy = loss(output, label)
        cross_entropy.backward()
        trainer.step(data.shape[0])
        #  Keep a moving average of the losses
        niter +=1
        moving_loss = .99 * moving_loss + .01 * nd.mean(cross_entropy).asscalar()
        est_loss = moving_loss/(1-0.99**niter)

    test_accuracy, test_loss = evaluate_accuracy(test_data, net, loss)
    train_accuracy, train_loss = evaluate_accuracy(train_data, net, loss)
    # save them for later
    loss_seq_train.append(train_loss)
    loss_seq_test.append(test_loss)
    acc_seq_train.append(train_accuracy)
    acc_seq_test.append(test_accuracy)
    if e % 20 == 0:
        print("Completed epoch %s. Train Loss: %s, Test Loss %s, Train_acc %s, Test_acc %s" %
              (e+1, train_loss, test_loss, train_accuracy, test_accuracy))

#Plotting the learning curves
plot_learningcurves(loss_seq_train,loss_seq_test,acc_seq_train,acc_seq_test)

##Regularization
#set the weight decay to something nonzero.
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx, force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01, 'wd': 0.001})

moving_loss = 0.
niter=0
loss_seq_train = []
loss_seq_test = []
acc_seq_train = []
acc_seq_test = []

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            cross_entropy = loss(output, label)
        cross_entropy.backward()
        trainer.step(data.shape[0])

        ##
        #  Keep a moving average of the losses
        ##
        niter +=1
        moving_loss = .99 * moving_loss + .01 * nd.mean(cross_entropy).asscalar()
        est_loss = moving_loss/(1-0.99**niter)

    test_accuracy, test_loss = evaluate_accuracy(test_data, net,loss)
    train_accuracy, train_loss = evaluate_accuracy(train_data, net, loss)

    # save them for later
    loss_seq_train.append(train_loss)
    loss_seq_test.append(test_loss)
    acc_seq_train.append(train_accuracy)
    acc_seq_test.append(test_accuracy)

    if e % 20 == 0:
        print("Completed epoch %s. Train Loss: %s, Test Loss %s, Train_acc %s, Test_acc %s" %
              (e+1, train_loss, test_loss, train_accuracy, test_accuracy))

## Plotting the learning curves
plot_learningcurves(loss_seq_train,loss_seq_test,acc_seq_train,acc_seq_test)

#the test accuracy improves a bit. 
#Note that the amount by which it improves actually depends on the amount of weight decay. 
#try and experiment with different extents of weight decay. 
#For instance, a larger weight decay (e.g. 0.01) will lead to inferior performance, 
#one that’s larger still (0.1) will lead to terrible results. 
#This is one of the reasons why tuning parameters is quite so important 
#in getting good experimental results in practice.





http://gluon.mxnet.io/chapter02_supervised-learning/environment.html
###Environment
#Basically learning can be expressed as , 
#given (x,y) and x  from p(x) distribution , train a model for p(y|x)
#Use that model for unseen data x from same p(x) distribution to predict y from p(y|x)


##Covariate Shift - training data and test data must come from same samples(having similar parameters/features)
#don't try to use a model trained on a some other distribution

#the situation where the distribution over the covariates (aka training data) 
#is shifted on test data relative to the training case. 
#Mathematically speaking, we are referring the case where p(x) changes 
#but p(y|x) remains unchanged.

##Concept Shift
#Now the distribution p(y|x) changes from training data to test data 


##Covariate Shift Correction
we want to estimate some dependency p(y|x) for which we have labeled data (xi,yi). 
the observations xi are drawn from some distribution q(x) rather 
than the ‘proper’ distribution p(x). 

To make progress, we need to reflect about what exactly is happening 
during training: we iterate over training data 
and associated labels {(x1,y1),…(ym,ym)} 
and update the weight vectors of the model after every minibatch. 

Depending on the situation we also apply some penalty to the parameters, 
e.g. L2 regularization

To correct Covariate Shift,  we need to re-weight each instance 
by the ratio of probabilities that it would have been drawn 
from the correct distribution β(x):=p(x)/q(x). 
we do not know that ratio, so before we can do anything useful 
we need to estimate it. 
 
Note that for any such approach, we need samples drawn 
from both distributions - the ‘true’ p, e.g. by access to training data, 
and the one used for generating the training set q 

In this case there exists a very effective approach that will give 
almost as good results: logistic regression. 
This is all that is needed to compute estimate probability ratios.
We learn a classifier to distinguish between data drawn from p(x) 
and data drawn from q(x). 

For simplicity’s sake assume that we have an equal number of instances 
from both distributions, 
denoted by xi~p(x) and xi'~q(x) respectively. 
Now denote by zi labels which are 1 for data drawn from p 
and -1 for data drawn from q

Then the probability in a mixed dataset is given by
p(z=1|x)=p(x)/(p(x)+q(x)) 
and hence 
p(z=1|x)/p(z=−1|x)=p(x)/q(x)

Hence, if we use a logistic regression approach where p(z=1|x)=1/(1+exp(−f(x))
it follows (after some simple algebra) that β(x)=exp(f(x)). 

In summary, we need to solve two problems: 
    first one to distinguish between data drawn from both distributions, 
    and then a reweighted minimization problem where we weigh terms by β , e.g. via the head gradients. 

#Here’s a prototypical algorithm for that purpose:

CovariateShiftCorrector(X, Z)
    X: Training dataset (without labels)
    Z: Test dataset (without labels)

    generate training set with {(x_i, -1) ... (z_j, 1)}
    train binary classifier using logistic regression to get function f
    weigh data using beta_i = exp(f(x_i)) or
                     beta_i = min(exp(f(x_i)), c)
    use weights beta_i for training on X with labels Y

#Generative Adversarial Networks use the very idea described above 
#to engineer a data generator 
#such that it cannot be distinguished from a reference dataset. 

##Concept Shift Correction

Concept shift is much harder to fix in a principled manner. 
For instance, in a situation where suddenly the problem changes 
from distinguishing cats from dogs to one of distinguishing white f
rom black animals, it will be unreasonable to assume 
that we can do much better than just training from scratch using the new labels. 

Fortunately, in practice, such extreme shifts almost never happen. 
Instead, what usually happens is that the task keeps on changing slowly. 
In such cases, we can use the same approach 
that we used for training networks to make them adapt to the change in the data. 

In other words, we use the existing network weights 
and simply perform a few update steps with the new data rather than training 
from scratch.




http://gluon.mxnet.io/chapter03_deep-neural-networks/mlp-scratch.html
###Multilayer perceptrons from scratch

#Linear regression: multiclass logistic regression (also called softmax regression) 
#inputs directly onto  outputs through a single linear transformation.
y' =softmax(Wx+b)

#Linearity means that given an output of interest, 
#for each input, increasing the value of the input should 
#either drive the value of the output up or drive it down, 
#irrespective of the value of the other inputs.
 
#To understand image generally requires more complex relationships between  inputs and outputs, 
#considering the possibility that  pattern might be characterized by interactions among the many features. 

In these cases, linear models will have low accuracy. 
We can model a more general class of functions by incorporating 
one or more hidden layers. 

The easiest way to do this is to stack a bunch of layers of neurons 
on top of each other. 
Each layer feeds into the layer above it, until we generate an output. 
This architecture is commonly called a “multilayer perceptron”. 

h1=ϕ(W1 * x+b1)
h2=ϕ(W2 * h1+b2)
...
hn=ϕ(Wn * hn−1+bn)

Note that each layer requires its own set of parameters. 
For each hidden layer, we calculate its value by first applying a linear function 
to the acivations of the layer below,
and then applying an element-wise nonlinear activation function,as ϕ

Finally, given the topmost hidden layer, we’ll generate an output. 
For  multiclass classification, softmax activation in the output layer.
y^=softmax(Wy * hn+by)

#Graphically, a multilayer perceptron could be depicted like this:
multilayer-perceptron.png

Multilayer perceptrons can account for complex interactions in the inputs 
because the hidden neurons depend on the values of each of the inputs. 
It’s easy to design a hidden node that that does arbitrary computation, 
such as, for instance, logical operations on its inputs. 
And it’s even widely known that multilayer perceptrons are universal approximators. That means that even for a single-hidden-layer neural network, with enough nodes, and the right set of weights, it could model any function at all! Actually learning that function is the hard part. And it turns out that we can approximate functions much more compactly if we use deeper (vs wider) neural network


##Imports
from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon



data_ctx = mx.cpu()
model_ctx = mx.cpu()
# model_ctx = mx.gpu(1)



num_inputs = 784
num_outputs = 10
batch_size = 64
num_examples = 60000
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)
train_data = gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)

##Allocate parameters
#  Set some constants so it's easy to modify the network later
num_hidden = 256
weight_scale = .01


#  Allocate parameters for the first hidden layer
W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale, ctx=model_ctx)
b1 = nd.random_normal(shape=num_hidden, scale=weight_scale, ctx=model_ctx)


#  Allocate parameters for the second hidden layer
W2 = nd.random_normal(shape=(num_hidden, num_hidden), scale=weight_scale, ctx=model_ctx)
b2 = nd.random_normal(shape=num_hidden, scale=weight_scale, ctx=model_ctx)


#  Allocate parameters for the output layer
W3 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale, ctx=model_ctx)
b3 = nd.random_normal(shape=num_outputs, scale=weight_scale, ctx=model_ctx)

params = [W1, b1, W2, b2, W3, b3]

for param in params:
    param.attach_grad()

##Activation functions

#If we compose a multi-layer network but use only linear operations, 
#then our entire network will still be a linear function. 

#To give our model the capacity to capture nonlinear functions, 
#we’ll need to interleave our linear operations with activation functions. 
#In this case, we’ll use the rectified linear unit (ReLU):

def relu(X):
    return nd.maximum(X, nd.zeros_like(X))

##Softmax output

def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear))
    partition = nd.nansum(exp, axis=0, exclude=True).reshape((-1, 1))
    return exp / partition

##The softmax cross-entropy loss function
#we calculated our model’s output and then ran this output 
#through the cross-entropy loss function:

def cross_entropy(yhat, y):
    return - nd.nansum(y * nd.log(yhat), axis=0, exclude=True)

#computationally, above may cause underflow or overflow 
#Solution is to merge both 

def softmax_cross_entropy(yhat_linear, y):
    return - nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)

##Define the model
def net(X):
    #  Compute the first hidden layer
    h1_linear = nd.dot(X, W1) + b1
    h1 = relu(h1_linear)

    #  Compute the second hidden layer
    h2_linear = nd.dot(h1, W2) + b2
    h2 = relu(h2_linear)

    #  Compute the output layer.
    #  We will omit the softmax function here
    #  because it will be applied
    #  in the softmax_cross_entropy loss
    yhat_linear = nd.dot(h2, W3) + b3
    return yhat_linear

##Optimizer

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

##Evaluation metric

def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()

##Execute the training loop

epochs = 10
learning_rate = .001
smoothing_constant = .01

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        label_one_hot = nd.one_hot(label, 10)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)
        cumulative_loss += nd.sum(loss).asscalar()


    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
          (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))

Epoch 0. Loss: 1.20780775437, Train_acc 0.883633, Test_acc 0.8877
Epoch 1. Loss: 0.328388882202, Train_acc 0.924017, Test_acc 0.9244
Epoch 2. Loss: 0.22106400394, Train_acc 0.949033, Test_acc 0.9464
Epoch 3. Loss: 0.162594895309, Train_acc 0.957433, Test_acc 0.9535
Epoch 4. Loss: 0.129279144899, Train_acc 0.96935, Test_acc 0.9637
Epoch 5. Loss: 0.105187748659, Train_acc 0.9739, Test_acc 0.9703
Epoch 6. Loss: 0.0890154179106, Train_acc 0.979033, Test_acc 0.9728
Epoch 7. Loss: 0.076162833334, Train_acc 0.982283, Test_acc 0.9723
Epoch 8. Loss: 0.0654618650412, Train_acc 0.984717, Test_acc 0.9728
Epoch 9. Loss: 0.0572528594335, Train_acc 0.987433, Test_acc 0.9751

#With just two hidden layers containing 256 hidden nodes, respectively, 
#we can achieve over 95% accuracy on this task.





http://gluon.mxnet.io/chapter03_deep-neural-networks/mlp-gluon.html
###Multilayer perceptrons in gluon



from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon

data_ctx = mx.cpu()
model_ctx = mx.cpu()
# model_ctx = mx.gpu(0)


batch_size = 64
num_inputs = 784
num_outputs = 10
num_examples = 60000
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)
train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)

##Define the model with gluon.Block
#In gluon a Block has one main job 
#- define a forward method that takes some NDArray input x 
#and generates an NDArray output. 

#Because the output and input are related to each other via NDArray operations, 
#MXNet can take derivatives through the block automatically. 

#A Block can just do something simple like apply an activation function. 
#But it can also combine a bunch of other Blocks together in creative ways. 


class MLP(gluon.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = gluon.nn.Dense(64)
            self.dense1 = gluon.nn.Dense(64)
            self.dense2 = gluon.nn.Dense(10)

    def forward(self, x):
        x = nd.relu(self.dense0(x))
        x = nd.relu(self.dense1(x))
        x = self.dense2(x)
        return x

net = MLP()
net.collect_params().initialize(mx.init.Normal(sigma=.01), ctx=model_ctx)

#And we can synthesize some gibberish data just 
#to demonstrate one forward pass through the network.
data = nd.ones((1,784))
net(data.as_in_context(model_ctx))
#Out[57]:
[[  4.40923759e-05  -8.20533780e-04   9.26479988e-04   8.04695825e-04
   -7.55993300e-04  -6.38230820e-04   5.50494005e-05  -1.17325678e-03
    7.58020557e-04   2.63349182e-04]]
<NDArray 1x10 @gpu(0)>

#or with print statement as debugging 

class MLP(gluon.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = gluon.nn.Dense(64, activation="relu")
            self.dense1 = gluon.nn.Dense(64, activation="relu")
            self.dense2 = gluon.nn.Dense(10)

    def forward(self, x):
        x = self.dense0(x)
        print("Hidden Representation 1: %s" % x)
        x = self.dense1(x)
        print("Hidden Representation 2: %s" % x)
        x = self.dense2(x)
        print("Network output: %s" % x)
        return x

net = MLP()
net.collect_params().initialize(mx.init.Normal(sigma=.01), ctx=model_ctx)
net(data.as_in_context(model_ctx))
#output 
Hidden Representation 1:
[[ 0.          0.21691252  0.          0.33119828  0.          0.          0.
   0.21983771  0.          0.          0.4556309   0.          0.08249515
   0.31085208  0.04958198  0.          0.330221    0.          0.          0.
   0.13425761  0.37306851  0.04791637  0.          0.          0.          0.
   0.23431879  0.          0.          0.          0.0448049   0.14588076
   0.          0.0239118   0.          0.25473717  0.03351231  0.20005098
   0.          0.          0.00603895  0.10416938  0.10464748  0.23973437
   0.          0.33381382  0.          0.24913697  0.29079285  0.12793788
   0.29657096  0.07166591  0.          0.43335861  0.32743987  0.          0.
   0.          0.          0.04985283  0.10861691  0.          0.        ]]
<NDArray 1x64 @gpu(0)>
Hidden Representation 2:
[[ 0.          0.          0.01573334  0.          0.          0.02613701
   0.00248956  0.          0.          0.02152583  0.          0.
   0.01183741  0.00089611  0.00513365  0.00952989  0.          0.          0.
   0.00989626  0.          0.00950431  0.          0.          0.
   0.01269766  0.00485498  0.          0.          0.00033371  0.00123863
   0.02299101  0.          0.01520418  0.          0.00365212  0.00016546
   0.00049757  0.00220794  0.          0.01853371  0.02050827  0.00796316
   0.02365419  0.          0.          0.          0.          0.00056281
   0.          0.0158518   0.00588764  0.02745012  0.02089521  0.02061545
   0.01254779  0.00096457  0.          0.00426208  0.          0.          0.
   0.00827779  0.00288925]]
<NDArray 1x64 @gpu(0)>
Network output:
[[  8.51602003e-04   4.21012577e-04  -3.94555100e-05   4.91072249e-04
   -2.73533806e-05  -9.80906654e-04  -2.85841583e-04  -1.03790930e-03
   -5.04873577e-04   7.01223849e-04]]
<NDArray 1x10 @gpu(0)>

Out[58]:
[[  8.51602003e-04   4.21012577e-04  -3.94555100e-05   4.91072249e-04
   -2.73533806e-05  -9.80906654e-04  -2.85841583e-04  -1.03790930e-03
   -5.04873577e-04   7.01223849e-04]]
<NDArray 1x10 @gpu(0)>




##Faster modeling with gluon.nn.Sequential
#    Instantiate a Sequential (let’s call it net)
#    Add a bunch of layers to it using net.add(...)

#Sequential assumes that the layers arrive bottom to top 
#(with input at the very bottom). 



num_hidden = 64
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(gluon.nn.Dense(num_outputs))

net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=model_ctx)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

##Training loop
epochs = 10
smoothing_constant = .01

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        cumulative_loss += nd.sum(loss).asscalar()


    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
          (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))
#output
Epoch 0. Loss: 1.27231270386, Train_acc 0.836933333333, Test_acc 0.846
Epoch 1. Loss: 0.477833755287, Train_acc 0.881066666667, Test_acc 0.8889
Epoch 2. Loss: 0.381976018492, Train_acc 0.89735, Test_acc 0.9035
Epoch 3. Loss: 0.33866001844, Train_acc 0.907533333333, Test_acc 0.9125
Epoch 4. Loss: 0.309403327727, Train_acc 0.913033333333, Test_acc 0.9165
Epoch 5. Loss: 0.285777178836, Train_acc 0.92025, Test_acc 0.9219
Epoch 6. Loss: 0.266318054875, Train_acc 0.925, Test_acc 0.9281
Epoch 7. Loss: 0.249801190837, Train_acc 0.931183333333, Test_acc 0.9323
Epoch 8. Loss: 0.235263404306, Train_acc 0.935483333333, Test_acc 0.9357
Epoch 9. Loss: 0.222571320128, Train_acc 0.9379, Test_acc 0.936

#convolutional neural networks which are especialy handy for working with images, 
#recurrent neural networks, which are especially useful for natural language processing.











###Dropout regularization from scratch

#With great flexibility comes overfitting liability
Given many more features than examples, linear models can overfit. 
But when there are many more examples than features, 
linear models can usually be counted on not to overfit.
Linear models have high bias, (they can only represent a small class of functions), 
but low variance (they give similar results across different random samples of the data).

Deep neural networks, however, occupy the opposite end of the bias-variance spectrum. 
Neural networks are so flexible because they aren’t confined to looking at each feature individually. 
Instead, they can learn complex interactions among groups of features
Even for a small number of features, deep neural networks are capable of overfitting.

##Dropping out activation
When a neural network overfits badly to training data, 
each layer depends too heavily on the exact configuration of features 
in the previous layer.

To prevent the neural network from depending too much 
on any exact activation pathway, 
Hinton and Srivastava proposed randomly dropping out (i.e. setting to 0) 
the hidden nodes in every layer with probability .5. 

Given a network with n nodes 
we are sampling uniformly at random from the 2**n networks 
in which a subset of the nodes are turned off.

##Making predictions with dropout models
#However, when it comes time to make predictions, 
#we want to use the full representational power of our model. 
#In other words, we don’t want to drop out activations at test time

from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
mx.random.seed(1)
ctx = mx.cpu()


mnist = mx.test_utils.get_mnist()
batch_size = 64
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)
train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)

#input 784(28*28 image size), output 256 
W1 = nd.random_normal(shape=(784,256), ctx=ctx) *.01
b1 = nd.random_normal(shape=256, ctx=ctx) * .01

W2 = nd.random_normal(shape=(256,128), ctx=ctx) *.01
b2 = nd.random_normal(shape=128, ctx=ctx) * .01

W3 = nd.random_normal(shape=(128,10), ctx=ctx) *.01
b3 = nd.random_normal(shape=10, ctx=ctx) *.01

params = [W1, b1, W2, b2, W3, b3]


for param in params:
    param.attach_grad()

##Activation functions
def relu(X):
    return nd.maximum(X, 0)

##Dropout

def dropout(X, drop_probability):
    keep_probability = 1 - drop_probability
    mask = nd.random_uniform(0, 1.0, X.shape, ctx=X.context) < keep_probability
    #  Avoid division by 0 when scaling
    if keep_probability > 0.0:
        scale = (1/keep_probability)
    else:
        scale = 0.0
    return mask * X * scale


#Check dropout functionality 
A = nd.arange(20).reshape((5,4))
dropout(A, 0.0)
#Out[7]:
[[  0.   1.   2.   3.]
 [  4.   5.   6.   7.]
 [  8.   9.  10.  11.]
 [ 12.  13.  14.  15.]
 [ 16.  17.  18.  19.]]
<NDArray 5x4 @cpu(0)>


dropout(A, 0.5)
#Out[8]:
[[  0.   0.   0.   6.]
 [  0.   0.   0.  14.]
 [ 16.  18.  20.  22.]
 [  0.   0.  28.   0.]
 [  0.   0.   0.  38.]]
<NDArray 5x4 @cpu(0)>



dropout(A, 1.0)
#Out[9]:
[[ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]]
<NDArray 5x4 @cpu(0)>

##Softmax output
def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear))
    partition = nd.nansum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition

##The softmax cross-entropy loss function
def softmax_cross_entropy(yhat_linear, y):
    return - nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)

##Define the model

def net(X, drop_prob=0.0):
    #  Compute the first hidden layer
    h1_linear = nd.dot(X, W1) + b1
    h1 = relu(h1_linear)
    h1 = dropout(h1, drop_prob)

    #  Compute the second hidden layer
    h2_linear = nd.dot(h1, W2) + b2
    h2 = relu(h2_linear)
    h2 = dropout(h2, drop_prob)

    #  Compute the output layer.
    #  We will omit the softmax function here
    #  because it will be applied
    #  in the softmax_cross_entropy loss
    yhat_linear = nd.dot(h2, W3) + b3
    return yhat_linear

##Optimizer
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

##Evaluation metric
def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()

##Execute the training loop
epochs = 10
moving_loss = 0.
learning_rate = .001

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        with autograd.record():
            #   Drop out 50% of hidden activations on the forward pass
            output = net(data, drop_prob=.5)
            loss = softmax_cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)
        #  Keep a moving average of the losses
        if i == 0:
            moving_loss = nd.mean(loss).asscalar()
        else:
            moving_loss = .99 * moving_loss + .01 * nd.mean(loss).asscalar()

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))
#Output 
Epoch 0. Loss: 0.758967668216, Train_acc 0.846667, Test_acc 0.8511
Epoch 1. Loss: 0.390177211666, Train_acc 0.921167, Test_acc 0.9208
Epoch 2. Loss: 0.294796600001, Train_acc 0.946517, Test_acc 0.9452
Epoch 3. Loss: 0.242323151582, Train_acc 0.956183, Test_acc 0.9532
Epoch 4. Loss: 0.214829764158, Train_acc 0.963917, Test_acc 0.96
Epoch 5. Loss: 0.18131017732, Train_acc 0.969083, Test_acc 0.9651
Epoch 6. Loss: 0.171195733796, Train_acc 0.972717, Test_acc 0.9672
Epoch 7. Loss: 0.161012466308, Train_acc 0.9754, Test_acc 0.9699
Epoch 8. Loss: 0.148282158084, Train_acc 0.978567, Test_acc 0.9729
Epoch 9. Loss: 0.137073164457, Train_acc 0.98015, Test_acc 0.9726

##Conclusion
#With just two hidden layers containing 256 and 128 hidden nodes, respectively, 
#we can achieve over 95% accuracy on this task.


###Dropout regularization with gluon
from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon


ctx = mx.cpu()

batch_size = 64
num_inputs = 784  #28*28 image size 
num_outputs = 10  #class 0...9 

#each data element is one byte 
#make data 0..1 and label as float 
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)
    
train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)

##Define the model
#Setting the dropout probability to .6 would mean that 60% of activations are dropped (set to zero) out and 40% are kept.

In [ ]:

num_hidden = 256
net = gluon.nn.Sequential()
with net.name_scope():
    # Adding first hidden layer
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    # Adding dropout with rate .5 to the first hidden layer
    net.add(gluon.nn.Dropout(.5))

    # Adding first hidden layer
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    # Adding dropout with rate .5 to the second hidden layer
    net.add(gluon.nn.Dropout(.5))

    # Adding the output layer
    net.add(gluon.nn.Dense(num_outputs))

##Parameter initialization

##net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

#To see what effect dropout is having on our predictions, 
#it’s instructive to pass the same example through our net multiple times.


for x, _ in train_data:
    x = x.as_in_context(ctx)
    break
print(net(x[0:1]))
print(net(x[0:1]))

#Note that we got the exact same answer on both forward passes through the net! 
#That’s because by, default, mxnet assumes that we are in predict mode. 
#We can explicitly invoke this scope by placing code within a 
#with autograd.predict_mode(): block.

with autograd.predict_mode():
    print(net(x[0:1]))
    print(net(x[0:1]))

#We can also run the code in train mode. 
#This tells MXNet to run our Blocks as they would run during training.
with autograd.train_mode():
    print(net(x[0:1]))
    print(net(x[0:1]))

##Accessing is_training() status
#autograd maintains a Boolean state that can be accessed via autograd.is_training(). 
#By default this value is False in the global scope. 


with autograd.predict_mode():
    print(autograd.is_training())

with autograd.train_mode():
    print(autograd.is_training())

##Integration with autograd.record

#When we train neural network models, we nearly always enter record() blocks. 
#The purpose of record() is to build the computational graph. 
#And the purpose of train is to indicate that we are training our model

#  Writing this every time could get cumbersome
with autograd.record():
    with autograd.train_mode():
        yhat = net(x)

#record() takes one argument, train_mode, which has a default value of True. 
#this by default turns on train_mode 
#(with autograd.record() is equivalent to with autograd.record(train_mode=True):). 


#Softmax cross-entropy loss
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

#Optimizer
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})

#Evaluation metric

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

#Training loop
epochs = 10
smoothing_constant = .01

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
            loss.backward()
        trainer.step(data.shape[0])
        #  Keep a moving average of the losses
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
          (e, moving_loss, train_accuracy, test_accuracy))
#Output 
Epoch 9. Loss: 0.121087726722, Train_acc 0.986133333333, Test_acc 0.9774



http://gluon.mxnet.io/chapter03_deep-neural-networks/plumbing.html
###A look under the hood of gluon

#Load up the data
from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, Block


#  Specify the context we'll be using
ctx = mx.cpu()

#  Load up our dataset
batch_size = 64
#data is 28x28x1 , each element is uint8, 0 to 255 
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)
    
#Each sample is an image (in 3D NDArray) with shape (28, 28, 1).
train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)
  
#train_data is iterator, each element is data_with_batch_size,label 
#ie <NDArray 64x28x28x1 @cpu(0)>, <NDArray 64 @cpu(0)>


                               
#Build Model 
#with net1.name_scope(): block. 
#This coerces gluon to give each parameter an appropriate name, 
#indicating which model it belongs to, e.g. sequential8_dense2_weight

net1 = gluon.nn.Sequential()
with net1.name_scope():
    net1.add(gluon.nn.Dense(128, activation="relu"))
    net1.add(gluon.nn.Dense(64, activation="relu"))
    net1.add(gluon.nn.Dense(10))
    
#OR by gluon.Block
class MLP(Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = nn.Dense(128)
            self.dense1 = nn.Dense(64)
            self.dense2 = nn.Dense(10)

    def forward(self, x):
        x = nd.relu(self.dense0(x))
        x = nd.relu(self.dense1(x))
        return self.dense2(x)
        
net2 = MLP()
net2.initialize(ctx=ctx)

#pass data through the network by calling it like a function
#gluon.Block.__call__(x) is defined 
#so that net(data) behaves identically to net.forward(data)
for data, _ in train_data:
    data = data.as_in_context(ctx)
    break
net2(data[0:1])

#The entire network is a Block, each layer is a Block

#Each neural network(even each layer) has to do the following things: 
 1. Store parameters 
 2. Accept inputs 
 3. Produce outputs (the forward pass) 
 4. Take derivatives (the backward pass)
 
#A single fully-connected layer is parameterized by a weight matrix and a bias vector, 
#produces outputs from inputs, 
#and, given the derivative of some objective with respect to its outputs, 
#can calculate the derivative with respect to its inputs.

#We only have to define the forward pass (forward(self, x)). 
#As long as the result is an NDArray, 
#using mxnet.autograd, gluon can handle the backward pass.
 
#Sequential itself subclasses Block and maintains a list of _children
#When we call forward on a Sequential, it executes the following code:
def forward(self, x):
    for block in self._children:
        x = block(x)
    return x


##Shape inference
#we only specified the number of nodes output, we never specified how many input nodes
print(net1.collect_params())
#the shapes of the weight matrices: (128,0), (64, 0), (10, 0). 
#0 means marking that the shape of these matrices is not yet known. 
#The shape will be inferred on the fly once the network is provided with some input.

#To get exact shape 
net1.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
#pass some data and then check params 
net1(data)
print(net1.collect_params())

#OR Specifying shape manually
net2 = gluon.nn.Sequential()
with net2.name_scope():
    net2.add(gluon.nn.Dense(128, in_units=784, activation="relu"))
    net2.add(gluon.nn.Dense(64, in_units=128, activation="relu"))
    net2.add(gluon.nn.Dense(10, in_units=64))

#Note that the parameters from this network can be initialized before we see any real data.
net2.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
print(net2.collect_params())




###Serialization - saving, loading and checkpointing


from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
ctx = mx.cpu()

#use ndarray.save and ndarray.load.
X = nd.ones((100, 100))
Y = nd.zeros((100, 100))


import os
os.makedirs('checkpoints', exist_ok=True)
filename = "checkpoints/test1.params"
nd.save(filename, [X, Y])

#to load a saved NDArray.
A, B = nd.load(filename)
print(A)
print(B)

#We can also save a dictionary where the keys are strings and the values are NDArrays.
mydict = {"X": X, "Y": Y}
filename = "checkpoints/test2.params"
nd.save(filename, mydict)

C = nd.load(filename)
print(C)


##Saving and loading the parameters of gluon models
#.save_params() and .load_params() methods.

num_hidden = 256
num_outputs = 1
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(gluon.nn.Dense(num_outputs))

net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=ctx)
net(nd.ones((1, 100), ctx=ctx))
#Out[7]:
[[ 362.53265381]]
<NDArray 1x1 @cpu(0)>

#save and load 

filename = "checkpoints/testnet.params"
net.save_params(filename)
net2 = gluon.nn.Sequential()
with net2.name_scope():
    net2.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net2.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net2.add(gluon.nn.Dense(num_outputs))
net2.load_params(filename, ctx=ctx)
net2(nd.ones((1, 100), ctx=ctx))
#Out[8]:
[[ 362.53265381]]
<NDArray 1x1 @cpu(0)>





###Convolutional neural networks from scratch

# the models people really use for classifying images.

from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
ctx = mx.cpu()
mx.random.seed(1)

batch_size = 64
num_inputs = 784
num_outputs = 10
#2nd arg is axes=(2,0,1) 3rd dimension first, then 1st dimension, then 2nd dimension
def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)

#<NDArray batch_sizex1x28x28 @cpu(0)>, <NDArray batch_size @cpu(0)>]
train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)
                                     
##Multilayer Perceptron 
Every node in each layer was connected to every node in the subsequent layers.  
This can require a lot of parameters 
If our input were a 256x256 color image and our network had 1,000 nodes in the first hidden layer, 
then our first weight matrix would require (256x256x3)x1000 parameters. 
That’s nearly 200 million.

##Convolutional neural networks
Convolutional neural networks incorporate convolutional layers. 
These layers associate each of their nodes with a small window, 
called a receptive field, in the previous layer, instead of connecting to the full layer. 

This allows us to first learn local features via transformations 
that are applied in the same way for the top right corner as for the bottom left. 
Then we collect all this local information to predict global qualities of the image

Second, we’ll be interleaving them with pooling layers.

Each node in convolutional layer is associated with a 3D block 
(height x width x channel) in the input tensor. 
Moreover, the convolutional layer itself has multiple output channels. 
So the layer is parameterized by a 4 dimensional weight tensor, called a convolutional kernel.

The output tensor is produced by sliding the kernel across the input image 
skipping locations according to a pre-defined stride 


#  Set the scale for weight initialization and choose
#  the number of hidden units in the fully-connected layer
weight_scale = .01
num_fc = 128

#W1,W2 is for conv layer 
#(output_channel, input_channel, height_window, width_window)
#(height_window, width_window) is receptive field size, called kernel 
W1 = nd.random_normal(shape=(20, 1, 3,3), scale=weight_scale, ctx=ctx)
#shape = output_channel 
b1 = nd.random_normal(shape=20, scale=weight_scale, ctx=ctx)

W2 = nd.random_normal(shape=(50, 20, 5, 5), scale=weight_scale, ctx=ctx)
b2 = nd.random_normal(shape=50, scale=weight_scale, ctx=ctx)

W3 = nd.random_normal(shape=(800, num_fc), scale=weight_scale, ctx=ctx)
b3 = nd.random_normal(shape=128, scale=weight_scale, ctx=ctx)

W4 = nd.random_normal(shape=(num_fc, num_outputs), scale=weight_scale, ctx=ctx)
b4 = nd.random_normal(shape=10, scale=weight_scale, ctx=ctx)

params = [W1, b1, W2, b2, W3, b3, W4, b4]

for param in params:
    param.attach_grad()

##Convolving with MXNet’s NDArrray
# use the function nd.Convolution(). 
#This function takes a few important arguments: inputs (data), 
#a 4D weight matrix (weight), a bias (bias), 
#the shape of the kernel (kernel), and a number of filters (num_filter).


for data, _ in train_data:
    data = data.as_in_context(ctx)
    break

conv = nd.Convolution(data=data, weight=W1, bias=b1, kernel=(3,3), num_filter=20)
print(conv.shape) #(64L, 20L, 26L, 26L)

#The number of examples (64) remains unchanged. 
#The number of channels (also called filters) has increased to 20. 
#And because the (3,3) kernel can only be applied in 26 different heights and widths (out of 28x28 input size )
#(without the kernel busting over the image border), our output is 26,26. 

#padding tricks can be used when we want the input and output to have the same height and width dimensions

#Pooling gives us a way to downsample in the spatial dimensions.
#Early convnets typically used average pooling, 
#but max pooling tends to give better results.


pool = nd.Pooling(data=conv, pool_type="max", kernel=(2,2), stride=(2,2))
print(pool.shape) #(64L, 20L, 13L, 13L)
#the height and width have been downsampled from (26,26) to (13,13) because kernel is (2,2) ie /ht and /width 

##Activation function
def relu(X):
    return nd.maximum(X,nd.zeros_like(X))

##Softmax output
def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear))
    partition = nd.sum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition

##Softmax cross-entropy loss
def softmax_cross_entropy(yhat_linear, y):
    return - nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)

##Define the model
def net(X, debug=False):
    #  Define the computation of the first convolutional layer
    h1_conv = nd.Convolution(data=X, weight=W1, bias=b1, kernel=(3,3), num_filter=20)
    h1_activation = relu(h1_conv)
    h1 = nd.Pooling(data=h1_activation, pool_type="avg", kernel=(2,2), stride=(2,2))
    if debug:
        print("h1 shape: %s,%s" % (np.array(h1_conv.shape), np.array(h1.shape)))
    #  Define the computation of the second convolutional layer
    h2_conv = nd.Convolution(data=h1, weight=W2, bias=b2, kernel=(5,5), num_filter=50)
    h2_activation = relu(h2_conv)
    h2 = nd.Pooling(data=h2_activation, pool_type="avg", kernel=(2,2), stride=(2,2))
    if debug:
        print("h2 shape: %s,%s" % (np.array(h2_conv.shape), np.array(h2.shape)))
    #  Flattening h2 so that we can feed it into a fully-connected layer
    h2 = nd.flatten(h2)
    if debug:
        print("Flat h2 shape: %s" % (np.array(h2.shape)))
    #  Define the computation of the third (fully-connected) layer
    h3_linear = nd.dot(h2, W3) + b3
    h3 = relu(h3_linear)
    if debug:
        print("h3 shape: %s" % (np.array(h3.shape)))
    #  Define the computation of the output layer
    yhat_linear = nd.dot(h3, W4) + b4
    if debug:
        print("yhat_linear shape: %s" % (np.array(yhat_linear.shape)))
    return yhat_linear

##Test run
output = net(data, debug=True)
#output 
>>> output = net(data, debug=True)
h1 shape: [64 20 26 26],[64 20 13 13]
h2 shape: [64 50  9  9],[64 50  4  4]
Flat h2 shape: [ 64 800]
h3 shape: [ 64 128]
yhat_linear shape: [64 10]

##Optimizer
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

##Evaluation metric

def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()

##The training loop
epochs = 1
learning_rate = .01
smoothing_constant = .01

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, num_outputs)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)

        ##
        #  Keep a moving average of the losses
        ##
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)


    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))

    
    
    
    
    
###Convolutional Neural Networks in gluon
from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
mx.random.seed(1)

ctx = mx.cpu()


batch_size = 64
num_inputs = 784
num_outputs = 10
def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)
train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)

##Define a convolutional neural network


num_fc = 512
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    # The Flatten layer collapses all axis, except the first one, into one axis.
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(num_fc, activation="relu"))
    net.add(gluon.nn.Dense(num_outputs))


net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})


def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

##Training Loop
epochs = 1
smoothing_constant = .01

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        #  Keep a moving average of the losses
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))


    
    
    
    
###Deep convolutional neural networks
#Previous chapter - we implemented a CNN with two convolutional layers 
#interleaved with pooling layers, a singly fully-connected hidden layer, 
#and a softmax output layer - called  LeNet

##AlexNet - In 2012, cuda-convnet implementation on an eight-layer CNN

from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
mx.random.seed(1)


ctx = mx.cpu()

#Cifar is a much smaller color dataset
#It contains 50,000 training and 10,000 test images. 
#The images belong in equal quantities to 10 categories. 
#we’ll upsample the images to 224x224 

def transformer(data, label):
    data = mx.image.imresize(data, 224, 224)
    data = mx.nd.transpose(data, (2,0,1))  #axis=(2,0,1) ie 3rd domention to first place , .. 
    data = data.astype(np.float32)
    return data, label

#Each sample is an image (in 3D NDArray) with shape (32, 32, 3), 3 channels for color 
#then transformed to 224x224x3 and then transposed to 3x224x224

batch_size = 64
train_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10('./data', train=True, transform=transformer),
    batch_size=batch_size, shuffle=True, last_batch='discard')

test_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10('./data', train=False, transform=transformer),
    batch_size=batch_size, shuffle=False, last_batch='discard')



for d, l in train_data:
    break


print(d.shape, l.shape) #(64L, 3L, 224L, 224L) (64L,)

print(d.dtype) #<type 'numpy.float32'>



##The AlexNet architecture
#contains 8 layers of transformations, five convolutional layers followed by two fully connected hidden layers and an output layer.

#The convolutional kernels in the first convolutional layer are 11×11
#in the second they are 5×5 and thereafter they are 3×3. 
#Moreover, the first, second, and fifth convolutional layers are each followed by 
#overlapping pooling operations with pool size 3×3 and stride (2×2).


alex_net = gluon.nn.Sequential()
with alex_net.name_scope():
    #  First convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=96, kernel_size=11, strides=(4,4), activation='relu'))
    alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
    #  Second convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=192, kernel_size=5, activation='relu'))
    alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=(2,2)))
    # Third convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=384, kernel_size=3, activation='relu'))
    # Fourth convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=384, kernel_size=3, activation='relu'))
    # Fifth convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=256, kernel_size=3, activation='relu'))
    alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
    # Flatten and apply fullly connected layers
    alex_net.add(gluon.nn.Flatten())
    alex_net.add(gluon.nn.Dense(4096, activation="relu"))
    alex_net.add(gluon.nn.Dense(4096, activation="relu"))
    alex_net.add(gluon.nn.Dense(10))


alex_net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

trainer = gluon.Trainer(alex_net.collect_params(), 'sgd', {'learning_rate': .001})

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()


def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for d, l in data_iterator:
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

##Training loop
#Only one epoch so tests can run quickly, increase this variable to actually run
epochs = 1
smoothing_constant = .01


for e in range(epochs):
    for i, (d, l) in enumerate(train_data):
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        with autograd.record():
            output = alex_net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        #  Keep a moving average of the losses
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

    test_accuracy = evaluate_accuracy(test_data, alex_net)
    train_accuracy = evaluate_accuracy(train_data, alex_net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))



###Very deep networks with repeating elements


from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
mx.random.seed(1)


ctx = mx.cpu()


batch_size = 64

def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)

train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)

##The VGG architecture
#A key aspect of VGG was to use many convolutional blocks 
#with relatively narrow kernels, 
#followed by a max-pooling step 
#and to repeat this block multiple times. 


from mxnet.gluon import nn

def vgg_block(num_convs, channels):
    out = nn.Sequential()
    for _ in range(num_convs):
        out.add(nn.Conv2D(channels=channels, kernel_size=3,
                      padding=1, activation='relu'))
    out.add(nn.MaxPool2D(pool_size=2, strides=2))
    return out

def vgg_stack(architecture):
    out = nn.Sequential()
    for (num_convs, channels) in architecture:
        out.add(vgg_block(num_convs, channels))
    return out

num_outputs = 10
architecture = ((1,64), (1,128), (2,256), (2,512))
net = nn.Sequential()
with net.name_scope():
    net.add(vgg_stack(architecture))
    net.add(nn.Flatten())
    net.add(nn.Dense(512, activation="relu"))
    net.add(nn.Dropout(.5))
    net.add(nn.Dense(512, activation="relu"))
    net.add(nn.Dropout(.5))
    net.add(nn.Dense(num_outputs))


net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .05})

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()


def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for d, l in data_iterator:
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


#  Only one epoch so tests can run quickly, increase this variable to actually run
epochs = 1
smoothing_constant = .01

for e in range(epochs):
    for i, (d, l) in enumerate(train_data):
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        #  Keep a moving average of the losses
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

        if i > 0 and i % 200 == 0:
            print('Batch %d. Loss: %f' % (i, moving_loss))

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))

    
    
    
    
    
###Batch Normalization from scratch
#For example - A DNN with three layers 
#After each training iteration, we update the weights in all the layers, 
#including the first and the second. 

#That means that over the course of training, 
#as the weights for the first two layers are learned, 
#the inputs to the third layer might look dramatically different 
#than they did at the beginning. 

#they might take values on a scale orders of magnitudes different 
#from when we started training. 
#And this shift in feature scale might have serious implications, 
#say for the ideal learning rate at each time.

#Sergey Ioffe and Christian Szegedy proposed Batch Normalization, 
#a technique that normalizes the mean and variance of each of the features 
#at every level of representation during training

#Empirically it appears to stabilize the gradient 
#(less exploding or vanishing values) 
#and batch-normalized models appear to overfit less. 

#In fact, batch-normalized models seldom even use dropout. 




from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd
mx.random.seed(1)


ctx = mx.cpu()


batch_size = 64
num_inputs = 784
num_outputs = 10
def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)
train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)

##Batch Normalization layer
#The layer, unlike Dropout, is usually used before the activation layer 
#(according to the authors’ original paper), instead of after activation layer.

#The basic idea is doing the normalization 
#then applying a linear scale and shift to the mini-batch:

#when it comes to (2D) CNN, 
#we normalize batch_size * height * width over each channel. 
#So that gamma and beta have the lengths the same as channel_count. 

#we need to manually reshape gamma and beta 
#so that they could (be automatically broadcast and) multipy the matrices 
#in the desired way.


#For input mini-batch B={x1,...,m}, with the parameter γ and β. 
#The output of the layer is {yi=BNγ,β(xi)},


def pure_batch_norm(X, gamma, beta, eps = 1e-5):
    if len(X.shape) not in (2, 4):
        raise ValueError('only supports dense or 2dconv')

    # dense
    if len(X.shape) == 2:
        # mini-batch mean
        mean = nd.mean(X, axis=0)
        # mini-batch variance
        variance = nd.mean((X - mean) ** 2, axis=0)
        # normalize
        X_hat = (X - mean) * 1.0 / nd.sqrt(variance + eps)
        # scale and shift
        out = gamma * X_hat + beta

    # 2d conv
    elif len(X.shape) == 4:
        # extract the dimensions
        N, C, H, W = X.shape
        # mini-batch mean
        mean = nd.mean(X, axis=(0, 2, 3))
        # mini-batch variance
        variance = nd.mean((X - mean.reshape((1, C, 1, 1))) ** 2, axis=(0, 2, 3))
        # normalize
        X_hat = (X - mean.reshape((1, C, 1, 1))) * 1.0 / nd.sqrt(variance.reshape((1, C, 1, 1)) + eps)
        # scale and shift
        out = gamma.reshape((1, C, 1, 1)) * X_hat + beta.reshape((1, C, 1, 1))

    return out

#We expect each column of the input matrix to be normalized.
A = nd.array([1,7,5,4,6,10], ctx=ctx).reshape((3,2))
A

pure_batch_norm(A,
    gamma = nd.array([1,1], ctx=ctx),
    beta=nd.array([0,0], ctx=ctx))


ga = nd.array([1,1], ctx=ctx)
be = nd.array([0,0], ctx=ctx)

B = nd.array([1,6,5,7,4,3,2,5,6,3,2,4,5,3,2,5,6], ctx=ctx).reshape((2,2,2,2))
B

pure_batch_norm(B, ga, be)


#In the testing process, 
#we want to use the mean and variance of the complete dataset, instead of those of mini batches. 
#In the implementation, we use moving statistics as a trade off, 
#because we don’t want to or don’t have the ability to compute the statistics of the complete 

#we need to maintain the moving statistics along with multiple runs of the BN
#In this quick-and-dirty implementation, 
#we use the global dictionary variables to store the statistics, 
#in which each key is the name of the layer (scope_name), and the value is the statistics

def batch_norm(X,
               gamma,
               beta,
               momentum = 0.9,
               eps = 1e-5,
               scope_name = '',
               is_training = True,
               debug = False):
    """compute the batch norm """
    global _BN_MOVING_MEANS, _BN_MOVING_VARS
    # the usual batch norm transformation

    if len(X.shape) not in (2, 4):
        raise ValueError('the input data shape should be one of:\n' +
                         'dense: (batch size, # of features)\n' +
                         '2d conv: (batch size, # of features, height, width)'
                        )

    # dense
    if len(X.shape) == 2:
        # mini-batch mean
        mean = nd.mean(X, axis=0)
        # mini-batch variance
        variance = nd.mean((X - mean) ** 2, axis=0)
        # normalize
        if is_training:
            # while training, we normalize the data using its mean and variance
            X_hat = (X - mean) * 1.0 / nd.sqrt(variance + eps)
        else:
            # while testing, we normalize the data using the pre-computed mean and variance
            X_hat = (X - _BN_MOVING_MEANS[scope_name]) *1.0 / nd.sqrt(_BN_MOVING_VARS[scope_name] + eps)
        # scale and shift
        out = gamma * X_hat + beta

    # 2d conv
    elif len(X.shape) == 4:
        # extract the dimensions
        N, C, H, W = X.shape
        # mini-batch mean
        mean = nd.mean(X, axis=(0,2,3))
        # mini-batch variance
        variance = nd.mean((X - mean.reshape((1, C, 1, 1))) ** 2, axis=(0, 2, 3))
        # normalize
        X_hat = (X - mean.reshape((1, C, 1, 1))) * 1.0 / nd.sqrt(variance.reshape((1, C, 1, 1)) + eps)
        if is_training:
            # while training, we normalize the data using its mean and variance
            X_hat = (X - mean.reshape((1, C, 1, 1))) * 1.0 / nd.sqrt(variance.reshape((1, C, 1, 1)) + eps)
        else:
            # while testing, we normalize the data using the pre-computed mean and variance
            X_hat = (X - _BN_MOVING_MEANS[scope_name].reshape((1, C, 1, 1))) * 1.0 \
                / nd.sqrt(_BN_MOVING_VARS[scope_name].reshape((1, C, 1, 1)) + eps)
        # scale and shift
        out = gamma.reshape((1, C, 1, 1)) * X_hat + beta.reshape((1, C, 1, 1))

    # to keep the moving statistics
    # init the attributes
    try: # to access them
        _BN_MOVING_MEANS, _BN_MOVING_VARS
    except: # error, create them
        _BN_MOVING_MEANS, _BN_MOVING_VARS = {}, {}

    # store the moving statistics by their scope_names, inplace
    if scope_name not in _BN_MOVING_MEANS:
        _BN_MOVING_MEANS[scope_name] = mean
    else:
        _BN_MOVING_MEANS[scope_name] = _BN_MOVING_MEANS[scope_name] * momentum + mean * (1.0 - momentum)
    if scope_name not in _BN_MOVING_VARS:
        _BN_MOVING_VARS[scope_name] = variance
    else:
        _BN_MOVING_VARS[scope_name] = _BN_MOVING_VARS[scope_name] * momentum + variance * (1.0 - momentum)

    # debug info
    if debug:
        print('== info start ==')
        print('scope_name = {}'.format(scope_name))
        print('mean = {}'.format(mean))
        print('var = {}'.format(variance))
        print('_BN_MOVING_MEANS = {}'.format(_BN_MOVING_MEANS[scope_name]))
        print('_BN_MOVING_VARS = {}'.format(_BN_MOVING_VARS[scope_name]))
        print('output = {}'.format(out))
        print('== info end ==')

    # return
    return out

##Parameters and gradients

#  Set the scale for weight initialization and choose
#  the number of hidden units in the fully-connected layer
weight_scale = .01
num_fc = 128

W1 = nd.random_normal(shape=(20, 1, 3,3), scale=weight_scale, ctx=ctx)
b1 = nd.random_normal(shape=20, scale=weight_scale, ctx=ctx)

gamma1 = nd.random_normal(shape=20, loc=1, scale=weight_scale, ctx=ctx)
beta1 = nd.random_normal(shape=20, scale=weight_scale, ctx=ctx)

W2 = nd.random_normal(shape=(50, 20, 5, 5), scale=weight_scale, ctx=ctx)
b2 = nd.random_normal(shape=50, scale=weight_scale, ctx=ctx)

gamma2 = nd.random_normal(shape=50, loc=1, scale=weight_scale, ctx=ctx)
beta2 = nd.random_normal(shape=50, scale=weight_scale, ctx=ctx)

W3 = nd.random_normal(shape=(800, num_fc), scale=weight_scale, ctx=ctx)
b3 = nd.random_normal(shape=num_fc, scale=weight_scale, ctx=ctx)

gamma3 = nd.random_normal(shape=num_fc, loc=1, scale=weight_scale, ctx=ctx)
beta3 = nd.random_normal(shape=num_fc, scale=weight_scale, ctx=ctx)

W4 = nd.random_normal(shape=(num_fc, num_outputs), scale=weight_scale, ctx=ctx)
b4 = nd.random_normal(shape=10, scale=weight_scale, ctx=ctx)

params = [W1, b1, gamma1, beta1, W2, b2, gamma2, beta2, W3, b3, gamma3, beta3, W4, b4]

for param in params:
    param.attach_grad()

##Activation functions

def relu(X):
    return nd.maximum(X, 0)

##Softmax output
def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear))
    partition = nd.nansum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition

##The softmax cross-entropy loss function

def softmax_cross_entropy(yhat_linear, y):
    return - nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)

##Define the model
#We insert the BN layer right after each linear layer.


def net(X, is_training = True, debug=False):
    #  Define the computation of the first convolutional layer
    h1_conv = nd.Convolution(data=X, weight=W1, bias=b1, kernel=(3,3), num_filter=20)
    h1_normed = batch_norm(h1_conv, gamma1, beta1, scope_name='bn1', is_training=is_training)
    h1_activation = relu(h1_normed)
    h1 = nd.Pooling(data=h1_activation, pool_type="avg", kernel=(2,2), stride=(2,2))
    if debug:
        print("h1 shape: %s" % (np.array(h1.shape)))
    #  Define the computation of the second convolutional layer
    h2_conv = nd.Convolution(data=h1, weight=W2, bias=b2, kernel=(5,5), num_filter=50)
    h2_normed = batch_norm(h2_conv, gamma2, beta2, scope_name='bn2', is_training=is_training)
    h2_activation = relu(h2_normed)
    h2 = nd.Pooling(data=h2_activation, pool_type="avg", kernel=(2,2), stride=(2,2))
    if debug:
        print("h2 shape: %s" % (np.array(h2.shape)))
    #  Flattening h2 so that we can feed it into a fully-connected layer
    h2 = nd.flatten(h2)
    if debug:
        print("Flat h2 shape: %s" % (np.array(h2.shape)))
    #  Define the computation of the third (fully-connected) layer
    h3_linear = nd.dot(h2, W3) + b3
    h3_normed = batch_norm(h3_linear, gamma3, beta3, scope_name='bn3', is_training=is_training)
    h3 = relu(h3_normed)
    if debug:
        print("h3 shape: %s" % (np.array(h3.shape)))
    #  Define the computation of the output layer
    yhat_linear = nd.dot(h3, W4) + b4
    if debug:
        print("yhat_linear shape: %s" % (np.array(yhat_linear.shape)))

    return yhat_linear

##Test run

for data, _ in train_data:
    data = data.as_in_context(ctx)
    break


output = net(data, is_training=True, debug=True)

##Optimizer

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

##Evaluation metric
def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        output = net(data, is_training=False) # attention here!
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()

##Execute the training loop

#use a gpu to run the code below. (And remember to set the ctx = mx.gpu() accordingly in the very beginning of this article.)


epochs = 1
moving_loss = 0.
learning_rate = .001

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, num_outputs)
        with autograd.record():
            # we are in training process,
            # so we normalize the data using batch mean and variance
            output = net(data, is_training=True)
            loss = softmax_cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)
        #  Keep a moving average of the losses
        if i == 0:
            moving_loss = nd.mean(loss).asscalar()
        else:
            moving_loss = .99 * moving_loss + .01 * nd.mean(loss).asscalar()

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))

    
    
    
    
###Batch Normalization in gluon
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
mx.random.seed(1)

ctx = mx.cpu()



batch_size = 64
num_inputs = 784
num_outputs = 10
def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)
train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)

##Define a CNN with Batch Normalization

num_fc = 512
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Conv2D(channels=20, kernel_size=5))
    net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net.add(gluon.nn.Activation(activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    net.add(gluon.nn.Conv2D(channels=50, kernel_size=5))
    net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net.add(gluon.nn.Activation(activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    # The Flatten layer collapses all axis, except the first one, into one axis.
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(num_fc))
    net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net.add(gluon.nn.Activation(activation='relu'))
    net.add(gluon.nn.Dense(num_outputs))



net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


epochs = 1
smoothing_constant = .01

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        #  Keep a moving average of the losses
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))


    
    
    
    
    
@@@
http://gluon.mxnet.io/chapter05_recurrent-neural-networks/simple-rnn.html
###Recurrent Neural Networks (RNNs) for Language Modeling

#Earlier chapter - feedforward networks 
#because each layer feeds into the next layer in a chain connecting the inputs 
#to the outputs.

At each iteration t, we feed in a new example xt, by setting the values of the input nodes . 
We then feed the activation forward by successively calculating the activations of each higher layer in the network. 
Finally, we read the outputs from the topmost layer.

So when we feed the next example xt+1, we overwrite all of the previous activations. 
If consecutive inputs to our network have no special relationship to each other 
(say, images uploaded by unrelated users), t
hen this is perfectly acceptable behavior. 

But what if our inputs exhibit a sequential relationship?

Recurrent neural networks provide a way to incorporate sequential structure. 
At each time step t, each hidden layer ht (typically) will receive input 
from both the current input xt and from that same hidden layer 
at the previous time step ht−1

Even though the neural network contains loops 
(the hidden layer is connected to itself), 
because this connection spans a time step our network is still a feedforward network. 
Thus we can still train by backpropagration just as we normally would with an MLP. Typically the loss function will be an average of the losses at each time step.

#Example 
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
import numpy as np
mx.random.seed(1)
ctx = mx.gpu(0)  #requires GPU , check https://aws.amazon.com/ec2/instance-types/p2/

with open("../data/nlp/timemachine.txt") as f:
    time_machine = f.read()
#check 
print(time_machine[0:500])
time_machine = time_machine[:-38083]  #last few lines are consist entirely of legalese from the Gutenberg gang.

#get vocab size 
character_list = list(set(time_machine))
vocab_size = len(character_list)
print(character_list)
print("Length of vocab: %s" % vocab_size)

#create dict with char as key and index as value 
character_dict = {}
for e, char in enumerate(character_list):
    character_dict[char] = e
print(character_dict)

#create numerical representation of text 
time_numerical = [character_dict[char] for char in time_machine]
#  Convert back to text
print("".join([character_list[idx] for idx in time_numerical[:39]]))

#One-hot representations - 2d structure where each row for one char
def one_hots(numerical_list, vocab_size=vocab_size):
    result = nd.zeros((len(numerical_list), vocab_size), ctx=ctx)
    for i, idx in enumerate(numerical_list):
        result[i, idx] = 1.0
    return result

print(one_hots(time_numerical[:2]))

#conversion from one hot to normal text 
def textify(embedding):
    result = ""
    indices = nd.argmax(embedding, axis=1).asnumpy()
    for idx in indices:
        result += character_list[int(idx)]
    return result
#Check 
textify(one_hots(time_numerical[0:40]))

#You might think we could just feed in the entire dataset as one gigantic input 
#and backpropagate across the entire sequence. 
#When you try to backpropagate across thousands of steps a few things go wrong: 
#(1) The time it takes to compute a single gradient update will be unreasonably long 
#(2) The gradient across thousands of recurrent steps has a tendency to either blow up, 
#causing NaN errors due to losing precision, or to vanish.

#Thus we’re going to look at feeding in our data in reasonably short sequences.


seq_length = 64
# -1 here so we have enough characters for labels later
num_samples = (len(time_numerical) - 1) // seq_length
dataset = one_hots(time_numerical[:seq_length*num_samples]).reshape((num_samples, seq_length, vocab_size))
#check 
textify(dataset[0])

#To make computation efficient 
#we’ll want to feed through a batch of sequences at the same time. 


batch_size = 32
print('# of sequences in dataset: ', len(dataset))
num_batches = len(dataset) // batch_size
print('# of batches: ', num_batches)
train_data = dataset[:num_batches*batch_size].reshape((batch_size, num_batches, seq_length, vocab_size))
# swap batch_size and seq_length axis to make later access easier
train_data = nd.swapaxes(train_data, 0, 1)
train_data = nd.swapaxes(train_data, 1, 2)
print('Shape of data set: ', train_data.shape)

# of sequences in dataset:  2805
# of batches:  87
#Shape of data set:  (87, 64, 32, 77)

#sanity check 
for i in range(3):
    print("***Batch %s:***\n %s \n %s \n\n" % (i, textify(train_data[i, :, 0]), textify(train_data[i, :, 1])))
    

##Preparing our labels
#our target at every time step is to predict the next character in the sequence. 
#So our labels should look just like our inputs but offset by one character.


labels = one_hots(time_numerical[1:seq_length*num_samples+1])
train_label = labels.reshape((batch_size, num_batches, seq_length, vocab_size))
train_label = nd.swapaxes(train_label, 0, 1)
train_label = nd.swapaxes(train_label, 1, 2)
print(train_label.shape) #(87, 64, 32, 77)
print(textify(train_data[10, :, 3]))
print(textify(train_label[10, :, 3]))
#output 
te, but the twisted crystalline bars lay unfinished upon the
ben
e, but the twisted crystalline bars lay unfinished upon the
benc

##Recurrent neural networks

#update for an ordinary hidden layer in a neural network with activation function ϕ
h=ϕ(x * W+b)

#To make this a recurrent neural network, 
ht=ϕ(xt * Wxh + ht−1 * Whh + bh)

#Then at every time set t, we’ll calculate the output as:
y^t=softmaxone−hot(ht * Why+by)


##Allocate parameters


num_inputs = vocab_size
num_hidden = 256
num_outputs = vocab_size


#  Weights connecting the inputs to the hidden layer
Wxh = nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01


#  Recurrent weights connecting the hidden layer across time steps
Whh = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx) * .01

#  Bias vector for hidden layer
bh = nd.random_normal(shape=num_hidden, ctx=ctx) * .01


# Weights to the output nodes
Why = nd.random_normal(shape=(num_hidden,num_outputs), ctx=ctx) * .01
by = nd.random_normal(shape=num_outputs, ctx=ctx) * .01

# NOTE: to keep notation consistent,
# we should really use capital letters
# for hidden layers and outputs,
# since we are doing batchwise computations]

##Attach the gradients
params = [Wxh, Whh, bh, Why, by]

for param in params:
    param.attach_grad()

##Softmax Activation
def softmax(y_linear, temperature=1.0):
    lin = (y_linear-nd.max(y_linear, axis=1).reshape((-1,1))) / temperature # shift each row of y_linear by its max
    exp = nd.exp(lin)
    partition =nd.sum(exp, axis=1).reshape((-1,1))
    return exp / partition


# With a temperature of 1 (always 1 during training), we get back some set of probabilities
softmax(nd.array([[1, -1], [-1, 1]]), temperature=1.0)
#Out[22]:
[[ 0.88079703  0.11920292]
 [ 0.11920292  0.88079703]]
<NDArray 2x2 @cpu(0)>




# If we set a high temperature, we can get more entropic (*noisier*) probabilities
softmax(nd.array([[1,-1],[-1,1]]), temperature=1000.0)

#Out[23]:
[[ 0.50049996  0.49949998]
 [ 0.49949998  0.50049996]]
<NDArray 2x2 @cpu(0)>



# Often we want to sample with low temperatures to produce sharp probabilities
softmax(nd.array([[10,-10],[-10,10]]), temperature=.1)

#Out[24]:
[[ 1.  0.]
 [ 0.  1.]]
<NDArray 2x2 @cpu(0)>

##Define the model

def simple_rnn(inputs, state, temperature=1.0):
    outputs = []
    h = state
    for X in inputs:
        h_linear = nd.dot(X, Wxh) + nd.dot(h, Whh) + bh
        h = nd.tanh(h_linear)
        yhat_linear = nd.dot(h, Why) + by
        yhat = softmax(yhat_linear, temperature=temperature)
        outputs.append(yhat)
    return (outputs, h)

##Cross-entropy loss function

#At every time step our task is to predict the next character, 
#given the string up to that point. 
#This is the familiar multi-task classification 
#that we introduced for handwritten digit classification. 
#Accordingly, we’ll rely on the same loss function, cross-entropy.


# def cross_entropy(yhat, y):
#     return - nd.sum(y * nd.log(yhat))

def cross_entropy(yhat, y):
    return - nd.mean(nd.sum(y * nd.log(yhat), axis=0, exclude=True))


cross_entropy(nd.array([[.2,.5,.3], [.2,.5,.3]]), nd.array([[1.,0,0], [0, 1.,0]]))
#Out[27]:
[ 1.15129256]
<NDArray 1 @cpu(0)>

##Averaging the loss over the sequence

#Because the unfolded RNN has multiple outputs (one at every time step) 
#we can calculate a loss at every time step. 
#The weights corresponding to the net at time step t
#influence both the loss at time step t and the loss at time step t+1
#To combine our losses into a single global loss, we’ll take the average of the losses at each time step.


def average_ce_loss(outputs, labels):
    assert(len(outputs) == len(labels))
    total_loss = 0.
    for (output, label) in zip(outputs,labels):
        total_loss = total_loss + cross_entropy(output, label)
    return total_loss / len(outputs)

##Optimizer

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

##Generating text by sampling
#we can now generate strings of plausible text. 
#The generation procedure goes as follows. 
#Say our string begins with the character “T”. 
#We can feed the letter “T” and get a conditional probability distribution 
#over the next character P(x2|x1="T"). 
#We can the sample from this distribution, e.g. producing an “i”, 
#and then assign x2="i", feeding this to the network at the next time step.



def sample(prefix, num_chars, temperature=1.0):
    # Initialize the string that we'll return to the supplied prefix
    string = prefix

    # Prepare the prefix as a sequence of one-hots for ingestion by RNN
    prefix_numerical = [character_dict[char] for char in prefix]
    input = one_hots(prefix_numerical)

    # Set the initial state of the hidden representation ($h_0$) to the zero vector
    sample_state = nd.zeros(shape=(1, num_hidden), ctx=ctx)

    # For num_chars iterations,
    #     1) feed in the current input
    #     2) sample next character from from output distribution
    #     3) add sampled character to the decoded string
    #     4) prepare the sampled character as a one_hot (to be the next input)
    for i in range(num_chars):
        outputs, sample_state = simple_rnn(input, sample_state, temperature=temperature)
        choice = np.random.choice(vocab_size, p=outputs[-1][0].asnumpy())
        string += character_list[choice]
        input = one_hots([choice])
    return string


##training loop 
epochs = 2000
moving_loss = 0.

learning_rate = .5

# state = nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)
for e in range(epochs):
    # Attenuate the learning rate by a factor of 2 every 100 epochs.
    if ((e+1) % 100 == 0):
        learning_rate = learning_rate / 2.0
    state = nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)
    for i in range(num_batches):
        data_one_hot = train_data[i]
        label_one_hot = train_label[i]
        with autograd.record():
            outputs, state = simple_rnn(data_one_hot, state)
            loss = average_ce_loss(outputs, label_one_hot)
            loss.backward()
        SGD(params, learning_rate)

        #  Keep a moving average of the losses
        if (i == 0) and (e == 0):
            moving_loss = np.mean(loss.asnumpy()[0])
        else:
            moving_loss = .99 * moving_loss + .01 * np.mean(loss.asnumpy()[0])

    print("Epoch %s. Loss: %s" % (e, moving_loss))
    print(sample("The Time Ma", 1024, temperature=.1))
    print(sample("The Medical Man rose, came to the lamp,", 1024, temperature=.1))

##Conclusions
#Once you start running this code, 
#it will spit out a sample at the end of each epoch.



###Long short-term memory (LSTM) RNNs


from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
import numpy as np
mx.random.seed(1)
ctx = mx.gpu(0)



with open("../data/nlp/timemachine.txt") as f:
    time_machine = f.read()
time_machine = time_machine[:-38083]

##Numerical representations of characters

character_list = list(set(time_machine))
vocab_size = len(character_list)
character_dict = {}
for e, char in enumerate(character_list):
    character_dict[char] = e
time_numerical = [character_dict[char] for char in time_machine]

##One-hot representations

def one_hots(numerical_list, vocab_size=vocab_size):
    result = nd.zeros((len(numerical_list), vocab_size), ctx=ctx)
    for i, idx in enumerate(numerical_list):
        result[i, idx] = 1.0
    return result


def textify(embedding):
    result = ""
    indices = nd.argmax(embedding, axis=1).asnumpy()
    for idx in indices:
        result += character_list[int(idx)]
    return result

##Preparing the data for training

batch_size = 32
seq_length = 64
# -1 here so we have enough characters for labels later
num_samples = (len(time_numerical) - 1) // seq_length
dataset = one_hots(time_numerical[:seq_length*num_samples]).reshape((num_samples, seq_length, vocab_size))
num_batches = len(dataset) // batch_size
train_data = dataset[:num_batches*batch_size].reshape((num_batches, batch_size, seq_length, vocab_size))
# swap batch_size and seq_length axis to make later access easier
train_data = nd.swapaxes(train_data, 1, 2)

##Preparing our labels
#our target at every time step is to predict the next character in the sequence. 
#So our labels should look just like our inputs but offset by one character.

labels = one_hots(time_numerical[1:seq_length*num_samples+1])
train_label = labels.reshape((num_batches, batch_size, seq_length, vocab_size))
train_label = nd.swapaxes(train_label, 1, 2)

##Long short-term memory (LSTM) RNNs
#An LSTM block has mechanisms to enable “memorizing” information 
#for an extended number of time steps


num_inputs = vocab_size
num_hidden = 256
num_outputs = vocab_size

#  Weights connecting the inputs to the hidden layer
Wxg = nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01
Wxi = nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01
Wxf = nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01
Wxo = nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01

#  Recurrent weights connecting the hidden layer across time steps
Whg = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx)* .01
Whi = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx)* .01
Whf = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx)* .01
Who = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx)* .01

#  Bias vector for hidden layer
bg = nd.random_normal(shape=num_hidden, ctx=ctx) * .01
bi = nd.random_normal(shape=num_hidden, ctx=ctx) * .01
bf = nd.random_normal(shape=num_hidden, ctx=ctx) * .01
bo = nd.random_normal(shape=num_hidden, ctx=ctx) * .01

# Weights to the output nodes
Why = nd.random_normal(shape=(num_hidden,num_outputs), ctx=ctx) * .01
by = nd.random_normal(shape=num_outputs, ctx=ctx) * .01


params = [Wxg, Wxi, Wxf, Wxo, Whg, Whi, Whf, Who, bg, bi, bf, bo, Why, by]

for param in params:
    param.attach_grad()

##Softmax Activation

def softmax(y_linear, temperature=1.0):
    lin = (y_linear-nd.max(y_linear)) / temperature
    exp = nd.exp(lin)
    partition = nd.sum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition

##Define the model

def lstm_rnn(inputs, h, c, temperature=1.0):
    outputs = []
    for X in inputs:
        g = nd.tanh(nd.dot(X, Wxg) + nd.dot(h, Whg) + bg)
        i = nd.sigmoid(nd.dot(X, Wxi) + nd.dot(h, Whi) + bi)
        f = nd.sigmoid(nd.dot(X, Wxf) + nd.dot(h, Whf) + bf)
        o = nd.sigmoid(nd.dot(X, Wxo) + nd.dot(h, Who) + bo)

        c = f * c + i * g
        h = o * nd.tanh(c)

        yhat_linear = nd.dot(h, Why) + by
        yhat = softmax(yhat_linear, temperature=temperature)
        outputs.append(yhat)
    return (outputs, h, c)

##Cross-entropy loss function

def cross_entropy(yhat, y):
    return - nd.mean(nd.sum(y * nd.log(yhat), axis=0, exclude=True))

##Averaging the loss over the sequence

def average_ce_loss(outputs, labels):
    assert(len(outputs) == len(labels))
    total_loss = 0.
    for (output, label) in zip(outputs,labels):
        total_loss = total_loss + cross_entropy(output, label)
    return total_loss / len(outputs)

##Optimizer

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

##Generating text by sampling

def sample(prefix, num_chars, temperature=1.0):
    # Initialize the string that we'll return to the supplied prefix
    string = prefix

    # Prepare the prefix as a sequence of one-hots for ingestion by RNN
    prefix_numerical = [character_dict[char] for char in prefix]
    input = one_hots(prefix_numerical)

    # Set the initial state of the hidden representation ($h_0$) to the zero vector
    h = nd.zeros(shape=(1, num_hidden), ctx=ctx)
    c = nd.zeros(shape=(1, num_hidden), ctx=ctx)

    # For num_chars iterations,
    #     1) feed in the current input
    #     2) sample next character from from output distribution
    #     3) add sampled character to the decoded string
    #     4) prepare the sampled character as a one_hot (to be the next input)
    for i in range(num_chars):
        outputs, h, c = lstm_rnn(input, h, c, temperature=temperature)
        choice = np.random.choice(vocab_size, p=outputs[-1][0].asnumpy())
        string += character_list[choice]
        input = one_hots([choice])
    return string



epochs = 2000
moving_loss = 0.

learning_rate = 2.0

# state = nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)
for e in range(epochs):
    # Attenuate the learning rate by a factor of 2 every 100 epochs.
    if ((e+1) % 100 == 0):
        learning_rate = learning_rate / 2.0
    h = nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)
    c = nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)
    for i in range(num_batches):
        data_one_hot = train_data[i]
        label_one_hot = train_label[i]
        with autograd.record():
            outputs, h, c = lstm_rnn(data_one_hot, h, c)
            loss = average_ce_loss(outputs, label_one_hot)
            loss.backward()
        SGD(params, learning_rate)

        #  Keep a moving average of the losses
        if (i == 0) and (e == 0):
            moving_loss = nd.mean(loss).asscalar()
        else:
            moving_loss = .99 * moving_loss + .01 * nd.mean(loss).asscalar()

    print("Epoch %s. Loss: %s" % (e, moving_loss))
    print(sample("The Time Ma", 1024, temperature=.1))
    print(sample("The Medical Man rose, came to the lamp,", 1024, temperature=.1))


    
    
    
###Gated recurrent unit (GRU) RNNs

from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
import numpy as np
mx.random.seed(1)
ctx = mx.gpu(0)



with open("../data/nlp/timemachine.txt") as f:
    time_machine = f.read()
time_machine = time_machine[:-38083]

##Numerical representations of characters


character_list = list(set(time_machine))
vocab_size = len(character_list)
character_dict = {}
for e, char in enumerate(character_list):
    character_dict[char] = e
time_numerical = [character_dict[char] for char in time_machine]

##One-hot representations


def one_hots(numerical_list, vocab_size=vocab_size):
    result = nd.zeros((len(numerical_list), vocab_size), ctx=ctx)
    for i, idx in enumerate(numerical_list):
        result[i, idx] = 1.0
    return result


def textify(embedding):
    result = ""
    indices = nd.argmax(embedding, axis=1).asnumpy()
    for idx in indices:
        result += character_list[int(idx)]
    return result

##Preparing the data for training

batch_size = 32
seq_length = 64
# -1 here so we have enough characters for labels later
num_samples = (len(time_numerical) - 1) // seq_length
dataset = one_hots(time_numerical[:seq_length*num_samples]).reshape((num_samples, seq_length, vocab_size))
num_batches = len(dataset) // batch_size
train_data = dataset[:num_batches*batch_size].reshape((num_batches, batch_size, seq_length, vocab_size))
# swap batch_size and seq_length axis to make later access easier
train_data = nd.swapaxes(train_data, 1, 2)

##Preparing our labels

labels = one_hots(time_numerical[1:seq_length*num_samples+1])
train_label = labels.reshape((num_batches, batch_size, seq_length, vocab_size))
train_label = nd.swapaxes(train_label, 1, 2)

##Gated recurrent units (GRU) RNNs
#Similar to LSTM blocks, the GRU also has mechanisms to enable “memorizing” information 
#for an extended number of time steps. 
#However, it does so in a more expedient way

#Empirically, GRUs have similar performance to LSTMs, 
#while requiring less parameters and forgoing an internal time state. 
#Intuitively, GRUs have enough gates/state for long-term retention, 
#but not too much, so that training and convergence remain fast and convex.


num_inputs = vocab_size
num_hidden = 256
num_outputs = vocab_size

#  Weights connecting the inputs to the hidden layer
Wxz = nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01
Wxr = nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01
Wxh = nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01

#  Recurrent weights connecting the hidden layer across time steps
Whz = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx)* .01
Whr = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx)* .01
Whh = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx)* .01

#  Bias vector for hidden layer
bz = nd.random_normal(shape=num_hidden, ctx=ctx) * .01
br = nd.random_normal(shape=num_hidden, ctx=ctx) * .01
bh = nd.random_normal(shape=num_hidden, ctx=ctx) * .01

# Weights to the output nodes
Why = nd.random_normal(shape=(num_hidden,num_outputs), ctx=ctx) * .01
by = nd.random_normal(shape=num_outputs, ctx=ctx) * .01


params = [Wxz, Wxr, Wxh, Whz, Whr, Whh, bz, br, bh, Why, by]

for param in params:
    param.attach_grad()

##Softmax Activation
def softmax(y_linear, temperature=1.0):
    lin = (y_linear-nd.max(y_linear)) / temperature
    exp = nd.exp(lin)
    partition = nd.sum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition

##Define the model
def gru_rnn(inputs, h, temperature=1.0):
    outputs = []
    for X in inputs:
        z = nd.sigmoid(nd.dot(X, Wxz) + nd.dot(h, Whz) + bz)
        r = nd.sigmoid(nd.dot(X, Wxr) + nd.dot(h, Whr) + br)
        g = nd.tanh(nd.dot(X, Wxh) + nd.dot(r * h, Whh) + bh)
        h = z * h + (1 - z) * g

        yhat_linear = nd.dot(h, Why) + by
        yhat = softmax(yhat_linear, temperature=temperature)
        outputs.append(yhat)
    return (outputs, h)

##Cross-entropy loss function
def cross_entropy(yhat, y):
    return - nd.mean(nd.sum(y * nd.log(yhat), axis=0, exclude=True))

##Averaging the loss over the sequence

def average_ce_loss(outputs, labels):
    assert(len(outputs) == len(labels))
    total_loss = nd.array([0.], ctx=ctx)
    for (output, label) in zip(outputs,labels):
        total_loss = total_loss + cross_entropy(output, label)
    return total_loss / len(outputs)

##Optimizer

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

##Generating text by sampling

def sample(prefix, num_chars, temperature=1.0):
    # Initialize the string that we'll return to the supplied prefix
    string = prefix

    # Prepare the prefix as a sequence of one-hots for ingestion by RNN
    prefix_numerical = [character_dict[char] for char in prefix]
    input = one_hots(prefix_numerical)

    # Set the initial state of the hidden representation ($h_0$) to the zero vector
    h = nd.zeros(shape=(1, num_hidden), ctx=ctx)
    c = nd.zeros(shape=(1, num_hidden), ctx=ctx)

    # For num_chars iterations,
    #     1) feed in the current input
    #     2) sample next character from from output distribution
    #     3) add sampled character to the decoded string
    #     4) prepare the sampled character as a one_hot (to be the next input)
    for i in range(num_chars):
        outputs, h = gru_rnn(input, h, temperature=temperature)
        choice = np.random.choice(vocab_size, p=outputs[-1][0].asnumpy())
        string += character_list[choice]
        input = one_hots([choice])
    return string


epochs = 2000
moving_loss = 0.

learning_rate = 2.0

# state = nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)
for e in range(epochs):
    # Attenuate the learning rate by a factor of 2 every 100 epochs.
    if ((e+1) % 100 == 0):
        learning_rate = learning_rate / 2.0
    h = nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)
    for i in range(num_batches):
        data_one_hot = train_data[i]
        label_one_hot = train_label[i]
        with autograd.record():
            outputs, h = gru_rnn(data_one_hot, h)
            loss = average_ce_loss(outputs, label_one_hot)
            loss.backward()
        SGD(params, learning_rate)

        #  Keep a moving average of the losses
        if (i == 0) and (e == 0):
            moving_loss = nd.mean(loss).asscalar()
        else:
            moving_loss = .99 * moving_loss + .01 * nd.mean(loss).asscalar()

    print("Epoch %s. Loss: %s" % (e, moving_loss))
    print(sample("The Time Ma", 1024, temperature=.1))
    print(sample("The Medical Man rose, came to the lamp,", 1024, temperature=.1))


    
    
    
###Recurrent Neural Networks with gluon
Example - predict the distribution of the next word given a sequence of previous words.


import math
import os
import time
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn

##Define classes for indexing words of the input document

#Dictionary class is for word indexing: words in the documents can be converted 
#from the string format to the integer format.

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

#Corpus class to index the words of the input document.

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(path + 'train.txt')
        self.valid = self.tokenize(path + 'valid.txt')
        self.test = self.tokenize(path + 'test.txt')

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = np.zeros((tokens,), dtype='int32')
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return mx.nd.array(ids, dtype='int32')


#Based on the gluon.Block class

class RNNModel(gluon.Block):
    """A model with an encoder, recurrent layer, and a decoder."""

    def __init__(self, mode, vocab_size, num_embed, num_hidden,
                 num_layers, dropout=0.5, tie_weights=False, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(vocab_size, num_embed,
                                        weight_initializer = mx.init.Uniform(0.1))
            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(num_hidden, num_layers, activation='relu', dropout=dropout,
                                   input_size=num_embed)
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(num_hidden, num_layers, dropout=dropout,
                                    input_size=num_embed)
            elif mode == 'gru':
                self.rnn = rnn.GRU(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            else:
                raise ValueError("Invalid mode %s. Options are rnn_relu, "
                                 "rnn_tanh, lstm, and gru"%mode)
            if tie_weights:
                self.decoder = nn.Dense(vocab_size, in_units = num_hidden,
                                        params = self.encoder.params)
            else:
                self.decoder = nn.Dense(vocab_size, in_units = num_hidden)
            self.num_hidden = num_hidden

    def forward(self, inputs, hidden):
        emb = self.drop(self.encoder(inputs))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.reshape((-1, self.num_hidden)))
        return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

        
        
##Select an RNN model and configure parameters

#to obtain a better performance, as reflected in a lower loss or perplexity, 
#one can set args_epochs to a larger value.

#In this demonstration, LSTM is the chosen type of RNN. 
#For other RNN options, one can replace the 'lstm' string to 'rnn_relu', 'rnn_tanh', or 'gru' 


args_data = '../data/nlp/ptb.'
args_model = 'rnn_relu'
args_emsize = 100
args_nhid = 100
args_nlayers = 2
args_lr = 1.0
args_clip = 0.2
args_epochs = 1
args_batch_size = 32
args_bptt = 5
args_dropout = 0.2
args_tied = True
args_cuda = 'store_true'
args_log_interval = 500
args_save = 'model.param'

##Load data as batches
#To speed up the subsequent data flow in the RNN model, 
#we pre-process the loaded data as batches


context = mx.cpu(0)
corpus = Corpus(args_data)

def batchify(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data

train_data = batchify(corpus.train, args_batch_size).as_in_context(context)
val_data = batchify(corpus.valid, args_batch_size).as_in_context(context)
test_data = batchify(corpus.test, args_batch_size).as_in_context(context)

##Build the model


ntokens = len(corpus.dictionary)

model = RNNModel(args_model, ntokens, args_emsize, args_nhid,
                       args_nlayers, args_dropout, args_tied)
model.collect_params().initialize(mx.init.Xavier(), ctx=context)
trainer = gluon.Trainer(model.collect_params(), 'sgd',
                        {'learning_rate': args_lr, 'momentum': 0, 'wd': 0})
loss = gluon.loss.SoftmaxCrossEntropyLoss()



def get_batch(source, i):
    seq_len = min(args_bptt, source.shape[0] - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len]
    return data, target.reshape((-1,))

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden

#model evaluation:It returns the loss of the model prediction. 


def eval(data_source):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(func = mx.nd.zeros, batch_size = args_batch_size, ctx=context)
    for i in range(0, data_source.shape[0] - 1, args_bptt):
        data, target = get_batch(data_source, i)
        output, hidden = model(data, hidden)
        L = loss(output, target)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal



def train():
    best_val = float("Inf")
    for epoch in range(args_epochs):
        total_L = 0.0
        start_time = time.time()
        hidden = model.begin_state(func = mx.nd.zeros, batch_size = args_batch_size, ctx = context)
        for ibatch, i in enumerate(range(0, train_data.shape[0] - 1, args_bptt)):
            data, target = get_batch(train_data, i)
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, target)
                L.backward()

            grads = [i.grad(context) for i in model.collect_params().values()]
            # Here gradient is for the whole batch.
            # So we multiply max_norm by batch_size and bptt size to balance it.
            gluon.utils.clip_global_norm(grads, args_clip * args_bptt * args_batch_size)

            trainer.step(args_batch_size)
            total_L += mx.nd.sum(L).asscalar()

            if ibatch % args_log_interval == 0 and ibatch > 0:
                cur_L = total_L / args_bptt / args_batch_size / args_log_interval
                print('[Epoch %d Batch %d] loss %.2f, perplexity %.2f' % (
                    epoch + 1, ibatch, cur_L, math.exp(cur_L)))
                total_L = 0.0

        val_L = eval(val_data)

        print('[Epoch %d] time cost %.2fs, validation loss %.2f, validation perplexity %.2f' % (
            epoch + 1, time.time() - start_time, val_L, math.exp(val_L)))

        if val_L < best_val:
            best_val = val_L
            test_L = eval(test_data)
            model.save_params(args_save)
            print('test loss %.2f, test perplexity %.2f' % (test_L, math.exp(test_L)))
        else:
            args_lr = args_lr * 0.25
            trainer._init_optimizer('sgd',
                                    {'learning_rate': args_lr,
                                     'momentum': 0,
                                     'wd': 0})
            model.load_params(args_save, context)

            
            
##Recall that the RNN model training is based on maximization likelihood of observations. 
#For evaluation purposes, we have used the following two measures:
    Loss: the loss function is defined as the average negative log likelihood of the words under prediction:
            loss=-(1/N)* SUM(log p-predicted-i), i = 1..N ,
            where N is the number of predictions and p-predicted-i 
            the likelihood of observing the next word in the i-th prediction.
    Perplexity: the average per-word perplexity is exp(loss)

##train and validate the model 

train()
model.load_params(args_save, context)
test_L = eval(test_data)
print('Best test loss %.2f, test perplexity %.2f'%(test_L, math.exp(test_L)))




http://gluon.mxnet.io/chapter07_distributed-learning/hybridize.html
###Fast, portable neural networks with Gluon HybridBlocks

##Imperative, or define-by-run, programming paradigm
#But the disadvantage is that it’s slow, constantly engaging the Python execution environment (which is slow) even though our entire function performs the same three low-level operations in the same sequence every time.
#It’s also holding on to all the intermediate values D and E until the function returns even though we can see that they’re not needed.
#But Imperative Programs Tend to be More Flexible
def our_function(A, B, C, D):
    # Compute some intermediate values
    E = basic_function1(A, B)
    F = basic_function2(C, D)

    # Finally, produce the thing you really care about
    G = basic_function3(E, F)
    return G

# Load up some data
W = some_stuff()
X = some_stuff()
Y = some_stuff()
Z = some_stuff()

result = our_function(W, X, Y, Z)

##symbolic programming - Theano and Tensorflow, 
#referred also as declarative programming or define-then-run programming. 
# The approach consists of three basic steps:
    Define a computation workflow, like a pass through a neural network, 
        using placeholder data
    Compile the program into a front-end language, e.g. Python, 
        independent format
    Invoke the compiled function, feeding it real data
    
#Symbolic Programs Tend to be More Efficient

#a symbolic version of the same program might look something like this:

# Create some placeholders to stand in for real data that might be supplied to the compiled function.
A = placeholder()
B = placeholder()
C = placeholder()
D = placeholder()

# Compute some intermediate values
E = symbolic_function1(A, B)
F = symbolic_function2(C, D)

# Finally, produce the thing you really care about
G = symbolic_function3(E, F)
# till this time no numerical computation actually happens

our_function = library.compile(inputs=[A, B, C, D], outputs=[G])

# Load up some data
W = some_stuff()
X = some_stuff()
Y = some_stuff()
Z = some_stuff()

result = our_function(W, X, Y, Z)

##Getting the best of both worlds with MXNet Gluon’s HybridBlocks
#Theano and those frameworks it inspired, like TensorFlow, run with the symbolic way
#Chainer and its descendants like PyTorch are fully imperative way

#MXNet accomplishes "best of both worlds" through the use of HybridBlocks. 
#Each HybridBlock can run fully imperatively defining their computation with real functions acting on real inputs. 
#But they’re also capable of running symbolically, acting on placeholders. 
#Gluon hides most of this under the hood 


#Given a HybridBlock whose forward computation consists of going through other HybridBlocks, 
#you can compile that section of the network by calling the HybridBlocks .hybridize() method.
 
##HybridSequential - counterpart of Sequential

#imperative style 
import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd

def get_net():
    # construct a MLP
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Dense(256, activation="relu"))
        net.add(nn.Dense(128, activation="relu"))
        net.add(nn.Dense(2))
    # initialize the parameters
    net.collect_params().initialize()
    return net

# forward
x = nd.random_normal(shape=(1, 512))
net = get_net()
print('=== net(x) ==={}'.format(net(x)))
#out 
=== net(x) ===
[[ 0.16526183 -0.14005636]]
<NDArray 1x2 @cpu(0)>

#To compile and optimize the HybridSequential
net.hybridize()
print('=== net(x) ==={}'.format(net(x)))
#output
=== net(x) ===
[[ 0.16526183 -0.14005636]]
<NDArray 1x2 @cpu(0)>

##Performance
from time import time
def bench(net, x):
    mx.nd.waitall()
    start = time()
    for i in range(1000):
        y = net(x)
    mx.nd.waitall()
    return time() - start

net = get_net()
print('Before hybridizing: %.4f sec'%(bench(net, x)))
net.hybridize()
print('After hybridizing: %.4f sec'%(bench(net, x)))
#output 
Before hybridizing: 0.4646 sec
After hybridizing: 0.2424 sec

##Get the symbolic program
#Previously, we feed net with NDArray data x, 
#and then net(x) returned the forward results. 

#Now if we feed it with a Symbol placeholder, 
#then the corresponding symbolic program will be returned.



from mxnet import sym
x = sym.var('data')
print('=== input data holder ===')
print(x)

y = net(x)
print('\n=== the symbolic program of net===')
print(y)

y_json = y.tojson()
print('\n=== the according json definition===')
print(y_json)
#output 
=== input data holder ===
<Symbol data>

=== the symbolic program of net===
<Symbol hybridsequential1_dense2_fwd>

=== the according json definition===
{
  "nodes": [
    {
      "op": "null",
      "name": "data",
      "inputs": []
    },
    {
      "op": "null",
      "name": "hybridsequential1_dense0_weight",
      "attr": {
        "__dtype__": "0",
        "__lr_mult__": "1.0",
        "__shape__": "(256, 0)",
        "__wd_mult__": "1.0"
      },
      "inputs": []
    },
    {
      "op": "null",
      "name": "hybridsequential1_dense0_bias",
      "attr": {
        "__dtype__": "0",
        "__init__": "zeros",
        "__lr_mult__": "1.0",
        "__shape__": "(256,)",
        "__wd_mult__": "1.0"
      },
      "inputs": []
    },
    {
      "op": "FullyConnected",
      "name": "hybridsequential1_dense0_fwd",
      "attr": {"num_hidden": "256"},
      "inputs": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    },
    {
      "op": "Activation",
      "name": "hybridsequential1_dense0_relu_fwd",
      "attr": {"act_type": "relu"},
      "inputs": [[3, 0, 0]]
    },
    {
      "op": "null",
      "name": "hybridsequential1_dense1_weight",
      "attr": {
        "__dtype__": "0",
        "__lr_mult__": "1.0",
        "__shape__": "(128, 0)",
        "__wd_mult__": "1.0"
      },
      "inputs": []
    },
    {
      "op": "null",
      "name": "hybridsequential1_dense1_bias",
      "attr": {
        "__dtype__": "0",
        "__init__": "zeros",
        "__lr_mult__": "1.0",
        "__shape__": "(128,)",
        "__wd_mult__": "1.0"
      },
      "inputs": []
    },
    {
      "op": "FullyConnected",
      "name": "hybridsequential1_dense1_fwd",
      "attr": {"num_hidden": "128"},
      "inputs": [[4, 0, 0], [5, 0, 0], [6, 0, 0]]
    },
    {
      "op": "Activation",
      "name": "hybridsequential1_dense1_relu_fwd",
      "attr": {"act_type": "relu"},
      "inputs": [[7, 0, 0]]
    },
    {
      "op": "null",
      "name": "hybridsequential1_dense2_weight",
      "attr": {
        "__dtype__": "0",
        "__lr_mult__": "1.0",
        "__shape__": "(2, 0)",
        "__wd_mult__": "1.0"
      },
      "inputs": []
    },
    {
      "op": "null",
      "name": "hybridsequential1_dense2_bias",
      "attr": {
        "__dtype__": "0",
        "__init__": "zeros",
        "__lr_mult__": "1.0",
        "__shape__": "(2,)",
        "__wd_mult__": "1.0"
      },
      "inputs": []
    },
    {
      "op": "FullyConnected",
      "name": "hybridsequential1_dense2_fwd",
      "attr": {"num_hidden": "2"},
      "inputs": [[8, 0, 0], [9, 0, 0], [10, 0, 0]]
    }
  ],
  "arg_nodes": [0, 1, 2, 5, 6, 9, 10],
  "node_row_ptr": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12
  ],
  "heads": [[11, 0, 0]],
  "attrs": {"mxnet_version": ["int", 1001]}
}

#Now we can save both the program and parameters onto disk, 
#so that it can be loaded later not only in Python,
#but in all other supported languages, such as C++, R, and Scala, as well.

y.save('model.json')
net.save_params('model.params')

##HybridBlock
#With normal Blocks, we just need to define a forward function 
#that takes an input x and computes the result of the forward pass 
#through the network. 
#MXNet can figure out the backward pass for us automatically with autograd.

#To define a HybridBlock, we instead have a hybrid_forward function:
from mxnet import gluon

class Net(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            self.fc1 = nn.Dense(256)
            self.fc2 = nn.Dense(128)
            self.fc3 = nn.Dense(2)

    def hybrid_forward(self, F, x):
        # F is a function space that depends on the type of x
        # If x's type is NDArray, then F will be mxnet.nd
        # If x's type is Symbol, then F will be mxnet.sym
        print('type(x): {}, F: {}'.format(
                type(x).__name__, F.__name__))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

#The hybrid_forward function takes an additional input, F, which stands for a backend. 

#MXNet has both a symbolic API (mxnet.symbol) and an imperative API (mxnet.ndarray).
#They have many of same functions (currently about 90% overlap) 
#and when they do, they support the same arguments in the same order. 

#When we define hybrid_forward, we pass in F. 
#When running in imperative mode, hybrid_forward is called with F as mxnet.ndarray 
#and x as some ndarray input. 

#When we compile with hybridize, F will be mxnet.symbol 
#and x will be some placeholder or intermediate symbolic value. 


net = Net()
net.collect_params().initialize()
x = nd.random_normal(shape=(1, 512))
print('=== 1st forward ===')
y = net(x)
print('=== 2nd forward ===')
y = net(x)
#output
=== 1st forward ===
type(x): NDArray, F: mxnet.ndarray
=== 2nd forward ===
type(x): NDArray, F: mxnet.ndarray

#Now run it again after hybridizing.

net.hybridize()
print('=== 1st forward ===')
y = net(x)
print('=== 2nd forward ===')
y = net(x)
#output
=== 1st forward ===
type(x): Symbol, F: mxnet.symbol
=== 2nd forward ===

#It differs from the previous execution in two aspects:
    the input data type now is Symbol even when we fed an NDArray into net, 
        because gluon implicitly constructed a symbolic data placeholder.
    hybrid_forward is called once at the first time we run net(x). 
        It is because gluon will construct the symbolic program on the first forward, 
        and then keep it for reuse later.

#One main reason that the network is faster after hybridizing is because 
#we don’t need to repeatedly invoke the Python forward function, 
#while keeping all computations within the highly efficient C++ backend engine.

#But the potential drawback is the loss of flexibility 
#to write the forward function. In other ways, 
#inserting print for debugging or control logic such as if and for into the forward function is not possible now.




http://gluon.mxnet.io/chapter07_distributed-learning/multiple-gpus-scratch.html
###Training with multiple GPUs from scratch

##Data Parallelism

#It works like this: Assume that we have k GPUs. 
    We split the examples in a data batch into k parts, 
    and send each part to a different GPUs 
    which then computes the gradient that part of the batch. 
    Finally, we collect the gradients from each of the GPUs 
    and sum them together before updating the weights.

def train_batch(data, k):
    split data into k parts
    for i = 1, ..., k:  # run in parallel
        compute grad_i w.r.t. weight_i using data_i on the i-th GPU
    grad = grad_1 + ... + grad_k
    for i = 1, ..., k:  # run in parallel
        copy grad to i-th GPU
        update weight_i by using grad

##Automatic Parallelization
#MXNet is able to automatically parallelize the workloads. 

#First, workloads, such as nd.dot are pushed into the backend engine for lazy evaluation. 
#and returns immediately without waiting for the computation to be finished. 
#We keep pushing until the results need to be copied out from MXNet, 
#such as print(x) or are converted into numpy by x.asnumpy(). 
#At that time, the Python thread is blocked until the results are ready.


from mxnet import nd
from time import time

start = time()
x = nd.random_uniform(shape=(2000,2000))
y = nd.dot(x, x)
print('=== workloads are pushed into the backend engine ===\n%f sec' % (time() - start))
z = y.asnumpy()
print('=== workloads are finished ===\n%f sec' % (time() - start))
#output
=== workloads are pushed into the backend engine ===
0.001160 sec
=== workloads are finished ===
0.174040 sec

#Second, MXNet depends on a powerful scheduling algorithm 
#that analyzes the dependencies of the pushed workloads. 
#This scheduler checks to see if two workloads are independent of each other. 
#If they are, then the engine may run them in parallel. 
#If a workload depend on results that have not yet been computed, 
#it will be made to wait until its inputs are ready.


a = nd.random_uniform(...)
b = nd.random_uniform(...)
c = a + b

#Then the computation for a and b may run in parallel, 
#while c cannot be computed until both a and b are ready.

#The following code shows that the engine effectively parallelizes 
#the dot operations on two GPUs:


from mxnet import gpu

def run(x):
    """push 10 matrix-matrix multiplications"""
    return [nd.dot(x,x) for i in range(10)]

def wait(x):
    """explicitly wait until all results are ready"""
    for y in x:
        y.wait_to_read()

x0 = nd.random_uniform(shape=(4000, 4000), ctx=gpu(0))
x1 = x0.copyto(gpu(1))

print('=== Run on GPU 0 and 1 in sequential ===')
start = time()
wait(run(x0))
wait(run(x1))
print('time: %f sec' %(time() - start))

print('=== Run on GPU 0 and 1 in parallel ===')
start = time()
y0 = run(x0)
y1 = run(x1)
wait(y0)
wait(y1)
print('time: %f sec' %(time() - start))
#output 
=== Run on GPU 0 and 1 in sequential ===
time: 1.842752 sec
=== Run on GPU 0 and 1 in parallel ===
time: 0.396227 sec

##

from mxnet import cpu

def copy(x, ctx):
    """copy data to a device"""
    return [y.copyto(ctx) for y in x]

print('=== Run on GPU 0 and then copy results to CPU in sequential ===')
start = time()
y0 = run(x0)
wait(y0)
z0 = copy(y0, cpu())
wait(z0)
print(time() - start)

print('=== Run and copy in parallel ===')
start = time()
y0 = run(x0)
z0 = copy(y0, cpu())
wait(z0)
print(time() - start)
#output 
=== Run on GPU 0 and then copy results to CPU in sequential ===
0.6489872932434082
=== Run and copy in parallel ===
0.39962267875671387


##Data parallelism using Gluon 

#example CNN
import mxnet as mx
from mxnet import nd, gluon, autograd
net = gluon.nn.Sequential(prefix='cnn_')
with net.name_scope():
    net.add(gluon.nn.Conv2D(channels=20, kernel_size=3, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(128, activation="relu"))
    net.add(gluon.nn.Dense(10))

loss = gluon.loss.SoftmaxCrossEntropyLoss()

##Initialize on multiple devices

#by passing in an array of device contexts, instead of the single contexts
GPU_COUNT = 2 # increase if you have more
ctx = [mx.gpu(i) for i in range(GPU_COUNT)]
net.collect_params().initialize(ctx=ctx)

#Given a batch of input data, we can split it into parts 
#(equal to the number of contexts) by calling gluon.utils.split_and_load(batch, ctx). 
#The split_and_load function doesn’t just split the data, 
#it also loads each part onto the appropriate device context.

#So now when we call the forward pass on two separate parts, 
#each one is computed on the appropriate corresponding device 
#and using the version of the parameters stored there.



from mxnet.test_utils import get_mnist
mnist = get_mnist()
batch = mnist['train_data'][0:GPU_COUNT*2, :]
data = gluon.utils.split_and_load(batch, ctx)
print(net(data[0]))
print(net(data[1]))

#output 
[[-0.01876061 -0.02165037 -0.01293943  0.03837404 -0.00821797 -0.00911531
   0.00416799 -0.00729158 -0.00232711 -0.00155549]
 [ 0.00441474 -0.01953595 -0.00128483  0.02768224  0.01389615 -0.01320441
  -0.01166505 -0.00637776  0.0135425  -0.00611765]]
<NDArray 2x10 @gpu(0)>

[[ -6.78736670e-03  -8.86893831e-03  -1.04004676e-02   1.72976423e-02
    2.26115398e-02  -6.36630831e-03  -1.54974898e-02  -1.22633884e-02
    1.19591374e-02  -6.60043515e-05]
 [ -1.17358668e-02  -2.16879714e-02   1.71219767e-03   2.49827504e-02
    1.16810966e-02  -9.52543691e-03  -1.03610428e-02   5.08510228e-03
    7.06662657e-03  -9.25292261e-03]]
<NDArray 2x10 @gpu(1)>

#At any time, we can access the version of the parameters stored on each device. 
#Recall ,our weights  may not actually be initialized 
#when we call initialize because the parameter shapes may not yet be known. 
#In these cases, initialization is deferred pending shape inference.


weight = net.collect_params()['cnn_conv0_weight']

for c in ctx:
    print('=== channel 0 of the first conv on {} ==={}'.format(
        c, weight.data(ctx=c)[0]))
#output 
=== channel 0 of the first conv on gpu(0) ===
[[[ 0.04118239  0.05352169 -0.04762455]
  [ 0.06035256 -0.01528978  0.04946674]
  [ 0.06110793 -0.00081179  0.02191102]]]
<NDArray 1x3x3 @gpu(0)>
=== channel 0 of the first conv on gpu(1) ===
[[[ 0.04118239  0.05352169 -0.04762455]
  [ 0.06035256 -0.01528978  0.04946674]
  [ 0.06110793 -0.00081179  0.02191102]]]
<NDArray 1x3x3 @gpu(1)>

#Similarly, we can access the gradients on each of the GPUs. 
#Because each GPU gets a different part of the batch 
#(a different subset of examples), the gradients on each GPU vary.

def forward_backward(net, data, label):
    with autograd.record():
        losses = [loss(net(X), Y) for X, Y in zip(data, label)]
    for l in losses:
        l.backward()

label = gluon.utils.split_and_load(mnist['train_label'][0:4], ctx)
forward_backward(net, data, label)
for c in ctx:
    print('=== grad of channel 0 of the first conv2d on {} ==={}'.format(
        c, weight.grad(ctx=c)[0]))
#output
=== grad of channel 0 of the first conv2d on gpu(0) ===
[[[-0.02078936 -0.00562428  0.01711007]
  [ 0.01138539  0.0280002   0.04094725]
  [ 0.00993335  0.01218192  0.02122578]]]
<NDArray 1x3x3 @gpu(0)>
=== grad of channel 0 of the first conv2d on gpu(1) ===
[[[-0.02543036 -0.02789939 -0.00302115]
  [-0.04816786 -0.03347274 -0.00403483]
  [-0.03178394 -0.01254033  0.00855637]]]
<NDArray 1x3x3 @gpu(1)>

##Put all things together

#Now we can implement the remaining functions. 
# a gluon trainer recognizes multi-devices, 
#it will automatically aggregate the gradients and synchronize the parameters.


from mxnet.io import NDArrayIter
from time import time

def train_batch(batch, ctx, net, trainer):
    # split the data batch and load them on GPUs
    data = gluon.utils.split_and_load(batch.data[0], ctx)
    label = gluon.utils.split_and_load(batch.label[0], ctx)
    # compute gradient
    forward_backward(net, data, label)
    # update parameters
    trainer.step(batch.data[0].shape[0])

def valid_batch(batch, ctx, net):
    data = batch.data[0].as_in_context(ctx[0])
    pred = nd.argmax(net(data), axis=1)
    return nd.sum(pred == batch.label[0].as_in_context(ctx[0])).asscalar()

def run(num_gpus, batch_size, lr):
    # the list of GPUs will be used
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    print('Running on {}'.format(ctx))

    # data iterator
    mnist = get_mnist()
    train_data = NDArrayIter(mnist["train_data"], mnist["train_label"], batch_size)
    valid_data = NDArrayIter(mnist["test_data"], mnist["test_label"], batch_size)
    print('Batch size is {}'.format(batch_size))

    net.collect_params().initialize(force_reinit=True, ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    for epoch in range(5):
        # train
        start = time()
        train_data.reset()
        for batch in train_data:
            train_batch(batch, ctx, net, trainer)
        nd.waitall()  # wait until all computations are finished to benchmark the time
        print('Epoch %d, training time = %.1f sec'%(epoch, time()-start))

        # validating
        valid_data.reset()
        correct, num = 0.0, 0.0
        for batch in valid_data:
            correct += valid_batch(batch, ctx, net)
            num += batch.data[0].shape[0]
        print('         validation accuracy = %.4f'%(correct/num))

run(1, 64, .3)
run(GPU_COUNT, 64*GPU_COUNT, .3)
#output 
Running on [gpu(0)]
Batch size is 64
Epoch 0, training time = 5.0 sec
         validation accuracy = 0.9738
Epoch 1, training time = 4.8 sec
         validation accuracy = 0.9841
Epoch 2, training time = 4.7 sec
         validation accuracy = 0.9863
Epoch 3, training time = 4.7 sec
         validation accuracy = 0.9868
Epoch 4, training time = 4.7 sec
         validation accuracy = 0.9877
Running on [gpu(0), gpu(1)]
Batch size is 128



###Distributed training with multiple machines

##Key-value store

#MXNet provides a key-value store to synchronize data among devices. 
#The following code initializes an ndarray associated with the key “weight” 
#on a key-value store.

from mxnet import kv, nd
store = kv.create('local')
shape = (2, 3)
x = nd.random_uniform(shape=shape)
store.init('weight', x)
print('=== init "weight" ==={}'.format(x))
#output
=== init "weight" ===
[[ 0.54881352  0.59284461  0.71518934]
 [ 0.84426576  0.60276335  0.85794562]]
<NDArray 2x3 @cpu(0)>

##After initialization, we can pull the value to multiple devices.

from mxnet import gpu
ctx = [gpu(0), gpu(1)]
y = [nd.zeros(shape, ctx=c) for c in ctx]
store.pull('weight', out=y)
print('=== pull "weight" to {} ===\n{}'.format(ctx, y))
#output
=== pull "weight" to [gpu(0), gpu(1)] ===
[
[[ 0.54881352  0.59284461  0.71518934]
 [ 0.84426576  0.60276335  0.85794562]]
<NDArray 2x3 @gpu(0)>,
[[ 0.54881352  0.59284461  0.71518934]
 [ 0.84426576  0.60276335  0.85794562]]
<NDArray 2x3 @gpu(1)>]

#We can also push new data value into the store. 
#It will first sum the data on the same key and then overwrite the current value.

z = [nd.ones(shape, ctx=ctx[i])+i for i in range(len(ctx))]
store.push('weight', z)
print('=== push to "weight" ===\n{}'.format(z))
store.pull('weight', out=y)
print('=== pull "weight" ===\n{}'.format(y))
#output
=== push to "weight" ===
[
[[ 1.  1.  1.]
 [ 1.  1.  1.]]
<NDArray 2x3 @gpu(0)>,
[[ 2.  2.  2.]
 [ 2.  2.  2.]]
<NDArray 2x3 @gpu(1)>]
=== pull "weight" ===
[
[[ 3.  3.  3.]
 [ 3.  3.  3.]]
<NDArray 2x3 @gpu(0)>,
[[ 3.  3.  3.]
 [ 3.  3.  3.]]
<NDArray 2x3 @gpu(1)>]

#With push and pull we can replace the allreduce function defined 
#in multiple-gpus-scratch by

def allreduce(data, data_name, store):
    store.push(data_name, data)
    store.pull(data_name, out=data)

##Distributed key-value store
#Not only can we synchronize data within a machine, 
#with the key-value store we can facilitate inter-machine communication. 

#To use it, one can create a distributed kvstore by using the following command: 
#(Note: distributed key-value store requires MXNet to be compiled with 
#the flag USE_DIST_KVSTORE=1, e.g. make USE_DIST_KVSTORE=1.)

store = kv.create('dist')

#Now if we run the code from the previous section on two machines at the same time, 
#then the store will aggregate the two ndarrays pushed from each machine, 
#and after that, the pulled results will be:
[[ 6.  6.  6.]
 [ 6.  6.  6.]]

#In the distributed setting, MXNet launches three kinds of processes 
#(each time, running python myprog.py will create a process). 
#One is a worker, which runs the user program, 
#such as the code in the previous section. 
#The other two are the server, which maintains the data pushed into the store, 
#and the scheduler, which monitors the aliveness of each node.

#It’s up to users which machines to run these processes on. 
#But to simplify the process placement and launching, 
#MXNet provides a tool located at tools/launch.py.

#Assume there are two machines, A and B. 
#They are ssh-able, and their IPs are saved in a file named hostfile. 
#Then we can start one worker in each machine through:

$ mxnet_path/tools/launch.py -H hostfile -n 2 python myprog.py

#It will also start a server in each machine, 
#and the scheduler on the same machine we are currently on.



##Using kvstore in gluon
#to change to multi-machine training 
#we only need to pass a distributed key-value store, for example,

store = kv.create('dist')
trainer = gluon.Trainer(..., kvstore=store)

#To split the data,
#One commonly used solution is to split the whole dataset into k parts at the beginning, 
#then let the i-th worker only read the i-th part of the data.

#We can obtain the total number of workers by reading the attribute num_workers 
#and the rank of the current worker from the attribute rank.

print('total number of workers: %d'%(store.num_workers))
print('my rank among workers: %d'%(store.rank))
#output
total number of workers: 1
my rank among workers: 0

#With this information, we can manually access the proper chunk of the input data. 
#In addition, several data iterators provided by MXNet already support reading 
#only part of the data
from mxnet.io import ImageRecordIter
data = ImageRecordIter(num_parts=store.num_workers, part_index=store.rank, ...)


###check other subjects 
http://gluon.mxnet.io/chapter08_computer-vision/object-detection.html