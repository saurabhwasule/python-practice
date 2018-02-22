###NN - Quick tour of NN 
#the activation function of a node defines the output of that node given an input 
#or set of inputs
#https://en.wikipedia.org/wiki/Activation_function

#If the activation function is linear, 
#then you can stack as many hidden layers in the neural network 
#and the final output is still a linear combination of the original input data

#Note: step function has no useful derivative (its derivative is 0 everywhere or undefined at the 0 point on x-axis). 
#It doesn’t work for backpropagation
#Perceptron with step function isn’t very “stable”
#a small change in any weight in the input layer of our perceptron network 
#could possibly lead to one neuron to suddenly flip from 0 to 1, 
#which could again affect the hidden layer’s behavior, and then affect the final outcome

#The solution is to use sigmoid function 
Zi = WiXi  where Wi is weights and Xi is input 
z = SUM(Wi*Xi) + bias , i=1..m 
sigmoid function , = 1/(1+exp(-z)), like S curve 

#few others 
tanh                                            f(z) = tanh(z) 
Rectified linear unit (ReLU)                    f(z) = 0 for z <0, z for z>=0
Leaky rectified linear unit (Leaky ReLU)        f(z) = 0.01z for z <0, z for z>=0
softmax                                         Fi(z) = exp(Zi)/(SUM(Zi)), i=1..m, then max of Fi(z)
maxout                                          f(z) = max of Zi 

#A deep neural network (DNN) is an NN with multiple hidden layers between the input and output layers


##NN - A feedforward neural network 
#is an artificial neural network 
#wherein connections between the units do not form a cycle
#the information moves in only one direction, forward, from the input nodes, 
#through the hidden nodes (if any) and to the output nodes. 
#There are no cycles or loops in the network
 


##NN - Single-layer perceptron
#The simplest kind of neural network is a single-layer perceptron network, 
#which consists of a single layer of output nodes; 
#the inputs are fed directly to the outputs via a series of weights

#The sum of the products of the weights and the inputs is calculated in each node, 
#and if the value is above some threshold (typically 0) the neuron fires 
#and takes the activated value (typically 1); 
#otherwise it takes the deactivated value (typically -1). 

#Neurons with this kind of activation function are also called artificial neurons 
#or linear threshold units



##NN - Multi-layer perceptron
#Contains hidden layers between input and output nodes 
#Check perceptron_node.png
#check https://deeplearning4j.org/neuralnet-overview


#Multi-layer networks use a variety of learning techniques, 
#the most popular being back-propagation. 

#Here, the output values are compared with the correct answer to compute the value of some predefined error-function.
#By various techniques, the error is then fed back through the network. 
#Using this information, the algorithm adjusts the weights of each connection 
#in order to reduce the value of the error function by some small amount. 
#After repeating this process for a sufficiently large number of training cycles, 
#the network will usually converge to some state 
#where the error of the calculations is small. 
#In this case, one would say that the network has learned a certain target function. 

#To adjust weights properly, one applies a general method for non-linear optimization 
#that is called gradient descent. 
#For this, the network calculates the derivative of the error function 
#with respect to the network weights, 
#and changes the weights such that the error decreases 
#(thus going downhill on the surface of the error function). 
#For this reason, back-propagation can only be applied on networks 
#with differentiable activation functions


##NN - Recurrent neural network
#A recurrent neural network (RNN) is a class of artificial neural network 
#where connections between units form a directed cycle. 
#This allows it to exhibit dynamic temporal behavior. 
#Unlike feedforward neural networks, 
#RNNs can use their internal memory to process arbitrary sequences of inputs. 
#This makes them applicable to tasks such as unsegmented, 
#connected handwriting recognition or speech recognition


##NN - Long short-term memory
#Long short-term memory (LSTM) block or network is a recurrent neural network 
#which can be used as a building component or block (of hidden layers) for  bigger recurrent neural network

#An LSTM block is composed of four main components: 
#a cell, an input gate, an output gate and a forget gate. 
#The cell is responsible for "remembering" values over arbitrary time intervals; 
#hence the word "memory" in LSTM. 
#Each of the three gates can be thought of as a "conventional" artificial neuron, 
#as in a multi-layer (or feedforward) neural network: 
#that is, they compute an activation (using an activation function) of a weighted sum. 

#The expression long short-term refers to the fact that LSTM is a model 
#for the short-term memory which can last for a long period of time. 

#An LSTM is well-suited to classify, process and predict time series 
#given time lags of unknown size and duration between important events.


##NN-Convolutional neural network
#A CNN consists of an input and an output layer, as well as multiple hidden layers. 
#The hidden layers of a CNN typically consist of convolutional(actually cross-correlation)
#layers, pooling layers, fully connected layers and normalization layers.

#Convolutional
#Convolutional layers apply a convolution operation to the input, 
#passing the result to the next layer

#Each convolutional neuron processes data only for its receptive field. 
#Tiling allows CNNs to tolerate translation of the input image 
#(e.g. translation, rotation, perspective distortion)

#Pooling
#Convolutional networks may include local or global pooling layers, 
#which combine the outputs of neuron clusters at one layer into a single neuron 
#in the next layer.
#For example, max pooling uses the maximum value from each of a cluster of neurons 
#at the prior layer.
#Another example is average pooling, which uses the average value from each of a cluster of neurons at the prior layer

#Fully connected
#Fully connected layers connect every neuron in one layer to every neuron in another layer.
#It is in principle the same as the traditional multi-layer perceptron neural network (MLP)


##one hot encoding 

#multiclass  is the problem of classifying instances into one of the more than two classes
#classifying instances into one of the two classes is called binary classification
#multi-label classification:  multiple labels are to be predicted for each instance.

#Given below table 
CompanyName   Price  
 VW           20000  
 Acura        10011  
 Honda        50000  
 Honda        10000  
 
#LabelEncoder 
CompanyName   Price  Categoricalvalue
 VW           20000  1
 Acura        10011  2
 Honda        50000  3
 Honda        10000  3

#One hot encoding - CompanyName feature column becomes three features 
#required in classification
VW  Acura Honda Price  
 1   0     0     20000  
 0   1     0     10011  
 0   0     1     50000  
 0   0     1     10000 

#multihot - summation of  one hot encoding feature column
VW  Acura Honda Price  
 1   0     0     20000  
 0   1     0     10011  
 0   0     2     50000  
 0   0     2     10000 
 
#Multi-label classification can be performed 
#with just a single one-hot vector for each label. 
#But the one-hot-vector has multiple ones for each sample rather than just single one. 
 
 





###Tensorflow - Installation only on 3.5.2 x64 bit 
$ pip3 install --upgrade tensorflow

#MSVCP140.DLL  must be in system32 
#https://www.microsoft.com/en-us/download/details.aspx?id=53587

##https://www.tensorflow.org/get_started/get_started
##The lowest level API--TensorFlow Core-- fine levels of control over  models.
#The higher level APIs, tf.estimator, tf.layers are built on top of TensorFlow Core. 


 


###Tensflow -Introduction - Tensors
#A tensor consists of a set of primitive values shaped into an array of any number of dimensions. 
#A tensor's rank is its number of dimensions.

3               # a rank 0 tensor; this is a scalar with shape []
[1., 2., 3.]    # a rank 1 tensor; this is a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]]        # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]]    # a rank 3 tensor with shape [2, 1, 3]


##Tensflow - Importing TensorFlow

import tensorflow as tf

##Tensflow - The Computational Graph
#A computational graph is a series of TensorFlow operations 
#arranged into a graph of nodes.
#TensorFlow Core programs  consisting of two discrete sections:
1.Building the computational graph.
2.Running the computational graph.

##Tensflow - constant 
#Each node takes zero or more tensors as inputs and produces a tensor as an output. 
#One type of node is a constant:
    #it takes no inputs, and it outputs a value it stores internally. 
#Constants are initialized with tf.constant, 
#and their value can never change.

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
>>> print(node1, node2)
Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)

#To evaluate the nodes, 
#we must run the computational graph within a session. 
#A session encapsulates the control and state of the TensorFlow runtime.

sess = tf.Session()
>>> print(sess.run([node1, node2])) # running multiple operation via list, returns multiple values 
[3.0, 4.0]


#Example - add two constant nodes and produce a new graph 
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))
#output
node3: Tensor("Add:0", shape=(), dtype=float32)
sess.run(node3): 7.0
#OR 
#t.eval(feed_dict=None,session=None) is a shortcut for calling 
#tf.get_default_session().run(t).
with sess.as_default():
    node3.eval()

##Tensflow - placeholder
##A graph can be parameterized to accept external inputs, known as placeholders. 
#A placeholder is a promise to provide a value later.

a = tf.placeholder(tf.float32) #a can be array as well, dtype must be type of each element 
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

#evaluate this graph 
print(sess.run(adder_node, {a: 3, b: 4.5})) #note key is object ,a, not string 'a'
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
#output
7.5
[ 3.  7.]
#Or 
with sess.as_default():
    adder_node.eval(feed_dict={a: 3, b: 4.5})

#adding another operation
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))
#output
22.5

##Tensflow - variable 
#Variables allow to add trainable parameters to a graph. 
#They are constructed with a type and initial value:

#basically W, b are mutable and value can be changed in place 
#Note placeholders are meant for input from users 
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b


#To initialize all the variables in a TensorFlow program
init = tf.global_variables_initializer()
sess.run(init)

#Since x is a placeholder,
# evaluate linear_model for several values of x simultaneously as follows:
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
#output
[ 0.          0.30000001  0.60000002  0.90000004]


##Tensflow - true y and Loss function  
#To evaluate the model on training data, 
#we need a y placeholder to provide the desired values, 
#and we need to write a loss function.

#A loss function measures how far apart the current model is from the provided data. 
#standard loss model for linear regression:
#which sums the squares of the deltas between the current model and the provided data. 


y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
#output
23.66

##Tensflow - Training 
#Vary values of W and b to get best values where loss is zero
#machine learning is used to find the correct model parameters automatically(here W,b) 

#A variable is initialized to the value provided to tf.Variable 
#but can be changed using operations like tf.assign. 

#with -1, 1 values of W, b 
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])  #Execute assignments 
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
#output, perfectly minimised
0.0




###Tensflow - tf.train API
#TensorFlow provides optimizers that slowly change each variable 
#in order to minimize the loss function. 
#The simplest optimizer is gradient descent. 
#It modifies each variable according to the magnitude of the derivative of loss with respect to that variable. 

#In general, computing symbolic derivatives manually is tedious and error-prone. 
#TensorFlow can automatically produce derivatives by tf.gradients for some models

#OR use optimizer 

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# reset values to  defaults.
sess.run(init) 

for i in range(1000):
  sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
#check value of W, b after 1000 run 
>>> print(sess.run([W, b])) 
[array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]


###Tensflow - Complete program for linear model optimization- manual training

import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

#model 
linear_model = W * x + b


# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values
#after below operations - W, b would contain end values where loss is minimised
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train}) #note key is not string 

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
#W: [-0.9999969] b: [ 0.99999082] loss: 5.69997e-11



###Tensflow - tf.estimator
#tf.estimator defines many common models.
#tf.estimator is a high-level TensorFlow library 
#that simplifies the mechanics of machine learning eg running training loop etc 


##Tensflow - Few Terminologies 
#Recommended practice 
#           batch_size  num_epochs              shuffle     Steps   Output
TRAIN       x           None(=autocalculated)   True        1000    Update trainable parameters(variables) 
EVALUATE    x           Finite                  False       x       Dict containing evaluation metrics, eg 'loss', 'accuracy'(=float)
PREDICT     x           1                       False       NA      list of dict containing 'predictions'(=np.ndarray) for each input x 


#There are n samples and the batch_size is b , num_epoch = steps*b/n
#Example 
#Training for 5 epochs on a 1000 samples 10 samples per batch will take 500 steps(num_epoch*n/b)

#Meaning 
1) Steps - number of times the training loop(forward and backword) in learning algorithm 
   will run to update the parameters in the model. 
   In each loop(step), it will process only one chunk of data(batch)
   Usually, this loop is based on the Gradient Descent algorithm.

2) Batch size - the size of the chunk of data in each loop  of the learning algorithm. 
   You can feed the whole data set, in which case the batch size 
   is equal to the data set size.
   You can also feed one example/observations at a time. 
   Or you can feed some number N of examples. 

3) Epoch - the number of times the full data set 
   is used to generate batches for training loops(steps)
   
4) Shuffle : if True shuffles the input data queue.

#Example 
#Note no of inputs to input layer needs to equal to batch size(Check?)

from get_mnist_data_tf import read_data_sets
mnist = read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.initialize_all_variables())
y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(accuracy.eval(feed_dict={x: mnist.test.images,
                               y_: mnist.test.labels}))
You want to leave the Batch size, to None, that means that you can run the model with a variable number of inputs (one or more). Batching is important to efficiently use your computing resources.
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

The next important line is:
batch = mnist.train.next_batch(50)

Here you are sending 50 elements as input but you can also change that to just one 
batch = mnist.train.next_batch(1)

Without modifying the graph. If you specify the Batch size (some number instead of None in the first snippet), then you would have to change each time and that is not ideal, specially in production.



##Example - using tf.estimator for the above 
import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np

# Declare list of features. We only have one numeric feature. 
#input data key is 'x'
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])] # each x is of size 1 (1D) ie scalar 

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# linear classification, and many neural network classifiers and regressors.
# The following code provides an estimator that does linear regression.
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use two data sets: one for training and one for evaluation
# We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

#batch_size = 4 means full data output in one execution of *_fn 
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True) #num_epochs=None to get infinite stream of data 
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)
    
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": x_eval}, batch_size=4, num_epochs=1, shuffle=False)

# We can invoke 1000 training steps by invoking the  method and passing the
# training data set.
estimator.train(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did.
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn) #{'global_step': 1000, 'average_loss': 0.002559511, 'loss':.010238044}
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)

##When run, it produces
train metrics: {'loss': 1.2712867e-09, 'global_step': 1000}
eval metrics: {'loss': 0.0025279333, 'global_step': 1000}
#predict 
predict_y = estimator.predict(input_fn=predict_input_fn)
list(predict_y) 
#generator , four prediction points for x_eval = np.array([2., 5., 8., 1.])
[{'predictions': array([-1.00027156], dtype=float32)}, {'predictions': array([-3.99930096], dtype=float32)}, {'predictions': array([-6.99833012], dtype=float32)}, {'predictions': array([-0.00059521], dtype=float32)}]



###A custom model using tf.estimator  

#To define a custom model that works with tf.estimator, 
#we need to use tf.estimator.Estimator. 

#tf.estimator.LinearRegressor is actually a sub-class of tf.estimator.Estimator. 

#Instead of sub-classing Estimator, 
#we simply provide Estimator a function model_fn that tells tf.estimator 
#how it can evaluate predictions, training steps, and loss. 

import numpy as np
import tensorflow as tf

# Declare list of features, we only have one real-valued feature
#mode is tf.estimator.ModeKeys, Used to differentiate PREDICT, TRAIN, EVALUATE phase 
#features, labels are user defined structure, access them as per own definitions 
#arg order can be random 
def model_fn(features, labels, mode):
  # Build a linear model and predict values
  #get_variable is used to create a Variable - here of size 1 (1D) ie scalar 
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W * features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  #tf_group is used to group a number of operations 
  train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
  # EstimatorSpec connects subgraphs we built to the appropriate functionality.
  #we should return loss operation, training operation and prediction operation 
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.estimator.Estimator(model_fn=model_fn)
# define our data sets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# train
estimator.train(input_fn=input_fn, steps=1000)
# Here we evaluate how well our model did.
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)





###Tensorflow - Tensors 

#A tf.Tensor has the following properties:
    A data type (float32, int32, or string, for example)
    A shape

#Each element in the Tensor has the same data type, 
#and the data type is always known. 

#Some types of tensors are special, 
#With the exception of tf.Variable, the value of a tensor is immutable,
    tf.Variable
    tf.Constant
    tf.Placeholder
    tf.SparseTensor

##Tensor like objects - below can be passed in place of tf.Tensor in Tensorflow, 
#and it converts them to tf.Tensor
    tf.Tensor
    tf.Variable
    numpy.ndarray
    list (and lists of tensor-like objects) => Tensor of rank 1
    Scalar Python types: bool, float, int, str => Tensor of rank 0

#OR register additional tensor-like types using tf.register_tensor_conversion_function.

 
##Rank
#The rank of a tf.Tensor object is its number of dimensions. 
#Synonyms for rank include order or degree or n-dimension. 
#Rank
0   Scalar (magnitude only) 
1   Vector (magnitude and direction) 
2   Matrix (table of numbers) 
3   3-Tensor (cube of numbers) 
n   n-Tensor (you get the idea) 

##Rank 0
mammal = tf.Variable("Elephant", tf.string)
ignition = tf.Variable(451, tf.int16)
floating = tf.Variable(3.14159265359, tf.float64)
its_complicated = tf.Variable((12.3, -4.85), tf.complex64)

##Rank 1
mystr = tf.Variable(["Hello"], tf.string)
cool_numbers  = tf.Variable([3.14159, 2.71828], tf.float32)
first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
its_very_complicated = tf.Variable([(12.3, -4.85), (7.5, -6.23)], tf.complex64)


##Higher ranks
#A rank 2 tf.Tensor object consists of at least one row and at least one column:


mymat = tf.Variable([[7],[11]], tf.int16)
myxor = tf.Variable([[False, True],[True, False]], tf.bool)
linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
squarish_squares = tf.Variable([ [4, 9], [16, 25] ], tf.int32)
rank_of_squares = tf.rank(squarish_squares)
mymatC = tf.Variable([[7],[11]], tf.int32)


#For example, during image processing, many tensors of rank 4 are used, 
#with dimensions corresponding batch_size, image width, image height, and color channel.
my_image = tf.zeros([10, 299, 299, 3])  # batch_size x height x width x color


##Getting a tf.Tensor object's rank
r = tf.rank(my3d)
# After the graph runs, r will hold the value 3.


##Referring to tf.Tensor slices, [dimension1,dimension2,...] 
#or [scalar tf.Tensor1, scalar tf.Tensor2, scalar tf.Tensor3,...]
#For a rank 0 tensor (a scalar), no indices are necessary

#For a rank 1 tensor (a vector), passing a single index allows you to access a number:
my_scalar = my_vector[2]


#For tensors of rank 2 or higher
my_scalar = my_matrix[1, 2]

#Passing a single number, returns a subvector of a matrix

my_row_vetor = my_matrix[2]
my_column_vector = my_matrix[:, 3]


##Shape
#The shape of a tensor is the number of elements in each dimension
#Shapes can be represented via Python lists / tuples of ints, or with the tf.TensorShape.

#TensorFlow automatically infers shapes during graph construction. 
#These inferred shapes might have known or unknown rank. 
#If the rank is known, the sizes of each dimension might be known or unknown.

#Rank       Shape               Dimension number    Example
0           []                  0-D                 A 0-D tensor. A scalar. 
1           [D0]                1-D                 A 1-D tensor with shape [5]. 
2           [D0, D1]            2-D                 A 2-D tensor with shape [3, 4]. 
3           [D0, D1, D2]        3-D A               3-D tensor with shape [1, 4, 3]. 
n           [D0, D1, ... Dn-1]  n-D                 A tensor with shape [D0, D1, ... Dn-1]. 


##Getting a tf.Tensor object's shape

tensor.shape #returns a TensorShape object, might be  partially-specified shapes 
tf.shape(tensor)
#Example 
zeros = tf.zeros(tf.shape(my_matrix)[1])


##Changing the shape of a tf.Tensor - use tf.reshape 
#The number of elements of a tensor is the product of the sizes of all its shapes.

rank_three_tensor = tf.ones([3, 4, 5])
matrix = tf.reshape(rank_three_tensor, [6, 10])  # Reshape existing content into
                                                 # a 6x10 matrix
matrixB = tf.reshape(matrix, [3, -1])  #  Reshape existing content into a 3x20
                                       # matrix. -1 tells reshape to calculate
                                       # the size of this dimension.
matrixAlt = tf.reshape(matrixB, [4, 3, -1])  # Reshape existing content into a
                                             #4x3x5 tensor

# Note that the number of elements of the reshaped Tensors has to match the
# original number of elements. Therefore, the following example generates an
# error because no possible value for the last dimension will match the number
# of elements.
yet_another = tf.reshape(matrixAlt, [13, 2, -1])  # ERROR!


##Data types
#Tensors have one single data type
#It is possible to cast tf.Tensors from one datatype to another using tf.cast:

# Cast a constant integer tensor into floating point.
float_tensor = tf.cast(tf.constant([1, 2, 3]), dtype=tf.float32)
float_tensor.dtype  #tf.float32

#list of datatypes , instance of tf.DType 
tf.float16: 16-bit half-precision floating-point.
tf.float32: 32-bit single-precision floating-point.
tf.float64: 64-bit double-precision floating-point.
tf.bfloat16: 16-bit truncated floating-point.
tf.complex64: 64-bit single-precision complex.
tf.complex128: 128-bit double-precision complex.
tf.int8: 8-bit signed integer.
tf.uint8: 8-bit unsigned integer.
tf.uint16: 16-bit unsigned integer.
tf.int16: 16-bit signed integer.
tf.int32: 32-bit signed integer.
tf.int64: 64-bit signed integer.
tf.bool: Boolean.
tf.string: String.
tf.qint8: Quantized 8-bit signed integer.
tf.quint8: Quantized 8-bit unsigned integer.
tf.qint16: Quantized 16-bit signed integer.
tf.quint16: Quantized 16-bit unsigned integer.
tf.qint32: Quantized 32-bit signed integer.
tf.resource: Handle to a mutable resource.

#variants of these types with the _ref suffix are defined for reference-typed tensors.
#To convert numpy types and string type names to a DType object.
tf.as_dtype (type_value)
    A value that can be converted to a tf.DType object.
    This may  be a tf.DType object, types_pb2.DataType  enum, a string type name, 
    or a numpy.dtype.

#Attributes of tf.Dtype 
as_datatype_enum
as_numpy_dtype
base_dtype
is_bool
is_complex
is_floating
is_integer
is_numpy_compatible
is_quantized
is_unsigned
limits  #Return intensity limits, i.e. (min, max) tuple, of the dtype.
max
min  #Returns the minimum representable value in this data type.
name  #Returns the string name for this DType.
real_dtype
size
__init__(type_enum)    #A types_pb2.DataType enum value
__eq__(other)
__int__()
__ne__(other)
is_compatible_with(other)
    DType(T)       .is_compatible_with(DType(T))        == True
    DType(T)       .is_compatible_with(DType(T).as_ref) == True
    DType(T).as_ref.is_compatible_with(DType(T))        == False
    DType(T).as_ref.is_compatible_with(DType(T).as_ref) == True




##Evaluating Tensors
#run the computation that produces a particular tf.Tensor 
#valid when a default tf.Session is active(ie with tf.Session().as_default():)
# .eval(feed_dict) equivalent to tf.get_default_session().run(tensor)
constant = tf.constant([1, 2, 3])
tensor = constant * constant
with tf.Session().as_default():
    print tensor.eval() #returns a numpy array with the same contents as the tensor.
#else use a session 
sess = tf.Session()
print sess.run(tensor)

#Sometimes it is not possible to evaluate a tf.Tensor with no context 
#because its value might depend on dynamic information that is not available. 
#For example, tensors that depend on Placeholders 
p = tf.placeholder(tf.float32)
t = p + 1.0
with tf.Session().as_default():
    t.eval()  # This will fail, since the placeholder did not get a value.
    t.eval(feed_dict={p:2.0})  # This will succeed Note p is object, not string 


#If a tf.Tensor depends on a value from a queue, 
#evaluating the tf.Tensor will only work once something has been enqueued; 
#remember to call tf.train.start_queue_runners before evaluating any tf.Tensors.


##Printing Tensors

#This code prints the tf.Tensor object (which represents deferred computation) and not its value.
t = <<some tensorflow operation>>
print t  # This will print the symbolic tensor when the graph is being built.
         # This tensor does not have a value in this context.


 
#Solution - Use tf.Print(input_,  data,  message=None,  first_n=None,  summarize=None,  name=None)
    input_: A tensor passed through this op.
    data: A list of tensors to print out when op is evaluated.
    Returns:Same tensor as input_.
#To correctly use tf.Print its return value must be used
t = <<some tensorflow operation>>
tf.Print(t, [t])  # This does nothing
t = tf.Print(t, [t])  # Here we are using the value returned by tf.Print
result = t + 1  # Now when result is evaluated the value of `t` will be printed.

#Now evaluate , and above tensor is printed 
sess = tf.Session()
print sess.run(result)


 
##Reference - Tensor 
#A Tensor can be passed as an input to another Operation.

# Build a dataflow graph.
c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
e = tf.matmul(c, d)

# Construct a `Session` to execute the graph.
sess = tf.Session()

# Execute the graph and store the value that `e` represents in `result`.
result = sess.run(e)

##Has below methods/atributes 
set_shape(shape)
    Updates the shape of this tensor.
    _, image_data = tf.TFRecordReader(...).read(...)
    image = tf.image.decode_png(image_data, channels=3)
    # The height and width dimensions of `image` are data dependent, and
    # cannot be computed without executing the op.
    print(image.shape)
    ==> TensorShape([Dimension(None), Dimension(None), Dimension(3)])
    # We know that each image in this dataset is 28 x 28 pixels.
    image.set_shape([28, 28, 3])
    print(image.shape)
    ==> TensorShape([Dimension(28), Dimension(28), Dimension(3)])



get_shape()/.shape
    Returns the TensorShape that represents the shape of this tensor.
    c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(c.shape)
    ==> TensorShape([Dimension(2), Dimension(3)])

    d = tf.constant([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    print(d.shape)
    ==> TensorShape([Dimension(4), Dimension(2)])

    # Raises a ValueError, because `c` and `d` do not have compatible
    # inner dimensions.
    e = tf.matmul(c, d)
    f = tf.matmul(c, d, transpose_a=True, transpose_b=True)
    print(f.shape)
    ==> TensorShape([Dimension(3), Dimension(4)])


##has below method,  which does operation elementwise  
abs(t)
    # tensor 'x' is [[-2.25 + 4.75j], [-3.25 + 5.75j]]
    tf.complex_abs(x) ==> [5.25594902, 6.60492229]

#Arithmatic operation - elementwise  
t1 + t2 , t1 - t2 
t1 and t2 , t1 or t2 , not t1 
does not have bool(t), iter(t1)
t1/t2 , t1//t2 , t1 % t2 , t1 * t2 
t1 == t2 , t1 != t2, t1 <= t2 etc 
-t1 
t1 ** t2 
    # tensor 'x' is [[2, 2], [3, 3]]
    # tensor 'y' is [[8, 16], [2, 3]]
    tf.pow(x, y) ==> [[256, 65536], [9, 27]]

getitem/slicing [start:stop:end]
    # strip leading and trailing 2 elements
    foo = tf.constant([1,2,3,4,5,6])
    print(foo[2:-2].eval()) # => [3,4]

    # skip every row and reverse every column
    foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
    print(foo[::2,::-1].eval()) # => [[3,2,1], [9,8,7]]

    # Insert another dimension
    foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
    print(foo[tf.newaxis, :, :].eval()) # => [[[1,2,3], [4,5,6], [7,8,9]]]
    print(foo[:, tf.newaxis, :].eval()) # => [[[1,2,3]], [[4,5,6]], [[7,8,9]]]
    print(foo[:, :, tf.newaxis].eval()) # => [[[1],[2],[3]], [[4],[5],[6]], [[7],[8],[9]]]

    # Ellipses (3 equivalent operations)
    foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
    print(foo[tf.newaxis, :, :].eval()) # => [[[1,2,3], [4,5,6], [7,8,9]]]
    print(foo[tf.newaxis, ...].eval()) # => [[[1,2,3], [4,5,6], [7,8,9]]]
    print(foo[tf.newaxis].eval()) # => [[[1,2,3], [4,5,6], [7,8,9]]]


matmul/@ 
    mat multiplication 
    # 2-D tensor `a`
    a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3]) => [[1. 2. 3.]
                                                          [4. 5. 6.]]
    # 2-D tensor `b`
    b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2]) => [[7. 8.]
                                                             [9. 10.]
                                                             [11. 12.]]
    c = tf.matmul(a, b) => [[58 64]
                            [139 154]]
    # 3-D tensor `a`
    a = tf.constant(np.arange(1, 13, dtype=np.int32),
                    shape=[2, 2, 3])                  => [[[ 1.  2.  3.]
                                                           [ 4.  5.  6.]],
                                                          [[ 7.  8.  9.]
                                                           [10. 11. 12.]]]

    # 3-D tensor `b`
    b = tf.constant(np.arange(13, 25, dtype=np.int32),
                    shape=[2, 3, 2])                   => [[[13. 14.]
                                                            [15. 16.]
                                                            [17. 18.]],
                                                           [[19. 20.]
                                                            [21. 22.]
                                                            [23. 24.]]]
    c = tf.matmul(a, b) => [[[ 94 100]
                             [229 244]],
                            [[508 532]
                             [697 730]]]

    # Since python >= 3.5 the @ operator is supported (see PEP 465).
    # In TensorFlow, it simply calls the `tf.matmul()` function, so the
    # following lines are equivalent:
    d = a @ b @ [[10.], [11.]]
    d = tf.matmul(tf.matmul(a, b), [[10.], [11.]])







###Tensorflow - Variables 
#https://www.tensorflow.org/programmers_guide/variables

#The best way to create a variable is to call the tf.get_variable
tf.get_variable(
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=True,
    collections=None,
    caching_device=None,
    partitioner=None,
    validate_shape=True,
    use_resource=None,
    custom_getter=None
)

##To create a variable with tf.get_variable
my_variable = tf.get_variable("my_variable", [1, 2, 3])  #3D tensor , 
#dtype tf.float32 and its initial value is randomized via tf.glorot_uniform_initializer 


#OR specify manually 
my_int_variable = tf.get_variable("my_int_variable", [1, 2, 3], dtype=tf.int32, 
  initializer=tf.zeros_initializer)
  
#Other initializer 
tf.random_normal_initializer
tf.truncated_normal_initializer
tf.random_uniform_initializer
tf.uniform_unit_scaling_initializer
tf.zeros_initializer
tf.ones_initializer
tf.orthogonal_initializer


#OR initialize a tf.Variable to have the value of a tf.Tensor
#do not specify the variable's shape,
other_variable = tf.get_variable("other_variable", dtype=tf.int32, 
  initializer=tf.constant([23, 42]))





##Variable collections
#it is sometimes useful to have a single way to access all of them

#By default every tf.Variable gets placed in the following two collections: 
* tf.GraphKeys.GLOBAL_VARIABLES --- variables that can be shared across multiple devices, 
* tf.GraphKeys.TRAINABLE_VARIABLES--- variables for which TensorFlow will calculate gradients.

#Or manually , add it to the tf.GraphKeys.LOCAL_VARIABLES

my_local = tf.get_variable("my_local", shape=(), collections=[tf.GraphKeys.LOCAL_VARIABLES])


#OR specify trainable=False as an argument to tf.get_variable:
my_non_trainable = tf.get_variable("my_non_trainable", 
                                   shape=(), 
                                   trainable=False)


#You can also use your own collections. 
#Any string is a valid collection name, 
#and there is no need to explicitly create a collection. 

#To add a variable (or any other object) to a collection after creating the variable
tf.add_to_collection("my_collection_name", my_local)


#to retrieve a list of all the variables (or other objects)
tf.get_collection("my_collection_name")


##Device placement
#Just like any other TensorFlow operation, 
#you can place variables on particular devices. 

#For example, to create a variable named v and places it on the second GPU device:
with tf.device("/gpu:1"):
  v = tf.get_variable("v", [1])


#It is particularly important for variables to be in the correct device in distributed settings. 
#For this reason , use tf.train.replica_device_setter, 
#which can automatically place variables in parameter servers

cluster_spec = {
    "ps": ["ps0:2222", "ps1:2222"],
    "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
  v = tf.get_variable("v", shape=[20, 20])  # this variable is placed 
                                            # in the parameter server
                                            # by the replica_device_setter


##Initializing variables
#Before you can use a variable, it must be initialized
#Most high-level frameworks such as tf.contrib.slim, tf.estimator.Estimator and Keras automatically initialize variables 

#Or for Explicit initialization 
#call tf.global_variables_initializer() for initializing all variables in the tf.GraphKeys.GLOBAL_VARIABLES collection. 
#Running this operation again  re-initializes all variables

session.run(tf.global_variables_initializer())
# Now all variables are initialized.

#OR for individual variable initialization 
session.run(my_variable.initializer)


#To know  which variables have still not been initialized
print(session.run(tf.report_uninitialized_variables()))


#by default tf.global_variables_initializer does not specify the order in which variables are initialized
#if you need order, use following 
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
w = tf.get_variable("w", initializer=v.initialized_value() + 1)


##Using variables
#simply treat it like a normal tf.Tensor:

v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
w = v + 1  # w is a tf.Tensor which is computed based on the value of v.
           # Any time a variable is used in an expression it gets automatically
           # converted to a tf.Tensor representing its value.


#To assign a value to a variable, 
#use the methods assign, assign_add, etc in tf.Variable class

v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
assignment = v.assign_add(1)
tf.global_variables_initializer().run()
assignment.run()


##To force a re-read of the value of a variable 
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
assignment = v.assign_add(1)
with tf.control_dependencies([assignment]):
  w = v.read_value()  # w is guaranteed to reflect v's value after the
                      # assign_add operation.


##Sharing variables
#TensorFlow supports two ways of sharing variables:
    Explicitly passing tf.Variable objects around.
    Implicitly wrapping tf.Variable objects within tf.variable_scope objects.


#Variable scopes allow you to control variable reuse 
#when calling functions which implicitly create and use variables. 


#For example, a function to create a convolutional / relu layer:
def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)


#In a real model, we want many such convolutional layers, 
#and calling this function repeatedly would not work:
#WRONG 
input1 = tf.random_normal([1,10,10,32])
input2 = tf.random_normal([1,20,20,32])
x = conv_relu(input1, kernel_shape=[5, 5, 1, 32], bias_shape=[32])
x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape = [32])  # This fails.

#Since the desired behavior is unclear 
#(create new variables or reuse the existing ones?) TensorFlow will fail. 

#Calling conv_relu in different scopes, clarifies that we want to create new variables:
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 1, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])


#If you do want the variables to be shared, 
#you have two options. 
#First, you can create a scope with the same name using reuse=True:

with tf.variable_scope("model"):
  output1 = my_image_filter(input1)
with tf.variable_scope("model", reuse=True):
  output2 = my_image_filter(input2)


#OR call scope.reuse_variables() to trigger a reuse:

with tf.variable_scope("model") as scope:
  output1 = my_image_filter(input1)
  scope.reuse_variables()
  output2 = my_image_filter(input2)

#it's also possible to initialize a variable scope based on another one
#Without using exact name 
with tf.variable_scope("model") as scope:
  output1 = my_image_filter(input1)
with tf.variable_scope(scope, reuse=True):
  output2 = my_image_filter(input2)




##Reference - Variables
tf.Variable
    # Create a variable.
    w = tf.Variable(<initial-value>, name=<optional-name>)

    # Use the variable in the graph like any Tensor.
    y = tf.matmul(w, ...another variable or tensor...)

    # The overloaded operators are available too.
    z = tf.sigmoid(w + y)

    # Assign a new value to the variable with `assign()` or a related method.
    w.assign(w + 1.0)
    w.assign_add(1.0)
    # Launch the graph in a session.
    with tf.Session() as sess:
        # Run the variable initializer.
        sess.run(w.initializer)
        # ...you now can run ops that use the value of 'w'...
        w.op.run()
        y.op.run()
        z.op.run()
        
    ##OR 
    # Add an Op to initialize global variables.
    init_op = tf.global_variables_initializer()

    # Launch the graph in a session.
    with tf.Session() as sess:
        # Run the Op that initializes global variables.
        sess.run(init_op)
        # ...you can now run any Op that uses variable values...
        w.op.run()  #shortcut for calling tf.get_default_session().run(op)
        y.op.run()
        z.op.run()
        
    #Attributes 
    device
        The device of this variable.
    dtype
        The DType of this variable.
    graph
        The Graph of this variable.
    initial_value
        Returns the Tensor used as the initial value for the variable.
    initializer
        The initializer operation for this variable.
    name
        The name of this variable.
    op
        The Operation of this variable.
        Use .op.run() to execute the variable 
        which is a shortcut for calling tf.get_default_session().run(op)
    shape
        The TensorShape of this variable.
    __init__(initial_value=None, trainable=True, collections=None,
                validate_shape=True, caching_device=None,
                name=None, variable_def=None,  dtype=None, 
                expected_shape=None,   import_scope=None)
        Creates a new variable with value initial_value.
        The new variable is added to the graph collections listed in collections, which defaults to [GraphKeys.GLOBAL_VARIABLES].
        If trainable is True the variable is also added to the graph collection GraphKeys.TRAINABLE_VARIABLES.
    __abs__( a, *args)
    __add__( a, *args)
        Returns x + y element-wise.
    __and__( a, *args)
        Returns the truth value of x AND y element-wise.
    __div__( a, *args)
    __floordiv__( a, *args)
        Divides x / y elementwise, rounding toward the most negative integer.
    __ge__( a, *args)
        Returns the truth value of (x >= y) element-wise.
    __getitem__( var,slice_spec)
        Creates a slice helper object given a variable.
        import tensorflow as tf
        A = tf.Variable([[1,2,3], [4,5,6], [7,8,9]], dtype=tf.float32)
        with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          print(sess.run(A[:2, :2]))  # => [[1,2], [4,5]]
          op = A[:2,:2].assign(22. * tf.ones((2, 2)))
          print(sess.run(op))  # => [[22, 22, 3], [22, 22, 6], [7,8,9]]
    __gt__( a, *args)
        Returns the truth value of (x > y) element-wise.
    __invert__( a, *args)
        Returns the truth value of NOT x element-wise.
    __iter__()
        Dummy method to prevent iteration. Do not call.
    __le__( a,*args)
        Returns the truth value of (x <= y) element-wise.
    __lt__( a, *args)
        Returns the truth value of (x < y) element-wise.
    __matmul__( a,*args)
        Multiplies matrix a by matrix b, producing a * b.
    __mod__( a, *args)
        Returns element-wise remainder of division. 
    __mul__( a, *args)
        Dispatches cwise mul for "DenseDense" and "DenseSparse".
    __neg__( a, *args)
        Computes numerical negative value element-wise.
    __or__( a, *args)
        Returns the truth value of x OR y element-wise.
    __pow__( a, *args)
        Computes the power of one value to another.
        # tensor 'x' is [[2, 2], [3, 3]]
        # tensor 'y' is [[8, 16], [2, 3]]
        tf.pow(x, y) ==> [[256, 65536], [9, 27]]
    __sub__( a, *args)
        Returns x - y element-wise.
    __truediv__( a, *args)
    __xor__( a, *args)
    assign( value, use_locking=False)
        Assigns a new value to the variable.
    assign_add( delta, use_locking=False)
        Adds a value to this variable.
    assign_sub( delta, use_locking=False)
        Subtracts a value from this variable.
    count_up_to(limit)
        Increments this variable until it reaches limit.
        Returns:A Tensor that will hold the variable value before the increment. 
    eval(session=None)
        In a session, computes and returns the value of this variable.
        v = tf.Variable([1, 2])
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            # Usage passing the session explicitly.
            print(v.eval(sess))
            # Usage with the default session.  The 'with' block
            # above makes 'sess' the default session.
            print(v.eval())
    from_proto(  variable_def,  import_scope=None)
        Returns a Variable object created from variable_def.
    get_shape()
        Alias of Variable.shape.
    initialized_value()
        Returns the value of the initialized variable.
        Use this instead of the variable itself to initialize another variable 
        with a value that depends on the value of this variable.
        # Initialize 'v' with a random tensor.
        v = tf.Variable(tf.truncated_normal([10, 40]))
        # Use `initialized_value` to guarantee that `v` has been
        # initialized before its value is used to initialize `w`.
        # The random values are picked only once.
        w = tf.Variable(v.initialized_value() * 2.0)
    load( value, session=None)
        Load new value into this variable
        v = tf.Variable([1, 2])
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            # Usage passing the session explicitly.
            v.load([2, 3], sess)
            print(v.eval(sess)) # prints [2 3]
            # Usage with the default session.  The 'with' block
            # above makes 'sess' the default session.
            v.load([3, 4], sess)
            print(v.eval()) # prints [3 4]
    read_value()
        Returns the value of this variable, read in the current context.
    scatter_sub(  sparse_delta,  use_locking=False)
        sparse_delta: IndexedSlices to be subtracted from this variable.
        use_locking: If True, use locking during the operation.
        Returns:A Tensor that will hold the new value of this variable 
        after the scattered subtraction has completed.
    set_shape(shape)
        Overrides the shape for this variable.
    to_proto(export_scope=None)
        Converts a Variable to a VariableDef protocol buffer.
    value()
        Returns the last snapshot of this variable.
        

        
##Variable helper functions
tf.global_variables()
    Returns global variables.
tf.local_variables()
tf.model_variables
tf.trainable_variables
tf.moving_average_variables
tf.global_variables_initializer
tf.local_variables_initializer
tf.variables_initializer
tf.is_variable_initialized
tf.report_uninitialized_variables
tf.assert_variables_initialized
tf.assign(ref,value,validate_shape=None,use_locking=None,name=None)
tf.assign_add(ref,value,use_locking=None,name=None)
    Update 'ref' by adding 'value' to it.
tf.assign_sub(ref,value,use_locking=None,name=None)

##Saving and Restoring Variables
tf.train.Saver
tf.train.latest_checkpoint
tf.train.get_checkpoint_state
tf.train.update_checkpoint_state

##Sharing Variables
#to create variables 
tf.get_variable(name,shape=None,dtype=None,initializer=None,regularizer=None,trainable=True,
    collections=None,caching_device=None,partitioner=None,validate_shape=True,use_resource=None,custom_getter=None)
tf.get_local_variable
tf.VariableScope
tf.variable_scope
tf.variable_op_scope
tf.get_variable_scope
tf.make_template
tf.no_regularizer
tf.constant_initializer
tf.random_normal_initializer
tf.truncated_normal_initializer
tf.random_uniform_initializer
tf.uniform_unit_scaling_initializer
tf.zeros_initializer
tf.ones_initializer
tf.orthogonal_initializer

##Variable Partitioners for Sharding
tf.fixed_size_partitioner
tf.variable_axis_size_partitioner
tf.min_max_variable_partitioner




###Tensorflow - Opertaions 
class tf.Operation
        Represents a graph node that performs computation on tensors.

        An Operation is a node in a TensorFlow Graph that takes zero or more Tensor objects 
        as input, and produces zero or more Tensor objects as output. 
        Objects of type Operation are created by calling a Python op constructor 
        (such as tf.matmul) or tf.Graph.create_op.

        For example c = tf.matmul(a, b) creates an Operation of type "MatMul" 
        that takes tensors a and b as input, and produces c as output.

        After the graph has been launched in a session, 
        an Operation can be executed by passing it to tf.Session.run. 
        op.run() is a shortcut for calling tf.get_default_session().run(op)
        (set default session at first eg 'with sess.asdefault():')
    control_inputs
        The Operation objects on which this op has a control dependency.
        Returns:A list of Operation objects.
    device
        The name of the device to which this op has been assigned, if any.
        Returns:The string name of the device to which this op has been assigned, or an empty string if it has not been assigned to a device.
    graph
        The Graph that contains this operation.
    inputs
        The list of Tensor objects representing the data inputs of this op.
    name
        The full name of this operation.
    node_def
        Returns a serialized NodeDef representation of this operation.
    op_def
        Returns the OpDef proto that represents the type of this op.
    outputs
        The list of Tensor objects representing the outputs of this op.
    traceback
        Returns the call stack from when this operation was constructed.
    traceback_with_start_lines
        Same as traceback but includes start line of function definition.
        Returns:A list of 5-tuples (filename, lineno, name, code, func_start_lineno).
    type
        The type of the op (e.g. "MatMul").
    __init__( node_def,  g, inputs=None,  output_types=None,  control_inputs=None, input_types=None,
            original_op=None, op_def=None)
        Creates an Operation.
    colocation_groups()
        Returns the list of colocation groups of the op.
    get_attr(name)
        Returns the value of the attr of this op with the given name.
    run( feed_dict=None, session=None)
        Runs this operation in a Session.
        Calling this method will execute all preceding operations 
        that produce the inputs needed for this operation.
         After the graph has been launched in a session, 
        an Operation can be executed by passing it to tf.Session.run. 
        op.run() is a shortcut for calling tf.get_default_session().run(op)
        (set default session at first eg 'with sess.asdefault():')
    values()
        DEPRECATED: Use outputs.

        
##Constant Value Tensors
tf.zeros(shape,dtype=tf.float32,name=None)
tf.zeros_like
tf.ones
tf.ones_like
tf.fill
tf.constant(value,dtype=None,shape=None,name='Const',verify_shape=False)

##Sequences
tf.linspace/lin_space( start,stop, num, name=None)
    tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0 11.0 12.0]
tf.range(limit, delta=1, dtype=None, name='range')
tf.range(start, limit, delta=1, dtype=None, name='range')


##Random Tensors
tf.random_normal
tf.truncated_normal
tf.random_uniform
tf.random_shuffle
tf.random_crop
tf.multinomial
tf.random_gamma
tf.set_random_seed

#Example 
# Create a tensor of shape [2, 3] consisting of random normal values, with mean
# -1 and standard deviation 4.
norm = tf.random_normal([2, 3], mean=-1, stddev=4)

# Shuffle the first dimension of a tensor
c = tf.constant([[1, 2], [3, 4], [5, 6]])
shuff = tf.random_shuffle(c)

# Each time we run these ops, different results are generated
sess = tf.Session()
print(sess.run(norm))
print(sess.run(norm))

# Set an op-level seed to generate repeatable sequences across sessions.
norm = tf.random_normal([2, 3], seed=1234)
sess = tf.Session()
print(sess.run(norm))
print(sess.run(norm))
sess = tf.Session()
print(sess.run(norm))
print(sess.run(norm))


#Another common use of random values is the initialization of variables
# Use random uniform values in [0, 1) as the initializer for a variable of shape
# [2, 3]. The default type is float32.
var = tf.Variable(tf.random_uniform([2, 3]), name="var")
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
print(sess.run(var))



##Arithmetic Operators of two Tensors
tf.add(x,y,name=None)
tf.subtract
tf.multiply
tf.scalar_mul
tf.div
tf.divide
tf.truediv
tf.floordiv
tf.realdiv
tf.truncatediv
tf.floor_div
tf.truncatemod
tf.floormod
tf.mod
tf.cross

##Basic Math Functions
tf.add_n
tf.abs
tf.negative
tf.sign
tf.reciprocal
tf.square( x, name=None)
    Returns the square element-wise
tf.round
tf.sqrt
tf.rsqrt
tf.pow
tf.exp
tf.expm1
tf.log
tf.log1p
tf.ceil
tf.floor
tf.maximum
tf.minimum
tf.cos
tf.sin
tf.lbeta
tf.tan
tf.acos
tf.asin
tf.atan
tf.cosh
tf.sinh
tf.asinh
tf.acosh
tf.atanh
tf.lgamma
tf.digamma
tf.erf
tf.erfc
tf.squared_difference
tf.igamma
tf.igammac
tf.zeta
tf.polygamma
tf.betainc
tf.rint

##Matrix Math Functions
tf.diag
tf.diag_part
tf.trace
tf.transpose
tf.eye
tf.matrix_diag
tf.matrix_diag_part
tf.matrix_band_part
tf.matrix_set_diag
tf.matrix_transpose
tf.matmul
tf.norm
tf.matrix_determinant
tf.matrix_inverse
tf.cholesky
tf.cholesky_solve
tf.matrix_solve(matrix, rhs, adjoint=None,name=None)
        matrix: A Tensor. Must be one of the following types: 
                float64, float32, complex64, complex128. 
                Shape is [..., M, M].
        rhs: A Tensor. Must have the same type as matrix. 
             Shape is [..., M, K]
        Returns:A Tensor. Has the same type as matrix. Shape is [..., M, K].
tf.matrix_triangular_solve
tf.matrix_solve_ls
tf.qr
tf.self_adjoint_eig
tf.self_adjoint_eigvals
tf.svd

##Tensor Math Function
tf.tensordot

##Complex Number Functions
tf.complex
tf.conj
tf.imag
tf.real

##Reduction
tf.reduce_sum(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)
    If axis has no entries, all dimensions are reduced, and a tensor with a single element is returned.
    axis = 0, means row changing/column order 
    # 'x' is [[1, 1, 1]
    #         [1, 1, 1]]
    tf.reduce_sum(x) ==> 6
    tf.reduce_sum(x, 0) ==> [2, 2, 2]
    tf.reduce_sum(x, 1) ==> [3, 3]
    tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
    tf.reduce_sum(x, [0, 1]) ==> 6
tf.reduce_prod
tf.reduce_min
tf.reduce_max
tf.reduce_mean
tf.reduce_all
    # 'x' is [[True,  True]
    #         [False, False]]
    tf.reduce_all(x) ==> False
    tf.reduce_all(x, 0) ==> [False, False]
    tf.reduce_all(x, 1) ==> [True, False]
tf.reduce_any
tf.reduce_logsumexp
tf.count_nonzero
tf.accumulate_n
tf.einsum

##Scan
#perform scans (running totals) across one axis of a tensor.
tf.cumsum
    tf.cumsum([a, b, c])  # => [a, a + b, a + b + c]
    tf.cumsum([a, b, c], reverse=True)  # => [a + b + c, b + c, c]
tf.cumprod

##Segmentation
#perform common math computations on tensor segments. 
#a segmentation is a partitioning of a tensor along the first dimension, 
#i.e. it defines a mapping from the first dimension onto segment_ids. 
#The segment_ids tensor should be the size of the first dimension, d0, 
#with consecutive IDs in the range 0 to k, where k<d0. 
#In particular, a segmentation of a matrix tensor is a mapping of rows to segments.


tf.segment_sum(data, segment_ids, name=None)
    Computes the sum along segments of a tensor.
    c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
    tf.segment_sum(c, tf.constant([0, 0, 1]))
        ==>  [[0 0 0 0]
              [5 6 7 8]]
tf.segment_prod
tf.segment_min
tf.segment_max
tf.segment_mean
tf.unsorted_segment_sum
tf.sparse_segment_sum
tf.sparse_segment_mean
tf.sparse_segment_sqrt_n

#Comparison and Indexing
#to add sequence comparison and index extraction to your graph. 
#You can use these operations to determine sequence differences and determine the indexes of specific values in a tensor.
tf.argmin
tf.argmax(input, axis=None,name=None,)
    Returns the index with the largest value across axes of a tensor.
    pred = np.array([[31, 23,  4, 24, 27, 34],
                [18,  3, 25,  0,  6, 35],
                [28, 14, 33, 22, 20,  8],
                [13, 30, 21, 19,  7,  9],
                [16,  1, 26, 32,  2, 29],
                [17, 12,  5, 11, 10, 15]])
     tf.argmax(pred, 1) #=1 means - column changng, #([5, 5, 2, 1, 3, 0])           
tf.setdiff1d
tf.where(condition,x=None,y=None,name=None)
    Return the elements, either from x or y, depending on the condition
    x and y must have the same shape.
    condition must have same shape as x 
    condition = np.random.normal(0, 1, (7,11)) < 0
    x = np.zeros((7, 11))
    y = np.ones((7, 11))
    tf.where(condition, x, y)
tf.edit_distance
tf.invert_permutation

##Casting
#to cast tensor data types in your graph.
tf.string_to_number(string_tensor,out_type=None,name=None)
tf.to_double
tf.to_float
tf.to_bfloat16
tf.to_int32
tf.to_int64
tf.cast
tf.bitcast
tf.saturate_cast

##Shapes and Shaping
tf.broadcast_dynamic_shape
tf.broadcast_static_shape
tf.shape
tf.shape_n
tf.size
tf.rank
tf.squeeze
tf.expand_dims
tf.meshgrid
tf.reshape(tensor, shape, name=None)
    Reshapes a tensor.
    # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # tensor 't' has shape [9]
    reshape(t, [3, 3]) ==> [[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]]

    # tensor 't' is [[[1, 1], [2, 2]],
    #                [[3, 3], [4, 4]]]
    # tensor 't' has shape [2, 2, 2]
    reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
                            [3, 3, 4, 4]]
    # tensor 't' is [[[1, 1, 1],
    #                 [2, 2, 2]],
    #                [[3, 3, 3],
    #                 [4, 4, 4]],
    #                [[5, 5, 5],
    #                 [6, 6, 6]]]
    # tensor 't' has shape [3, 2, 3]
    # pass '[-1]' to flatten 't'
    reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]

    # -1 can also be used to infer the shape
    # -1 is inferred to be 9:
    reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                             [4, 4, 4, 5, 5, 5, 6, 6, 6]]
    # -1 is inferred to be 2:
    reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                             [4, 4, 4, 5, 5, 5, 6, 6, 6]]
    # -1 is inferred to be 3:
    reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
                                  [2, 2, 2],
                                  [3, 3, 3]],
                                 [[4, 4, 4],
                                  [5, 5, 5],
                                  [6, 6, 6]]]
    # tensor 't' is [7]
    # shape `[]` reshapes to a scalar
    reshape(t, []) ==> 7




##Slicing and Joining
tf.slice(input_, begin, size, name=None)
    # 'input' is [[[1, 1, 1], [2, 2, 2]],
    #             [[3, 3, 3], [4, 4, 4]],
    #             [[5, 5, 5], [6, 6, 6]]]
    tf.slice(input, [1, 0, 0], [1, 1, 3]) ==> [[[3, 3, 3]]]
    tf.slice(input, [1, 0, 0], [1, 2, 3]) ==> [[[3, 3, 3],[4, 4, 4]]]
    tf.slice(input, [1, 0, 0], [2, 1, 3]) ==> [[[3, 3, 3]],[[5, 5, 5]]]
tf.strided_slice
tf.split(value, num_or_size_splits, axis=0, num=None, name='split')
    Splits a tensor into sub tensors
    # 'value' is a tensor with shape [5, 30]
    # Split 'value' into 3 tensors with sizes [4, 15, 11] along dimension 1
    split0, split1, split2 = tf.split(value, [4, 15, 11], 1)
    tf.shape(split0) ==> [5, 4]
    tf.shape(split1) ==> [5, 15]
    tf.shape(split2) ==> [5, 11]
    # Split 'value' into 3 tensors along dimension 1
    split0, split1, split2 = tf.split(value, num_or_size_splits=3, axis=1)
    tf.shape(split0) ==> [5, 10]    
tf.tile
tf.pad
tf.concat
tf.stack(values,  axis=0, name='stack')
    Stacks a list of rank-R tensors into one rank-(R+1) tensor.
    # 'x' is [1, 4]
    # 'y' is [2, 5]
    # 'z' is [3, 6]
    stack([x, y, z])  # => [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
    stack([x, y, z], axis=1)  # => [[1, 2, 3], [4, 5, 6]]
tf.parallel_stack
tf.unstack
tf.reverse_sequence
tf.reverse
tf.reverse_v2
tf.transpose
tf.extract_image_patches
tf.space_to_batch_nd
tf.space_to_batch
tf.required_space_to_batch_paddings
tf.batch_to_space_nd
tf.batch_to_space
tf.space_to_depth
tf.depth_to_space
tf.gather
tf.gather_nd
tf.unique_with_counts
tf.scatter_nd
tf.dynamic_partition
tf.dynamic_stitch
tf.boolean_mask
tf.one_hot
tf.sequence_mask
tf.dequantize
tf.quantize_v2
tf.quantized_concat
tf.setdiff1d

##Fake quantization
#Operations used to help train for better quantization accuracy.
tf.fake_quant_with_min_max_args
tf.fake_quant_with_min_max_args_gradient
tf.fake_quant_with_min_max_vars
tf.fake_quant_with_min_max_vars_gradient
tf.fake_quant_with_min_max_vars_per_channel
tf.fake_quant_with_min_max_vars_per_channel_gradient

##Control Flow Operations
#to control the execution of operations 
#and add conditional dependencies to your graph.
tf.identity
tf.tuple
tf.group
tf.no_op
tf.count_up_to(ref,limit,name=None)
tf.cond(pred, true_fn=None, false_fn=None, strict=False, name=None, fn1=None, fn2=None)
    Return true_fn() if the predicate pred is true else false_fn()
      x = tf.constant(2)
      y = tf.constant(5)
      def f1(): return tf.multiply(x, 17)
      def f2(): return tf.add(y, 23)
      r = tf.cond(tf.less(x, y), f1, f2)
      # r is set to f1().
      # Operations in f2 (e.g., tf.add) are not executed.

tf.case(pred_fn_pairs,default,exclusive=False,strict=False,name='case')
    Pseudocode:
          if (x < y) return 17;
          else return 23;
    Expressions:
          f1 = lambda: tf.constant(17)
          f2 = lambda: tf.constant(23)
          r = case([(tf.less(x, y), f1)], default=f2)

    Example 2: Pseudocode:
          if (x < y && x > z) raise OpError("Only one predicate may evaluate true");
          if (x < y) return 17;
          else if (x > z) return 23;
          else return -1;
    Expressions:
          def f1(): return tf.constant(17)
          def f2(): return tf.constant(23)
          def f3(): return tf.constant(-1)
          r = case({tf.less(x, y): f1, tf.greater(x, z): f2},
                   default=f3, exclusive=True)


tf.while_loop(cond,body,loop_vars,shape_invariants=None,parallel_iterations=10,back_prop=True,swap_memory=False,name=None)
    Repeat body while the condition cond is true.
    i = tf.constant(0)
    c = lambda i: tf.less(i, 10)
    b = lambda i: tf.add(i, 1)
    r = tf.while_loop(c, b, [i])

    Example with nesting and a namedtuple:
    import collections
    Pair = collections.namedtuple('Pair', 'j, k')
    ijk_0 = (tf.constant(0), Pair(tf.constant(1), tf.constant(2)))
    c = lambda i, p: i < 10
    b = lambda i, p: (i + 1, Pair((p.j + p.k), (p.j - p.k)))
    ijk_final = tf.while_loop(c, b, ijk_0)

    Example using shape_invariants:
    i0 = tf.constant(0)
    m0 = tf.ones([2, 2])
    c = lambda i, m: i < 10
    b = lambda i, m: [i+1, tf.concat([m, m], axis=0)]
    tf.while_loop(
        c, b, loop_vars=[i0, m0],
        shape_invariants=[i0.get_shape(), tf.TensorShape([None, 2])])

    
##Logical Operators
#to add logical operators to your graph.
tf.logical_and(x,y,name=None)
    Returns the truth value of x AND y element-wise.
tf.logical_not
tf.logical_or
tf.logical_xor

#Comparison Operators
#to add comparison operators to your graph.
tf.equal(x,y,name=None)
    Returns the truth value of (x == y) element-wise
tf.not_equal
tf.less
tf.less_equal
tf.greater
tf.greater_equal
tf.where

##Debugging Operations
#to validate values and debug your graph.
tf.is_finite
tf.is_inf
tf.is_nan
tf.verify_tensor_all_finite
tf.check_numerics
tf.add_check_numerics_ops
tf.Assert
tf.Print(input_, data, message=None, first_n=None, summarize=None, name=None)


##Higher Order Operators
tf.map_fn(fn,elems,dtype=None,
    parallel_iterations=10, back_prop=True, swap_memory=False, infer_shape=True,name=None)
    elems = np.array([1, 2, 3, 4, 5, 6])
    squares = map_fn(lambda x: x * x, elems)
    # squares == [1, 4, 9, 16, 25, 36]

    elems = (np.array([1, 2, 3]), np.array([-1, 1, -1]))
    alternate = map_fn(lambda x: x[0] * x[1], elems, dtype=tf.int64)
    # alternate == [-1, 2, -3]

    elems = np.array([1, 2, 3])
    alternates = map_fn(lambda x: (x, -x), elems, dtype=(tf.int64, tf.int64))
    # alternates[0] == [1, 2, 3]
    # alternates[1] == [-1, -2, -3]
tf.foldl(fn, elems, initializer=None,parallel_iterations=10, back_prop=True, swap_memory=False,  name=None)
tf.foldr(fn, elems, initializer=None,parallel_iterations=10, back_prop=True, swap_memory=False,  name=None)
    elems = [1, 2, 3, 4, 5, 6]
    sum = foldr(lambda a, x: a + x, elems)
    # sum == 21
tf.scan(fn, elems, initializer=None,parallel_iterations=10, back_prop=True, swap_memory=False,  name=None)
    scan on the list of tensors unpacked from elems on dimension 0.
    elems = np.array([1, 2, 3, 4, 5, 6])
    sum = scan(lambda a, x: a + x, elems)
    # sum == [1, 3, 6, 10, 15, 21]

    elems = np.array([1, 2, 3, 4, 5, 6])
    initializer = np.array(0)
    sum_one = scan(
        lambda a, x: x[0] - x[1] + a, (elems + 1, elems), initializer)
    # sum_one == [1, 2, 3, 4, 5, 6]

    elems = np.array([1, 0, 0, 0, 0, 0])
    initializer = (np.array(0), np.array(1))
    fibonaccis = scan(lambda a, _: (a[1], a[0] + a[1]), elems, initializer)
    # fibonaccis == ([1, 1, 2, 3, 5, 8], [1, 2, 3, 5, 8, 13])

##Hashing
#String hashing ops take a string input tensor and map each element to an integer.
tf.string_to_hash_bucket_fast
tf.string_to_hash_bucket_strong
tf.string_to_hash_bucket

##Joining
tf.reduce_join(inputs,axis=None,keep_dims=False,separator='',name=None,reduction_indices=None)
    # tensor `a` is [["a", "b"], ["c", "d"]]
    tf.reduce_join(a, 0) ==> ["ac", "bd"]
    tf.reduce_join(a, 1) ==> ["ab", "cd"]
    tf.reduce_join(a, -2) = tf.reduce_join(a, 0) ==> ["ac", "bd"]
    tf.reduce_join(a, -1) = tf.reduce_join(a, 1) ==> ["ab", "cd"]
    tf.reduce_join(a, 0, keep_dims=True) ==> [["ac", "bd"]]
    tf.reduce_join(a, 1, keep_dims=True) ==> [["ab"], ["cd"]]
    tf.reduce_join(a, 0, separator=".") ==> ["a.c", "b.d"]
    tf.reduce_join(a, [0, 1]) ==> ["acbd"]
    tf.reduce_join(a, [1, 0]) ==> ["abcd"]
    tf.reduce_join(a, []) ==> ["abcd"]
tf.string_join(inputs, separator=None, name=None)
    Joins the strings in the given list of string tensors into one tensor;
    
    
##Splitting
tf.string_split(source, delimiter=' ')
    Split elements of source based on delimiter into a SparseTensor
tf.substr(input, pos, len, name=None)
    Return substrings from Tensor of strings
    
##Conversion
tf.as_string(input,precision=None,scientific=None,shortest=None,width=None,
    fill=None,name=None)
    Converts each entry in the given tensor to strings. Supports many numeric types and boolean.
    input: A Tensor. Must be one of the following types: int32, int64, complex64, float32, float64, bool, int8.
tf.encode_base64
tf.decode_base64


###Tensorflow- SparseTensor
#which is instantiated with three arguments:
dense_shape
    The shape of the tensor. 
    Takes a list indicating the number of elements in each dimension. 
    For example, dense_shape=[3,6] specifies a two-dimensional 3x6 tensor, 
    dense_shape=[2,3,4] specifies a three-dimensional 2x3x4 tensor, 
    and dense_shape=[9] specifies a one-dimensional tensor with 9 elements.
indices
    The indices of the elements in the tensor that contain nonzero values. 
    Takes a list of terms, where each term is itself a list containing the index of a nonzero element. 
    (Elements are zero-indexed—i.e., [0,0] is the index value for the element in the first column of the first row in a two-dimensional tensor.) 
    For example, indices=[[1,3], [2,4]] specifies that the elements 
    with indexes of [1,3] and [2,4] have nonzero values.
values
    A one-dimensional tensor of values. 
    Term i in values corresponds to term i in indices and specifies its value. 
    For example, given indices=[[1,3], [2,4]], 
    the parameter values=[18, 3.6] specifies that element [1,3] of the tensor has a value of 18, and element [2,4] of the tensor has a value of 3.6.

#Note tf.SparseTensor can be used with / , // , *, .eval()
#eg : * - Component-wise multiplies a SparseTensor by a dense Tensor.
##tf.SparseTensor - Conversion
    tf.sparse_to_dense(sparse_indices,output_shape,sparse_values, default_value=0, validate_indices=True, name=None)
    tf.sparse_tensor_to_dense(sp_input, default_value=0,validate_indices=True,name=None)
    tf.sparse_to_indicator( sp_input, vocab_size, name=None )
    tf.sparse_merge( sp_ids, sp_values, vocab_size, name=None, already_sorted=False ) 

##tf.SparseTensor - Manipulation
    tf.sparse_concat( axis, sp_inputs, name=None, expand_nonconcat_dim=False, concat_dim=None ) 
    tf.sparse_fill_empty_rows( sp_input, default_value, name=None ) 
    tf.sparse_mask( a, mask_indices, name=None ) 
    tf.sparse_placeholder( dtype, shape=None, name=None ) 
    tf.sparse_reorder( sp_input, name=None ) 
    tf.sparse_reset_shape( sp_input, new_shape=None ) 
    tf.sparse_reshape( sp_input, shape, name=None ) 
    tf.sparse_retain( sp_input, to_retain ) 
    tf.sparse_slice( sp_input, start, size, name=None )  
    tf.sparse_split( keyword_required=KeywordRequired(), sp_input=None, num_split=None, axis=None, name=None, split_dim=None ) 
    tf.sparse_transpose( sp_input, perm=None, name=None ) 
    tf.sparse_segment_mean( data, indices, segment_ids, name=None ) 
    tf.sparse_segment_sqrt_n( data, indices, segment_ids, name=None ) 
    tf.sparse_segment_sum( data, indices, segment_ids, name=None ) 
        c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
        # Select two rows, one segment.
        tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
        # => [[0 0 0 0]]
        # Select two rows, two segment.
        tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
        # => [[ 1  2  3  4]
        #     [-1 -2 -3 -4]]
        # Select all rows, two segments.
        tf.sparse_segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
        # => [[0 0 0 0]
        #     [5 6 7 8]]
        # Which is equivalent to:
        tf.segment_sum(c, tf.constant([0, 0, 1]))
        
##tf.SparseTensor - Reduction
    tf.sparse_reduce_max( sp_input, axis=None, keep_dims=False, reduction_axes=None )
        # 'x' represents [[1, ?, 2]
        #                 [?, 3, ?]]
        # where ? is implicitly-zero.
        tf.sparse_reduce_max(x) ==> 3
        tf.sparse_reduce_max(x, 0) ==> [1, 3, 2]
        tf.sparse_reduce_max(x, 1) ==> [2, 3]  # Can also use -1 as the axis.
        tf.sparse_reduce_max(x, 1, keep_dims=True) ==> [[2], [3]]
        tf.sparse_reduce_max(x, [0, 1]) ==> 3   
    tf.sparse_reduce_max_sparse( sp_input, axis=None, keep_dims=False, reduction_axes=None ) 
    tf.sparse_reduce_sum( sp_input, axis=None, keep_dims=False, reduction_axes=None ) 
        # 'x' represents [[1, ?, 1]
        #                 [?, 1, ?]]
        # where ? is implicitly-zero.
        tf.sparse_reduce_sum(x) ==> 3
        tf.sparse_reduce_sum(x, 0) ==> [1, 1, 1]
        tf.sparse_reduce_sum(x, 1) ==> [2, 1]  # Can also use -1 as the axis.
        tf.sparse_reduce_sum(x, 1, keep_dims=True) ==> [[2], [1]]
        tf.sparse_reduce_sum(x, [0, 1]) ==> 3
    tf.sparse_reduce_sum_sparse( sp_input, axis=None, keep_dims=False, reduction_axes=None ) 


##Math Operations
    tf.sparse_add( a, b, thresh=0 ) 
    tf.sparse_softmax( sp_input, name=None )  
        # First batch:
        # [?   e.]
        # [1.  ? ]
        # Second batch:
        # [e   ? ]
        # [e   e ]
        shape = [2, 2, 2]  # 3-D SparseTensor
        values = np.asarray([[[0., np.e], [1., 0.]], [[np.e, 0.], [np.e, np.e]]])
        indices = np.vstack(np.where(values)).astype(np.int64).T

        result = tf.sparse_softmax(tf.SparseTensor(indices, values, shape))
        # ...returning a 3-D SparseTensor, equivalent to:
        # [?   1.]     [1    ?]
        # [1.  ? ] and [.5  .5]
        # where ? means implicitly zero.    
    tf.sparse_tensor_dense_matmul( sp_a, b, adjoint_a=False, adjoint_b=False, name=None ) 
    tf.sparse_matmul( a, b, transpose_a=None, transpose_b=None, a_is_sparse=None, b_is_sparse=None, name=None ) 
    tf.sparse_minimum( sp_a, sp_b, name=None ) 
    tf.sparse_maximum( sp_a, sp_b, name=None ) 
        Returns the element-wise max of two SparseTensors
        sp_zero = sparse_tensor.SparseTensor([[0]], [0], [7])
        sp_one = sparse_tensor.SparseTensor([[1]], [1], [7])
        res = tf.sparse_maximum(sp_zero, sp_one).eval()
        # "res" should be equal to SparseTensor([[0], [1]], [0, 1], [7]).


 
 
 
 
###Tensorflow - Graphs and Sessions - Low level API 
#Higher-level APIs such as tf.estimator.Estimator and Keras hide the details of graphs and sessions 

#Dataflow is a common programming model for parallel computing. 
#In a dataflow graph, the nodes represent units of computation, 
#and the edges represent the data consumed or produced by a computation. 
#
#For example, in a TensorFlow graph, the tf.matmul operation 
#would correspond to a single node with two incoming edges 
#(the matrices to be multiplied) and one outgoing edge (the result of the multiplication).
 

##tf.Graph
#A tf.Graph contains two relevant kinds of information:

#Graph structure. The nodes(tf.Operation) and edges(tf.Tensor) of the graph
#Graph collections. for storing collections of metadata in a tf.Graph. 
        The tf.add_to_collection associates a list of objects with a key 
        (where tf.GraphKeys defines some of the standard keys), 
        and tf.get_collection looks up all objects associated with a key. 

##tf.GraphKeys
    GLOBAL_VARIABLES: the default collection of Variable objects, 
                shared across distributed environment (model variables are subset of these). 
                Commonly, all TRAINABLE_VARIABLES variables will be in MODEL_VARIABLES, 
                and all MODEL_VARIABLES variables will be in GLOBAL_VARIABLES.
    LOCAL_VARIABLES: the subset of Variable objects that are local to each machine. 
                Usually used for temporarily variables, like counters. 
                use tf.contrib.framework.local_variable to add to this collection.
    MODEL_VARIABLES: the subset of Variable objects that are used in the model for inference (feed forward).
                use tf.contrib.framework.model_variable to add to this collection.
    TRAINABLE_VARIABLES: the subset of Variable objects that will be trained by an optimizer. 
    SUMMARIES: the summary Tensor objects that have been created in the graph. S
    QUEUE_RUNNERS: the QueueRunner objects that are used to produce input for a computation. 
    MOVING_AVERAGE_VARIABLES: the subset of Variable objects that will also keep moving averages. 
    REGULARIZATION_LOSSES: regularization losses collected during graph construction.
    WEIGHTS : not populated 
    BIASES  : not populated 
    ACTIVATIONS : not populated 

##Building a tf.Graph -
# tf.Operation (node) and tf.Tensor (edge) objects and add them to a tf.Graph instance

#Example 
Calling tf.constant(42.0) creates a single tf.Operation that produces the value 42.0, 
    adds it to the default graph, 
    and returns a tf.Tensor that represents the value of the constant.

Calling tf.matmul(x, y) creates a single tf.Operation 
    that multiplies the values of tf.Tensor objects x and y, 
    adds it to the default graph, 
    and returns a tf.Tensor that represents the result of the multiplication.

Executing v = tf.Variable(0) adds to the graph a tf.Operation 
    that will store a writeable tensor value 
    that persists between tf.Session.run calls. 

Calling tf.train.Optimizer.minimize will add operations 
    and tensors to the default graph that calculate gradients, 
    and return a tf.Operation that, 
    when run, will apply those gradients to a set of variables.


##Naming operations

#A tf.Graph object defines a namespace for the tf.Operation objects it contains. 
#TensorFlow automatically chooses a unique name for each operation 

#tf.Tensor objects are implicitly named after the tf.Operation 
#that produces the tensor as output. 
#A tensor name has the form "<OP_NAME>:<i>" where:
    "<OP_NAME>" is the name of the operation that produces it.
    "<i>" is an integer representing the index of that tensor among the operation's outputs.



#OR explicitly create the name by using 'name' argument of tf.Operation, tf.Tensor
#For example, tf.constant(42.0, name="answer") creates a new tf.Operation named "answer" 
#and returns a tf.Tensor named "answer:0". 
#If the default graph already contained an operation named "answer", 
#the TensorFlow would append "_1", "_2", and so on to the name

#OR  tf.name_scope function adds a name scope prefix to all operations created in a particular context. 
#The current name scope prefix is a "/"-delimited list of the names of all active tf.name_scope context managers. 
#If a name scope has already been used in the current context, 
#TensorFlow appens "_1", "_2", and so on. 
#The graph visualizer uses name scopes to group operations and reduce the visual complexity of a graph.

c_0 = tf.constant(0, name="c")  # => operation named "c"

# Already-used names will be "uniquified".
c_1 = tf.constant(2, name="c")  # => operation named "c_1"

# Name scopes add a prefix to all operations created in the same context.
with tf.name_scope("outer"):
  c_2 = tf.constant(2, name="c")  # => operation named "outer/c"

  # Name scopes nest like paths in a hierarchical file system.
  with tf.name_scope("inner"):
    c_3 = tf.constant(3, name="c")  # => operation named "outer/inner/c"

  # Exiting a name scope context will return to the previous prefix.
  c_4 = tf.constant(4, name="c")  # => operation named "outer/c_1"

  # Already-used name scopes will be "uniquified".
  with tf.name_scope("inner"):
    c_5 = tf.constant(5, name="c")  # => operation named "outer/inner_1/c"



##Placing operations on different devices - use tf.device
#A device specification has the following form:
/job:<JOB_NAME>/task:<TASK_INDEX>/device:<DEVICE_TYPE>:<DEVICE_INDEX>
    <JOB_NAME> is an alpha-numeric string that does not start with a number.
    <DEVICE_TYPE> is a registered device type (such as GPU or CPU).
    <TASK_INDEX> is a non-negative integer representing the index of the task in the job named <JOB_NAME>. See tf.train.ClusterSpec for an explanation of jobs and tasks.
    <DEVICE_INDEX> is a non-negative integer representing the index of the device, for example, to distinguish between different GPU devices used in the same process.

#Operations created outside either context will run on the "best possible" device. 
#For example, if you have a GPU and a CPU available, and the operation
# has a GPU implementation, TensorFlow will choose the GPU.
weights = tf.random_normal(...)

with tf.device("/device:CPU:0"):
  # Operations created in this context will be pinned to the CPU.
  img = tf.decode_jpeg(tf.read_file("img.jpg"))

with tf.device("/device:GPU:0"):
  # Operations created in this context will be pinned to the GPU.
  result = tf.matmul(weights, img)


#in a distributed configuration, 
#specify the job name and task ID to place variables on a task in the parameter server job ("/job:ps"), 
#and the other operations on task in the worker job ("/job:worker"):


with tf.device("/job:ps/task:0"):
  weights_1 = tf.Variable(tf.truncated_normal([784, 100]))
  biases_1 = tf.Variable(tf.zeroes([100]))

with tf.device("/job:ps/task:1"):
  weights_2 = tf.Variable(tf.truncated_normal([100, 10]))
  biases_2 = tf.Variable(tf.zeroes([10]))

with tf.device("/job:worker"):
  layer_1 = tf.matmul(train_batch, weights_1) + biases_1
  layer_2 = tf.matmul(train_batch, weights_2) + biases_2


##For example, the tf.train.replica_device_setter API can be used 
#with tf.device to place operations for data-parallel distributed training. 

#For example, the following code shows how tf.train.replica_device_setter 
#applies different placement policies to tf.Variable objects and other operations:

with tf.device(tf.train.replica_device_setter(ps_tasks=3)):
  # tf.Variable objects are, by default, placed on tasks in "/job:ps" in a
  # round-robin fashion.
  w_0 = tf.Variable(...)  # placed on "/job:ps/task:0"
  b_0 = tf.Variable(...)  # placed on "/job:ps/task:1"
  w_1 = tf.Variable(...)  # placed on "/job:ps/task:2"
  b_1 = tf.Variable(...)  # placed on "/job:ps/task:0"

  input_data = tf.placeholder(tf.float32)     # placed on "/job:worker"
  layer_0 = tf.matmul(input_data, w_0) + b_0  # placed on "/job:worker"
  layer_1 = tf.matmul(layer_0, w_1) + b_1     # placed on "/job:worker"
  
#Reference
tf.train.replica_device_setter(
    ps_tasks=0,
    ps_device='/job:ps',
    worker_device='/job:worker',
    merge_devices=True,
    cluster=None,
    ps_ops=None,
    ps_strategy=None
)
    Return a device function to use when building a Graph for replicas.
    Device Functions are used in with tf.device(device_function): statement
    to automatically assign devices to Operation objects as they are constructed
        ps_tasks: Number of tasks in the ps job. Ignored if cluster is provided.
        cluster: ClusterDef proto or ClusterSpec.
        ps_ops: List of strings representing Operation types that need to be placed on ps devices. If None, defaults to ["Variable"].
    # To build a cluster with two ps jobs on hosts ps0 and ps1, and 3 worker
    # jobs on hosts worker0, worker1 and worker2.
    cluster_spec = {
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
    with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
      # Build your graph
      v1 = tf.Variable(...)  # assigned to /job:ps/task:0
      v2 = tf.Variable(...)  # assigned to /job:ps/task:1
      v3 = tf.Variable(...)  # assigned to /job:ps/task:0
    # Run compute

  
  
  
  
  


##Tensor-like objects
#Many TensorFlow operations take one or more tf.Tensor objects as arguments. 
#For example, tf.matmul takes two tf.Tensor objects, and tf.add_n takes a list of n tf.Tensor objects. 

#OR pass tensor-like object in place of a tf.Tensor
#Tensorflow implicitly convert it to a tf.Tensor using the tf.convert_to_tensor method.
#By default, TensorFlow will create a new tf.Tensor each time you use the same tensor-like object. 
#To avoid this, manually call tf.convert_to_tensor on the tensor-like object once 
#and use the returned tf.Tensor instead.

#Tensor like objects 
    tf.Tensor
    tf.Variable
    numpy.ndarray
    list (and lists of tensor-like objects)
    Scalar Python types: bool, float, int, str

#OR register additional tensor-like types using tf.register_tensor_conversion_function.



##Executing a graph in a tf.Session

#TensorFlow uses the tf.Session class to represent a connection 
#between the client program---typically a Python program, ---and the C++ runtime. 

#A tf.Session object provides access to devices in the local machine, 
#and remote devices using the distributed TensorFlow runtime

#Higher-level APIs such as tf.train.MonitoredTrainingSession 
#or tf.estimator.Estimator will create and manage a tf.Session for you. 

#These APIs accept optional target and config arguments 
#(either directly, or as part of a tf.estimator.RunConfig object)


##Low level API for Creating a tf.Session

# Create a default in-process session.
#if not used with 'with', use sess.close() to close the session 
with tf.Session() as sess:
  # ...

# Create a remote session.
with tf.Session("grpc://example.org:2222"):
  # ...

 

  
##tf.Session.__init__ accepts three optional arguments:
    target. If this argument is left empty (the default), 
        the session will only use devices in the local machine. 
        OR specify a grpc:// URL to specify the address of a TensorFlow server, 
        which gives the session access to all devices on machines 
        that that server controls.  
    graph. By default, a new tf.Session will be bound to---
            and only able to run operations in---the current default graph. 
            For multiple graphs specifiy graph here 
    config. specify a tf.ConfigProto that controls the behavior of the session. 
            some of the configuration options include:
                Allow_soft_placement: Set this to True to enable a "soft" device placement algorithm, 
                    which ignores tf.device annotations that attempt to place CPU-only operations 
                    on a GPU device, and places them on the CPU instead.
                cluster_def: When using distributed TensorFlow, specify 
                    what machines to use in the computation, 
                    and provide a mapping between job names, task indices, and network addresses. 
                graph_options.optimizer_options: Provides control over the optimizations that TensorFlow performs on your graph before executing it.
                gpu_options.allow_growth: Set this to True to change the GPU memory allocator 
                    so that it gradually increases the amount of memory allocated, 
                    rather than allocating most of the memory at startup.


##Using tf.Session.run to execute operations

#tf.Session.run requires you to specify a list of fetches, 
#which determine the return values, 
#and may be a tf.Operation, a tf.Tensor, or a tensor-like type such as tf.Variable. 

    x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
    w = tf.Variable(tf.random_uniform([2, 2]))
    y = tf.matmul(x, w)
    output = tf.nn.softmax(y) #Computes softmax activations.
    init_op = w.initializer

    with tf.Session() as sess:
      # Run the initializer on `w`.
      sess.run(init_op)

      # Evaluate `output`. `sess.run(output)` will return a NumPy array containing
      # the result of the computation.
      print(sess.run(output))

      # Evaluate `y` and `output`. Note that `y` will only be computed once, and its
      # result used both to return `y_val` and as an input to the `tf.nn.softmax()`
      # op. Both `y_val` and `output_val` will be NumPy arrays.
      y_val, output_val = sess.run([y, output])


      
#tf.Session.run also optionally takes a dictionary of feeds, 
#which is a mapping from tf.Tensor objects (typically tf.placeholder tensors) 
#to values (typically Python scalars, lists, or NumPy arrays) 
#that will be substituted for those tensors in the execution

    # Define a placeholder that expects a vector of three floating-point values,
    # and a computation that depends on it.
    x = tf.placeholder(tf.float32, shape=[3]) #list of three elements
    y = tf.square(x)

    with tf.Session() as sess:
      # Feeding a value changes the result that is returned when you evaluate `y`.
      print(sess.run(y, {x: [1.0, 2.0, 3.0]})  # => "[1.0, 4.0, 9.0]"
      print(sess.run(y, {x: [0.0, 0.0, 5.0]})  # => "[0.0, 0.0, 25.0]"

      # Raises `tf.errors.InvalidArgumentError`, because you must feed a value for
      # a `tf.placeholder()` when evaluating a tensor that depends on it.
      sess.run(y)

      # Raises `ValueError`, because the shape of `37.0` does not match the shape
      # of placeholder `x`.
      sess.run(y, {x: 37.0})


#tf.Session.run also accepts an optional options argument 
#that enables  to specify options about the call, 
#and an optional run_metadata argument that enables  to collect metadata about the execution. 

#For example, use these options together to collect tracing information about the execution:

y = tf.matmul([[37.0, -23.0], [1.0, 4.0]], tf.random_uniform([2, 2]))

with tf.Session() as sess:
  # Define options for the `sess.run()` call.
  options = tf.RunOptions()
  options.output_partition_graphs = True
  options.trace_level = tf.RunOptions.FULL_TRACE

  # Define a container for the returned metadata.
  metadata = tf.RunMetadata()

  sess.run(y, options=options, run_metadata=metadata)

  # Print the subgraphs that executed on each device.
  print(metadata.partition_graphs)

  # Print the timings of each operation that executed.
  print(metadata.step_stats)


##Visualizing your graph
#The graph visualizer is a component of TensorBoard that renders the structure of a graph visually in a browser. 

#using a tf.estimator.Estimator, the graph (and any summaries) will be logged 
#automatically to the model_dir that you specified when creating the estimator.


#The easiest way to create a visualization is to pass a tf.Graph 
#when creating the tf.summary.FileWriter:


# Build your graph.
x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)
# ...
loss = ...
train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
  # `sess.graph` provides access to the graph used in a `tf.Session`.
  writer = tf.summary.FileWriter("/tmp/log/...", sess.graph)

  # Perform your computation...
  for i in range(1000):
    sess.run(train_op)
    # ...

  writer.close()



##Programming with multiple graphs

# When training a model, a common way of organizing code is to use one graph for training model, 
#and a separate graph for evaluating or performing inference with a trained model. 

#TensorFlow provides a "default graph" that is implicitly passed to all API functions in the same context. 
#For many applications, a single graph is sufficient. 

#OR install a different tf.Graph as the default graph, using tf.Graph.as_default context manager:


g_1 = tf.Graph()
with g_1.as_default():
  # Operations created in this scope will be added to `g_1`.
  c = tf.constant("Node in g_1")

  # Sessions created in this scope will run operations from `g_1`.
  sess_1 = tf.Session()

g_2 = tf.Graph()
with g_2.as_default():
  # Operations created in this scope will be added to `g_2`.
  d = tf.constant("Node in g_2")

# Alternatively, you can pass a graph when constructing a `tf.Session`:
# `sess_2` will run operations from `g_2`.
sess_2 = tf.Session(graph=g_2)

assert c.graph is g_1
assert sess_1.graph is g_1

assert d.graph is g_2
assert sess_2.graph is g_2


#To inspect the current default graph, 
#call tf.get_default_graph, which returns a tf.Graph object:

# Print all of the operations in the default graph.
g = tf.get_default_graph()
print(g.get_operations())


        
###Tensorflow - Estimators 
##https://www.tensorflow.org/programmers_guide/estimators

##Steps 
1.Write one or more input_fn functions
        feature_dict : A dictionary in which the keys are feature column names 
            and the values are Tensors (or SparseTensors) containing the corresponding feature data
        label: A Tensor containing one or more labels

        def my_training_set():
           ...  
           #feature_dict is a dict with keys as population, crime_rate, median_education as given in step 2
           return feature_dict, label  #label is list of data point for each of feature observations 

2.Define the feature columns. 
    Each tf.feature_column identifies a feature name, its type, and any input pre-processing. 
 
    # Define three numeric feature columns.
    population = tf.feature_column.numeric_column('population')
    crime_rate = tf.feature_column.numeric_column('crime_rate')
    median_education = tf.feature_column.numeric_column('median_education',
                        normalizer_fn='lambda x: x - global_education_mean') #input pre-processing


3.Instantiate the relevant pre-made Estimator. 
    # Instantiate an estimator, passing the feature columns.
    estimator = tf.estimator.Estimator.LinearClassifier(
        feature_columns=[population, crime_rate, median_education],
        )

4.Call a training, evaluation, or prediction method. 
    estimator.train(input_fn=my_training_set, steps=2000)
    metrics = estimator.evaluate(input_fn=my_evaluation_set)
    predict_y = estimator.train(input_fn=my_prediction_set)
    
    
    
###tf.estimator - Feature columns and transformations
#A FeatureColumn represents a single feature in  data. 

##Categorical Features 
#one-hot encoding 
#Categorical features in linear models are typically translated into a sparse vector 
#For example, if there are only three possible eye colors 
#you can represent 'eye_color' as a length 3 vector: 
#'brown' would become [1, 0, 0], 
#'blue' would become [0, 1, 0] , 'green' would become [0, 0, 1]

eye_color = tf.feature_column.categorical_column_with_vocabulary_list(
    "eye_color", vocabulary_list=["blue", "brown", "green"])

#OR generate FeatureColumns for categorical features without all possible values. 
education = tf.feature_column.categorical_column_with_hash_bucket(
    "education", hash_bucket_size=1000)

##Feature Crosses
#derived feature based on combinations of feature values
#Used for getting relative dependencies of features 

#Example - given 'favorite_sport', 'home_city'
#create a new feature 'favorite_sport_x_home_city' by concatenating 'favorite_sport', 'home_city'

sport_x_city = tf.feature_column.crossed_column(
    ["favorite_sport", "home_city"], hash_bucket_size=int(1e4))

##Continuous columns
#specify a continuous feature 
age = tf.feature_column.numeric_column("age")

##Bucketization
#Bucketization turns a continuous column into a categorical column. 
#This transformation lets you use continuous features in feature crosses, 
#or learn cases where specific value ranges have particular importance.

#Bucketization divides the range of possible values into subranges called buckets:
#The bucket into which a value falls becomes the categorical label for that value.
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])


    
    
###tf.estmate - Building Input Functions with tf.estimator 
#You provide the data through an input function.

#The input function must return (feature_cols, labels)
#feature_cols is a dictionary of tensors. 
#Each key corresponds to the name of a FeatureColumn. 
#Each key's value is a tensor containing the values of that feature 

#The input_fn passed to  train, evaluate, and predict methods of the Estimator. 


def my_input_fn():

    # Preprocess your data here...

    # ...then return 
    return feature_cols, labels

#meaning
feature_cols
    A dict containing key/value pairs that map feature column names to Tensors 
    (or SparseTensors) containing the corresponding feature data.
labels
    A Tensor containing your label (target) values: 
    the values that model aims to predict.



##Converting Feature Data( in numpy or pandas) to Tensors

import numpy as np

feature_columns = tf.feature_column.numeric_column("x", shape=[4]) #each feature is of 4 size

#abive 'x' is mentioned as key of dict 
# numpy input_fn.
my_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(x_data)},
    y=np.array(y_data),
    ...)



import pandas as pd
# pandas input_fn.
my_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=pd.DataFrame({"x": x_data}),
    y=pd.Series(y_data),
    ...)

##Reference - tf.estimator.inputs.numpy_input_fn
tf.estimator.inputs.numpy_input_fn(
    x,
    y=None,
    batch_size=128,
    num_epochs=1,
    shuffle=None,
    queue_capacity=1000,
    num_threads=1
)

#This returns a function outputting features and target based on the dict of numpy arrays. 
#The dict features has the same keys as the x.

#Args:
    x: dict of numpy array object.
    y: numpy array object. None if absent.
    batch_size: Integer, size of batches to return.
    num_epochs: Integer, number of epochs to iterate over data. 
                If None will run forever.
    shuffle: Boolean, if True shuffles the queue. 
             Avoid shuffle at prediction time.
    queue_capacity: Integer, size of queue to accumulate.
    num_threads: Integer, number of threads used for reading and enqueueing. 
                 In order to have predicted and repeatable order of reading and enqueueing, 
                 such as in prediction and evaluation mode, num_threads should be 1.

#Returns:Function, that has signature of ()->(dict of features, target)

#Example 
age = np.arange(4) * 1.0
height = np.arange(32, 36)
x = {'age': age, 'height': height}
y = np.arange(-32, -28)

with tf.Session() as session:
    input_fn = tf.estimator.inputs.numpy_input_fn(x, y, batch_size=2, shuffle=False, num_epochs=1)
    print(input_fn())
({'height': <tf.Tensor 'fifo_queue_DequeueUpTo:2' shape=(?,) dtype=int32>, 
  'age': <tf.Tensor 'fifo_queue_DequeueUpTo:1'shape=(?,) dtype=float64>}, 
  <tf.Tensor 'fifo_queue_DequeueUpTo:3' shape=(?,) dtype=int32>
 )    

##Reference - tf.estimator.inputs.pandas_input_fn

tf.estimator.inputs.pandas_input_fn(
    x,
    y=None,
    batch_size=128,
    num_epochs=1,
    shuffle=None,
    queue_capacity=1000,
    num_threads=1,
    target_column='target'
)

#Note: y's index must match x's index.
#Args:
    x: pandas DataFrame object.
    y: pandas Series object. None if absent.
    target_column: str, name to give the target column y.

#Returns:Function, that has signature of ()->(dict of features, target)


##For sparse, categorical data (data where the majority of values are 0), 
#Use SparseTensor

#Example - The following code defines a two-dimensional SparseTensor 
#with 3 rows and 5 columns. 
#The element with index [0,1] has a value of 6, 
#and the element with index [2,4] has a value of 0.5 (all other values are 0):
sparse_tensor = tf.SparseTensor(indices=[[0,1], [2,4]],
                                values=[6, 0.5],
                                dense_shape=[3, 5])


#This corresponds to the following dense tensor:
[[0, 6, 0, 0, 0]
 [0, 0, 0, 0, 0]
 [0, 0, 0, 0, 0.5]]


##to parameterize your input function
def my_input_fn(data_set):
  ...
  return feature_cols, labels

def my_input_fn_training_set():
  return my_input_fn(training_set)

classifier.train(input_fn=my_input_fn_training_set, steps=2000)


#OR  use Python's functools.partial 
classifier.train(
    input_fn=functools.partial(my_input_fn, data_set=training_set),
    steps=2000)


#OR to wrap your input_fn invocation in a lambda 
classifier.train(input_fn=lambda: my_input_fn(training_set), steps=2000)





    
## Reference: tf.feature_column
bucketized_column(source_column, boundaries)
    Represents discretized dense input.
    Buckets include the left boundary, and exclude the right boundary. 
    boundaries = [0, 10, 100]
    input tensor = [[-5, 10000]
                    [150,   10]
                    [5,    100]]
    #then the output will be
    output = [[0, 3]
              [3, 2]
              [1, 3]]

        
categorical_column_with_hash_bucket( key, hash_bucket_size, dtype=tf.string)
    Represents sparse feature where ids are set by hashing.    
    Use this when your sparse features are in string or integer format, 
    and you want to distribute your inputs into a finite number of buckets by hashing. 
    output_id = Hash(input_feature_string) % bucket_size


categorical_column_with_identity( key, num_buckets, default_value=None)
    A _CategoricalColumn that returns identity values.    
    Use this when your inputs are integers in the range [0, num_buckets), 
    and you want to use the input value itself as the categorical ID. 
    Values outside this range will result in default_value if specified, 
    otherwise it will fail.

    Example :, each input in the range [0, 1000000) 
    is assigned the same value. 
    All other inputs are assigned default_value 0. 
    Note that a literal 0 in inputs will result in the same default ID.

    Linear model:
    video_id = categorical_column_with_identity(
        key='video_id', num_buckets=1000000, default_value=0)
    Embedding for a DNN model:
    columns = [embedding_column(video_id, 9),...]


categorical_column_with_vocabulary_file( key, vocabulary_file, vocabulary_size, num_oov_buckets=0, default_value=None, dtype=tf.string)
    A _CategoricalColumn with a vocabulary file.

    Use this when your inputs are in string or integer format, 
    and you have a vocabulary file that maps each value to an integer ID. 
    By default, out-of-vocabulary values are ignored. 
    Use either (but not both) of num_oov_buckets 
    and default_value to specify how to include out-of-vocabulary values.

    Example with default_value: File '/us/states.txt' contains 51 lines 
    - the first line is 'XX', and the other 50 each have a 2-character U.S. state abbreviation. 
    Both a literal 'XX' in input, and other values missing from the file, 
    will be assigned ID 0. 
    All others are assigned the corresponding line number 1-50.
    states = categorical_column_with_vocabulary_file(
        key='states', vocabulary_file='/us/states.txt', vocabulary_size=51,
        default_value=0)
    And to make an embedding with either:
    columns = [embedding_column(states, 3),...]
  

categorical_column_with_vocabulary_list( key, vocabulary_list,  dtype=None, default_value=-1,  num_oov_buckets=0)
    A _CategoricalColumn with in-memory vocabulary.

    Use this when your inputs are in string or integer format, 
    and you have an in-memory vocabulary mapping each value to an integer ID. 
    By default, out-of-vocabulary values are ignored. 
    Use either (but not both) of num_oov_buckets and default_value to specify how to include out-of-vocabulary values.

    Example with num_oov_buckets: 
    In the following example, each input in vocabulary_list is assigned 
    an ID 0-3 corresponding to its index (e.g., input 'B' produces output 2). 
    All other inputs are hashed and assigned an ID 4-5.

    colors = categorical_column_with_vocabulary_list(
        key='colors', vocabulary_list=('R', 'G', 'B', 'Y'),
        num_oov_buckets=2)

    And to make an embedding with either:
    columns = [embedding_column(colors, 3),...]


crossed_column(  keys, hash_bucket_size,  hash_key=None)
    Returns a column for performing crosses of categorical features.

    Crossed features will be hashed according to hash_bucket_size. 
    Conceptually, the transformation can be thought of as: 
    Hash(cartesian product of features) % hash_bucket_size

    For example, if the input features are:
        SparseTensor referred by first key:
        shape = [2, 2]
        {
            [0, 0]: "a"
            [1, 0]: "b"
            [1, 1]: "c"
        }
        SparseTensor referred by second key:
        shape = [2, 1]
        {
            [0, 0]: "d"
            [1, 0]: "e"
        }
    then crossed feature will look like:
     shape = [2, 2]
    {
        [0, 0]: Hash64("d", Hash64("a")) % hash_bucket_size
        [1, 0]: Hash64("e", Hash64("b")) % hash_bucket_size
        [1, 1]: Hash64("e", Hash64("c")) % hash_bucket_size
    }

    keywords_x_doc_terms = crossed_column(['keywords', 'doc_terms'], 50K)
 
    You could also use vocabulary lookup before crossing:
    keywords = categorical_column_with_vocabulary_file(
        'keywords', '/path/to/vocabulary/file', vocabulary_size=1K)
    keywords_x_doc_terms = crossed_column([keywords, 'doc_terms'], 50K)
    
    If an input feature is of numeric type, 
    you can use categorical_column_with_identity, or bucketized_column, 
    # vertical_id is an integer categorical feature.
    vertical_id = categorical_column_with_identity('vertical_id', 10K)
    price = numeric_column('price')
    # bucketized_column converts numerical feature to a categorical one.
    bucketized_price = bucketized_column(price, boundaries=[...])
    vertical_id_x_price = crossed_column([vertical_id, bucketized_price], 50K)
   
    To use crossed column in DNN model, 
    you need to add it in an embedding column as in this example:
    vertical_id_x_price = crossed_column([vertical_id, bucketized_price], 50K)
    vertical_id_x_price_embedded = embedding_column(vertical_id_x_price, 10)
    

tf.feature_column.embedding_column( categorical_column, dimension,  combiner='mean', initializer=None,  ckpt_to_load_from=None,  tensor_name_in_ckpt=None,  max_norm=None,  trainable=True)
    _DenseColumn that converts from sparse, categorical input.
        dimension: An integer specifying dimension of the embedding, must be > 0.
    Use this when your inputs are sparse, 
    but you want to convert them to a dense representation (e.g., to feed to a DNN).

    Inputs must be a _CategoricalColumn created 
    by any of the categorical_column_* function. 

    video_id = categorical_column_with_identity(
        key='video_id', num_buckets=1000000, default_value=0)
    columns = [embedding_column(video_id, 9),...]
    
    
tf.feature_column.indicator_column(categorical_column)
    Represents multi-hot representation of given categorical column.

    Used to wrap any categorical_column_* (e.g., to feed to DNN). 
    Use embedding_column if the inputs are sparse.

    name = indicator_column(categorical_column_with_vocabulary_list(
        'name', ['bob', 'george', 'wanda'])
    columns = [name, ...]
    features = tf.parse_example(..., features=make_parse_example_spec(columns))
    dense_tensor = input_layer(features, columns)

    dense_tensor == [[1, 0, 0]]  # If "name" bytes_list is ["bob"]
    dense_tensor == [[1, 0, 1]]  # If "name" bytes_list is ["bob", "wanda"]
    dense_tensor == [[2, 0, 0]]  # If "name" bytes_list is ["bob", "bob"]

   
    
input_layer(  features, feature_columns,  weight_collections=None, trainable=True)
    Returns a dense Tensor as input layer based on given feature_columns.
        features: A mapping from key to tensors. 
                  _FeatureColumns look up via these keys. 
                  For example numeric_column('price') will look at 'price' key in this dict. 
                  Values can be a SparseTensor or a Tensor depends on corresponding _FeatureColumn.
        feature_columns: An iterable containing the FeatureColumns to use as inputs to your model. 
                  All items should be instances of classes derived from _DenseColumn 
                  such as numeric_column, embedding_column, bucketized_column, indicator_column. 
                  If you have categorical features, you can wrap them with an embedding_column 
                  or indicator_column.

    Generally a single example in training data is described with FeatureColumns. 
    At the first layer of the model, 
    this column oriented data should be converted to a single Tensor.

    price = numeric_column('price')
    keywords_embedded = embedding_column(
        categorical_column_with_hash_bucket("keywords", 10K), dimensions=16)
    columns = [price, keywords_embedded, ...]
    features = tf.parse_example(..., features=make_parse_example_spec(columns))
    dense_tensor = input_layer(features, columns)
    for units in [128, 64, 32]:
      dense_tensor = tf.layers.dense(dense_tensor, units, tf.nn.relu)
    prediction = tf.layers.dense(dense_tensor, 1)
    
    

linear_model(  features,  feature_columns,  units=1,  sparse_combiner='sum',  weight_collections=None,  trainable=True)
    Returns a linear prediction Tensor based on given feature_columns.
        units: An integer, dimensionality of the output space. Default value is 1.
    This function generates a weighted sum based on output dimension units. 
    Weighted sum refers to logits in classification problems. 
    It refers to the prediction itself for linear regression problems.

    linear_model treats categorical columns as indicator_columns while input_layer explicitly requires wrapping each of them with an embedding_column or an indicator_column.

    price = numeric_column('price')
    price_buckets = bucketized_column(price, boundaries=[0., 10., 100., 1000.])
    keywords = categorical_column_with_hash_bucket("keywords", 10K)
    keywords_price = crossed_column('keywords', price_buckets, ...)
    columns = [price_buckets, keywords, keywords_price ...]
    features = tf.parse_example(..., features=make_parse_example_spec(columns))
    prediction = linear_model(features, columns)


make_parse_example_spec(feature_columns)
    Creates parsing spec dictionary from input feature_columns.
            feature_columns: An iterable containing all feature columns. 
                    All items should be instances of classes derived from _FeatureColumn.
                    
    The returned dictionary can be used as arg 'features' in tf.parse_example.
    # Define features and transformations
    feature_b = numeric_column(...)
    feature_c_bucketized = bucketized_column(numeric_column("feature_c"), ...)
    feature_a_x_feature_c = crossed_column(
        columns=["feature_a", feature_c_bucketized], ...)

    feature_columns = set(
        [feature_b, feature_c_bucketized, feature_a_x_feature_c])
    features = tf.parse_example(
        serialized=serialized_examples,
        features=make_parse_example_spec(feature_columns))

    For the above example, make_parse_example_spec would return the dict:
    {
        "feature_a": parsing_ops.VarLenFeature(tf.string),
        "feature_b": parsing_ops.FixedLenFeature([1], dtype=tf.float32),
        "feature_c": parsing_ops.FixedLenFeature([1], dtype=tf.float32)
    }

    
    

numeric_column( key,  shape=(1,),  default_value=None,  dtype=tf.float32,  normalizer_fn=None)
    Represents real valued or numerical features.
        key: A unique string identifying the input feature. 
             It is used as the column name and the dictionary key 
             for feature parsing configs, feature Tensor objects, and feature columns.
        shape: An iterable of integers specifies the shape of the Tensor. 
           An integer can be given which means a single dimension Tensor 
           with given width. The Tensor representing the column 
           will have the shape of [batch_size] + shape.
    price = numeric_column('price')
    columns = [price, ...]
    features = tf.parse_example(..., features=make_parse_example_spec(columns))
    dense_tensor = input_layer(features, columns)

    # or
    bucketized_price = bucketized_column(price, boundaries=[...])
    columns = [bucketized_price, ...]
    features = tf.parse_example(..., features=make_parse_example_spec(columns))
    linear_prediction = linear_model(features, columns)




    
weighted_categorical_column(categorical_column, weight_feature_key, dtype=tf.float32)
    Applies weight values to a _CategoricalColumn.
        categorical_column: A _CategoricalColumn created by categorical_column_with_* functions.

    Use this when each of your sparse inputs has both an ID and a value. 
    For example, if you are representing text documents 
    as a collection of word frequencies, 
    you can provide 2 parallel sparse input features 
    ('terms' and 'frequencies' below).

    This assumes the input dictionary contains a SparseTensor for key 'terms', 
    and a SparseTensor for key 'frequencies'. 
    These 2 tensors must have the same indices and dense shape.

    Input tf.Example objects:
    [
      features {
        feature {
          key: "terms"
          value {bytes_list {value: "very" value: "model"}}
        }
        feature {
          key: "frequencies"
          value {float_list {value: 0.3 value: 0.1}}
        }
      },
      features {
        feature {
          key: "terms"
          value {bytes_list {value: "when" value: "course" value: "human"}}
        }
        feature {
          key: "frequencies"
          value {float_list {value: 0.4 value: 0.1 value: 0.2}}
        }
      }
    ]

    categorical_column = categorical_column_with_hash_bucket(
        column_name='terms', hash_bucket_size=1000)
    weighted_column = weighted_categorical_column(
        categorical_column=categorical_column, weight_feature_key='frequencies')
    columns = [weighted_column, ...]
    features = tf.parse_example(..., features=make_parse_example_spec(columns))
    linear_prediction, _, _ = linear_model(features, columns)

    
    

###TensorFlow- Example of LinearClassifier, DNNClassifier, DNNLinearCombinedClassifier
#https://www.tensorflow.org/tutorials/wide#how_logistic_regression_works

#Example file 
#Column Name 	Type 	        Description
age 	        Continuous 	    The age of the individual
workclass 	    Categorical 	The type of employer the individual has (government, military, private, etc.).
fnlwgt 	        Continuous 	    The number of people the census takers believe that observation represents (sample weight). This variable will not be used.
education 	    Categorical 	The highest level of education achieved for that individual.
education_num 	Continuous 	    The highest level of education in numerical form.
marital_status 	Categorical 	Marital status of the individual.
occupation 	    Categorical 	The occupation of the individual.
relationship 	Categorical 	Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race 	        Categorical 	White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
gender 	        Categorical 	Female, Male.
capital_gain 	Continuous 	    Capital gains recorded.
capital_loss 	Continuous 	    Capital Losses recorded.
hours_per_week 	Continuous 	    Hours worked per week.
native_country 	Categorical 	Country of origin of the individual.
income 	        Categorical 	">50K" or "<=50K", meaning whether the person makes more than \$50,000 annually.

## Module: tf.compat
#Functions for Python 2 vs. 3 compatibility.
#Types
#The compatibility module also provides the following types:
    bytes_or_text_types
    complex_types
    integral_types
    real_types

#Functions
as_bytes(...): Converts either bytes or unicode to bytes, using utf-8 encoding for text.
as_str(...): Converts either bytes or unicode to bytes, using utf-8 encoding for text.
as_str_any(...): Converts to str as str(value), but use as_str for bytes.
as_text(...): Returns the given argument as a unicode string.


#Function usage 
tf.feature_column.categorical_column_with_vocabulary_list(  key,  vocabulary_list,)
    A _CategoricalColumn with in-memory vocabulary(ie list of words)
tf.feature_column.categorical_column_with_hash_bucket(key, hash_bucket_size,)
    Returns _HashedCategoricalColumn
    Distributes inputs into a finite number of buckets by hashing.
tf.feature_column.numeric_column(key,  shape=(1,),)
    Represents real valued or numerical features.
tf.feature_column.bucketized_column( source_column,  boundaries)
    Converts  dense input into discretized values
tf.feature_column.crossed_column(keys,  hash_bucket_size,)
    Returns a column(_CrossedColumn) for performing crosses of categorical features.
tf.feature_column.indicator_column(categorical_column)
    Represents multi-hot representation of given categorical column
tf.feature_column.embedding_column(categorical_column, dimension,...)
    _DenseColumn that converts from sparse, categorical input.
    Use this when your inputs are sparse, 
    but you want to convert them to a dense representation (e.g., to feed to a DNN).

#code 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import pandas as pd
from six.moves import urllib
import tensorflow as tf


CSV_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "gender",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income_bracket"
]

gender = tf.feature_column.categorical_column_with_vocabulary_list(
    "gender", ["Female", "Male"])
education = tf.feature_column.categorical_column_with_vocabulary_list(
    "education", [
        "Bachelors", "HS-grad", "11th", "Masters", "9th",
        "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
        "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
        "Preschool", "12th"
    ])
marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    "marital_status", [
        "Married-civ-spouse", "Divorced", "Married-spouse-absent",
        "Never-married", "Separated", "Married-AF-spouse", "Widowed"
    ])
relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    "relationship", [
        "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
        "Other-relative"
    ])
workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    "workclass", [
        "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
        "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
    ])

# To show an example of hashing:
occupation = tf.feature_column.categorical_column_with_hash_bucket(
    "occupation", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket(
    "native_country", hash_bucket_size=1000)

# Continuous base columns.
age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")

# Transformations.
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

# Wide columns and deep columns.
base_columns = [
    gender, education, marital_status, relationship, workclass, occupation,
    native_country, age_buckets,
]

crossed_columns = [
    tf.feature_column.crossed_column(
        ["education", "occupation"], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        [age_buckets, "education", "occupation"], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        ["native_country", "occupation"], hash_bucket_size=1000)
]

deep_columns = [
    tf.feature_column.indicator_column(workclass),
    tf.feature_column.indicator_column(education),
    tf.feature_column.indicator_column(gender),
    tf.feature_column.indicator_column(relationship),
    # To show an example of embedding
    tf.feature_column.embedding_column(native_country, dimension=8),
    tf.feature_column.embedding_column(occupation, dimension=8),
    age,
    education_num,
    capital_gain,
    capital_loss,
    hours_per_week,
]


def maybe_download(train_data, test_data):
  """Maybe downloads training data and returns train and test file names."""
  if train_data:
    train_file_name = train_data
  else:
    train_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        train_file.name)  # pylint: disable=line-too-long
    train_file_name = train_file.name
    train_file.close()
    print("Training data is downloaded to %s" % train_file_name)

  if test_data:
    test_file_name = test_data
  else:
    test_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        test_file.name)  # pylint: disable=line-too-long
    test_file_name = test_file.name
    test_file.close()
    print("Test data is downloaded to %s"% test_file_name)

  return train_file_name, test_file_name
  
#Function usage 
tf.gfile.Open or tf.gfile.GFile(name,  mode='r')
    File I/O wrappers without thread locking.
    Usage is exactly same as python file object 

  
#Since the task is a binary classification problem, 
#we'll construct a label column named "label" 
#whose value is 1 if the income is over 50K, and 0 otherwise.

train_labels = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
test_labels = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
 
#Code  
def build_estimator(model_dir, model_type):
  """Build an estimator."""
  if model_type == "wide":
    m = tf.estimator.LinearClassifier(
        model_dir=model_dir, feature_columns=base_columns + crossed_columns)
  elif model_type == "deep":
    m = tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=[100, 50])
  else:
    m = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=crossed_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])
  return m


def input_fn(data_file, num_epochs, shuffle):
  """Input builder function."""
  df_data = pd.read_csv(
      tf.gfile.Open(data_file),
      names=CSV_COLUMNS,
      skipinitialspace=True,
      engine="python",
      skiprows=1)
  # remove NaN elements
  df_data = df_data.dropna(how="any", axis=0)
  labels = df_data["income_bracket"].apply(lambda x: ">50K" in x).astype(int)
  return tf.estimator.inputs.pandas_input_fn(
      x=df_data,
      y=labels,
      batch_size=100,
      num_epochs=num_epochs,
      shuffle=shuffle,
      num_threads=5)


def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
  """Train and evaluate the model."""
  train_file_name, test_file_name = maybe_download(train_data, test_data)
  model_dir = tempfile.mkdtemp() if not model_dir else model_dir

  m = build_estimator(model_dir, model_type)
  # set num_epochs to None to get infinite stream of data.
  m.train(
      input_fn=input_fn(train_file_name, num_epochs=None, shuffle=True),
      steps=train_steps)
  # set steps to None to run evaluation until all data consumed.
  results = m.evaluate(
      input_fn=input_fn(test_file_name, num_epochs=1, shuffle=False),
      steps=None)
  print("model directory = %s" % model_dir)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))


FLAGS = None


def main(_):
  train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--model_type",
      type=str,
      default="wide_n_deep",
      help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=2000,
      help="Number of training steps."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      default="",
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="",
      help="Path to the test data."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  
  
##Adding Regularization to Prevent Overfitting
#Regularization is a technique used to avoid overfitting  
  
m = tf.estimator.LinearClassifier(
    model_dir=model_dir, feature_columns=base_columns + crossed_columns,
    optimizer=tf.train.FtrlOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=1.0, #A float value, must be greater than or equal to zero.
      l2_regularization_strength=1.0), #A float value, must be greater than or equal to zero.
    model_dir=model_dir)

#L1 regularization tends to make model weights stay at zero, creating sparser models, 
#L2 regularization also tries to make the model weights closer to zero but not necessarily zero. 
  
# if you increase the strength of L1 regularization, 
#you will have a smaller model size because many of the model weights will be zero.

#you should try various combinations of L1, L2 regularization strengths and find the best parameters 
#that best control overfitting and give you a desirable model size.








    
    
    
    
    
###tf.estimator - Feature format - class tf.Feature 
#Defined in tensorflow/core/example/feature.proto.
#Protocol messages(.proto file) for describing input data Examples for machine learning model training or inference.
#efficient binary format
#CHeck - https://developers.google.com/protocol-buffers/docs/pythontutorial


#There are three base Feature types:
   - bytes
   - float
   - int64

#A Feature contains Lists which may hold zero or more values.  
#These lists are the base values BytesList, FloatList, Int64List.

#Features are organized into categories by name.  
#The Features message contains the mapping from name to Feature.

#Example Features for a movie recommendation application:
   feature {
     key: "age"
     value { float_list {
       value: 29.0
     }}
   }
   feature {
     key: "movie"
     value { bytes_list {
       value: "The Shawshank Redemption"
       value: "Fight Club"
     }}
   }
   feature {
     key: "movie_ratings"
     value { float_list {
       value: 9.0
       value: 9.7
     }}
   }
   feature {
     key: "suggestion"
     value { bytes_list {
       value: "Inception"
     }}
   }
   feature {
     key: "suggestion_purchased"
     value { int64_list {
       value: 1
     }}
   }
   feature {
     key: "purchase_price"
     value { float_list {
       value: 9.99
     }}
   }   
##Python constructors 
tf.train.Feature(BytesList bytes_list or  FloatList float_list or Int64List int64_list )
tf.train.BytesList/FloatList/Int64List(value=[v1,v2,v3...])
tf.train.FeatureList(feature=[feature1,feature2,..])
tf.train.Features(feature={ "name": Feature,...}) # Map from feature name to feature.
tf.train.FeatureLists(feature_list={ "name": FeatureList,...}) #Map from feature name to feature list.
tf.train.Example(features =Features)
tf.train.SequenceExample(context=Features, feature_lists=FeatureLists)


    
    
###tf.estimator - Example format - class tf.Example
#Defined in https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/core/example/feature.proto


#Protocol messages(.proto file) for describing input data Examples for machine learning model training or inference.
#efficient binary format
#CHeck - https://developers.google.com/protocol-buffers/docs/pythontutorial
 
#It contains a key-value store (features); 
#where each key (string) maps to a Feature message 
#(which is one of packed BytesList,FloatList, or Int64List).  


#In TensorFlow, Examples are read in row-major format
#to store an M x N matrix of Bytes,
#the BytesList must contain M*N bytes, with M rows of N contiguous values for each row

#Example of Example file 
  features {
        feature {
          key: "age"
          value { float_list {  value: 29.0 }}
          }
    feature {
      key: "movie"
      value { bytes_list {
        value: "The Shawshank Redemption"
        value: "Fight Club"
      }}
    }
    feature {
      key: "movie_ratings"
      value { float_list {
        value: 9.0
        value: 9.7
      }}
    }
    feature {
      key: "suggestion"
      value { bytes_list {
        value: "Inception"
      }}
    }
    # Note that this feature exists to be used as a label in training.
    # E.g., if training a logistic regression model to predict purchase
    # probability in our learning tool we would set the label feature to
    # "suggestion_purchased".
    feature {
      key: "suggestion_purchased"
      value { float_list {
        value: 1.0
      }}
    }
    # Similar to "suggestion_purchased" above this feature exists to be used
    # as a label in training.
    # E.g., if training a linear regression model to predict purchase
    # price in our learning tool we would set the label feature to
    # "purchase_price".
    feature {
      key: "purchase_price"
      value { float_list {
        value: 9.99
      }}
    }
 }

#A conformant Example data set obeys the following conventions:
  - If a Feature K exists in one example with data type T, it must be of
      type T in all other examples when present. It may be omitted.
  - The number of instances of Feature K list data may vary across examples,
      depending on the requirements of the model.
  - If a Feature K doesn't exist in an example, a K-specific default will be
      used, if configured.
  - If a Feature K exists in an example but contains no items, the intent
      is considered to be an empty tensor and no default will be used.


#A SequenceExample is an Example representing one or more sequences and a context

#The feature_lists contain a key, value map 
#where each key is associated with a repeated set of Features (a FeatureList).
#A FeatureList thus represents the values of a feature identified by its key over time / frames.

#Eg: The time-independent features ("locale","age", "favorites") describing the user are part of the context. 
#The sequence of movies the user rated are part of the feature_lists. 

context: {
  feature: {
    key  : "locale"
    value: {
      bytes_list: {
        value: [ "pt_BR" ]
      }
    }
  }
  feature: {
    key  : "age"
    value: {
      float_list: {
        value: [ 19.0 ]
      }
    }
  }
  feature: {
    key  : "favorites"
    value: {
      bytes_list: {
        value: [ "Majesty Rose", "Savannah Outen", "One Direction" ]
      }
    }
  }
}
feature_lists: {
  feature_list: {
    key  : "movie_ratings"
    value: {
      feature: {
        float_list: {
          value: [ 4.5 ]
        }
      }
      feature: {
        float_list: {
          value: [ 5.0 ]
        }
      }
    }
  }
  feature_list: {
    key  : "movie_names"
    value: {
      feature: {
        bytes_list: {
          value: [ "The Shawshank Redemption" ]
        }
      }
      feature: {
        bytes_list: {
          value: [ "Fight Club" ]
        }
      }
    }
  }
  feature_list: {
    key  : "actors"
    value: {
      feature: {
        bytes_list: {
          value: [ "Tim Robbins", "Morgan Freeman" ]
        }
      }
      feature: {
        bytes_list: {
          value: [ "Brad Pitt", "Edward Norton", "Helena Bonham Carter" ]
        }
      }
    }
  }
}

#A conformant SequenceExample data set obeys the following conventions:
Context:
  - All conformant context features K must obey the same conventions as
    a conformant Example's features (see above).
Feature lists:
  - A FeatureList L may be missing in an example; it is up to the
    parser configuration to determine if this is allowed or considered
    an empty list (zero length).
  - If a FeatureList L exists, it may be empty (zero length).
  - If a FeatureList L is non-empty, all features within the FeatureList
    must have the same data type T. Even across SequenceExamples, the type T
    of the FeatureList identified by the same key must be the same. An entry
    without any values may serve as an empty feature.
  - If a FeatureList L is non-empty, it is up to the parser configuration
    to determine if all features within the FeatureList must
    have the same size.  The same holds for this FeatureList across multiple
    examples.

##Examples of conformant and non-conformant examples' FeatureLists:
#Conformant FeatureLists:
   feature_lists: { feature_list: {
     key: "movie_ratings"
     value: { feature: { float_list: { value: [ 4.5 ] } }
              feature: { float_list: { value: [ 5.0 ] } } }
   } }

#Non-conformant FeatureLists (mismatched types):
   feature_lists: { feature_list: {
     key: "movie_ratings"
     value: { feature: { float_list: { value: [ 4.5 ] } }
              feature: { int64_list: { value: [ 5 ] } } }
   } }
   
##Python constructors
tf.train.Feature(BytesList bytes_list or  FloatList float_list or Int64List int64_list )
tf.train.BytesList/FloatList/Int64List(value=[v1,v2,v3...])
tf.train.FeatureList(feature=[feature1,feature2,..])
tf.train.Features(feature={"name": Feature,...}) # Map from feature name to feature.
tf.train.FeatureLists(feature_list={ "name": FeatureList,...}) #Map from feature name to feature list.
tf.train.Example(features =Features)
tf.train.SequenceExample(context=Features, feature_lists=FeatureLists)





##Helper class to handle Example proto
class FixedLenFeature(shape,dtype,default_value=None)
    Configuration for parsing a fixed-length input feature,
    parsed to Tensor
    To treat sparse input as dense, provide a default_value; 
    otherwise, the parse functions will fail on any examples missing this feature.
    Fields:
        shape: Shape of input data.
        dtype: Data type of input.
        default_value: Value to be used if an example is missing this feature. 
         It must be compatible with dtype and of the specified shape.

class FixedLenSequenceFeature(shape,dtype,allow_missing=False,default_value=None)
    Configuration for parsing a variable-length input feature into a Tensor.
    Fields:
        shape: Shape of input data for dimension 2 and higher. First dimension is of variable length None.
        dtype: Data type of input.
        allow_missing: Whether to allow this feature to be missing from a feature list item. Is available only for parsing SequenceExample not for parsing Examples.
        Default_value: Scalar value to be used to pad multiple Examples to their maximum length. Irrelevant for parsing a single Example or SequenceExample. Defaults to "" for dtype string and 0 otherwise (optional).

class VarLenFeature(dtype)
    Configuration for parsing a variable-length input feature(note there is no shape argument)
    parsed to a SparseTensor
    Fields:
        dtype: Data type of input.
        
        
class SparseFeature(index_key, value_key, dtype, size, already_sorted=False)
    Configuration for parsing a sparse input feature from an Example.
    Note, preferably use VarLenFeature
        value_key: The name of key for a Feature in the Example 
                   whose parsed Tensor will be the resulting SparseTensor.values.
        index_key: A list of names - one for each dimension in the resulting SparseTensor 
                   whose indices[i][dim] indicating the position of the i-th value 
                   in the dim dimension will be equal to the i-th value in the Feature 
                   with key named index_key[dim] in the Example.
        size: A list of ints for the resulting SparseTensor.dense_shape.
    For example, we can represent the following 2D SparseTensor
    SparseTensor(indices=[[3, 1], [20, 0]],
                 values=[0.5, -1.0]
                 dense_shape=[100, 3])
    with an Example input proto
    features {
      feature { key: "val" value { float_list { value: [ 0.5, -1.0 ] } } }
      feature { key: "ix0" value { int64_list { value: [ 3, 20 ] } } }
      feature { key: "ix1" value { int64_list { value: [ 1, 0 ] } } }
    }
    and SparseFeature config with 2 index_keys
    SparseFeature(index_key=["ix0", "ix1"],
                  value_key="val",
                  dtype=tf.float32,
                  size=[100, 3])


tf.parse_example(serialized,features,name=None,example_names=None)
    Parses multiple Example protos given in serialized(binary format)
    into a dictionary mapping keys to Tensor and SparseTensor objects. 
    
    Returns features is a dict from keys to VarLenFeature, SparseFeature, 
    and FixedLenFeature objects. 
    
    Each VarLenFeature and SparseFeature is mapped to a SparseTensor, 
    and each FixedLenFeature is mapped to a Tensor
 
    For example, for 
    features = {"ft":tf.VarLenFeature(tf.float32)}
    and three serialized Examples (binary format)are provided:
    serialized = [
      features
        { feature { key: "ft" value { float_list { value: [1.0, 2.0] } } } },
      features
        { feature []},
      features
        { feature { key: "ft" value { float_list { value: [3.0] } } }
    ]
    parsed_features = tf.parse_example(serialized, features)
    #then the output will look like:
    {"ft": SparseTensor(indices=[[0, 0], [0, 1], [2, 0]],
                        values=[1.0, 2.0, 3.0],
                        dense_shape=(3, 2)) }

    If instead a FixedLenSequenceFeature with default_value = -1.0 
    and shape=[] is used then the output will look like:
    {"ft": [[1.0, 2.0], [3.0, -1.0]]}

    Given two Example input protos in serialized:
    [
      features {
        feature { key: "kw" value { bytes_list { value: [ "knit", "big" ] } } }
        feature { key: "gps" value { float_list { value: [] } } }
      },
      features {
        feature { key: "kw" value { bytes_list { value: [ "emmy" ] } } }
        feature { key: "dank" value { int64_list { value: [ 42 ] } } }
        feature { key: "gps" value { } }
      }
    ]
    And arguments
    example_names: ["input0", "input1"],
    features: {
        "kw": VarLenFeature(tf.string),
        "dank": VarLenFeature(tf.int64),
        "gps": VarLenFeature(tf.float32),
    }
    Then the output is a dictionary:
    {
      "kw": SparseTensor(
          indices=[[0, 0], [0, 1], [1, 0]],
          values=["knit", "big", "emmy"]
          dense_shape=[2, 2]),
      "dank": SparseTensor(
          indices=[[1, 0]],
          values=[42],
          dense_shape=[2, 1]),
      "gps": SparseTensor(
          indices=[],
          values=[],
          dense_shape=[2, 0]),
    }
    For dense results in two serialized Examples:
    [
      features {
        feature { key: "age" value { int64_list { value: [ 0 ] } } }
        feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
       },
       features {
        feature { key: "age" value { int64_list { value: [] } } }
        feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
      }
    ]
    and arguments:
    example_names: ["input0", "input1"],
    features: {
        "age": FixedLenFeature([], dtype=tf.int64, default_value=-1),
        "gender": FixedLenFeature([], dtype=tf.string),
    }
    And the expected output is:
    {
      "age": [[0], [-1]],
      "gender": [["f"], ["f"]],
    }
    An alternative to VarLenFeature to obtain a SparseTensor is SparseFeature. 
    For example, given two Example input protos in serialized:
    [
      features {
        feature { key: "val" value { float_list { value: [ 0.5, -1.0 ] } } }
        feature { key: "ix" value { int64_list { value: [ 3, 20 ] } } }
      },
      features {
        feature { key: "val" value { float_list { value: [ 0.0 ] } } }
        feature { key: "ix" value { int64_list { value: [ 42 ] } } }
      }
    ]
    And arguments
    example_names: ["input0", "input1"],
    features: {
        "sparse": SparseFeature(
            index_key="ix", value_key="val", dtype=tf.float32, size=100),
    }
    Then the output is a dictionary:
    {
      "sparse": SparseTensor(
          indices=[[0, 3], [0, 20], [1, 42]],
          values=[0.5, -1.0, 0.0]
          dense_shape=[2, 100]),
    }
    
    

tf.parse_single_example ( serialized, features, name=None, example_names=None)
    Parses a single Example proto.
    Example: 
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            # Label part
            'objects_number': tf.FixedLenFeature([], tf.int64),
            'bboxes': tf.VarLenFeature(tf.float32),
            'labels': tf.VarLenFeature(tf.int64),
            # Dense data
            'image_raw': tf.FixedLenFeature([],tf.string)

        })

    # Get metadata
    objects_number = tf.cast(features['objects_number'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)

    # Actual data
    image_shape = tf.parallel_stack([height, width, depth])
    bboxes_shape = tf.parallel_stack([objects_number, 4])

    # BBOX data is actually dense convert it to dense tensor
    bboxes = tf.sparse_tensor_to_dense(features['bboxes'], default_value=0)

    # Since information about shape is lost reshape it
    bboxes = tf.reshape(bboxes, bboxes_shape)
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, image_shape)





tf.parse_single_sequence_example(  serialized,  context_features=None,  sequence_features=None,  example_name=None,  name=None)
    Parses a single serialized SequenceExample proto given in serialized.

    This op parses a serialized sequence example into a tuple of dictionaries mapping keys to Tensor and SparseTensor objects respectively. The first dictionary contains mappings for keys appearing in context_features, and the second dictionary contains mappings for keys appearing in sequence_features.
    At least one of context_features and sequence_features must be provided and non-empty.

    The context_features keys are associated with a SequenceExample as a whole, independent of time / frame. In contrast, the sequence_features keys provide a way to access variable-length data within the FeatureList section of the SequenceExample proto. While the shapes of context_features values are fixed with respect to frame, the frame dimension (the first dimension) of sequence_features values may vary between SequenceExample protos, and even between feature_list keys within the same SequenceExample.

    context_features contains VarLenFeature and FixedLenFeature objects.
    sequence_features contains VarLenFeature and FixedLenSequenceFeature


tf.parse_tensor(  serialized,  out_type,  name=None)
    Transforms a serialized tf.TensorProto proto into a Tensor.
    Args:
    Serialized: A Tensor of type string. A scalar string containing a serialized TensorProto proto.
    out_type: A tf.DType. The type of the serialized tensor. The provided type must match the type of the serialized tensor and no implicit conversion will take place.
    name: A name for the operation (optional).
    Returns:A Tensor of type out_type. A Tensor of type out_type.
    
    
    
##Example of TFRecords and Example.proto and Feature.proto
#CHeck - http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
#in tensflow_many_examples
#train data - train.zip - https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data   
tf.train.Feature(BytesList bytes_list or  FloatList float_list or Int64List int64_list )
tf.train.BytesList/FloatList/Int64List(value=[v1,v2,v3...])
tf.train.FeatureList(feature=[feature1,feature2,..])
tf.train.Features(feature={"name": Feature,...}) # Map from feature name to feature.
tf.train.FeatureLists(feature_list={ "name": FeatureList,...}) #Map from feature name to feature list.
tf.train.Example(features =Features)
tf.train.SequenceExample(context=Features, feature_lists=FeatureLists)


##tfrecord_writer.py
from random import shuffle
import glob
import cv2  #image read/show , cv2.imread(), .imshow(), .imwrite()
import tensorflow as tf
import numpy as np
import sys

shuffle_data = True  # shuffle the addresses before saving
cat_dog_train_path = 'Cat vs Dog/train/*.jpg'
# read addresses and labels from the 'train' folder
addrs = glob.glob(cat_dog_train_path)
labels = [0 if 'cat' in addr else 1 for addr in addrs]  # 0 = Cat, 1 = Dog
# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

# Divide the data into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.6 * len(addrs))]
train_labels = labels[0:int(0.6 * len(labels))]
val_addrs = addrs[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
val_labels = labels[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
test_addrs = addrs[int(0.8 * len(addrs)):]
test_labels = labels[int(0.8 * len(labels)):]

def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


train_filename = 'train.tfrecords'  # address to save the TFRecords file
# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print 'Train data: {}/{}'.format(i, len(train_addrs))
        sys.stdout.flush()
    # Load the image
    img = load_image(train_addrs[i])
    label = train_labels[i]
    # Create a feature
    feature = {'train/label': _int64_feature(label),
               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}  #tf,compat for Py2/Py3 compatibility layer 
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()    
    
    
##tfrecord_reader.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
data_path = 'train.tfrecords'  # address to save the hdf5 file

with tf.Session() as sess:
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1) #A queue with the output strings
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) #read takes a queue, Returns the next record (key, value pair) produced by a reader.
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['train/image'], tf.float32) #Reinterpret the bytes of a string as a vector of numbers.

    # Cast label data into int32
    label = tf.cast(features['train/label'], tf.int32)
    # Reshape image data into the original shape
    image = tf.reshape(image, [224, 224, 3])

    # Any preprocessing here ...

    # Creates batches by randomly shuffling tensors
    #Returns:A list or dictionary of tensors with the types as tensors.
    images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)

    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator() #A coordinator for threads.
    threads = tf.train.start_queue_runners(coord=coord)  #Starts all queue runners collected in the graph.
    
    for batch_index in range(5):
        img, lbl = sess.run([images, labels])
        img = img.astype(np.uint8)
        for j in range(6):
            plt.subplot(2, 3, j + 1)
            plt.imshow(img[j, ...])
            plt.title('cat' if lbl[j] == 0 else 'dog')
        plt.show()
    # Stop the threads
    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)
    sess.close()

 
    
    
    
    
    
    
    


###tf.estimator - Class Estimator

class Estimator(model_fn,  model_dir=None,  config=None,  params=None)
    Base class for all Classifiers, Regression under tf.estimator module 

    The Estimator object wraps a model which is specified by a model_fn, 
    which, given inputs and a number of other parameters, 
    returns the ops necessary to perform training, evaluation, or predictions.

    All outputs (checkpoints, event files, etc.) are written to model_dir, 
    If model_dir is not set, a temporary directory is used.

    The config argument can be passed RunConfig object 
    It is passed on to the model_fn, if the model_fn has a parameter named "config" 
    (and input functions in the same manner). 
    If the config parameter is not passed, it is instantiated by the Estimator. 
    Estimator makes config available to the model 

    The params argument contains hyperparameters. 
    It is passed to the model_fn, if the model_fn has a parameter named "params", 
    and to the input functions in the same manner. 

    None of Estimator's methods can be overridden in subclasses 
    Subclasses should use model_fn to configure the base class, 
    and may add methods implementing specialized functionality.

    ##Attributes    
    config
    model_dir
    params
    __init__(  model_fn,  model_dir=None,  config=None,  params=None)
        Constructs an Estimator instance.
        model_fn: Model function. takes following Args in following orders:
            features: This is the first item returned from the input_fn passed to train, evaluate, and predict. 
                      This should be a single Tensor or dict of same.
            labels: This is the second item returned from the input_fn passed to train, evaluate, and predict. 
                    This should be a single Tensor or dict of same (for multi-head models). 
                    If mode is ModeKeys.PREDICT, labels=None will be passed. 
                    If the model_fn's signature does not accept mode, 
                    the model_fn must still be able to handle labels=None.
            mode: Optional. Specifies if this training, evaluation or prediction vis tf.ModeKeys.
            params: Optional dict of hyperparameters. 
                    Will receive what is passed to Estimator in params parameter. 
                    This allows to configure Estimators from hyper parameter tuning.
            config: Optional configuration object. Will receive what is passed to Estimator in config parameter, or the default config. Allows updating things in your model_fn based on configuration such as num_ps_replicas, or model_dir.
            Returns: EstimatorSpec
        model_dir: Directory to save model parameters, graph and etc. This can also be used to load checkpoints from the directory into a estimator to continue training a previously saved model. If None, the model_dir in config will be used if set. If both are set, they must be same. If both are None, a temporary directory will be used.
        config: Configuration object.
        params: dict of hyper parameters that will be passed into model_fn. 
                Keys are names of parameters, values are basic python types.                
    evaluate(input_fn,steps=None,hooks=None,checkpoint_path=None,name=None)
        Evaluates the model given evaluation data input_fn.
        For each step, calls input_fn, which returns one batch of data. 
        Evaluates until: - steps batches are processed, 
        or - input_fn raises an end-of-input exception (OutOfRangeError or StopIteration).
        Returns:
        A dict containing the evaluation metrics specified in model_fn keyed by name, as well as an entry global_step 
    export_savedmodel(export_dir_base,serving_input_receiver_fn,assets_extra=None,
            as_text=False,checkpoint_path=None)
        Exports inference graph as a SavedModel into given dir.    
    predict( input_fn, predict_keys=None, hooks=None, checkpoint_path=None)
        Yields:Evaluated values of predictions tensors.
    train( input_fn, hooks=None,  steps=None, max_steps=None)
        Trains a model given training data input_fn.
        Returns:self, for chaining.
    
##Builtin Estimator - usages are exactly same 
class DNNClassifier: 
    A classifier for TensorFlow DNN models.

class DNNLinearCombinedClassifier: 
    An estimator for TensorFlow Linear and DNN joined classification models.

class DNNLinearCombinedRegressor: 
    An estimator for TensorFlow Linear and DNN joined models for regression.

class DNNRegressor: 
    A regressor for TensorFlow DNN models.

class LinearClassifier: 
    Linear classifier model.

class LinearRegressor: 
    An estimator for TensorFlow Linear regression problems.






###tf.estimator - Class DNNClassifier - Deep Neural Network 
class DNNClassifier(  hidden_units, feature_columns,  model_dir=None,  
                n_classes=2,  weight_column=None,
                label_vocabulary=None,  optimizer='Adagrad',  
                activation_fn=tf.nn.relu,  dropout=None,  
                input_layer_partitioner=None,   config=None)
        A classifier for TensorFlow DNN models.
        Loss is calculated by using softmax cross entropy.
        sparse_feature_a = sparse_column_with_hash_bucket(...)
        sparse_feature_b = sparse_column_with_hash_bucket(...)

        sparse_feature_a_emb = embedding_column(sparse_id_column=sparse_feature_a,      ...)
        sparse_feature_b_emb = embedding_column(sparse_id_column=sparse_feature_b, ...)

        estimator = DNNClassifier(
            feature_columns=[sparse_feature_a_emb, sparse_feature_b_emb],
            hidden_units=[1024, 512, 256])

        # Or estimator using the ProximalAdagradOptimizer optimizer with
        # regularization.
        estimator = DNNClassifier(
            feature_columns=[sparse_feature_a_emb, sparse_feature_b_emb],
            hidden_units=[1024, 512, 256],
            optimizer=tf.train.ProximalAdagradOptimizer(
              learning_rate=0.1,
              l1_regularization_strength=0.001
            ))

        # Input builders
        def input_fn_train: # returns x, y
          pass
        estimator.train(input_fn=input_fn_train, steps=100)

        def input_fn_eval: # returns x, y
          pass
        metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)
        def input_fn_predict: # returns x, None
          pass
        predictions = estimator.predict(input_fn=input_fn_predict)

        Input of train and evaluate should have following features, 
        otherwise there will be a KeyError:
            if weight_column is not None, a feature with key=weight_column 
            whose value is a Tensor.
            for each column in feature_columns:
                if column is a _CategoricalColumn, 
                    a feature with key=column.name whose value is a SparseTensor.
                if column is a _WeightedCategoricalColumn, 
                    two features: the first with key the id column name, 
                    the second with key the weight column name. 
                    Both features' value must be a SparseTensor.
                if column is a _DenseColumn, 
                    a feature with key=column.name whose value is a Tensor.

        config
        model_dir
        params
    __init__(  hidden_units, feature_columns,  model_dir=None,  n_classes=2,  weight_column=None,
        label_vocabulary=None,  optimizer='Adagrad',  activation_fn=tf.nn.relu,  dropout=None,  input_layer_partitioner=None,   config=None)
        Initializes a DNNClassifier instance.
            hidden_units: Iterable of number hidden units per layer. 
                All layers are fully connected. Ex. [64, 32] means first layer has 64 nodes and second one has 32.
            feature_columns: An iterable containing all the feature columns used by the model. 
                All items in the set should be instances of classes derived from _FeatureColumn.
            model_dir: Directory to save model parameters, graph and etc. 
                This can also be used to load checkpoints from the directory into a estimator to continue training a previously saved model.
            n_classes: Number of label classes. Defaults to 2, namely binary classification. 
                Must be > 1.
            weight_column: A string or a _NumericColumn created by tf.feature_column.numeric_column 
                defining feature column representing weights. 
                It is used to down weight or boost examples during training. It will be multiplied by the loss of the example. If it is a string, it is used as a key to fetch weight tensor from the features. If it is a _NumericColumn, raw tensor is fetched by key weight_column.key, then weight_column.normalizer_fn is applied on it to get weight tensor.
            label_vocabulary: A list of strings represents possible label values. 
                If given, labels must be string type and have any value in label_vocabulary. 
                If it is not given, that means labels are already encoded as integer or float within [0, 1] for n_classes=2 and encoded as integer values in {0, 1,..., n_classes-1} for n_classes>2 . Also there will be errors if vocabulary is not provided and labels are string.
            optimizer: An instance of tf.Optimizer used to train the model. 
                Defaults to Adagrad optimizer.
            activation_fn: Activation function applied to each layer. 
                If None, will use tf.nn.relu.
            dropout: When not None, the probability we will drop out a given coordinate.
            input_layer_partitioner: Optional. Partitioner for input layer. 
                Defaults to min_max_variable_partitioner with min_slice_size 64 << 20.
            config: RunConfig object to configure the runtime settings.

    evaluate(input_fn,  steps=None,  hooks=None,  checkpoint_path=None,  name=None)
        Evaluates the model given evaluation data input_fn.
        For each step, calls input_fn, which returns one batch of data. 
        Evaluates until: - steps batches are processed, 
        or - input_fn raises an end-of-input exception (OutOfRangeError or StopIteration).
            input_fn: Input function returning a tuple of: 
                features - Dictionary of string feature name to Tensor or SparseTensor. 
                labels - Tensor or dictionary of Tensor with labels.
            steps: Number of steps for which to evaluate model. 
                If None, evaluates until input_fn raises an end-of-input exception.
            hooks: List of SessionRunHook subclass instances. 
                Used for callbacks inside the evaluation call.
            checkpoint_path: Path of a specific checkpoint to evaluate. 
                If None, the latest checkpoint in model_dir is used.
            name: Name of the evaluation if user needs to run multiple evaluations on different data sets, 
                such as on training data vs test data. Metrics for different evaluations are saved in separate folders, and appear separately in tensorboard.
        Returns:
            A dict containing the evaluation metrics specified in model_fn keyed by name, 
            as well as an entry global_step which contains the value of the global step for which this evaluation was performed.

    export_savedmodel(  export_dir_base,  serving_input_receiver_fn,  assets_extra=None, as_text=False,  checkpoint_path=None)
        Exports inference graph as a SavedModel into given dir.
    predict(  input_fn,  predict_keys=None,  hooks=None,  checkpoint_path=None)
        Returns predictions for given features.
            input_fn: Input function returning features which is a dictionary of string feature name to Tensor or SparseTensor. If it returns a tuple, first item is extracted as features. Prediction continues until input_fn raises an end-of-input exception (OutOfRangeError or StopIteration).
            predict_keys: list of str, name of the keys to predict. 
                It is used if the EstimatorSpec.predictions is a dict. 
                If predict_keys is used then rest of the predictions will be filtered from the dictionary. 
                If None, returns all.
            hooks: List of SessionRunHook subclass instances. 
                Used for callbacks inside the prediction call.
            checkpoint_path: Path of a specific checkpoint to predict. 
                If None, the latest checkpoint in model_dir is used.
        Yields:Evaluated values of predictions tensors.
    train( input_fn,  hooks=None,  steps=None,  max_steps=None)
        Trains a model given training data input_fn.
            input_fn: Input function returning a tuple of: 
                features - Tensor or dictionary of string feature name to Tensor. 
                labels - Tensor or dictionary of Tensor with labels.
            hooks: List of SessionRunHook subclass instances. 
                Used for callbacks inside the training loop.
            steps: Number of steps for which to train model. 
            If None, train forever or train until input_fn generates the OutOfRange or StopIteration error. 'steps' works incrementally. If you call two times train(steps=10) then training occurs in total 20 steps. If OutOfRange or StopIteration error occurs in the middle, training stops before 20 steps. If you don't want to have incremental behavior please set max_steps instead. If set, max_steps must be None.
            max_steps: Number of total steps for which to train model. 
                If None, train forever or train until input_fn generates the OutOfRange or StopIteration error. 
                If set, steps must be None. If OutOfRange or StopIteration error occurs in the middle, training stops before max_steps steps.
            Two calls to train(steps=100) means 200 training iterations. 
            On the other hand, two calls to train(max_steps=100) means 
            that the second call will not do any iteration since first call did all 100 steps.
        Returns:self, for chaining.


###Tensorflow - Dataloading -tf.contrib.learn.datasets.base.
Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


def load_csv_with_header(filename,target_dtype,features_dtype,target_column=-1):
  """Load dataset from CSV file with a header row."""
  ...
  return Dataset(data=data, target=target)


def load_csv_without_header(filename,target_dtype,features_dtype,target_column=-1):
  """Load dataset from CSV file without a header row."""
  ...
  return Dataset(data=data, target=target)


def shrink_csv(filename, ratio):
  """Create a smaller dataset of only 1/ratio of original data.
  write to 
  filename_small = filename.replace('.', '_small.')
  """



def load_iris(data_path=None):
  """Load Iris dataset.
  Args:   data_path: string, path to iris dataset, default ./data/iris.csv
  Returns:   Dataset object containing data in-memory.
  """
 

def load_boston(data_path=None):
  """Load Boston housing dataset.
  Args:   data_path: string, path to boston dataset, default ./data/boston_house_prices.csv
  Returns:    Dataset object containing data in-memory.
  """
 
 
 
 
 
 
###Tensorflow- Creating checkpoints - Saves and restores variables.
#A training program that saves regularly looks like:

...
# Create a saver.
saver = tf.train.Saver()
# Launch the graph and train, saving the model every 1,000 steps.
sess = tf.Session()
for step in xrange(1000000):
    sess.run(..training_op..)
    if step % 1000 == 0:
        # Append the step number to the checkpoint name:
        saver.save(sess, 'my-model', global_step=step)

        
#Only few variables can be saved 
v1 = tf.Variable(..., name='v1')
v2 = tf.Variable(..., name='v2')

# Pass the variables as a dict:
saver = tf.train.Saver({'v1': v1, 'v2': v2})

# Or pass them as a list.
saver = tf.train.Saver([v1, v2])
# Passing a list is equivalent to passing a dict with the variable op names
# as keys:
saver = tf.train.Saver({v.op.name: v for v in [v1, v2]})

  
###***TensorFlow - Tensorboard - Low Level API 
#tf.estimator auto manages tensorboard via model_dir 

#Tensorflow summaries are essentially logs. 
#to write logs we need a log writer

#Add below lines after sess.run(tf.global_variables_initializer())
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter("C:/tmp/tensorflow/")
writer.add_graph(sess.graph)
#Whereas sess is 
sess = tf.Session()


#start tensorboard.
$ tensorboard --logdir=C:/tmp/tensorflow/ --port 6006
#check in firefox

##Make Your Tensorflow Graph Readable
#To clean up the visualization of model in tensorboard 
#add the scope of variables and a name for placeholders and variables. 

#By default, only the top of this hierarchy is shown

import tensorflow as tf

with tf.name_scope('hidden') as scope:
  a = tf.constant(5, name='alpha')
  W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights')
  b = tf.Variable(tf.zeros([1]), name='biases')

#This results in the following three op names:
#By default, the visualization will collapse all three into a node labeled hidden
    hidden/alpha
    hidden/weights
    hidden/biases



##Log Dynamic Values - Serializing the data

#TensorBoard operates by reading TensorFlow events files, 
#which contain summary data that are generated when running TensorFlow. 

#Class for writing Summaries
tf.summary.FileWriter
tf.summary.FileWriterCache
#Utilities
tf.summary.get_summary_description(node_def)
#Summary Ops
tf.summary.tensor_summary( name, tensor, summary_description=None,collections=None, summary_metadata=None, family=None,  display_name=None)
tf.summary.scalar( name,  tensor,  collections=None,  family=None)
    Outputs a Summary protocol buffer containing a single scalar value.
tf.summary.histogram( name, values, collections=None, family=None)
    Adding a histogram summary makes it possible to visualize data's distribution in TensorBoard. 
tf.summary.audio( name,  tensor,  sample_rate, max_outputs=3, collections=None, family=None)
    The audio is built from tensor which must be 3-D with shape [batch_size, frames, channels] or 2-D with shape [batch_size, frames].
tf.summary.image(name,  tensor,  max_outputs=3,  collections=None,  family=None)
    The images are built from tensor which must be 4-D with shape [batch_size, height, width, channels] and where channels can be:1,3,4 for Grayscale, RGB, RGBA
tf.summary.merge(inputs,  collections=None, name=None)
tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)



#Example 
tf.summary.scalar("accuracy", accuracy)
tf.summary.histogram("weights", w)




##Instead of executing every summary operation individually 
##we can merge them all together into a single merged summary operation.

summ = tf.summary.merge_all()

#then execute this operation together with train operation inside train cycle 
#and write the values into  log file 
for i in range(2001):
    [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y: batch[1]})
    writer.add_summary(s, i)


    
###Tensorflow - TensorBoard Histogram Dashboard 
#displays how the distribution of some Tensor in TensorFlow graph has changed over time. 
#It does this by showing many histograms visualizations of  tensor at different points in time.


import tensorflow as tf

k = tf.placeholder(tf.float32)

# Make a normal distribution, with a shifting mean
mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
# Record that distribution into a histogram summary
tf.summary.histogram("normal/moving_mean", mean_moving_normal)

# Setup a session and summary writer
sess = tf.Session()
writer = tf.summary.FileWriter("C:/tmp/tensorflow/histogram_example")

summaries = tf.summary.merge_all()

# Setup a loop and write the summaries to disk
N = 400
for step in range(N):
  k_val = step/float(N)
  summ = sess.run(summaries, feed_dict={k: k_val})
  writer.add_summary(summ, global_step=step)

#Load tensor board 
$ tensorboard --logdir=C:/tmp/tensorflow/histogram_example

#http://localhost:6006/ and navigate to the Histogram Dashboard

#There is a control on the left of the dashboard 
#that allows you to toggle the histogram mode from "offset" to "overlay":

#In "offset" mode, the visualization rotates 45 degrees,


##Multimodal Distributions

import tensorflow as tf

k = tf.placeholder(tf.float32)

# Make a normal distribution, with a shifting mean
mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
# Record that distribution into a histogram summary
tf.summary.histogram("normal/moving_mean", mean_moving_normal)

# Make a normal distribution with shrinking variance
variance_shrinking_normal = tf.random_normal(shape=[1000], mean=0, stddev=1-(k))
# Record that distribution too
tf.summary.histogram("normal/shrinking_variance", variance_shrinking_normal)

# Let's combine both of those distributions into one dataset
normal_combined = tf.concat([mean_moving_normal, variance_shrinking_normal], 0)
# We add another histogram summary to record the combined distribution
tf.summary.histogram("normal/bimodal", normal_combined)

summaries = tf.summary.merge_all()

# Setup a session and summary writer
sess = tf.Session()
writer = tf.summary.FileWriter("/tmp/histogram_example")

# Setup a loop and write the summaries to disk
N = 400
for step in range(N):
  k_val = step/float(N)
  summ = sess.run(summaries, feed_dict={k: k_val})
  writer.add_summary(summ, global_step=step)


##Some more distributions

import tensorflow as tf

k = tf.placeholder(tf.float32)

# Make a normal distribution, with a shifting mean
mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
# Record that distribution into a histogram summary
tf.summary.histogram("normal/moving_mean", mean_moving_normal)

# Make a normal distribution with shrinking variance
variance_shrinking_normal = tf.random_normal(shape=[1000], mean=0, stddev=1-(k))
# Record that distribution too
tf.summary.histogram("normal/shrinking_variance", variance_shrinking_normal)

# Let's combine both of those distributions into one dataset
normal_combined = tf.concat([mean_moving_normal, variance_shrinking_normal], 0)
# We add another histogram summary to record the combined distribution
tf.summary.histogram("normal/bimodal", normal_combined)

# Add a gamma distribution
gamma = tf.random_gamma(shape=[1000], alpha=k)
tf.summary.histogram("gamma", gamma)

# And a poisson distribution
poisson = tf.random_poisson(shape=[1000], lam=k)
tf.summary.histogram("poisson", poisson)

# And a uniform distribution
uniform = tf.random_uniform(shape=[1000], maxval=k*10)
tf.summary.histogram("uniform", uniform)

# Finally, combine everything together!
all_distributions = [mean_moving_normal, variance_shrinking_normal,
                     gamma, poisson, uniform]
all_combined = tf.concat(all_distributions, 0)
tf.summary.histogram("all_combined", all_combined)

summaries = tf.summary.merge_all()

# Setup a session and summary writer
sess = tf.Session()
writer = tf.summary.FileWriter("/tmp/histogram_example")

# Setup a loop and write the summaries to disk
N = 400
for step in range(N):
  k_val = step/float(N)
  summ = sess.run(summaries, feed_dict={k: k_val})
  writer.add_summary(summ, global_step=step) 
  
  

###*** Tensflow - tf.estimator with TensorBoard for MNIST data 

import os
import os.path
import shutil
import tensorflow as tf

LOGDIR = "C:/tmp/tensorflow/"
LABELS = os.path.join(os.getcwd(), "labels_1024.tsv")
SPRITES = os.path.join(os.getcwd(), "sprite_1024.png")
## MNIST EMBEDDINGS ##
mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + "data", one_hot=True)
## Get a sprite and labels file for the embedding projector ##

if not (os.path.isfile(LABELS) and os.path.isfile(SPRITES)):
  print("Necessary data files were not found. Run this command from inside the "
    "repo provided at "
    "https://github.com/dandelionmane/tf-dev-summit-tensorboard-tutorial.")
  exit(1)


# shutil.copyfile(LABELS, os.path.join(LOGDIR, LABELS))
# shutil.copyfile(SPRITES, os.path.join(LOGDIR, SPRITES))


def conv_layer(input, size_in, size_out, name="conv"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def fc_layer(input, size_in, size_out, name="fc"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    act = tf.matmul(input, w) + b
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return act


def mnist_model(learning_rate, use_two_fc, use_two_conv, hparam):
  tf.reset_default_graph()
  sess = tf.Session()

  # Setup placeholders, and reshape the data
  x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
  x_image = tf.reshape(x, [-1, 28, 28, 1])
  tf.summary.image('input', x_image, 3)
  y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

  if use_two_conv:
    conv1 = conv_layer(x_image, 1, 32, "conv1")
    conv_out = conv_layer(conv1, 32, 64, "conv2")
  else:
    conv1 = conv_layer(x_image, 1, 64, "conv")
    conv_out = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

  flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])


  if use_two_fc:
    fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, "fc1")
    relu = tf.nn.relu(fc1)
    embedding_input = relu
    tf.summary.histogram("fc1/relu", relu)
    embedding_size = 1024
    logits = fc_layer(fc1, 1024, 10, "fc2")
  else:
    embedding_input = flattened
    embedding_size = 7*7*64
    logits = fc_layer(flattened, 7*7*64, 10, "fc")

  with tf.name_scope("xent"):
    xent = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y), name="xent")
    tf.summary.scalar("xent", xent)

  with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

  with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

  summ = tf.summary.merge_all()


  embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
  assignment = embedding.assign(embedding_input)
  saver = tf.train.Saver()

  sess.run(tf.global_variables_initializer())
  writer = tf.summary.FileWriter(LOGDIR + hparam)
  writer.add_graph(sess.graph)

  config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
  embedding_config = config.embeddings.add()
  embedding_config.tensor_name = embedding.name
  embedding_config.sprite.image_path = SPRITES
  embedding_config.metadata_path = LABELS
  # Specify the width and height of a single thumbnail.
  embedding_config.sprite.single_image_dim.extend([28, 28])
  tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

  for i in range(2001):
    batch = mnist.train.next_batch(100)
    if i % 5 == 0:
      [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y: batch[1]})
      writer.add_summary(s, i)
    if i % 500 == 0:
      sess.run(assignment, feed_dict={x: mnist.test.images[:1024], y: mnist.test.labels[:1024]})
      saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
    sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
  conv_param = "conv=2" if use_two_conv else "conv=1"
  fc_param = "fc=2" if use_two_fc else "fc=1"
  return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)

def main():
  # You can try adding some more learning rates
  for learning_rate in [1E-3, 1E-4]:

    # Include "False" as a value to try different model architectures
    for use_two_fc in [True]:
      for use_two_conv in [False, True]:
        # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2")
        hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
        print('Starting run for %s' % hparam)

        # Actually run with the new settings
        mnist_model(learning_rate, use_two_fc, use_two_conv, hparam)
  print('Done training!')
  print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)
  print('Running on mac? If you want to get rid of the dialogue asking to give '
        'network permissions to TensorBoard, you can provide this flag: '
        '--host=localhost')

if __name__ == '__main__':
  main()





###*** Tensflow - MNIST example - Another way to implement

import tensorflow as tf

# reset everything to rerun in jupyter
tf.reset_default_graph()

# config
batch_size = 100
learning_rate = 0.5
training_epochs = 5
logs_path = "c:/tmp/tensorflow/mnist/2"

# load mnist data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# input images
with tf.name_scope('input'):
    # None -> batch size can be any size, 784 -> flattened mnist image
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input") 
    # target 10 output classes
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

# model parameters will change during training so we use tf.Variable
with tf.name_scope("weights"):
    W = tf.Variable(tf.zeros([784, 10]))

# bias
with tf.name_scope("biases"):
    b = tf.Variable(tf.zeros([10]))

# implement model
with tf.name_scope("softmax"):
    # y is our prediction
    y = tf.nn.softmax(tf.matmul(x,W) + b)

# specify cost function
with tf.name_scope('cross_entropy'):
    # this is our cost
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# specify optimizer
with tf.name_scope('train'):
    # optimizer is an "operation" which we can execute in a session
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope('Accuracy'):
    # Accuracy
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
# create a summary for our cost and accuracy
tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)

# merge all summaries into a single "operation" which we can execute in a session 
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    # variables need to be initialized before we can use them
    sess.run(tf.initialize_all_variables())

    # create log writer object
    writer = tf.summary.FileWriter(logs_path, sess.graph)
        
    # perform training cycles
    for epoch in range(training_epochs):
        
        # number of batches in one epoch
        batch_count = int(mnist.train.num_examples/batch_size)
        
        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            
            # perform the operations we defined earlier on batch
            _, summary = sess.run([train_op, summary_op], feed_dict={x: batch_x, y_: batch_y})
            
            # write log
            writer.add_summary(summary, epoch * batch_count + i)
            
        if epoch % 5 == 0: 
            print "Epoch: ", epoch 
    print "Accuracy: ", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print "done"

    
###*** Tensflow - MNIST - Another way to implement 
"""A simple MNIST classifier which displays summaries in TensorBoard.

This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.

It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)

  sess = tf.InteractiveSession()
  # Create a multilayer model.

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

  # We can't initialize these variables to 0 - the network will get stuck.
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations

  hidden1 = nn_layer(x, 784, 500, 'layer1')

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

  # Do not apply softmax activation yet, see below.
  y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

  with tf.name_scope('cross_entropy'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the
    # raw outputs of the nn_layer above, and then average across
    # the batch.
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to
  # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  tf.global_variables_initializer().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/input_data'),
      help='Directory for storing input data')
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/mnist_with_summaries'),
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  


  
  
  
  
  
###pandas - Quick pandas DF Reference 
#Using .loc and .iloc
df.loc[row_indexer,column_indexer]
    Any of the axes accessors may be the null slice ':'
    Axes left out of the specification are assumed to be ':' 
    (e.g. df.loc['a'] is equiv to p.loc['a', :])

.loc is primarily label based, but may also be used with a boolean array
    A single label, e.g. 5 or 'a', 
    (note that 5 is interpreted as a label of the index. This use is not an integer position along the index)
    A list or array of labels ['a', 'b', 'c']
    A slice object with labels 'a':'f', (note that contrary to usual python slices, both the start and the stop are included!)
    A boolean array
    A callable function with one argument (the calling Series, DataFrame or Panel) and that returns valid output for indexing (one of the above)

.iloc is primarily integer position based (from 0 to length-1 of the axis), but may also be used with a boolean array. 
    An integer e.g. 5
    A list or array of integers [4, 3, 0]
    A slice object with ints 1:7
    A boolean array
    A callable function with one argument (the calling Series, DataFrame or Panel) and that returns valid output for indexing (one of the above)

#Example of loc 
df1 = pd.DataFrame(np.random.randn(6,4),index=list('abcdef'),columns=list('ABCD'))
df1.loc[['a', 'b', 'd'], :]
df1.loc['d':, 'A':'C']
df1.loc['a'] #Series
df1.loc[:, df1.loc['a'] > 0]
##Example of iloc 
df1 = pd.DataFrame(np.random.randn(6,4),index=list(range(0,12,2)),columns=list(range(0,8,2)))
#slice starts with 0 to n-1 
#note these are not labels, but index of row or column   
df1.iloc[:3] #first three rows 
df1.iloc[1:5, 2:4]
df1.iloc[[1, 3, 5], [1, 3]]
df1.iloc[1:3, :]
df1.iloc[:, 1:3]
df1.iloc[1] #Series at index=1 
#with callable 
df1 = pd.DataFrame(np.random.randn(6, 4),index=list('abcdef'),columns=list('ABCD'))
df1.loc[lambda df: df.A > 0, :]
df1.loc[:, lambda df: ['A', 'B']]
df1.iloc[:, lambda df: [0, 1]]
df1[lambda df: df.columns[0]]
 
#ix, loc, iloc 
#.ix can decide to index positionally OR via labels depending on the data type of the index.
dfd = pd.DataFrame({'A': [1, 2, 3],'B': [4, 5, 6]},index=list('abc'))
 
#all are equivalents 
dfd.ix[[0, 2], 'A']
dfd.loc[dfd.index[[0, 2]], 'A']
dfd.iloc[[0, 2], dfd.columns.get_loc('A')]
dfd.iloc[[0, 2], dfd.columns.get_indexer(['A', 'B'])]


#[] accessing
Series 	    series[label] 	scalar value
DataFrame 	frame[colname] 	Series corresponding to colname, then use series.values to get values 
#pass a list of columns to [] to select columns in that order
df[['A', 'B']]
#access an index on a Series, column on a DataFrame directly as an attribute:
df.A
#index column 
df.index 
#or 
df['index']

#Update 
dfa.A = list(range(len(dfa.index)))     # update ok if A already exists
dfa['A'] = list(range(len(dfa.index)))  # use this form to create a new column

#With DataFrame, slicing inside of [] slices the rows.
df[:3]   #means df[:3, :]
df[::-1]

# .loc/[] operations can enlarge when setting a non-existant key for that axis.
dfi = pd.DataFrame(np.arange(6).reshape(3,2), columns=['A','B'])
dfi.loc[:,'C'] = dfi.loc[:,'A'] #creation of new column C 
dfi.loc[3] = 5   #updating row index 3 to 5 

##Fast scalar(single) value getting and setting
#at provides label based scalar lookups, 
#iat provides integer based lookups
dates = pd.date_range('1/1/2000', periods=8)
df = pd.DataFrame(np.random.randn(8, 4), index=dates, columns=['A', 'B', 'C', 'D'])
df.at[dates[5], 'A']
df.iat[3, 0]
df.at[dates[5], 'E'] = 7
df.iat[3, 0] = 7

##select method
#select takes a function which take each element of labels along axis and returns a boolean
df.select(lambda x: x == 'A', axis=1)

##lookup method 
#to extract a set of values(single values) given a sequence of row labels and column labels, 
dflookup = pd.DataFrame(np.random.rand(20,4), columns = ['A','B','C','D'])
dflookup.lookup(list(range(0,10,2)), ['B','C','A','B','D'])
#array([ 0.3506,  0.4779,  0.4825,  0.9197,  0.5019])

##DF - Index Object 
index = pd.Index(list(range(5)), name='rows')
columns = pd.Index(['A', 'B', 'C'], name='cols')
df = pd.DataFrame(np.random.randn(5, 3), index=index, columns=columns)
a = pd.Index(['c', 'b', 'a'])
b = pd.Index(['c', 'e', 'd'])
a | b  #Index(['a', 'b', 'c', 'd', 'e'], dtype='object')
a & b  #Index(['c'], dtype='object')
a.difference(b) #Index(['a', 'b'], dtype='object')
idx1 = pd.Index([1, 2, 3, 4])
idx2 = pd.Index([2, 3, 4, 5])
idx1.symmetric_difference(idx2)
idx1 ^ idx2
#set/reset Index 
data
Out[312]: 
     a    b  c    d
0  bar  one  z  1.0
1  bar  two  y  2.0
2  foo  one  x  3.0
3  foo  two  w  4.0

In [313]: indexed1 = data.set_index('c')

In [314]: indexed1
Out[314]: 
     a    b    d
c               
z  bar  one  1.0
y  bar  two  2.0
x  foo  one  3.0
w  foo  two  4.0

In [315]: indexed2 = data.set_index(['a', 'b'])

In [316]: indexed2
Out[316]: 
         c    d
a   b          
bar one  z  1.0
    two  y  2.0
foo one  x  3.0
    two  w  4.0


In [324]: indexed2.reset_index()
Out[324]: 
     a    b  c    d
0  bar  one  z  1.0
1  bar  two  y  2.0
2  foo  one  x  3.0
3  foo  two  w  4.0

##DF - Series 
In [271]: s = pd.Series([1,2,3], index=['a','b','c'])

In [272]: s.get('a')               # equivalent to s['a']
Out[272]: 1

In [273]: s.get('x', default=-1)
Out[273]: -1

##Duplicate Data
#duplicated returns a boolean vector whose length is the number of rows, 
    #and which indicates whether a row is duplicated.
#drop_duplicates removes duplicate rows.
1.keep='first' (default): mark / drop duplicates except for the first occurrence.
2.keep='last': mark / drop duplicates except for the last occurrence.
3.keep=False: mark / drop all duplicates.
#Example 
df2 = pd.DataFrame({'a': ['one', 'one', 'two', 'two', 'two', 'three', 'four'],
                            'b': ['x', 'y', 'x', 'y', 'x', 'x', 'x'],
                            'c': np.random.randn(7)})
        
df2.duplicated('a')
Out[257]: 
0    False
1     True
2    False
3     True
4     True
5    False
6    False
dtype: bool
df2.drop_duplicates('a')
Out[260]: 
       a  b         c
0    one  x -1.067137
2    two  x -0.211056
5  three  x -1.964475
6   four  x  1.298329
df2.duplicated(['a', 'b'])
Out[263]: 
0    False
1    False
2    False
3    False
4     True
5    False
6    False
dtype: bool

df2.drop_duplicates(['a', 'b'])
Out[264]: 
       a  b         c
0    one  x -1.067137
1    one  y  0.309500
2    two  x -0.211056
3    two  y -1.842023
5  three  x -1.964475
6   four  x  1.298329

##query() 
df = pd.DataFrame(np.random.randint(n, size=(n, 3)), columns=list('abc'))
#below are equivalent
df.query('(a < b) & (b < c)')
df[(df.a < df.b) & (df.b < df.c)]
df.query('a < b and b < c')
#below are equivalent
df.query('a in b')
df[df.a.isin(df.b)]
#below are equivalent
df.query('a not in b')
df[~df.a.isin(df.b)]
#below are equivalent
df.query('a in b and c < d')
df[df.b.isin(df.a) & (df.c < df.d)]

#Comparing a list of values to a column using ==/!= works similarly to in/not in
df.query('b == ["a", "b", "c"]')
df[df.b.isin(["a", "b", "c"])]
df.query('c == [1, 2]')

df = pd.DataFrame(np.random.rand(n, 3), columns=list('abc'))
df['bools'] = np.random.rand(len(df)) > 0.5
df.query('~bools')
df.query('not bools')
df.query('not bools') == df[~df.bools]
#below are equivalent
shorter = df.query('a < b < c and (not bools) or bools > 2')
longer = df[(df.a < df.b) & (df.b < df.c) & (~df.bools) | (df.bools > 2)]

#DF Main attributes 
T 	    Transpose index and columns
axes 	Return a list with the row axis labels and column axis labels as the only members.
ndim 	Number of axes / array dimensions
shape 	Return a tuple representing the dimensionality of the DataFrame.
size 	number of elements in the NDFrame
values 	Numpy representation of NDFrame
columns Column labels
index   Index 
data    underlying data 
#Series Main attributes
T 	    return the transpose, which is by definition self
axes 	Return a list of the row axis labels
data 	return the data pointer of the underlying data
ndim 	return the number of dimensions of the underlying data,
shape 	return a tuple of the shape of the underlying data
size 	return the number of elements in the underlying data
values 	Return Series as ndarray or ndarray-like

##Category series 
s = pd.Series(["a","b","c","a"], dtype="category")
Out[2]: 
0    a
1    b
2    c
3    a
dtype: category
Categories (3, object): [a, b, c]

df = pd.DataFrame({"A":["a","b","c","a"]})
df["B"] = df["A"].astype('category')
Out[5]: 
   A  B
0  a  a
1  b  b
2  c  c
3  a  a

raw_cat = pd.Categorical(["a","b","c","a"], categories=["b","c","d"],ordered=False)
s = pd.Series(raw_cat)
Out[12]: 
0    NaN
1      b
2      c
3    NaN
dtype: category
Categories (3, object): [b, c, d]

df = pd.DataFrame({"A":["a","b","c","a"]})
df["B"] = raw_cat
Out[15]: 
   A    B
0  a  NaN
1  b    b
2  c    c
3  a  NaN

#Categorical data has a specific category dtype:
df.dtypes
Out[19]: 
A      object
B    category
dtype: object

#To get back to the original Series or numpy array, 
#use Series.astype(original_dtype) 
#or np.asarray(categorical):


#Using .describe() on categorical data will produce similar output to a Series or DataFrame of type string.
cat = pd.Categorical(["a", "c", "c", np.nan], categories=["b", "a", "c"])
df = pd.DataFrame({"cat":cat, "s":["a", "c", "c", np.nan]})
df.describe()
Out[30]: 
       cat  s
count    3  3
unique   2  2
top      c  c
freq     2  2

df["cat"].describe()
Out[31]: 
count     3
unique    2
top       c
freq      2
Name: cat, dtype: object

#Working with categories
#Categorical data has .cat property, Many methods are available on that 
s = pd.Series(["a","b","c","a"], dtype="category")
s.cat.categories
Out[33]: Index(['a', 'b', 'c'], dtype='object')
s.cat.ordered
Out[34]: False





  
### Classification Example - Iris data with tf.estimate 
1.Load CSVs containing Iris training/test data into a TensorFlow Dataset
2.Construct a neural network classifier
3.Train the model using the training data
4.Evaluate the accuracy of the model
5.Classify new samples

#Iris data contains below fields 
#Species - setosa,versicolor,virginica
Sepal Length 	Sepal Width 	Petal Length 	Petal Width 	Species
6.4,            2.8,            5.6,            2.2,            2

#Code 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
  # If the training and test sets aren't stored locally, download them.
  if not os.path.exists(IRIS_TRAINING):
    raw = urllib.request.urlopen(IRIS_TRAINING_URL).read().decode()
    with open(IRIS_TRAINING, 'w') as f:
        f.write(raw)

  if not os.path.exists(IRIS_TEST):
    raw = urllib.request.urlopen(IRIS_TEST_URL).read().decode()
    with open(IRIS_TEST, 'w') as f:
        f.write(raw)

  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)

  # Specify that all features have real-value data
  #feature col name is 'x' and shape of x ie size is 4 for each row 
  #then 'x' must be key of dict passed to numpy_input_fn 
  feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]
  
  #Traditional machine learning relies on shallow nets, composed of one input and one output layer, 
  #and at most one hidden layer in between. 
  #More than three layers (including input and output) qualifies as “deepElearning.
  # Build 3 layer DNN(Deep Nural Network) with 10, 20, 10 units respectively.
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          model_dir="C:/tmp/iris_model")
  # Define the training inputs
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(training_set.data)},
      y=np.array(training_set.target),
      num_epochs=None,
      shuffle=True)

  # Train model.
  classifier.train(input_fn=train_input_fn, steps=2000)

  # Define the test inputs
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(test_set.data)},
      y=np.array(test_set.target),
      num_epochs=1,
      shuffle=False)

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  # Classify two new flower samples.
  new_samples = np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": new_samples},
      num_epochs=1,
      shuffle=False)

  predictions = list(classifier.predict(input_fn=predict_input_fn))
  predicted_classes = [p["classes"] for p in predictions]

  print(
      "New Samples, Class Predictions:    {}\n"
      .format(predicted_classes))

if __name__ == "__main__":
    main()



    
    
    








     
###Tensorflow Regresson Example - A Neural Network Model for Boston House Values

Feature     Description
CRIM        Crime rate per capita 
ZN          Fraction of residential land zoned to permit 25,000+ sq ft lots 
INDUS       Fraction of land that is non-retail business 
NOX         Concentration of nitric oxides in parts per 10 million 
RM          Average Rooms per dwelling 
AGE         Fraction of owner-occupied residences built before 1940 
DIS         Distance to Boston-area employment centers 
TAX         Property tax rate per $10,000 
PTRATIO     Student-teacher ratio 

#the label your model will predict is MEDV, 
#the median value of owner-occupied residences in thousands of dollars.


#Code 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"


def get_input_fn(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y=pd.Series(data_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle)


def main(unused_argv):
  #Load datasets
  #Read CSV (comma-separated) file into DataFrame
  #a 2D table where columns are features and 'names' give List of column names to use
  training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS)
  test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,
                         skiprows=1, names=COLUMNS)

  # Set of 6 examples for which to predict median house values
  prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)

  # create Feature columns names (which are numeric)
  feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

  # Build 2 layer fully connected DNN with 10, 10 units respectively.
  regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                        hidden_units=[10, 10],
                                        model_dir="/tmp/boston_model")

  # Train
  regressor.train(input_fn=get_input_fn(training_set), steps=5000)

  # Evaluate loss over one epoch of test_set.
  ev = regressor.evaluate(input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
  loss_score = ev["loss"]
  print("Loss: {0:f}".format(loss_score))

  # Print out predictions over a slice of prediction_set.
  y = regressor.predict(input_fn=get_input_fn(prediction_set, num_epochs=1, shuffle=False))
  # .predict() returns an iterator of dicts; convert to a list and print
  # predictions
  predictions = list(p["predictions"] for p in itertools.islice(y, 6))
  print("Predictions: {}".format(str(predictions)))

if __name__ == "__main__":
  tf.app.run()




    















 










###Tensorflow - Distributed TensorFlow 

# Start a TensorFlow server as a single-process "cluster".
$ python
>>> import tensorflow as tf
>>> c = tf.constant("Hello, distributed TensorFlow!")
>>> server = tf.train.Server.create_local_server() #creates a single-process cluster, with an in-process server.
>>> sess = tf.Session(server.target)  # Create a session on the server.
>>> sess.run(c)
'Hello, distributed TensorFlow!'

##Create a cluster
A TensorFlow cluster comprises a one or more "jobs", 
each divided into lists of one or more "tasks". 

Each task(ie machine) is associated with a TensorFlow "server", 
which contains a "master" that can be used to create sessions, 
and a "worker" that executes operations in the graph. 


#To create a cluster, start one TensorFlow server per task in the cluster. 
#Each task typically runs on a different machine, 
#but you can run multiple tasks on the same machine (e.g. different GPU devices). 

#In each task, do the following:
    Create a tf.train.ClusterSpec that describes all of the tasks in the cluster. 
    This should be the same for each task.

    Create a tf.train.Server, passing the tf.train.ClusterSpec to the constructor, 
    and identifying the local task with a job name and task index.


##Create a tf.train.ClusterSpec to describe the cluster
#The cluster specification dictionary maps job names to lists of network addresses. 

tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
#Available tasks
/job:local/task:0
/job:local/task:1

tf.train.ClusterSpec({
    "worker": [
        "worker0.example.com:2222",
        "worker1.example.com:2222",
        "worker2.example.com:2222"
    ],
    "ps": [
        "ps0.example.com:2222",
        "ps1.example.com:2222"
    ]})
#Available tasks
/job:worker/task:0
/job:worker/task:1
/job:worker/task:2
/job:ps/task:0
/job:ps/task:1


##Create a tf.train.Server instance in each task

#to launch a cluster with two servers running on localhost:2222 and localhost:2223, 
#run the following snippets in two different processes on the local machine:
#Check future version of Tensorflow for automated processing 

# In task 0:
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=0)

# In task 1:
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=1)

##Specifying distributed devices in your model

#To place operations on a particular process, use tf.device function 

#Example - the variables are created on two tasks in the ps job
with tf.device("/job:ps/task:0"):
  weights_1 = tf.Variable(...)
  biases_1 = tf.Variable(...)

with tf.device("/job:ps/task:1"):
  weights_2 = tf.Variable(...)
  biases_2 = tf.Variable(...)

#the compute-intensive part of the model is created in the worker job
with tf.device("/job:worker/task:0"):
  input, labels = ...
  layer_1 = tf.nn.relu(tf.matmul(input, weights_1) + biases_1)
  logits = tf.nn.relu(tf.matmul(layer_1, weights_2) + biases_2)
  # ...
  train_op = ...

#Below is master service 
#TensorFlow will insert the appropriate data transfers between the jobs 
#(from ps to worker for the forward pass, and from worker to ps for applying gradients).
with tf.Session("grpc://worker0.example.com:2222") as sess: #connects to above 
  for _ in range(10000):
    sess.run(train_op)



## Replicated training

#A common training configuration, called "data parallelism," 
#involves multiple tasks in a worker job training the same model 
#on different mini-batches of data, 
#updating shared parameters hosted in one or more tasks in a ps job. 
#All tasks typically run on different machines. 

#Possible approaches include:
    In-graph replication. In this approach, the client builds a single tf.Graph 
        that contains one set of parameters (in tf.Variable nodes pinned to /job:ps); 
        and multiple copies of the compute-intensive part of the model, 
        each pinned to a different task in /job:worker.

    Between-graph replication. In this approach, 
        there is a separate client for each /job:worker task, typically in the same process as the worker task. 
        Each client builds a similar graph containing the parameters 
        (pinned to /job:ps as before using tf.train.replica_device_setter to map them deterministically to the same tasks); 
        and a single copy of the compute-intensive part of the model, 
        pinned to the local task in /job:worker.

    Asynchronous training. In this approach, each replica of the graph has an independent training loop 
        that executes without coordination. 
        It is compatible with both forms of replication above.

    Synchronous training. In this approach, all of the replicas read the same values 
        for the current parameters, compute gradients in parallel, 
        and then apply them together. 
        It is compatible with in-graph replication 
        (e.g. using gradient averaging as in the CIFAR-10 multi-GPU trainer), 
        and between-graph replication (e.g. using the tf.train.SyncReplicasOptimizer).

##Example - implementing between-graph replication and asynchronous training. 

#trainer.py
import argparse
import sys

import tensorflow as tf

FLAGS = None

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Build model...
      loss = ...
      global_step = tf.contrib.framework.get_or_create_global_step()

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=1000000)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="/tmp/train_logs",
                                           hooks=hooks) as mon_sess:
      while not mon_sess.should_stop():
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        # mon_sess.run handles AbortedError in case of preempted PS.
        mon_sess.run(train_op)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

##To start the trainer with two parameter servers and two workers, 
#use the following command line 

# On ps0.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=ps --task_index=0
# On ps1.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=ps --task_index=1
# On worker0.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=worker --task_index=0
# On worker1.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=worker --task_index=1


##Reference - Class Server
class Server 
    An in-process TensorFlow server, for use in distributed training.
        server_def
            Returns the tf.train.ServerDef for this server.
        target
            Returns the target for a tf.Session to connect to this server.
            To create a tf.Session that connects to this server, use the following snippet:
            server = tf.train.Server(...)
            with tf.Session(server.target):
              # ...
        __init__( server_or_cluster_def,job_name=None, task_index=None, protocol=None, config=None,  start=True)
            Creates a new server with the given definition.
        create_local_server(  config=None,  start=True)
            Creates a new single-process cluster running on the local host.
            containing a single task in a job called "local".
            Returns:A local tf.train.Server.
        join()
            Blocks until the server has shut down.
        start()
            Starts this server.


tf.app
    Generic entry point script.
        run( main=None, argv=None)
            Runs the program with an optional 'main' function and 'argv' list.


            
            
###Tensorflow - Saving and Restoring 
#Estimators automatically saves and restores variables (in the model_dir).

##Saving variables
#Create a Saver with tf.train.Saver() to manage all variables in the model. 
#the use tf.train.Saver.save method to save variables to a checkpoint file

# Create some variables.
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  inc_v1.op.run()
  dec_v2.op.run()
  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in file: %s" % save_path)


##Restoring variables
#Note that when you restore variables from a file 
#you do not have to initialize them beforehand


tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())


##Choosing which variables to save and restore
#If you do not pass any arguments to tf.train.Saver(), 
#the saver handles all variables in the graph. 
#Each variable is saved under the name that was passed when the variable was created.

#Note 
1.You can create as many Saver objects as you want 
 if you need to save and restore different subsets of the model variables. 
 The same variable can be listed in multiple saver objects; 
 its value is only changed when the Saver.restore() method is run.

2.If you only restore a subset of the model variables at the start of a session, 
  you have to run an initialize op for the other variables. 
  eg using  op = tf.variables_initializer( var_list, name='init')
  After you launch the graph in a session, 
  you can run the returned Op(ie op.run()) to initialize all the variables in var_list. 
  This Op runs all the initializers of the variables in var_list in parallel.
  
3.To inspect the variables in a checkpoint, you can use the inspect_checkpoint library, 
  eg print_tensors_in_checkpoint_file function.

4.By default, Saver uses the value of the tf.Variable.name property for each variable. 
  However, when you create a Saver object, you may optionally choose names 
  for the variables in the checkpoint files.


tf.reset_default_graph()
# Create some variables.
v1 = tf.get_variable("v1", [3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", [5], initializer = tf.zeros_initializer)

# Add ops to save and restore only `v2` using the name "v2"
saver = tf.train.Saver({"v2": v2})

# Use the saver object normally after that.
with tf.Session() as sess:
  # Initialize v1 since the saver will not.
  v1.initializer.run()
  saver.restore(sess, "/tmp/model.ckpt")

  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())




    


##Overview of saving and restoring models - use SavedModel 

#For example, the following code suggests a typical way to use SavedModelBuilder 
#to build a SavedModel:

export_dir = ...
...
builder = tf.saved_model_builder.SavedModelBuilder(export_dir)
with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph_and_variables(sess,
                                       [tag_constants.TRAINING],
                                       signature_def_map=foo_signatures,
                                       assets_collection=foo_assets)
...
# Add a second MetaGraphDef for inference.
with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph([tag_constants.SERVING])
...
builder.save()


##Loading a SavedModel in Python

export_dir = ...
...
with tf.Session(graph=tf.Graph()) as sess:
  tf.saved_model.loader.load(sess, [tag_constants.TRAINING], export_dir)
  ...


##Using SavedModel with Estimators
#After training an Estimator model, you may want to create a service from that model 
#that takes requests and returns a result. 
#You can run such a service locally on your machine or deploy it scalably in the cloud.
#For details check 
#https://www.tensorflow.org/programmers_guide/saved_model#using_savedmodel_with_estimators









###Tensorflow - Importing Data - DataSet API 
#The preferred way to feed data into a tensorflow program is using the Datasets API.
   
1.A tf.contrib.data.Dataset represents a sequence of elements, 
  in which each element contains one or more Tensor objects. 

2.To start an input pipeline, define a source. 
  For example, to construct a Dataset from some tensors in memory, 
  you can use tf.contrib.data.Dataset.from_tensors() 
  or tf.contrib.data.Dataset.from_tensor_slices(). 
  Alternatively, if your input data are on disk in the recommend TFRecord format, 
  you can construct a tf.contrib.data.TFRecordDataset.

3.Once you have a Dataset object, you can transform it into a new Dataset 
  by chaining method calls on the tf.contrib.data.Dataset object. 
  For example, you can apply per-element transformations such as Dataset.map() 
  (to apply a function to each element), 
  and multi-element transformations such as Dataset.batch()

4.Consume values from a Dataset is to make an tf.contrib.data.Iterator object 
  that provides access to one element of the dataset at a time 
  (for example, by calling Dataset.make_one_shot_iterator()). 
  A tf.contrib.data.Iterator provides two operations: 
  Iterator.initializer, which enables to (re)initialize the iterator's state; 
  and Iterator.get_next(), which returns tf.Tensor objects that correspond to next element.

  
##Dataset structure
#An element contains one or more tf.Tensor objects, called components
class tf.contrib.data.Dataset
        Represents a potentially large set of elements.
        Most of the methods Returns:A Dataset., hence can be chained
    #Properties
    output_shapes
        Returns the shape,tf.TensorShape of each component of an element of this dataset.
    output_types
        Returns the type,tf.DType of each component of an element of this dataset.
    batch(batch_size)
        Combines consecutive elements of this dataset into batches.
        batch_size: A tf.int64 scalar tf.Tensor, representing the number of consecutive elements of this dataset to combine in a single batch.
        Returns:A Dataset.
    cache(filename='')
        Caches the elements in this dataset.
        filename: A tf.string scalar tf.Tensor, representing the name of a directory on the filesystem to use for caching tensors in this Dataset. If a filename is not provided, the dataset will be cached in memory.
        Returns:A Dataset.
    concatenate(dataset)
        Creates a Dataset by concatenating given dataset with this dataset.
        Returns:A Dataset.
        # NOTE: The following examples use `{ ... }` to represent the
        # contents of a dataset.
        a = { 1, 2, 3 }
        b = { 4, 5, 6, 7 }
        # Input dataset and dataset to be concatenated should have same
        # nested structures and output types.
        # c = { (8, 9), (10, 11), (12, 13) }
        # d = { 14.0, 15.0, 16.0 }
        # a.concatenate(c) and a.concatenate(d) would result in error.
        a.concatenate(b) == { 1, 2, 3, 4, 5, 6, 7 }
        
    dense_to_sparse_batch( batch_size,  row_shape)
        Batches ragged elements of this dataset into tf.SparseTensors.
        Like Dataset.padded_batch(), this method combines multiple consecutive elements 
        of this dataset, which might have different shapes, into a single element. 
        The resulting element has three components (indices, values, and dense_shape), 
        which comprise a tf.SparseTensor that represents the same data. 
        The row_shape represents the dense shape of each row in the resulting tf.SparseTensor,
        to which the effective batch size is prepended.

        # NOTE: The following examples use `{ ... }` to represent the
        # contents of a dataset.
        a = { ['a', 'b', 'c'], ['a', 'b'], ['a', 'b', 'c', 'd'] }
        a.dense_to_sparse_batch(batch_size=2, row_shape=[6]) == {
            ([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]],  # indices
             ['a', 'b', 'c', 'a', 'b'],                 # values
             [2, 6]),                                   # dense_shape
            ([[2, 0], [2, 1], [2, 2], [2, 3]],
             ['a', 'b', 'c', 'd'],
             [1, 6])
        }
    enumerate(start=0)
        Enumerate the elements of this dataset. Similar to python's enumerate.
        # NOTE: The following examples use `{ ... }` to represent the
        # contents of a dataset.
        a = { 1, 2, 3 }
        b = { (7, 8), (9, 10), (11, 12) }
        # The nested structure of the `datasets` argument determines the
        # structure of elements in the resulting dataset.
        a.enumerate(start=5) == { (5, 1), (6, 2), (7, 3) }
        b.enumerate() == { (0, (7, 8)), (1, (9, 10)), (2, (11, 12)) }
    filter(predicate)
        Filters this dataset according to predicate.
    flat_map(map_func)
        Maps map_func across this dataset and flattens the result.
    from_sparse_tensor_slices(sparse_tensor)
        Splits each rank-N tf.SparseTensor in this dataset row-wise.
        Returns:A Dataset of rank-(N-1) sparse tensors.
    from_tensor_slices(tensors)
        Creates a Dataset whose elements are slices of the given tensors.
    from_tensors(tensors)
        Creates a Dataset with a single element, comprising the given tensors.
    group_by_window( key_func, reduce_func,  window_size)
        Performs a windowed "group-by" operation on this dataset.
        This method maps each consecutive element in this dataset to a key using key_func 
        and groups the elements by key. 
        It then applies reduce_func to at most window_size elements matching the same key. 
        All execpt the final window for each key will contain window_size elements; 
        the final window may be smaller.
        key_func: A function mapping a nested structure of tensors 
                 (having shapes and types defined by self.output_shapes and self.output_types)
                 to a scalar tf.int64 tensor.
        reduce_func: A function mapping a key and a dataset of up to batch_size 
                    consecutive elements matching that key to another dataset.
        window_size: A tf.int64 scalar tf.Tensor
    ignore_errors()
        Creates a Dataset from this one and silently ignores any errors.
        dataset = tf.contrib.data.Dataset.from_tensor_slices([1., 2., 0., 4.])
        # Computing `tf.check_numerics(1. / 0.)` will raise an InvalidArgumentError.
        dataset = dataset.map(lambda x: tf.check_numerics(1. / x, "error"))
        # Using `ignore_errors()` will drop the element that causes an error.
        dataset = dataset.ignore_errors()  # ==> { 1., 0.5, 0.2 }
    interleave(  map_func,  cycle_length,  block_length=1)
        Maps map_func across this dataset, and interleaves the results.
        For example, you can use Dataset.interleave() to process many input files concurrently:
        # Preprocess 4 files concurrently, and interleave blocks of 16 records from
        # each file.
        filenames = ["/var/data/file1.txt", "/var/data/file2.txt", ..."]
        dataset = (Dataset.from_tensor_slices(filenames)
                   .interleave(
                       lambda x: TextLineDataset(x).map(parse_fn, num_threads=1),
                       cycle_length=4, block_length=16))

        The cycle_length and block_length arguments control the order 
        in which elements are produced. cycle_length controls the number of input elements 
        that are processed concurrently. 
        If you set cycle_length to 1, this transformation will handle one input element at a time,
        and will produce identical results = to tf.contrib.data.Dataset.flat_map. 
        In general, this transformation will apply map_func to cycle_length input elements, 
        open iterators on the returned Dataset objects, 
        and cycle through them producing block_length consecutive elements from each iterator, 
        and consuming the next input element each time it reaches the end of an iterator.
        # NOTE: The following examples use `{ ... }` to represent the
        # contents of a dataset.
        a = { 1, 2, 3, 4, 5 }

        # NOTE: New lines indicate "block" boundaries.
        a.interleave(lambda x: Dataset.from_tensors(x).repeat(6),
                     cycle_length=2, block_length=4) == {
            1, 1, 1, 1,
            2, 2, 2, 2,
            1, 1,
            2, 2,
            3, 3, 3, 3,
            4, 4, 4, 4,
            3, 3,
            4, 4,
            5, 5, 5, 5,
            5, 5,
        }

    list_files(file_pattern)
        A dataset of all files matching a pattern(glob pattern)
        Returns:A Dataset of strings corresponding to file names.
    make_dataset_resource()
        Creates a tf.Tensor of tf.resource tensor representing this dataset.
    make_initializable_iterator(shared_name=None)
        Creates an Iterator for enumerating the elements of this dataset.
    make_one_shot_iterator()
        Creates an Iterator for enumerating the elements of this dataset.
    map(  map_func,  num_threads=None,  output_buffer_size=None)
        Maps map_func across this datset.
    padded_batch(  batch_size,  padded_shapes,  padding_values=None)
        Combines consecutive elements of this dataset into padded batches.
        Like Dataset.dense_to_sparse_batch(), this method combines multiple consecutive elements of this dataset, 
        which might have different shapes, into a single element. 
        The tensors in the resulting element have an additional outer dimension, 
        and are padded to the respective shape in padded_shapes.
    range(*args)
        Creates a Dataset of a step-separated range of values.
        Dataset.range(5) == [0, 1, 2, 3, 4]
        Dataset.range(2, 5) == [2, 3, 4]
        Dataset.range(1, 5, 2) == [1, 3]
        Dataset.range(1, 5, -2) == []
        Dataset.range(5, 1) == []
        Dataset.range(5, 1, -2) == [5, 3]
    read_batch_features( file_pattern, batch_size, features, reader, reader_args=None, randomize_input=True,
                num_epochs=None, capacity=10000)
        Reads batches of Examples.
    repeat(count=None)
        Repeats this dataset count times.
    shuffle( buffer_size,  seed=None)
        Randomly shuffles the elements of this dataset.
    skip(count)
        Creates a Dataset that skips count elements from this dataset.
    take(count)
        Creates a Dataset with at most count elements from this dataset.
    unbatch()
        Splits elements of this dataset into sequences of consecutive elements.
        For example, if elements of this dataset are shaped [B, a0, a1, ...], 
        where B may vary from element to element, 
        then for each element in this dataset, 
        the unbatched dataset will contain B consecutive elements of shape [a0, a1, ...].
    zip(datasets)
        Creates a Dataset by zipping together the given datasets.

        # NOTE: The following examples use `{ ... }` to represent the
        # contents of a dataset.
        a = { 1, 2, 3 }
        b = { 4, 5, 6 }
        c = { (7, 8), (9, 10), (11, 12) }
        d = { 13, 14 }

        # The nested structure of the `datasets` argument determines the
        # structure of elements in the resulting dataset.
        Dataset.zip((a, b)) == { (1, 4), (2, 5), (3, 6) }
        Dataset.zip((b, a)) == { (4, 1), (5, 2), (6, 3) }

        # The `datasets` argument may contain an arbitrary number of
        # datasets.
        Dataset.zip((a, b, c)) == { (1, 4, (7, 8)),
                                    (2, 5, (9, 10)),
                                    (3, 6, (11, 12)) }

        # The number of elements in the resulting dataset is the same as
        # the size of the smallest dataset in `datasets`.
        Dataset.zip((a, d)) == { (1, 13), (2, 14) }
        
        
        
    
class tf.contrib.data.Iterator    
    get_next(name=None)
        Returns a nested structure of tf.Tensors containing the next element.
    make_initializer(dataset)
        Returns a tf.Operation that initializes this iterator on dataset.
    string_handle(name=None)
        Returns a string-valued tf.Tensor that represents this iterator.

from_dataset( dataset, shared_name=None)
    Creates a new, uninitialized Iterator from the given Dataset.
    To initialize this iterator, you must run its initializer:
    dataset = ...
    iterator = Iterator.from_dataset(dataset)
    # ...
    sess.run(iterator.initializer)
    
from_string_handle(  string_handle,  output_types, output_shapes=None)
    Creates a new, uninitialized Iterator based on the given handle.
    This method allows you to define a "feedable" iterator 
    where you can choose between concrete iterators by feeding a value 
    in a tf.Session.run call.
    In that case, string_handle would a tf.placeholder, 
    and you would feed it with the value of tf.contrib.data.Iterator.string_handle in each step.

    For example, if you had two iterators that marked the current position 
    in a training dataset and a test dataset, 
    you could choose which to use in each step as follows:
    train_iterator = tf.contrib.data.Dataset(...).make_one_shot_iterator()
    train_iterator_handle = sess.run(train_iterator.string_handle())
    test_iterator = tf.contrib.data.Dataset(...).make_one_shot_iterator()
    test_iterator_handle = sess.run(test_iterator.string_handle())
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.contrib.data.Iterator.from_string_handle(handle, train_iterator.output_types)
    next_element = iterator.get_next()
    loss = f(next_element)
    train_loss = sess.run(loss, feed_dict={handle: train_iterator_handle})
    test_loss = sess.run(loss, feed_dict={handle: test_iterator_handle})

from_structure( output_types, output_shapes=None,shared_name=None)
    Creates a new, uninitialized Iterator with the given structure.
    This iterator-constructing method can be used to create an iterator 
    that is reusable with many different datasets.
    
    The returned iterator is not bound to a particular dataset, 
    and it has no initializer. 
    To initialize the iterator,  run the operation returned by Iterator.make_initializer(dataset).
    iterator = Iterator.from_structure(tf.int64, tf.TensorShape([]))
    dataset_range = Dataset.range(10)
    range_initializer = iterator.make_initializer(dataset_range)

    dataset_evens = dataset_range.filter(lambda x: x % 2 == 0)
    evens_initializer = iterator.make_initializer(dataset_evens)
    # Define a model based on the iterator; in this example, the model_fn
    # is expected to take scalar tf.int64 Tensors as input (see
    # the definition of 'iterator' above).
    prediction, loss = model_fn(iterator.get_next())

    # Train for `num_epochs`, where for each epoch, we first iterate over
    # dataset_range, and then iterate over dataset_evens.
    for _ in range(num_epochs):
      # Initialize the iterator to `dataset_range`
      sess.run(range_initializer)
      while True:
        try:
          pred, loss_val = sess.run([prediction, loss])
        except tf.errors.OutOfRangeError:
          break

      # Initialize the iterator to `dataset_evens`
      sess.run(evens_initializer)
      while True:
        try:
          pred, loss_val = sess.run([prediction, loss])
        except tf.errors.OutOfRangeError:
          break


##Example 
sess = tf.Session()
#4 rows/Tensor of 10 random numbers 
dataset1 = tf.contrib.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(dataset1.output_types)  # ==> "tf.float32"
print(dataset1.output_shapes)  # ==> "(10,)"
#Use Dataset.make_initializable_iterator() 
iterator1 = dataset1.make_initializable_iterator() 
next_element1 = iterator1.get_next()
sess.run(iterator1.initializer) #reset, it can be called as many times and it reinitializes 
#four times - so four rows as Tensor of shape [10]
sess.run(next_element1)
sess.run(next_element1)
sess.run(next_element1)
sess.run(next_element1)
#5th times 
sess.run(next_element1)
OutOfRangeError (see above for traceback): End of sequence
#A common pattern is to wrap the "training loop" in a try-except block:
sess.run(iterator.initializer)
while True:
  try:
    sess.run(result)
  except tf.errors.OutOfRangeError:
    break

    
#diff with .from_tensors
dataset1 = tf.contrib.data.Dataset.from_tensors(tf.random_uniform([4, 10]))
iterator1 = dataset1.make_initializable_iterator() 
next_element1 = iterator1.get_next()
sess.run(iterator1.initializer)
sess.run(next_element1) #whole 2d matrix of 4x10 as Tensor [4,10]
sess.run(next_element1) #OutOfRangeError (see above for traceback): End of sequence
sess.run(iterator1.initializer) #reinitializes 
sess.run(next_element1)
#More example 
dataset2 = tf.contrib.data.Dataset.from_tensor_slices(
   (tf.random_uniform([4]),tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
print(dataset2.output_shapes)  # ==> "((), (100,))"

dataset3 = tf.contrib.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"

 
  
#It is often convenient to give names to each component of an element
dataset = tf.contrib.data.Dataset.from_tensor_slices(
   {"a": tf.random_uniform([4]),
    "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"









#The Dataset transformations support datasets of any structure. 
#When using the Dataset.map(), Dataset.flat_map(), and Dataset.filter() transformations, 
#which apply a function to each element, 
#the element structure determines the arguments of the function:

dataset1 = dataset1.map(lambda x: (x,len(x) ) #x is Tensor of shape 10

dataset2 = dataset2.flat_map(lambda x, y: ...)

# Note: Argument destructuring is not available in Python 3.(use tuple)
dataset3 = dataset3.filter(lambda x, (y, z): ...)





#If each element of the dataset has a nested structure, 
#the return value of Iterator.get_next() will be one or more tf.Tensor objects in the same nested structure:

#evaluating any of next1, next2, or next3 will advance the iterator for all components.
dataset1 = tf.contrib.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
dataset2 = tf.contrib.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100])))
dataset3 = tf.contrib.data.Dataset.zip((dataset1, dataset2))

iterator = dataset3.make_initializable_iterator()

sess.run(iterator.initializer)
next1, (next2, next3) = iterator.get_next()





##Initialization of  an iterator
#Supports below , in increasing level of sophistication:
    one-shot,
    initializable,
    reinitializable, and
    feedable.


##A one-shot iterator is the simplest form of iterator, 
#which only supports iterating once through a dataset, 
#with no need for explicit initialization. - CUrrent version gives Error for tf.Variable 
dataset = tf.contrib.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
sess = tf.Session()
for i in range(100):
  value = sess.run(next_element)
  assert i == value


##An initializable iterator requires you to run an explicit iterator.initializer operation before using it.
#it enables you to parameterize the definition of the dataset, using one or more tf.placeholder() tensors 
max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.contrib.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Initialize an iterator over a dataset with 10 elements.
sess.run(iterator.initializer, feed_dict={max_value: 10})
for i in range(10):
  value = sess.run(next_element)
  assert i == value

# Initialize the same iterator over a dataset with 100 elements.
sess.run(iterator.initializer, feed_dict={max_value: 100})
for i in range(100):
  value = sess.run(next_element)
  assert i == value


##A reinitializable iterator can be initialized from multiple different Dataset objects. 

# Define training and validation datasets with the same structure.
training_dataset = tf.contrib.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.contrib.data.Dataset.range(50)

# A reinitializable iterator is defined by its structure. We could use the
# `output_types` and `output_shapes` properties of either `training_dataset`
# or `validation_dataset` here, because they are compatible.
iterator = tf.contrib.data.Iterator.from_structure(training_dataset.output_types,
                                   training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

# Run 20 epochs in which the training dataset is traversed, followed by the
# validation dataset.
for _ in range(20):
  # Initialize an iterator over the training dataset.
  sess.run(training_init_op)
  for _ in range(100):
    sess.run(next_element)

  # Initialize an iterator over the validation dataset.
  sess.run(validation_init_op)
  for _ in range(50):
    sess.run(next_element)


##A feedable iterator can be used together with tf.placeholder 
#to select what Iterator to use in each call to tf.Session.run, 
#via the feed_dict mechanism. 
#It offers the same functionality as a reinitializable iterator, 
#but it does not require you to initialize the iterator 
#from the start of a dataset when you switch between iterators. 


# Define training and validation datasets with the same structure.
training_dataset = tf.contrib.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
validation_dataset = tf.contrib.data.Dataset.range(50)

# A feedable iterator is defined by a handle placeholder and its structure. We
# could use the `output_types` and `output_shapes` properties of either
# `training_dataset` or `validation_dataset` here, because they have
# identical structure.
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.contrib.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

# You can use feedable iterators with a variety of different kinds of iterator
# (such as one-shot and initializable iterators).
training_iterator = training_dataset.make_initializable_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

# The `Iterator.string_handle()` method returns a tensor that can be evaluated
# and used to feed the `handle` placeholder.
training_handle = sess.run(training_iterator.string_handle())
validation_handle = sess.run(validation_iterator.string_handle())

# Loop forever, alternating between training and validation.
while True:
  # Run 200 steps using the training dataset. Note that the training dataset is
  # infinite, and we resume from where we left off in the previous `while` loop
  # iteration.
  for _ in range(200):
    sess.run(next_element, feed_dict={handle: training_handle})

  # Run one pass over the validation dataset.
  sess.run(validation_iterator.initializer)
  for _ in range(50):
    sess.run(next_element, feed_dict={handle: validation_handle})







    
##Reading input data

#Consuming NumPy arrays - all fit in memory 

# Load the training data into two NumPy arrays, for example using `np.load()`.
# data is saved via np.savez('training_data.npz', features=a, labels=b)
with np.load("training_data.npz") as data:
  features = data["features"]
  labels = data["labels"]

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

#embed the features and labels arrays in graph as tf.constant() operations
#wastes memory 
dataset = tf.contrib.data.Dataset.from_tensor_slices((features, labels))

##OR to limit wastage of memory, define the Dataset in terms of tf.placeholder() tensors, 
#and feed the NumPy arrays when you initialize an Iterator over the dataset.

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.contrib.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# [Other transformations on `dataset`...]
dataset = ...
iterator = dataset.make_initializable_iterator()

sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})


                                       
                                          
                                          
##Consuming TFRecord data - use TFRecordDataset
#the TFRecord file format is a simple record-oriented binary format 
#Saving is done via tf.python_io.TFRecordWriter.write(record)

# Creates a dataset that reads all of the examples from two files.
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.contrib.data.TFRecordDataset(filenames)

#OR to change the filenames programmatically 
filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset = dataset.map(...)  # Parse the record into tensors.
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()

# You can feed the initializer with the appropriate filenames for the current
# phase of execution, e.g. training vs. validation.

# Initialize `iterator` with training data.
training_filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

# Initialize `iterator` with validation data.
validation_filenames = ["/var/data/validation1.tfrecord", ...]
sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})




##Consuming text data - use TextLineDataset
#Given one or more filenames, 
#a TextLineDataset will produce one string-valued element per line of those files. 
#TextLineDataset accepts filenames as a tf.Tensor, so you can parameterize it by passing a tf.placeholder(tf.string).
filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]
dataset = tf.contrib.data.TextLineDataset(filenames)


#By default, a TextLineDataset yields every line of each file, 
#for example if the file starts with a header line, or contains comments. 
#can be removed using the Dataset.skip() and Dataset.filter() transformations. 

#To apply these transformations to each file separately, use Dataset.flat_map() 
#to create a nested Dataset for each file.

filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]
dataset = tf.contrib.data.Dataset.from_tensor_slices(filenames)

# Use `Dataset.flat_map()` to transform each file as a separate nested dataset,
# and then concatenate their contents sequentially into a single "flat" dataset.
# * Skip the first line (header row).
# * Filter out lines beginning with "#" (comments).
dataset = dataset.flat_map(
    lambda filename: (
        tf.contrib.data.TextLineDataset(filename)
        .skip(1)
        .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))))

##Consuming csv data    
#Use above to handle csv file 
#OR use below low level api as well 

filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1], [1], [1]]
#list of Tensors 
col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4])

#must call tf.train.start_queue_runners to populate the queue 
#before you call run or eval to execute the read. 
#Otherwise read will block while it waits for filenames from the queue.

with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1200):
    # Retrieve a single instance:
    example, label = sess.run([features, col5])

  coord.request_stop()
  coord.join(threads)

        
        
 
##Dataset.map() -Parsing tf.Example protocol buffer messages

#Many input pipelines extract tf.train.Example protocol buffer messages 
#from a TFRecord-format file (written, for example, using tf.python_io.TFRecordWriter). 

#Each tf.train.Example record contains one or more "features", 
#and the input pipeline typically converts these features into tensors.



# Transforms a scalar string `example_proto` into a pair of a scalar string and
# a scalar integer, representing an image and its label, respectively.
def _parse_function(example_proto):
  features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
              "label": tf.FixedLenFeature((), tf.int32, default_value=0)}
  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features["image"], parsed_features["label"]

# Creates a dataset that reads all of the examples from two files, and extracts
# the image and label features.
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)


##Dataset.map() - Decoding image data and resizing it

#When training a neural network on real-world image data, 
#it is often necessary to convert images of different sizes to a common size, 
#so that they may be batched into a fixed size.


# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

# A vector of filenames.
filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([0, 37, ...])

dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)




##Dataset.map() - Applying arbitrary Python logic with tf.py_func()


import cv2

# Use a custom OpenCV function to read the image, instead of the standard
# TensorFlow `tf.read_file()` operation.
def _read_py_function(filename, label):
  image_decoded = cv2.imread(image_string, cv2.IMREAD_GRAYSCALE)
  return image_decoded, label

# Use standard TensorFlow operations to resize the image to a fixed shape.
def _resize_function(image_decoded, label):
  image_decoded.set_shape([None, None, None])
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

filenames = ["/var/data/image1.jpg", "/var/data/image2.jpg", ...]
labels = [0, 37, 29, 1, ...]

dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(
    lambda filename, label: tf.py_func(
        _read_py_function, [filename, label], [tf.uint8, label.dtype]))
dataset = dataset.map(_resize_function)

 
##Batching dataset elements - Simple batching - Dataset.batch()
#The simplest form of batching stacks n consecutive elements of a dataset into a single element. 
#like  tf.stack() operator, applied to each component of the elements: 
#i.e. for each component i, all elements must have a tensor of the exact same shape.


inc_dataset = tf.contrib.data.Dataset.range(100)
dec_dataset = tf.contrib.data.Dataset.range(0, -100, -1)
dataset = tf.contrib.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])


##Batching dataset elements - Batching tensors with padding - Dataset.padded_batch()
#for tensors that all don't have the same size. 
#to batch tensors of different shape by specifying one or more dimensions in which they may be padded.


dataset = tf.contrib.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.padded_batch(4, padded_shapes=[None])

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
                               #      [5, 5, 5, 5, 5, 0, 0],
                               #      [6, 6, 6, 6, 6, 6, 0],
                               #      [7, 7, 7, 7, 7, 7, 7]]


##Training workflows - Processing multiple epochs
#use the Dataset.repeat() transformation. 
#For example, to create a dataset that repeats its input for 10 epochs:

filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.repeat(10) #no arguments will repeat the input indefinitely
dataset = dataset.batch(32)


#If you want to receive a signal at the end of each epoch, 
#you can write a training loop that catches the tf.errors.OutOfRangeError at the end of a dataset. 
#At that point you might collect some statistics (e.g. the validation error) for the epoch.

filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Compute for 100 epochs.
for _ in range(100):
  sess.run(iterator.initializer)
  while True:
    try:
      sess.run(next_element)
    except tf.errors.OutOfRangeError:
      break

  # [Perform end-of-epoch calculations here.]


##Training workflows - Randomly shuffling input data - Dataset.shuffle()
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
dataset = dataset.repeat()


##Training workflows - Using high-level APIs
#The tf.train.MonitoredTrainingSession API simplifies many aspects of running TensorFlow in a distributed setting. 
#MonitoredTrainingSession uses the tf.errors.OutOfRangeError to signal that training has completed, 
#so to use it with the Dataset API, we recommend using Dataset.make_one_shot_iterator()

filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
dataset = dataset.repeat(num_epochs)
iterator = dataset.make_one_shot_iterator() #Error, use .make_initializable_iterator(), sess.run(iterator.initializer)

next_example, next_label = iterator.get_next()
loss = model_function(next_example, next_label)

training_op = tf.train.AdagradOptimizer(...).minimize(loss)

with tf.train.MonitoredTrainingSession(...) as sess:
  while not sess.should_stop():
    sess.run(training_op)


#To use a Dataset in the input_fn of a tf.estimator.Estimator, 
#we also recommend using Dataset.make_one_shot_iterator(). For example:
def dataset_input_fn():
  filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
  dataset = tf.contrib.data.TFRecordDataset(filenames)

  # Use `tf.parse_single_example()` to extract data from a `tf.Example`
  # protocol buffer, and perform any additional per-record preprocessing.
  def parser(record):
    keys_to_features = {
        "image_data": tf.FixedLenFeature((), tf.string, default_value=""),
        "date_time": tf.FixedLenFeature((), tf.int64, default_value=""),
        "label": tf.FixedLenFeature((), tf.int64,
                                    default_value=tf.zeros([], dtype=tf.int64)),
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    # Perform additional preprocessing on the parsed data.
    image = tf.decode_jpeg(parsed["image_data"])
    image = tf.reshape(image, [299, 299, 1])
    label = tf.cast(parsed["label"], tf.int32)

    return {"image_data": image, "date_time": parsed["date_time"]}, label

  # Use `Dataset.map()` to build a pair of a feature dictionary and a label
  # tensor for each example.
  dataset = dataset.map(parser)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(32)
  dataset = dataset.repeat(num_epochs)
  iterator = dataset.make_one_shot_iterator()

  # `features` is a dictionary in which each value is a batch of values for
  # that feature; `labels` is a batch of labels.
  features, labels = iterator.get_next()
  return features, labels



###Tensorflow - Threading and Queues 
#Use DataSet API, this is for old version 

###Tensorflow - Reading Data 
#Use DataSet API, this is for old version 


 
###Tensorflow - Embeddings 

#An embedding is a mapping from discrete objects, such as words, to vectors of real numbers. 

#For example, a 300-dimensional embedding for English words could include:
blue:  (0.01359, 0.00075997, 0.24608, ..., -0.2524, 1.0048, 0.06259)
blues:  (0.01396, 0.11887, -0.48963, ..., 0.033483, -0.10007, 0.1158)
orange:  (-0.24776, -0.12359, 0.20986, ..., 0.079717, 0.23865, -0.014213)
oranges:  (-0.35609, 0.21854, 0.080944, ..., -0.35413, 0.38511, -0.070976)

#Classifiers, and neural networks are designed to work with dense continuous vectors, 
#Embeddings let you apply machine learning to discrete inputs. 



##Training an Embedding


word_ids = ["I", "have", "a", "cat", "."]
word_embeddings = tf.get_variable("word_embeddings",  [vocabulary_size, embedding_size])
embedded_word_ids = tf.gather(word_embeddings, word_ids)

##Real world example 
#check - word2vec_basic.py
#https://www.tensorflow.org/tutorials/word2vec









 





###Tensorflow -  Building a Convolutional Neural Network 
#The MNIST dataset comprises 60,000 training examples and 10,000 test examples 
#of the handwritten digits 0-9, formatted as 28x28-pixel monochrome images.



##Introduction 
Convolutional neural networks (CNNs) are the current state-of-the-art model architecture
for image classification tasks. 

CNNs apply a series of filters to the raw pixel data of an image to extract 
and learn higher-level features, which the model can then use for classification. 
CNNs contains three components:
    1.Convolutional layers, which apply a specified number of convolution filters to the image. 
    For each subregion, the layer performs a set of mathematical operations 
    to produce a single value in the output feature map. 
    Convolutional layers then typically apply a ReLU activation function 
    to the output to introduce nonlinearities into the model.

    2.Pooling layers, which downsample the image data extracted by the convolutional layers 
    to reduce the dimensionality of the feature map in order to decrease processing time. 
    A commonly used pooling algorithm is max pooling, which extracts subregions 
    of the feature map (e.g., 2x2-pixel tiles), keeps their maximum value, 
    and discards all other values.

    3.Dense (fully connected) layers, which perform classification 
    on the features extracted by the convolutional layers 
    and downsampled by the pooling layers. 
    In a dense layer, every node in the layer is connected to every node 
    in the preceding layer.

Typically, a CNN is composed of a stack of convolutional modules 
that perform feature extraction. 
Each module consists of a convolutional layer followed by a pooling layer. 
The last convolutional module is followed by one or more dense layers 
that perform classification. 

The final dense layer in a CNN contains a single node for each target class in the model
(all the possible classes the model may predict), 
with a softmax activation function to generate a value between 0?1 for each node 
(the sum of all these softmax values is equal to 1). 

We can interpret the softmax values for a given image as relative measurements 
of how likely it is that the image falls into each target class.



##Example - Building CNN 
#consists of 
    Convolutional Layer #1: Applies 32 5x5 filters (extracting 5x5-pixel subregions), 
                            with ReLU activation function
    Pooling Layer #1: Performs max pooling with a 2x2 filter and stride of 2 
                     (which specifies that pooled regions do not overlap)
    Convolutional Layer #2: Applies 64 5x5 filters, with ReLU activation function
    Pooling Layer #2: Again, performs max pooling with a 2x2 filter and stride of 2
                      Stride means the step of the convolution operation
    Dense Layer #1: 1,024 neurons, with dropout regularization rate of 0.4 
                    (probability of 0.4 that any given element will be dropped during training)
    Dense Layer #2 (Logits Layer): 10 neurons, one for each digit target class (0?9).

#Uses following functions 
    tf.layers.conv2d(): Constructs a two-dimensional convolutional layer. 
                        Takes number of filters, filter kernel size, padding, and activation function as arguments.
    tf.layers.max_pooling2d(): Constructs a two-dimensional pooling layer 
                                using the max-pooling algorithm. 
                                Takes pooling filter size and stride as arguments.
    tf.layers.dense(): Constructs a dense layer. 
                       Takes number of neurons and activation function as arguments.

#Expects  input tensors to have a shape of [batch_size, image_width, image_height, channels], 
    batch_size. Size of the subset of examples to use 
                when performing gradient descent during training.
                -1 means that this dimension should be dynamically computed
    image_width. Width of the example images.
    image_height. Height of the example images.
    channels. Number of color channels in the example images. 
              For color images, the number of channels is 3 (red, green, blue). 
              For monochrome images, there is just 1 channel (black).

#Example 
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""

    #MNIST dataset is composed of monochrome 28x28 pixel images, 
    #so the desired shape for our input layer is [batch_size, 28, 28, 1]
    #To convert our input feature map (features) to this shape, 
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    #The filters argument specifies the number of filters to apply (here, 32), 
    #and kernel_size specifies the dimensions of the filters as [width, height] (here, [5, 5]).

    #To specify that the output tensor should have the same width and height values 
    #as the input tensor, set padding=same here, 
    #which instructs TensorFlow to add 0 values to the edges of the output tensor 
    #to preserve width and height of 28. 
    #(Without padding, a 5x5 convolution over a 28x28 tensor will produce a 24x24 tensor, 
    #as there are 24x24 locations to extract a 5x5 tile from a 28x28 grid.)

    #output tensor produced by conv2d() has a shape of [batch_size, 28, 28, 32]: the same width and height dimensions as the input, 
    #but now with 32 channels holding the output from each of the filters.
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #1
    #construct a layer that performs max pooling with a 2x2 filter and stride of 2
    #pool_size argument specifies the size of the max pooling filter as [width, height] (here, [2, 2]
    #set a stride of 2, which indicates that the subregions extracted 
    #by the filter should be separated by 2 pixels in both the width and height dimensions 
    #(for a 2x2 filter, this means that none of the regions extracted will overlap). 

    #output tensor produced by max_pooling2d() (pool1) 
    #has a shape of [batch_size, 14, 14, 32]: the 2x2 filter reduces width and height by 50% each.
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    #conv2 has a shape of [batch_size, 14, 14, 64], the same width and height as pool1 
    #(due to padding="same"), and 64 channels for the 64 filters applied.

    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    #pool2 has shape [batch_size, 7, 7, 64] (50% reduction of width and height from conv2).
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer for classification
    #flatten our feature map (pool2) to shape [batch_size, features], 
    #so that our tensor has only two dimensions

    #-1 signifies that the batch_size dimension will be dynamically calculated based
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    #The units argument specifies the number of neurons in the dense layer (1,024)
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    #To help improve the results of model, apply dropout regularization to  dense layer,

    #rate argument specifies the dropout rate; here, we use 0.4, 
    #which means 40% of the elements will be randomly dropped out during training.
    #dropout will only be performed if training is True

    #output tensor dropout has shape [batch_size, 1024].
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    #logits layer, which will return the raw values for  predictions
    #create a dense layer with 10 neurons (one for each target class 0?9), with linear activation (the default):

    #output tensor [batch_size, 10]-dimensional tensor.
    logits = tf.layers.dense(inputs=dropout, units=10)

    
    #Steps for using tf.estimator 
    #Step-1 - TRAIN 
    # 1. Create a optimizer from tf.train. eg GradientDescentOptimizer
    # 2. Create a loss function from tf.losses eg softmax_cross_entropy
    # 3. create train_op by optimizer.minimise passing loss 
    # 4. return tf.estimator.EstimatorSpec passing mode, loss, train_op    
    #Step-2: EVAL 
    # 1. Create metrics from tf.metrics. eg accuracy between actual and predictions
    # 2. Create eval_metric_ops as dict of "metric_name" with value above 
    # 3. Create a loss function from tf.losses eg softmax_cross_entropy
    # 4. return tf.estimator.EstimatorSpec passing mode, loss, eval_metric_ops 
    #Step-3: PREDICT 
    # 1. Create class based on some function and other eg probabilities based on some function 
    # 2. Create predictions as dict of "prediction_key" with value above 
    # 3. return tf.estimator.EstimatorSpec passing mode, predictions 


    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      #predicted class is the element in the corresponding 
      #row of the logits tensor with the highest raw value.
      #Since logits has [batch_size, 10], hence axis=1 means index=1 ie 10
      "classes": tf.argmax(input=logits, axis=1),
      
      
      # Add `softmax_tensor` to the graph. 
      #It is used for PREDICT and by the `logging_hook`.
      #probabilities from logits layer by applying softmax activation using tf.nn.softmax
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    #Return an EstimatorSpec object
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    # define a loss function that measures 
    #how closely the model's predictions match the target classes.
    
    #labels tensor contains a list of predictions for our examples, e.g. [1, 9, ...]
    #indices is just labels tensor, with values cast to integers. 
    #The depth is 10 because we have 10 possible target classes, one for each digit.
    
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy( onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    # use a learning rate of 0.001 
    # and stochastic gradient descent as the optimization algorithm:
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

#Reference 
#EstimatorSpec fully defines the model to be run by Estimator.
tf.estimator.EstimatorSpec ( mode, predictions=None,  loss=None, train_op=None, eval_metric_ops=None,
        export_outputs=None,  training_chief_hooks=None,  training_hooks=None,  
        scaffold=None)
    mode: A ModeKeys. Specifies if this is training, evaluation or prediction.
    predictions: Predictions Tensor or dict of Tensor.
    loss: Training loss Tensor. Must be either scalar, or with shape [1].
    train_op: Op for the training step.
    eval_metric_ops: Dict of metric results keyed by name. 
                    The values of the dict are the results of calling a metric function,
                    namely a (metric_tensor, update_op) tuple.
    Depending on the value of mode, different arguments are required. 
    Namely For mode == ModeKeys.TRAIN: required fields are loss and train_op. 
    For mode == ModeKeys.EVAL: required field isloss. 
    For mode == ModeKeys.PREDICT: required fields are predictions.

    model_fn can populate all arguments independent of mode. 
    In this case, some arguments will be ignored by Estimator. 
    E.g. train_op will be ignored in eval and infer modes. 
    
    #args might be in any order , but arg name must be same 
    def my_model_fn(mode, features, labels):
      predictions = ...
      loss = ...
      train_op = ...
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions,
          loss=loss,
          train_op=train_op)


##One-hot encoding 
#labels tensor contains a list of predictions for our examples, e.g. [1, 9, ...]. 
#In order to calculate cross-entropy, convert labels to the corresponding one-hot encoding:
[[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
 ...]

#tf.one_hot() has two required arguments:
    indices. The locations in the one-hot tensor that will have "on values"?
             i.e., the locations of 1 values in the tensor shown above.
    depth. The depth of the one-hot tensor?
            i.e., the number of target classes. Here, the depth is 10.



##Load Training and Test Data

def main(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator( model_fn=cnn_model_fn, model_dir="C:/tmp/mnist_convnet_model")

    # Set up logging for predictions
    # store a dict of the tensors we want to log in tensors_to_log
    #"softmax_tensor" is from earlier 'name' in predictions 
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)


    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])


    # Evaluate the model and print results
    #we set num_epochs=1, so that the model evaluates the metrics over one epoch of data 
    #and returns the result. 
    #set shuffle=False to iterate through the data sequentially.
    
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    
    
##Run the Model
#As the model trains, you'll see log output like the following:

INFO:tensorflow:loss = 2.36026, step = 1
INFO:tensorflow:probabilities = [[ 0.07722801  0.08618255  0.09256398, ...]]
...
INFO:tensorflow:loss = 2.13119, step = 101
INFO:tensorflow:global_step/sec: 5.44132
...
INFO:tensorflow:Loss for final step: 0.553216.

INFO:tensorflow:Restored model from /tmp/mnist_convnet_model
INFO:tensorflow:Eval steps [0,inf) for training step 20000.
INFO:tensorflow:Input iterator is exhausted.
INFO:tensorflow:Saving evaluation summary for step 20000: accuracy = 0.9733, loss = 0.0902271
{'loss': 0.090227105, 'global_step': 20000, 'accuracy': 0.97329998}

#Here, we've achieved an accuracy of 97.3% on our test data set.


##code 
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="c:/tmp/mnist_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()


##Reference  - tf.layers - New API for NN eg RNN and CNN 

tf.layers.average_pooling1d(inputs, pool_size, strides,  padding='valid',  data_format='channels_last',  name=None)
    Average Pooling layer for 1D inputs.
    Returns:The output tensor, of rank 3.
    inputs: The tensor over which to pool. Must have rank 3.
    pool_size: An integer or tuple/list of a single integer, representing the size of the pooling window.
    strides: An integer or tuple/list of a single integer, specifying the strides of the pooling operation.
      

tf.layers.average_pooling2d(inputs,  pool_size, strides, padding='valid',data_format='channels_last', name=None  )
    Average pooling layer for 2D inputs (e.g. images).

tf.layers.average_pooling3d( inputs,pool_size,strides,padding='valid',data_format='channels_last',name=None)
    Average pooling layer for 3D inputs (e.g. volumes).

tf.layers.batch_normalization(inputs,axis=-1,
    momentum=0.99,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer=tf.zeros_initializer(),
    gamma_initializer=tf.ones_initializer(),
    moving_mean_initializer=tf.zeros_initializer(),
    moving_variance_initializer=tf.ones_initializer(),
    beta_regularizer=None,
    gamma_regularizer=None,
    training=False,
    trainable=True,
    name=None,
    reuse=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=False
)
Functional interface for the batch normalization layer.

tf.layers.conv1d(inputs,
    filters,
    kernel_size,
    strides=1,
    padding='valid',
    data_format='channels_last',
    dilation_rate=1,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    trainable=True,
    name=None,
    reuse=None
)
    Functional interface for 1D convolution layer (e.g. temporal convolution).
    inputs: Tensor input.
    filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
    kernel_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.


tf.layers.conv2d( inputs,
    filters,
    kernel_size,
    strides=(1, 1),
    padding='valid',
    data_format='channels_last',
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    trainable=True,
    name=None,
    reuse=None
)
Functional interface for the 2D convolution layer.

tf.layers.conv2d_transpose(inputs,
    inputs,
    filters,
    kernel_size,
    strides=(1, 1),
    padding='valid',
    data_format='channels_last',
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    trainable=True,
    name=None,
    reuse=None

)
Functional interface for transposed 2D convolution layer.

tf.layers.conv3d(inputs,
    filters,
    kernel_size,
    strides=(1, 1, 1),
    padding='valid',
    data_format='channels_last',
    dilation_rate=(1, 1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    trainable=True,
    name=None,
    reuse=None
)
Functional interface for the 3D convolution layer.

tf.layers.conv3d_transpose(inputs,
    filters,
    kernel_size,
    strides=(1, 1, 1),
    padding='valid',
    data_format='channels_last',
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    trainable=True,
    name=None,
    reuse=None
)
Functional interface for transposed 3D convolution layer.

tf.layers.dense(inputs,
    units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    trainable=True,
    name=None,
    reuse=None
)
Functional interface for the densely-connected layer.

tf.layers.dropout(inputs,
    rate=0.5,
    noise_shape=None,
    seed=None,
    training=False,
    name=None
)
    Applies Dropout to the input.
    Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting. The units that are kept are scaled by 1 / (1 - rate), so that their sum is unchanged at training time and inference time.
    inputs: Tensor input.
    rate: The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out 10% of input units.
    
tf.layers.max_pooling1d(inputs,
    pool_size,
    strides,
    padding='valid',
    data_format='channels_last',
    name=None
)
Max Pooling layer for 1D inputs.

tf.layers.max_pooling2d(inputs,
    pool_size,
    strides,
    padding='valid',
    data_format='channels_last',
    name=None
)
Max pooling layer for 2D inputs (e.g. images).

tf.layers.max_pooling3d(inputs,
    pool_size,
    strides,
    padding='valid',
    data_format='channels_last',
    name=None
)
    Max pooling layer for 3D inputs (e.g. volumes).

tf.layers.separable_conv2d(inputs,
    filters,
    kernel_size,
    strides=(1, 1),
    padding='valid',
    data_format='channels_last',
    dilation_rate=(1, 1),
    depth_multiplier=1,
    activation=None,
    use_bias=True,
    depthwise_initializer=None,
    pointwise_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    depthwise_regularizer=None,
    pointwise_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    trainable=True,
    name=None,
    reuse=None
)
Functional interface for the depthwise separable 2D convolution layer.




##Reference -  Neural Network - low level API 
#Activation Functions
#All activation ops apply componentwise, and produce a tensor of the same shape as the input tensor.

    tf.nn.relu
    tf.nn.relu6
    tf.nn.crelu
    tf.nn.elu
    tf.nn.softplus
    tf.nn.softsign
    tf.nn.dropout
    tf.nn.bias_add
    tf.sigmoid
    tf.tanh

#Convolution
#The convolution ops sweep a 2-D filter over a batch of images, 
#applying the filter to each window of each image of the appropriate size. 
#The different ops trade off between generic vs. specific filters:
        conv2d: Arbitrary filters that can mix channels together.
        depthwise_conv2d: Filters that operate on each channel independently.
        separable_conv2d: A depthwise spatial filter followed by a pointwise filter.

    tf.nn.convolution
    tf.nn.conv2d
    tf.nn.depthwise_conv2d
    tf.nn.depthwise_conv2d_native
    tf.nn.separable_conv2d
    tf.nn.atrous_conv2d
    tf.nn.atrous_conv2d_transpose
    tf.nn.conv2d_transpose
    tf.nn.conv1d
    tf.nn.conv3d
    tf.nn.conv3d_transpose
    tf.nn.conv2d_backprop_filter
    tf.nn.conv2d_backprop_input
    tf.nn.conv3d_backprop_filter_v2
    tf.nn.depthwise_conv2d_native_backprop_filter
    tf.nn.depthwise_conv2d_native_backprop_input

#Pooling
#The pooling ops sweep a rectangular window over the input tensor, 
#computing a reduction operation for each window (average, max, or max with argmax). 
#Each pooling op uses rectangular windows of size ksize separated by offset strides. 
    tf.nn.avg_pool
    tf.nn.max_pool
    tf.nn.max_pool_with_argmax
    tf.nn.avg_pool3d
    tf.nn.max_pool3d
    tf.nn.fractional_avg_pool
    tf.nn.fractional_max_pool
    tf.nn.pool

#Morphological filtering
#Morphological operators are non-linear filters used in image processing.
    tf.nn.dilation2d
    tf.nn.erosion2d
    tf.nn.with_space_to_batch

#Normalization
#Normalization is useful to prevent neurons from saturating when inputs may have varying scale, and to aid generalization.
    tf.nn.l2_normalize
    tf.nn.local_response_normalization
    tf.nn.sufficient_statistics
    tf.nn.normalize_moments
    tf.nn.moments
    tf.nn.weighted_moments
    tf.nn.fused_batch_norm
    tf.nn.batch_normalization
    tf.nn.batch_norm_with_global_normalization

#Losses
#The loss ops measure error between two tensors, or between a tensor and zero. 
#These can be used for measuring accuracy of a network in a regression task or for regularization purposes (weight decay).

    tf.nn.l2_loss
    tf.nn.log_poisson_loss

#Classification
    tf.nn.sigmoid_cross_entropy_with_logits
    tf.nn.softmax
    tf.nn.log_softmax
    tf.nn.softmax_cross_entropy_with_logits
    tf.nn.sparse_softmax_cross_entropy_with_logits
    tf.nn.weighted_cross_entropy_with_logits

#Embeddings
#TensorFlow provides library support for looking up values in embedding tensors.
    tf.nn.embedding_lookup
    tf.nn.embedding_lookup_sparse

#Recurrent Neural Networks
#TensorFlow provides a number of methods for constructing Recurrent Neural Networks. 
#Most accept an RNNCell-subclassed object (see the documentation for tf.contrib.rnn).
    tf.nn.dynamic_rnn
    tf.nn.bidirectional_dynamic_rnn
    tf.nn.raw_rnn

#Connectionist Temporal Classification (CTC)
    tf.nn.ctc_loss
    tf.nn.ctc_greedy_decoder
    tf.nn.ctc_beam_search_decoder

#Evaluation
#The evaluation ops are useful for measuring the performance of a network. 
#They are typically used at evaluation time.
    tf.nn.top_k
    tf.nn.in_top_k

#Sampled Loss Functions
#TensorFlow provides the following sampled loss functions for faster training.
    tf.nn.nce_loss
    tf.nn.sampled_softmax_loss

#Candidate Samplers
#TensorFlow provides the following samplers for randomly sampling candidate classes when using one of the sampled loss functions above.
    tf.nn.uniform_candidate_sampler
    tf.nn.log_uniform_candidate_sampler
    tf.nn.learned_unigram_candidate_sampler
    tf.nn.fixed_unigram_candidate_sampler

#Miscellaneous candidate sampling utilities
    tf.nn.compute_accidental_hits

#Quantization ops

    tf.nn.quantized_conv2d
    tf.nn.quantized_relu_x
    tf.nn.quantized_max_pool
    tf.nn.quantized_avg_pool

##Reference -  Training - Optimizer
#The Optimizer base class provides methods to compute gradients for a loss 
#and apply gradients to variables. 
    tf.train.Optimizer
    tf.train.GradientDescentOptimizer
    tf.train.AdadeltaOptimizer
    tf.train.AdagradOptimizer
    tf.train.AdagradDAOptimizer
    tf.train.MomentumOptimizer
    tf.train.AdamOptimizer
    tf.train.FtrlOptimizer
    tf.train.ProximalGradientDescentOptimizer
    tf.train.ProximalAdagradOptimizer
    tf.train.RMSPropOptimizer

##Gradient Computation
#TensorFlow provides functions to compute the derivatives for a given TensorFlow computation graph, adding operations to the graph. The optimizer classes automatically compute derivatives on your graph, but creators of new Optimizers or expert users can call the lower-level functions below.
    tf.gradients
    tf.AggregationMethod
    tf.stop_gradient
    tf.hessians

##Gradient Clipping
#TensorFlow provides several operations that you can use to add clipping functions to your graph. You can use these functions to perform general data clipping, but they're particularly useful for handling exploding or vanishing gradients.
    tf.clip_by_value
    tf.clip_by_norm
    tf.clip_by_average_norm
    tf.clip_by_global_norm
    tf.global_norm

##Decaying the learning rate
    tf.train.exponential_decay
    tf.train.inverse_time_decay
    tf.train.natural_exp_decay
    tf.train.piecewise_constant
    tf.train.polynomial_decay

##Moving Averages
#Some training algorithms, such as GradientDescent and Momentum often benefit from maintaining a moving average of variables during optimization. Using the moving averages for evaluations often improve results significantly.
    tf.train.ExponentialMovingAverage

##Coordinator and QueueRunner
    tf.train.Coordinator
    tf.train.QueueRunner
    tf.train.LooperThread
    tf.train.add_queue_runner
    tf.train.start_queue_runners

##Distributed execution
    tf.train.Server
    tf.train.Supervisor
    tf.train.SessionManager
    tf.train.ClusterSpec
    tf.train.replica_device_setter
    tf.train.MonitoredTrainingSession
    tf.train.MonitoredSession
    tf.train.SingularMonitoredSession
    tf.train.Scaffold
    tf.train.SessionCreator
    tf.train.ChiefSessionCreator
    tf.train.WorkerSessionCreator

##Reading Summaries from Event Files
    tf.train.summary_iterator

##Training Hooks
#Hooks are tools that run in the process of training/evaluation of the model.
    tf.train.SessionRunHook
    tf.train.SessionRunArgs
    tf.train.SessionRunContext
    tf.train.SessionRunValues
    tf.train.LoggingTensorHook
    tf.train.StopAtStepHook
    tf.train.CheckpointSaverHook
    tf.train.NewCheckpointReader
    tf.train.StepCounterHook
    tf.train.NanLossDuringTrainingError
    tf.train.NanTensorHook
    tf.train.SummarySaverHook
    tf.train.GlobalStepWaiterHook
    tf.train.FinalOpsHook
    tf.train.FeedFnHook

##Training Utilities
    tf.train.global_step
    tf.train.basic_train_loop
    tf.train.get_global_step
    tf.train.assert_global_step
    tf.train.write_graph




##Reference  Module: tf.metrics
#Calculate a measurement between labels and predictions 
tf.metrics.accuracy(labels,
    predictions,
    weights=None,
    metrics_collections=None,
    updates_collections=None,
    name=None
)
    Calculates how often predictions matches labels.

tf.metrics.auc(labels,
    predictions,
    weights=None,
    num_thresholds=200,
    metrics_collections=None,
    updates_collections=None,
    curve='ROC',
    name=None
)
    Computes the approximate AUC via a Riemann sum.

tf.metrics.false_negatives(...)
    Computes the total number of false negatives.

tf.metrics.false_positives(...)
    Sum the weights of false positives.

tf.metrics.mean(...)
    Computes the (weighted) mean of the given values.

tf.metrics.mean_absolute_error(...)
    Computes the mean absolute error between the labels and predictions.

tf.metrics.mean_cosine_distance(...)
    Computes the cosine distance between the labels and predictions.

tf.metrics.mean_iou(...)
    Calculate per-step mean Intersection-Over-Union (mIOU).

tf.metrics.mean_per_class_accuracy(...)
    Calculates the mean of the per-class accuracies.

tf.metrics.mean_relative_error(...)
    Computes the mean relative error by normalizing with the given values.

tf.metrics.mean_squared_error(...)
    Computes the mean squared error between the labels and predictions.

tf.metrics.mean_tensor(...)
    Computes the element-wise (weighted) mean of the given tensors.

tf.metrics.percentage_below(...)
    Computes the percentage of values less than the given threshold.

tf.metrics.precision(...)
    Computes the precision of the predictions with respect to the labels.

tf.metrics.precision_at_thresholds(...)
    Computes precision values for different thresholds on predictions.

tf.metrics.recall(...): Computes the recall of the predictions with respect to the labels.

tf.metrics.recall_at_k(...)
    Computes recall@k of the predictions with respect to sparse labels.

tf.metrics.recall_at_thresholds(...)
    Computes various recall values for different thresholds on predictions.

tf.metrics.root_mean_squared_error(...)
    Computes the root mean squared error between the labels and predictions.

tf.metrics.sensitivity_at_specificity(...)
    Computes the specificity at a given sensitivity.

tf.metrics.sparse_average_precision_at_k(...)
    Computes average precision@k of predictions with respect to sparse labels.

tf.metrics.sparse_precision_at_k(...)
    Computes precision@k of the predictions with respect to sparse labels.

tf.metrics.specificity_at_sensitivity(...)
    Computes the specificity at a given sensitivity.

tf.metrics.true_positives(...)
    Sum the weights of true_positives.

    
    
##Reference  Module: tf.losses
#Loss operations for use in neural networks.
#All the losses are added to the GraphKeys.LOSSES collection by default.

tf.losses.absolute_difference(labels,
    predictions,
    weights=1.0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
)
Adds an Absolute Difference loss to the training procedure.

tf.losses.add_loss(...)
Adds a externally defined loss to the collection of losses.

tf.losses.compute_weighted_loss(...)
Computes the weighted loss.

tf.losses.cosine_distance(...)
Adds a cosine-distance loss to the training procedure.

tf.losses.get_losses(...)
Gets the list of losses from the loss_collection.

tf.losses.get_regularization_loss(...)
Gets the total regularization loss.

tf.losses.get_regularization_losses(...)
Gets the list of regularization losses.

tf.losses.get_total_loss(...)
Returns a tensor whose value represents the total loss.

tf.losses.hinge_loss(...)
Adds a hinge loss to the training procedure.

tf.losses.huber_loss(...)
Adds a Huber Loss term to the training procedure.

tf.losses.log_loss(...)
Adds a Log Loss term to the training procedure.

tf.losses.mean_pairwise_squared_error(...)
Adds a pairwise-errors-squared loss to the training procedure.

tf.losses.mean_squared_error(...)
Adds a Sum-of-Squares loss to the training procedure.

tf.losses.sigmoid_cross_entropy(...)
Creates a cross-entropy loss using tf.nn.sigmoid_cross_entropy_with_logits.

tf.losses.softmax_cross_entropy(onehot_labels,
    logits,
    weights=1.0,
    label_smoothing=0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
Creates a cross-entropy loss using tf.nn.softmax_cross_entropy_with_logits.

tf.losses.sparse_softmax_cross_entropy(...)
Cross-entropy loss using tf.nn.sparse_softmax_cross_entropy_with_logits.

##Reference -  Images
#Encoding and Decoding
#TensorFlow provides Ops to decode and encode JPEG and PNG formats. 
#Encoded images are represented by scalar string Tensors, 
#decoded images by 3-D uint8 tensors of shape [height, width, channels]. 
#(PNG also supports uint16.)
    tf.image.decode_gif
    tf.image.decode_jpeg
    tf.image.encode_jpeg
    tf.image.decode_png
    tf.image.encode_png
    tf.image.decode_image

#Resizing
# Decode a JPG image and resize it to 299 by 299 using default method.
image = tf.image.decode_jpeg(...)
resized_image = tf.image.resize_images(image, [299, 299])

    tf.image.resize_images
    tf.image.resize_area
    tf.image.resize_bicubic
    tf.image.resize_bilinear
    tf.image.resize_nearest_neighbor

#Cropping
    tf.image.resize_image_with_crop_or_pad
    tf.image.central_crop
    tf.image.pad_to_bounding_box
    tf.image.crop_to_bounding_box
    tf.image.extract_glimpse
    tf.image.crop_and_resize

#Flipping, Rotating and Transposing
    tf.image.flip_up_down
    tf.image.random_flip_up_down
    tf.image.flip_left_right
    tf.image.random_flip_left_right
    tf.image.transpose_image
    tf.image.rot90

#Converting Between Colorspaces
# Decode an image and convert it to HSV.
rgb_image = tf.image.decode_png(...,  channels=3)
rgb_image_float = tf.image.convert_image_dtype(rgb_image, tf.float32)
hsv_image = tf.image.rgb_to_hsv(rgb_image)

    tf.image.rgb_to_grayscale
    tf.image.grayscale_to_rgb
    tf.image.hsv_to_rgb
    tf.image.rgb_to_hsv
    tf.image.convert_image_dtype

#Image Adjustments
#TensorFlow provides functions to adjust images in various ways: brightness, contrast, hue, and saturation. 
#Each adjustment can be done with predefined parameters or with random parameters picked from predefined intervals. Random adjustments are often useful to expand a training set and reduce overfitting.

    tf.image.adjust_brightness
    tf.image.random_brightness
    tf.image.adjust_contrast
    tf.image.random_contrast
    tf.image.adjust_hue
    tf.image.random_hue
    tf.image.adjust_gamma
    tf.image.adjust_saturation
    tf.image.random_saturation
    tf.image.per_image_standardization

#Working with Bounding Boxes
    tf.image.draw_bounding_boxes
    tf.image.non_max_suppression
    tf.image.sample_distorted_bounding_box

#Denoising
    tf.image.total_variation

    
    
    
    


###Tensorflow - RNN -    recurrent neural network for language modeling. 
#The goal of the problem is to fit a probabilistic model which assigns probabilities to sentences. 
#It does so by predicting next words in a text given a history of previous words.  

#Use TFLearn
    
##Data 
#data/ directory of the http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
#The dataset is already preprocessed and contains overall 10000 different words, including the end-of-sentence marker 
#and a special symbol (\<unk>) for rare words.


##reader.py, 
#convert each word to a unique integer identifier
    
"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import tensorflow as tf

Py3 = sys.version_info[0] == 3

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y  
    
    
##ptb_word_lm.py
#check code advanced/tensflow/models-tutorials    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
###General Applications -  Mandelbrot Set
$ pip install ipython
$ pip install jupyter
#To start ipython 
$ ipython 
#to start jupyter 
$ jupyter notebook 
#then browser http://localhost:8888  
#first time  token URL is displayed in console, use that eg    http://localhost:8888/?token=d91243f68b99dded7c3ce87a679a953fd56588f1e65a0ea3
#then New -> Python3 ..
#To save as pdf in jupyter , install pandoc and MikTex 
# https://github.com/jgm/pandoc/releases/latest
# https://miktex.org/download
# then start jupyter - Note first time while saving, it may download many files, hence may fail couple of time, but keep on trying 

##code 
# Import libraries for simulation
import tensorflow as tf
import numpy as np

# Imports for visualization
import PIL.Image
from io import BytesIO
from IPython.display import Image, display


def DisplayFractal(a, fmt='jpeg'):
    """Display an array of iteration counts as a
    colorful picture of a fractal."""
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic),
                        30+50*np.sin(a_cyclic),
                        155-80*np.cos(a_cyclic)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

##Session and Variable Initialization
sess = tf.InteractiveSession()

# Use NumPy to create a 2D array of complex numbers
Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
Z = X+1j*Y

#initialize TensorFlow tensors.
xs = tf.constant(Z.astype(np.complex64))
zs = tf.Variable(xs)
ns = tf.Variable(tf.zeros_like(xs, tf.float32))

tf.global_variables_initializer().run()

#Defining and Running the Computation

# Compute the new values of z: z^2 + x
zs_ = zs*zs + xs

# Have we diverged with this new value?
not_diverged = tf.abs(zs_) < 4

# Operation to update the zs and the iteration count.
#
#Create an op that groups multiple operations.
# tf.group(*inputs,**kwargs)

step = tf.group(
  zs.assign(zs_),
  ns.assign_add(tf.cast(not_diverged, tf.float32))
  )

for i in range(200): step.run()
DisplayFractal(ns.eval())  #returns a numpy array with the same contents as the tensor.



















  
    
###Keras
#high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK

$ pip install keras

#By default uses Tensorflow backend 
#change it in $HOME(or %USERPROFILE%)/.keras/keras.json
#change the field backend to "theano", "tensorflow", or "cntk
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
#details
•image_data_format: String, either "channels_last" or "channels_first". It specifies which data format convention Keras will follow. (keras.backend.image_data_format() returns it.)
•For 2D data (e.g. image), "channels_last" assumes (rows, cols, channels) while "channels_first" assumes (channels, rows, cols). 
•For 3D data, "channels_last" assumes (conv_dim1, conv_dim2, conv_dim3, channels) while "channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).
•epsilon: Float, a numeric fuzzing constant used to avoid dividing by zero in some operations.
•floatx: String, "float16", "float32", or "float64". Default float precision.
•backend: String, "tensorflow", "theano", or "cntk".



###Keras- Getting started with the Keras Sequential model

#The Sequential model is a linear stack of layers.

from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

#OR add layers via the .add() method:

model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))

##Specifying the input shape for first layer 
#only the first, because following layers can do automatic shape inference
    Pass an input_shape argument to the first layer. 
        This is a shape tuple (a tuple of integers or None entries, 
        where None indicates that any positive integer may be expected). 
        In input_shape, the batch dimension is not included.
    Some 2D layers, such as Dense, support the specification of their input shape 
        via the argument input_dim, 
        and some 3D temporal layers support the arguments input_dim and input_length.
    If you ever need to specify a fixed batch size for your inputs 
        (this is useful for stateful recurrent networks), 
        you can pass a batch_size argument to a layer. 
        If you pass both batch_size=32 and input_shape=(6, 8) to a layer, 
        it will then expect every batch of inputs to have the batch shape (32, 6, 8).

#equivalent code
model = Sequential()
model.add(Dense(32, input_shape=(784,)))

model = Sequential()
model.add(Dense(32, input_dim=784))

##Compilation - before training 
#pass three arguments:
    An optimizer. 
        This could be the string identifier of an existing optimizer (such as rmsprop or adagrad), 
        or an instance of the Optimizer class. 
    A loss function. 
        This is the objective that the model will try to minimize. 
        It can be the string identifier of an existing loss function (such as categorical_crossentropy or mse), 
        or it can be an objective function.
    A list of metrics. 
        For any classification problem you will want to set this to metrics=['accuracy']. 
        A metric could be the string identifier of an existing metric 
        or a custom metric function.

# For a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# For a mean squared error regression problem
model.compile(optimizer='rmsprop',
              loss='mse')

# For custom metrics
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])

##Training
#use Numpy arrays of input data and labels. 

# For a single-input model with 2 classes (binary classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)

# For a single-input model with 10 classes (categorical classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)


###Multilayer Perceptron (MLP) for multi-class softmax classification:

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
              
              
#In the neural network terminology:
#•one epoch = one forward pass and one backward pass of all the training examples
#•batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.
#•number of iterations = number of passes, each pass using [batch size] number of examples. 
#To be clear, one pass = one forward pass + one backward pass (we do not count the forward pass and backward pass as two different passes).

#Example: if you have 1000 training examples, and your batch size is 500, 
#then it will take 2 iterations to complete 1 epoch.


model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)

y_predict = model.predict(x_test, batch_size=32, verbose=0)


###MLP for binary classification:

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Generate dummy data
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)

##VGG-like convnet:

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

# Generate dummy data
x_train = np.random.random((100, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)

##Sequence classification with LSTM(Long short-term memory )

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)

##Sequence classification with 1D convolutions:

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(seq_length, 100)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)

##Stacked LSTM for sequence classification

#In this model, we stack 3 LSTM layers on top of each other, 
#making the model capable of learning higher-level temporal representations.

#The first two LSTMs return their full output sequences, 
#but the last one only returns the last step in its output sequence, 
#thus dropping the temporal dimension 
#(i.e. converting the input sequence into a single vector).

#stacked LSTM

from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))

# Generate dummy validation data
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))

model.fit(x_train, y_train,
          batch_size=64, epochs=5,
          validation_data=(x_val, y_val))

##Same stacked LSTM model, rendered "stateful"
#A stateful recurrent model is one for which the internal states (memories) 
#obtained after processing a batch of samples are reused as initial states 
#for the samples of the next batch. 

#This allows to process longer sequences while keeping computational complexity manageable.


from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10
batch_size = 32

# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
x_train = np.random.random((batch_size * 10, timesteps, data_dim))
y_train = np.random.random((batch_size * 10, num_classes))

# Generate dummy validation data
x_val = np.random.random((batch_size * 3, timesteps, data_dim))
y_val = np.random.random((batch_size * 3, num_classes))

model.fit(x_train, y_train,
          batch_size=batch_size, epochs=5, shuffle=False,
          validation_data=(x_val, y_val))



###Keras - Getting started with the Keras functional API
#for defining complex models, 
#such as multi-output models, directed acyclic graphs, or models with shared layers.

##a densely-connected network
#The Sequential model is probably a better choice to implement such a network, 
1.A layer instance is callable (on a tensor), and it returns a tensor
2.Input tensor(s) and output tensor(s) can then be used to define a Model
3.Such a model can be trained just like Keras Sequential models.

from keras.layers import Input, Dense
from keras.models import Model

# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels)  # starts training

##All models are callable, just like layers
#you can treat any model as if it were a layer, by calling it on a tensor. 
#Note that by calling a model you aren't just reusing the architecture of the model, 
#you are also reusing its weights.

x = Input(shape=(784,))
# This works, and returns the 10-way softmax we defined above.
y = model(x)

#This can allow, for instance, to quickly create models 
#that can process sequences of inputs. 
#You could turn an image classification model into a video classification model, 
#in just one line.

from keras.layers import TimeDistributed

# Input tensor for sequences of 20 timesteps,
# each containing a 784-dimensional vector
input_sequences = Input(shape=(20, 784))

# This applies our previous model to every timestep in the input sequences.
# the output of the previous model was a 10-way softmax,
# so the output of the layer below will be a sequence of 20 vectors of size 10.
processed_sequences = TimeDistributed(model)(input_sequences)




##Multi-input and multi-output models

#Example - to predict how many retweets and likes a news headline will receive on Twitter. 
#The main input to the model will be the headline itself, as a sequence of words, 
#will also have an auxiliary input, receiving extra data such as the time of day when the headline was posted, etc. 

#The model will also be supervised via two loss functions. 
#Using the main loss function earlier in a model is a good regularization mechanism for deep models.


                    main input(InputLayer)
                            |
                            v
                    embedding_1(Embedding)
                            |
                            v
aux_input(InputLayer)   lstm_1(LSTM)
              \             /        \
               v           v          v
                merge_(Merge)      aux_output(Dense)
                     |
                     v
                 dense_1(Dense)
                     |
                     v
                 dense_2(Dense)
                     |
                     v
                 dense_3(Dense)
                     |
                     v
                 main_output(Dense)




#The main input will receive the headline, as a sequence of integers 
#(each integer encodes a word). 
#The integers will be between 1 and 10,000 (a vocabulary of 10,000 words) 
#and the sequences will be 100 words long.

from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
main_input = Input(shape=(100,), dtype='int32', name='main_input')

# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# A LSTM(Long short-term memory ) will transform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = LSTM(32)(x)

#insert the auxiliary loss and get aux output 
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

#feed into the model our auxiliary input data by concatenating it with the LSTM output

auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# And finally we add the main logistic regression layer
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

#This defines a model with two inputs and two outputs:
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1., 0.2])

#train the model by passing it lists of input arrays and target arrays:

model.fit([headline_data, additional_data], [labels, labels],
          epochs=50, batch_size=32)

#Since our inputs and outputs are named 
#We could also have compiled the model via:

model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})

# And trained it via:
model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': labels, 'aux_output': labels},
          epochs=50, batch_size=32)

          
      




      
##Shared layers

#Example - to build a model that can tell whether two tweets are from the same person or not 
#(this can allow us to compare users by the similarity of their tweets, for instance).

#One way to achieve this is to build a model that encodes two tweets into two vectors, 
#concatenates the vectors and then adds a logistic regression; 
#this outputs a probability that the two tweets share the same author. 
#The model would then be trained on positive tweet pairs and negative tweet pairs.

#Because the problem is symmetric, 
#the mechanism that encodes the first tweet should be reused (weights and all) 
#to encode the second tweet. 
#Here we use a shared LSTM layer to encode the tweets.


#We will take as input for a tweet a binary matrix of shape (140, 256), 
#i.e. a sequence of 140 vectors of size 256, 
#where each dimension in the 256-dimensional vector encodes the presence/absence of a character (out of an alphabet of 256 frequent characters).

import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

tweet_a = Input(shape=(140, 256))
tweet_b = Input(shape=(140, 256))

#To share a layer across different inputs, 
#simply instantiate the layer once, then call it on as many inputs as you want:

# This layer can take as input a matrix
# and will return a vector of size 64
shared_lstm = LSTM(64)

# When we reuse the same layer instance
# multiple times, the weights of the layer
# are also being reused
# (it is effectively *the same* layer)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

# We can then concatenate the two vectors:
merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)

# And add a logistic regression on top
predictions = Dense(1, activation='sigmoid')(merged_vector)

# We define a trainable model linking the
# tweet inputs to the predictions
model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([data_a, data_b], labels, epochs=10)


##The concept of layer "node"

#Whenever you are calling a layer on some input, 
#you are creating a new tensor (the output of the layer), 
#and you are adding a "node" to the layer, linking the input tensor to the output tensor.

#When you are calling the same layer multiple times, 
#that layer owns multiple nodes indexed as 0, 1, 2...


#As long as a layer is only connected to one input, 
#and layer.output will return the one output of the layer:

a = Input(shape=(140, 256))

lstm = LSTM(32)
encoded_a = lstm(a)

assert lstm.output == encoded_a

#if the layer has multiple inputs:

a = Input(shape=(140, 256))
b = Input(shape=(140, 256))

lstm = LSTM(32)
encoded_a = lstm(a)
encoded_b = lstm(b)

lstm.output

>> AttributeError: Layer lstm_1 has multiple inbound nodes,
hence the notion of "layer output" is ill-defined.
Use `get_output_at(node_index)` instead.

#The following works:

assert lstm.get_output_at(0) == encoded_a
assert lstm.get_output_at(1) == encoded_b

#The same is true for the properties input_shape and output_shape: 
#as long as the layer has only one node, 
#or as long as all nodes have the same input/output shape, 
#then the notion of "layer output/input shape" is well defined, 
#and that one shape will be returned by layer.output_shape/layer.input_shape. 

#But if, for instance, you apply the same Conv2D layer to an input of shape (32, 32, 3),
#and then to an input of shape (64, 64, 3), 
#the layer will have multiple input/output shapes, 
#and you will have to fetch them by specifying the index of the node they belong to:

a = Input(shape=(32, 32, 3))
b = Input(shape=(64, 64, 3))

conv = Conv2D(16, (3, 3), padding='same')
conved_a = conv(a)

# Only one input so far, the following will work:
assert conv.input_shape == (None, 32, 32, 3)

conved_b = conv(b)
# now the `.input_shape` property wouldn't work, but this does:
assert conv.get_input_shape_at(0) == (None, 32, 32, 3)
assert conv.get_input_shape_at(1) == (None, 64, 64, 3)


##Inception module

#https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf

from keras.layers import Conv2D, MaxPooling2D, Input

input_img = Input(shape=(256, 256, 3))

tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

##Residual connection on a convolution layer
#https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf


from keras.layers import Conv2D, Input

# input tensor for a 3-channel 256x256 image
x = Input(shape=(256, 256, 3))
# 3x3 conv with 3 output channels (same as input channels)
y = Conv2D(3, (3, 3), padding='same')(x)
# this returns x + y.
z = keras.layers.add([x, y])

##Shared vision model
#This model reuses the same image-processing module on two inputs, 
#to classify whether two MNIST digits are the same digit or different digits.

from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model

# First, define the vision modules
digit_input = Input(shape=(27, 27, 1))
x = Conv2D(64, (3, 3))(digit_input)
x = Conv2D(64, (3, 3))(x)
x = MaxPooling2D((2, 2))(x)
out = Flatten()(x)

vision_model = Model(digit_input, out)

# Then define the tell-digits-apart model
digit_a = Input(shape=(27, 27, 1))
digit_b = Input(shape=(27, 27, 1))

# The vision model will be shared, weights and all
out_a = vision_model(digit_a)
out_b = vision_model(digit_b)

concatenated = keras.layers.concatenate([out_a, out_b])
out = Dense(1, activation='sigmoid')(concatenated)

classification_model = Model([digit_a, digit_b], out)

##Visual question answering model
#This model can select the correct one-word answer 
#when asked a natural-language question about a picture.

#It works by encoding the question into a vector, encoding the image into a vector, 
#concatenating the two, and training on top a logistic regression over some vocabulary of potential answers.

from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential

# First, let's define a vision model using a Sequential model.
# This model will encode an image into a vector.
vision_model = Sequential()
vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
vision_model.add(Conv2D(64, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(128, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Flatten())

# Now let's get a tensor with the output of our vision model:
image_input = Input(shape=(224, 224, 3))
encoded_image = vision_model(image_input)

# Next, let's define a language model to encode the question into a vector.
# Each question will be at most 100 word long,
# and we will index words as integers from 1 to 9999.
question_input = Input(shape=(100,), dtype='int32')
embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
encoded_question = LSTM(256)(embedded_question)

# Let's concatenate the question vector and the image vector:
merged = keras.layers.concatenate([encoded_question, encoded_image])

# And let's train a logistic regression over 1000 words on top:
output = Dense(1000, activation='softmax')(merged)

# This is our final model:
vqa_model = Model(inputs=[image_input, question_input], outputs=output)

# The next stage would be training this model on actual data.

##Video question answering model
#we can quickly turn it into a video QA model. With appropriate training, 
#you will be able to show it a short video (e.g. 100-frame human action) 
#and ask a natural language question about the video (e.g. "what sport is the boy playing?" -> "football").

from keras.layers import TimeDistributed

video_input = Input(shape=(100, 224, 224, 3))
# This is our video encoded via the previously trained vision_model (weights are reused)
encoded_frame_sequence = TimeDistributed(vision_model)(video_input)  # the output will be a sequence of vectors
encoded_video = LSTM(256)(encoded_frame_sequence)  # the output will be a vector

# This is a model-level representation of the question encoder, reusing the same weights as before:
question_encoder = Model(inputs=question_input, outputs=encoded_question)

# Let's use it to encode the question:
video_question_input = Input(shape=(100,), dtype='int32')
encoded_video_question = question_encoder(video_question_input)

# And this is our video question answering model:
merged = keras.layers.concatenate([encoded_video, encoded_video_question])
output = Dense(1000, activation='softmax')(merged)
video_qa_model = Model(inputs=[video_input, video_question_input], outputs=output)





###Keras - Models 

#There are two types of models available in Keras: 
#the Sequential model and the Model class used with functional API.

#These models have a number of methods in common:
    model.summary(): prints a summary representation of your model. Shortcut for utils.print_summary
    model.get_config(): returns a dictionary containing the configuration of the model. The model can be reinstantiated from its config via:

config = model.get_config()
model = Model.from_config(config)
# or, for Sequential:
model = Sequential.from_config(config)

#Other common methods 
    model.get_weights(): returns a list of all weight tensors in the model, as Numpy arrays.
    model.set_weights(weights): sets the values of the weights of the model, from a list of Numpy arrays. The arrays in the list should have the same shape as those returned by get_weights().
    model.to_json(): returns a representation of the model as a JSON string. Note that the representation does not include the weights, only the architecture. You can reinstantiate the same model (with reinitialized weights) from the JSON string via:

from models import model_from_json
json_string = model.to_json()
model = model_from_json(json_string)


#Other common methods 
    model.to_yaml(): returns a representation of the model as a YAML string. Note that the representation does not include the weights, only the architecture. You can reinstantiate the same model (with reinitialized weights) from the YAML string via:

from models import model_from_yaml

yaml_string = model.to_yaml()
model = model_from_yaml(yaml_string)

#Other common methods 
    model.save_weights(filepath): saves the weights of the model as a HDF5 file.
    model.load_weights(filepath, by_name=False): loads the weights of the model from a HDF5 file (created by save_weights). By default, the architecture is expected to be unchanged. To load weights into a different architecture (with some layers in common), use by_name=True to load only those layers with the same name.


    

###Keras - Models - The Sequential model API

#Useful attributes of Model
    model.layers is a list of the layers added to the model.

##Sequential model methods

compile(self, optimizer, loss, metrics=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
    Configures the model for training.
    Arguments
        optimizer: String (name of optimizer) or optimizer object. See optimizers.
        loss: String (name of objective function) or objective function. See losses. If the model has multiple outputs, you can use a different loss on each output by passing a dictionary or a list of losses. The loss value that will be minimized by the model will then be the sum of all individual losses.
        metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use metrics=['accuracy']. To specify different metrics for different outputs of a multi-output model, you could also pass a dictionary, such as metrics={'output_a': 'accuracy'}.
        sample_weight_mode: If you need to do timestep-wise sample weighting (2D weights), set this to "temporal". None defaults to sample-wise weights (1D). If the model has multiple outputs, you can use a different sample_weight_mode on each output by passing a dictionary or a list of modes.
        weighted_metrics: List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
        target_tensors: By default, Keras will create a placeholder for the model's target, which will be fed with the target data during training. If instead you would like to use your own target tensor (in turn, Keras will not expect external Numpy data for these targets at training time), you can specify them via the target_tensors argument. It should be a single tensor (for a single-output Sequential model).
        **kwargs: When using the Theano/CNTK backends, these arguments are passed into K.function. When using the TensorFlow backend, these arguments are passed into tf.Session.run.
    Raises
        ValueError: In case of invalid arguments for
    #Example
    model = Sequential()
    model.add(Dense(32, input_shape=(500,)))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])



fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
    Trains the model for a fixed number of epochs (iterations on a dataset).
    Arguments
        x: Numpy array of training data. If the input layer in the model is named, you can also pass a dictionary mapping the input name to a Numpy array. x can be None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).
        y: Numpy array of target (label) data. If the output layer in the model is named, you can also pass a dictionary mapping the output name to a Numpy array. y can be None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).
        batch_size: Integer or None. Number of samples per gradient update. If unspecified, it will default to 32.
        epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
        verbose: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        callbacks: List of keras.callbacks.Callback instances. List of callbacks to apply during training. See callbacks.
        validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling.
        validation_data: tuple (x_val, y_val) or tuple (x_val, y_val, val_sample_weights) on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. This will override validation_split.
        shuffle: Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). 'batch' is a special option for dealing with the limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not None.
        class_weight: Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
        sample_weight: Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only). You can either pass a flat (1D) Numpy array with the same length as the input samples (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile().
        initial_epoch: Epoch at which to start training (useful for resuming a previous training run).
        steps_per_epoch: Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.
        validation_steps: Only relevant if steps_per_epoch is specified. Total number of steps (batches of samples) to validate before stopping.
    Returns
    A History object. 
    Its History.history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
    Raises
        RuntimeError: If the model was never compiled.
        ValueError: In case of mismatch between the provided input data and what the model expects.



evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)
    Computes the loss on some input data, batch by batch.
    Arguments
        x: input data, as a Numpy array or list of Numpy arrays (if the model has multiple inputs).
        y: labels, as a Numpy array.
        batch_size: integer. Number of samples per gradient update.
        verbose: verbosity mode, 0 or 1.
        sample_weight: sample weights, as a Numpy array.
    Returns
    Scalar test loss (if the model has no metrics) or list of scalars (if the model computes other metrics). The attribute model.metrics_names will give you the display labels for the scalar outputs.
    Raises
        RuntimeError: if the model was never compiled.



predict(self, x, batch_size=32, verbose=0)
    Generates output predictions for the input samples.
    The input samples are processed batch by batch.
    Arguments
        x: the input data, as a Numpy array.
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.
    Returns
    A Numpy array of predictions.




train_on_batch(self, x, y, class_weight=None, sample_weight=None)
    Single gradient update over one batch of samples.
    Arguments
        x: input data, as a Numpy array or list of Numpy arrays (if the model has multiple inputs).
        y: labels, as a Numpy array.
        class_weight: dictionary mapping classes to a weight value, used for scaling the loss function (during training only).
        sample_weight: sample weights, as a Numpy array.
    Returns
    Scalar training loss (if the model has no metrics) or list of scalars (if the model computes other metrics). The attribute model.metrics_names will give you the display labels for the scalar outputs.
    Raises
        RuntimeError: if the model was never compiled.



test_on_batch(self, x, y, sample_weight=None)
    Evaluates the model over a single batch of samples.
    Arguments
        x: input data, as a Numpy array or list of Numpy arrays (if the model has multiple inputs).
        y: labels, as a Numpy array.
        sample_weight: sample weights, as a Numpy array.
    Returns
    Scalar test loss (if the model has no metrics) or list of scalars (if the model computes other metrics). The attribute model.metrics_names will give you the display labels for the scalar outputs.
    Raises
        RuntimeError: if the model was never compiled.



predict_on_batch(self, x)
    Returns predictions for a single batch of samples.
    Arguments
        x: input data, as a Numpy array or list of Numpy arrays (if the model has multiple inputs).
    Returns
    A Numpy array of predictions.


fit_generator(self, generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
    Fits the model on data generated batch-by-batch by a Python generator.
    The generator is run in parallel to the model, for efficiency. For instance, this allows you to do real-time data augmentation on images on CPU in parallel to training your model on GPU.
    Arguments
        generator: A generator. The output of the generator must be either
        a tuple (inputs, targets)
        a tuple (inputs, targets, sample_weights). All arrays should contain the same number of samples. The generator is expected to loop over its data indefinitely. An epoch finishes when steps_per_epoch batches have been seen by the model.
        steps_per_epoch: Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch. It should typically be equal to the number of samples of your dataset divided by the batch size. Optional for Sequence: if unspecified, will use the len(generator) as a number of steps.
        epochs: Integer, total number of iterations on the data. Note that in conjunction with initial_epoch, the parameter epochs is to be understood as "final epoch". The model is not trained for n steps given by epochs, but until the epoch epochs is reached.
        verbose: Verbosity mode, 0, 1, or 2.
        callbacks: List of callbacks to be called during training.
        validation_data: This can be either
        A generator for the validation data
        A tuple (inputs, targets)
        A tuple (inputs, targets, sample_weights).
        validation_steps: Only relevant if validation_data is a generator. Number of steps to yield from validation generator at the end of every epoch. It should typically be equal to the number of samples of your validation dataset divided by the batch size. Optional for Sequence: if unspecified, will use the len(validation_data) as a number of steps.
        class_weight: Dictionary mapping class indices to a weight for the class.
        max_queue_size: Maximum size for the generator queue
        workers: Maximum number of processes to spin up
        use_multiprocessing: if True, use process based threading. Note that because this implementation relies on multiprocessing, you should not pass non picklable arguments to the generator as they can't be passed easily to children processes.
        shuffle: Whether to shuffle the order of the batches at the beginning of each epoch. Only used with instances of Sequence (keras.utils.Sequence).
        initial_epoch: Epoch at which to start training (useful for resuming a previous training run).

    Returns
        A History object.
    Raises
        RuntimeError: if the model was never compiled.
    #Example
    def generate_arrays_from_file(path):
        while 1:
            f = open(path)
            for line in f:
                # create Numpy arrays of input data
                # and labels, from each line in the file
                x, y = process_line(line)
                yield (x, y)
            f.close()

    model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                        steps_per_epoch=1000, epochs=10)



evaluate_generator(self, generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False)
    Evaluates the model on a data generator.
    The generator should return the same kind of data as accepted by test_on_batch.
    Arguments
        generator: Generator yielding tuples (inputs, targets) or (inputs, targets, sample_weights)
        steps: Total number of steps (batches of samples) to yield from generator before stopping. Optional for Sequence: if unspecified, will use the len(generator) as a number of steps.
        max_queue_size: maximum size for the generator queue
        workers: maximum number of processes to spin up
        use_multiprocessing: if True, use process based threading. Note that because this implementation relies on multiprocessing, you should not pass non picklable arguments to the generator as they can't be passed easily to children processes.
    Returns
        Scalar test loss (if the model has no metrics) or list of scalars (if the model computes other metrics). The attribute model.metrics_names will give you the display labels for the scalar outputs.
    Raises
        RuntimeError: if the model was never compiled.



predict_generator(self, generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    Generates predictions for the input samples from a data generator.
    The generator should return the same kind of data as accepted by predict_on_batch.
    Arguments
        generator: generator yielding batches of input samples.
        steps: Total number of steps (batches of samples) to yield from generator before stopping. Optional for Sequence: if unspecified, will use the len(generator) as a number of steps.
        max_queue_size: maximum size for the generator queue
        workers: maximum number of processes to spin up
        use_multiprocessing: if True, use process based threading. Note that because this implementation relies on multiprocessing, you should not pass non picklable arguments to the generator as they can't be passed easily to children processes.
        verbose: verbosity mode, 0 or 1.
    Returns
        A Numpy array of predictions.


get_layer(self, name=None, index=None)
    Retrieve a layer that is part of the model.
    Returns a layer based on either its name (unique) or its index in the graph. 
    Indices are based on order of horizontal graph traversal (bottom-up).
    Arguments
        name: string, name of layer.
        index: integer, index of layer.
    Returns
        A layer instance.




###Keras - Functional model API 

#In the functional API, given some input tensor(s) and output tensor(s), 
#you can instantiate a Model via:

from keras.models import Model
from keras.layers import Input, Dense

a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(inputs=a, outputs=b)

#This model will include all layers required in the computation of b given a.
#In the case of multi-input or multi-output models, you can use lists as well:

model = Model(inputs=[a1, a2], outputs=[b1, b3, b3])

##Useful attributes of Model
    model.layers is a flattened list of the layers comprising the model graph.
    model.inputs is the list of input tensors.
    model.outputs is the list of output tensors.

##Methods

compile(self, optimizer, loss, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
    Configures the model for training.
    Arguments
        optimizer: String (name of optimizer) or optimizer instance. See optimizers.
        loss: String (name of objective function) or objective function. See losses. If the model has multiple outputs, you can use a different loss on each output by passing a dictionary or a list of losses. The loss value that will be minimized by the model will then be the sum of all individual losses.
        metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use metrics=['accuracy']. To specify different metrics for different outputs of a multi-output model, you could also pass a dictionary, such as metrics={'output_a': 'accuracy'}.
        loss_weights: Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs. The loss value that will be minimized by the model will then be the weighted sum of all individual losses, weighted by the loss_weights coefficients. If a list, it is expected to have a 1:1 mapping to the model's outputs. If a tensor, it is expected to map output names (strings) to scalar coefficients.
        sample_weight_mode: If you need to do timestep-wise sample weighting (2D weights), set this to "temporal". None defaults to sample-wise weights (1D). If the model has multiple outputs, you can use a different sample_weight_mode on each output by passing a dictionary or a list of modes.
        weighted_metrics: List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
        target_tensors: By default, Keras will create placeholders for the model's target, which will be fed with the target data during training. If instead you would like to use your own target tensors (in turn, Keras will not expect external Numpy data for these targets at training time), you can specify them via the target_tensors argument. It can be a single tensor (for a single-output model), a list of tensors, or a dict mapping output names to target tensors.
        **kwargs: When using the Theano/CNTK backends, these arguments are passed into K.function. When using the TensorFlow backend, these arguments are passed into tf.Session.run.
    Raises
        ValueError: In case of invalid arguments for optimizer, loss, metrics or sample_weight_mode.



fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
    Trains the model for a fixed number of epochs (iterations on a dataset).
    Arguments
        x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs). If input layers in the model are named, you can also pass a dictionary mapping input names to Numpy arrays. x can be None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).
        y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs). If output layers in the model are named, you can also pass a dictionary mapping output names to Numpy arrays. y can be None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).
        batch_size: Integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32.
        epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
        verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        callbacks: List of keras.callbacks.Callback instances. List of callbacks to apply during training. See callbacks.
        validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling.
        validation_data: tuple (x_val, y_val) or tuple (x_val, y_val, val_sample_weights) on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. validation_data will override validation_split.
        shuffle: Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). 'batch' is a special option for dealing with the limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not None.
        class_weight: Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
        sample_weight: Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only). You can either pass a flat (1D) Numpy array with the same length as the input samples (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile().
        initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        steps_per_epoch: Integer or None. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.
        validation_steps: Only relevant if steps_per_epoch is specified. Total number of steps (batches of samples) to validate before stopping.
    Returns
        A History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
    Raises
        RuntimeError: If the model was never compiled.
        ValueError: In case of mismatch between the provided input data and what the model expects.



evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None)
    Returns the loss value & metrics values for the model in test mode.
    Computation is done in batches.
    Arguments
        x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs). If input layers in the model are named, you can also pass a dictionary mapping input names to Numpy arrays. x can be None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).
        y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs). If output layers in the model are named, you can also pass a dictionary mapping output names to Numpy arrays. y can be None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).
        batch_size: Integer or None. Number of samples per evaluation step. If unspecified, batch_size will default to 32.
        verbose: 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar.
        sample_weight: Optional Numpy array of weights for the test samples, used for weighting the loss function. You can either pass a flat (1D) Numpy array with the same length as the input samples (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile().
        steps: Integer or None. Total number of steps (batches of samples) before declaring the evaluation round finished. The default None is equal to the number of samples in your dataset divided by the batch size.
    Returns
        Scalar test loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs and/or metrics). The attribute model.metrics_names will give you the display labels for the scalar outputs.


predict(self, x, batch_size=None, verbose=0, steps=None)
    Generates output predictions for the input samples.
    Computation is done in batches.
    Arguments
        x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple outputs).
        batch_size: Integer. If unspecified, it will default to 32.
        verbose: Verbosity mode, 0 or 1.
        steps: Total number of steps (batches of samples) before declaring the prediction round finished. Ignored with the default value of None.
    Returns
    Numpy array(s) of predictions.
    Raises
        ValueError: In case of mismatch between the provided input data and the model's expectations, or in case a stateful model receives a number of samples that is not a multiple of the batch size.



train_on_batch(self, x, y, sample_weight=None, class_weight=None)
    Runs a single gradient update on a single batch of data.
    Arguments
        x: Numpy array of training data, or list of Numpy arrays if the model has multiple inputs. If all inputs in the model are named, you can also pass a dictionary mapping input names to Numpy arrays.
        y: Numpy array of target data, or list of Numpy arrays if the model has multiple outputs. If all outputs in the model are named, you can also pass a dictionary mapping output names to Numpy arrays.
        sample_weight: Optional array of the same length as x, containing weights to apply to the model's loss for each sample. In the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile().
        class_weight: Optional dictionary mapping class indices (integers) to a weight (float) to apply to the model's loss for the samples from this class during training. This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
    Returns
        Scalar training loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs and/or metrics). The attribute model.metrics_names will give you the display labels for the scalar outputs.


test_on_batch(self, x, y, sample_weight=None)
    Test the model on a single batch of samples.
    Arguments
        x: Numpy array of test data, or list of Numpy arrays if the model has multiple inputs. If all inputs in the model are named, you can also pass a dictionary mapping input names to Numpy arrays.
        y: Numpy array of target data, or list of Numpy arrays if the model has multiple outputs. If all outputs in the model are named, you can also pass a dictionary mapping output names to Numpy arrays.
        sample_weight: Optional array of the same length as x, containing weights to apply to the model's loss for each sample. In the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile().
    Returns
        Scalar test loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs and/or metrics). The attribute model.metrics_names will give you the display labels for the scalar outputs.



predict_on_batch(self, x)
    Returns predictions for a single batch of samples.
    Arguments
        x: Input samples, as a Numpy array.
    Returns
        Numpy array(s) of predictions.


fit_generator(self, generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
    Fits the model on data yielded batch-by-batch by a Python generator.
    The generator is run in parallel to the model, for efficiency. For instance, this allows you to do real-time data augmentation on images on CPU in parallel to training your model on GPU.
    The use of keras.utils.Sequence guarantees the ordering and guarantees the single use of every input per epoch when using use_multiprocessing=True.
    Arguments
        generator: A generator or an instance of Sequence (keras.utils.Sequence) object in order to avoid duplicate data when using multiprocessing. The output of the generator must be either
        a tuple (inputs, targets)
        a tuple (inputs, targets, sample_weights). This tuple (a single output of the generator) makes a single batch. Therefore, all arrays in this tuple must have the same length (equal to the size of this batch). Different batches may have different sizes. For example, the last batch of the epoch is commonly smaller than the others, if the size of the dataset is not divisible by the batch size. The generator is expected to loop over its data indefinitely. An epoch finishes when steps_per_epoch batches have been seen by the model.
        steps_per_epoch: Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch. It should typically be equal to the number of samples of your dataset divided by the batch size. Optional for Sequence: if unspecified, will use the len(generator) as a number of steps.
        epochs: Integer, total number of iterations on the data.
        verbose: Verbosity mode, 0, 1, or 2.
        callbacks: List of callbacks to be called during training.
        validation_data: This can be either
        a generator for the validation data
        a tuple (inputs, targets)
        a tuple (inputs, targets, sample_weights).
        validation_steps: Only relevant if validation_data is a generator. Total number of steps (batches of samples) to yield from generator before stopping. Optional for Sequence: if unspecified, will use the len(validation_data) as a number of steps.
        class_weight: Dictionary mapping class indices to a weight for the class.
        max_queue_size: Integer. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
        workers: Integer. Maximum number of processes to spin up when using process based threading. If unspecified, workers will default to 1. If 0, will execute the generator on the main thread.
        use_multiprocessing: Boolean. If True, use process based threading. If unspecified, workers will default to False. Note that because this implementation relies on multiprocessing, you should not pass non picklable arguments to the generator as they can't be passed easily to children processes.
        shuffle: Whether to shuffle the order of the batches at the beginning of each epoch. Only used with instances of Sequence (keras.utils.Sequence).
        initial_epoch: Epoch at which to start training (useful for resuming a previous training run)
    Returns
        A History object.

    #Example
    def generate_arrays_from_file(path):
        while 1:
            f = open(path)
            for line in f:
                # create numpy arrays of input data
                # and labels, from each line in the file
                x1, x2, y = process_line(line)
                yield ({'input_1': x1, 'input_2': x2}, {'output': y})
            f.close()

    model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                        steps_per_epoch=10000, epochs=10)



evaluate_generator(self, generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False)
    Evaluates the model on a data generator.
    The generator should return the same kind of data as accepted by test_on_batch.
    Arguments
        generator: Generator yielding tuples (inputs, targets) or (inputs, targets, sample_weights) or an instance of Sequence (keras.utils.Sequence) object in order to avoid duplicate data when using multiprocessing.
        steps: Total number of steps (batches of samples) to yield from generator before stopping. Optional for Sequence: if unspecified, will use the len(generator) as a number of steps.
        max_queue_size: maximum size for the generator queue
        workers: Integer. Maximum number of processes to spin up when using process based threading. If unspecified, workers will default to 1. If 0, will execute the generator on the main thread.
        use_multiprocessing: if True, use process based threading. Note that because this implementation relies on multiprocessing, you should not pass non picklable arguments to the generator as they can't be passed easily to children processes.
    Returns
        Scalar test loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs and/or metrics). The attribute model.metrics_names will give you the display labels for the scalar outputs.


predict_generator(self, generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    Generates predictions for the input samples from a data generator.
    The generator should return the same kind of data as accepted by predict_on_batch.
    Arguments
        generator: Generator yielding batches of input samples or an instance of Sequence (keras.utils.Sequence) object in order to avoid duplicate data when using multiprocessing.
        steps: Total number of steps (batches of samples) to yield from generator before stopping. Optional for Sequence: if unspecified, will use the len(generator) as a number of steps.
        max_queue_size: Maximum size for the generator queue.
        workers: Integer. Maximum number of processes to spin up when using process based threading. If unspecified, workers will default to 1. If 0, will execute the generator on the main thread.
        use_multiprocessing: If True, use process based threading. Note that because this implementation relies on multiprocessing, you should not pass non picklable arguments to the generator as they can't be passed easily to children processes.
        verbose: verbosity mode, 0 or 1.
    Returns
    Numpy array(s) of predictions.
    Raises
        ValueError: In case the generator yields data in an invalid format.



get_layer(self, name=None, index=None)
    Retrieves a layer based on either its name (unique) or index.
    Indices are based on order of horizontal graph traversal (bottom-up).
    Arguments
        name: String, name of layer.
        index: Integer, index of layer.
    Returns
    A layer instance.






###Keras - Keras layers
#All Keras layers have a number of methods in common:
    layer.get_weights(): returns the weights of the layer as a list of Numpy arrays.
    layer.set_weights(weights): sets the weights of the layer from a list of Numpy arrays (with the same shapes as the output of get_weights).
    layer.get_config(): returns a dictionary containing the configuration of the layer. The layer can be reinstantiated from its config via:


layer = Dense(32)
config = layer.get_config()
reconstructed_layer = Dense.from_config(config)

#Or:
from keras import layers

config = layer.get_config()
layer = layers.deserialize({'class_name': layer.__class__.__name__,'config': config})


#If a layer has a single node 
#you can get its input tensor, output tensor, input shape and output shape via:
    layer.input
    layer.output
    layer.input_shape
    layer.output_shape

#If the layer has multiple nodes 
    layer.get_input_at(node_index)
    layer.get_output_at(node_index)
    layer.get_input_shape_at(node_index)
    layer.get_output_shape_at(node_index)








###Keras -Core Layers 

keras.layers.Dense(units, activation=None, use_bias=True, 
        kernel_initializer='glorot_uniform', bias_initializer='zeros', 
        kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
    Densely-connected NN layer.
    Dense implements the operation: output = activation(dot(input, kernel) + bias) 
    where activation is the element-wise activation function passed 
    as the activation argument, 
    kernel is a weights matrix created by the layer, 
    and bias is a bias vector created by the layer 
    (only applicable if use_bias is True).
    Note: if the input to the layer has a rank greater than 2, 
    then it is flattened prior to the initial dot product with kernel.    
    Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use . If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the kernel weights matrix (see initializers).
        bias_initializer: Initializer for the bias vector (see initializers).
        kernel_regularizer: Regularizer function applied to the kernel weights matrix (see regularizer).
        bias_regularizer: Regularizer function applied to the bias vector (see regularizer).
        activity_regularizer: Regularizer function applied to the output of the layer (its "activation"). (see regularizer).
        kernel_constraint: Constraint function applied to the kernel weights matrix (see constraints).
        bias_constraint: Constraint function applied to the bias vector (see constraints).
    Input shape
        nD tensor with shape: (batch_size, ..., input_dim). 
        The most common situation would be a 2D input with shape (batch_size, input_dim).
    Output shape
        nD tensor with shape: (batch_size, ..., units). 
        For instance, for a 2D input with shape (batch_size, input_dim), the output would have shape (batch_size, units).
    #Example
    # as first layer in a sequential model:
    model = Sequential()
    model.add(Dense(32, input_shape=(16,)))
    # now the model will take as input arrays of shape (*, 16)
    # and output arrays of shape (*, 32)
    # after the first layer, you don't need to specify
    # the size of the input anymore:
    model.add(Dense(32))


keras.layers.Activation(activation)
    Applies an activation function to an output.
    Arguments
        activation: name of activation function to use 
        or alternatively, a Theano or TensorFlow operation.
    Available activations
        softmax(x, axis=-1)
            Softmax activation function.
        elu(x, alpha=1.0)
        selu(x)
            Scaled Exponential Linear Unit. (Klambauer et al., 2017)
        softplus(x)
        softsign(x)
        relu(x, alpha=0.0, max_value=None)
        tanh(x)
        sigmoid(x)
        hard_sigmoid(x)
        linear(x)




keras.layers.LeakyReLU(alpha=0.3)
    Leaky version of a Rectified Linear Unit.
    

keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
    Parametric Rectified Linear Unit.
    

keras.layers.ELU(alpha=1.0)
    Exponential Linear Unit.

keras.layers.ThresholdedReLU(theta=1.0)
    Thresholded Rectified Linear Unit.

keras.layers.Dropout(rate, noise_shape=None, seed=None)
    Applies Dropout to the input.
    Dropout consists in randomly setting a fraction rate of input units 
    to 0 at each update during training time, which helps prevent overfitting.
    Arguments
        rate: float between 0 and 1. Fraction of the input units to drop.
        noise_shape: 1D integer tensor representing the shape of the binary dropout mask that will be multiplied with the input. For instance, if your inputs have shape (batch_size, timesteps, features) and you want the dropout mask to be the same for all timesteps, you can use noise_shape=(batch_size, 1, features).
        seed: A Python integer to use as random seed.


keras.layers.Flatten()
    Flattens the input. Does not affect the batch size.
    #Example
    model = Sequential()
    model.add(Conv2D(64, 3, 3,
                     border_mode='same',
                     input_shape=(3, 32, 32)))
    # now: model.output_shape == (None, 64, 32, 32)

    model.add(Flatten())
    # now: model.output_shape == (None, 65536)



keras.layers.Reshape(target_shape)
    Reshapes an output to a certain shape.
    Arguments
        target_shape: target shape. Tuple of integers. Does not include the batch axis.
    Input shape
        Arbitrary, although all dimensions in the input shaped must be fixed. 
        Use the keyword argument input_shape 
        (tuple of integers, does not include the batch axis) 
        when using this layer as the first layer in a model.
    Output shape
    (batch_size,) + target_shape
    #Example
    # as first layer in a Sequential model
    model = Sequential()
    model.add(Reshape((3, 4), input_shape=(12,)))
    # now: model.output_shape == (None, 3, 4)
    # note: `None` is the batch dimension

    # as intermediate layer in a Sequential model
    model.add(Reshape((6, 2)))
    # now: model.output_shape == (None, 6, 2)

    # also supports shape inference using `-1` as dimension
    model.add(Reshape((-1, 2, 2)))
    # now: model.output_shape == (None, 3, 2, 2)


keras.layers.Permute(dims)
    Permutes the dimensions of the input according to a given pattern.
    Useful for e.g. connecting RNNs and convnets together.
    #Example
    model = Sequential()
    model.add(Permute((2, 1), input_shape=(10, 64)))
    # now: model.output_shape == (None, 64, 10)
    # note: `None` is the batch dimension
    Arguments
        dims: Tuple of integers. Permutation pattern, does not include the samples dimension. Indexing starts at 1. 
        For instance, (2, 1) permutes the first and second dimension of the input.


keras.layers.RepeatVector(n)
    Repeats the input n times.
    #Example
    model = Sequential()
    model.add(Dense(32, input_dim=32))
    # now: model.output_shape == (None, 32)
    # note: `None` is the batch dimension
    model.add(RepeatVector(3))
    # now: model.output_shape == (None, 3, 32)
    Arguments
        n: integer, repetition factor.
    Input shape
        2D tensor of shape (num_samples, features).
    Output shape
        3D tensor of shape (num_samples, n, features).

        

keras.layers.Lambda(function, output_shape=None, mask=None, arguments=None)
    Wraps arbitrary expression as a Layer object.
    #Examples
    # add a x -> x^2 layer
    model.add(Lambda(lambda x: x ** 2))
    # add a layer that returns the concatenation
    # of the positive part of the input and
    # the opposite of the negative part
    def antirectifier(x):
        x -= K.mean(x, axis=1, keepdims=True)
        x = K.l2_normalize(x, axis=1)
        pos = K.relu(x)
        neg = K.relu(-x)
        return K.concatenate([pos, neg], axis=1)
    def antirectifier_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  # only valid for 2D tensors
        shape[-1] *= 2
        return tuple(shape)
    model.add(Lambda(antirectifier, output_shape=antirectifier_output_shape))
    Arguments
        function: The function to be evaluated. Takes input tensor as first argument.
        output_shape: Expected output shape from function. Only relevant when using Theano. Can be a tuple or function. If a tuple, it only specifies the first dimension onward; sample dimension is assumed either the same as the input: output_shape = (input_shape[0], ) + output_shape or, the input is None and the sample dimension is also None: output_shape = (None, ) + output_shape If a function, it specifies the entire shape as a function of the input shape: output_shape = f(input_shape)
        arguments: optional dictionary of keyword arguments to be passed to the function.



keras.layers.ActivityRegularization(l1=0.0, l2=0.0)
    Layer that applies an update to the cost function based input activity.
    Arguments
        l1: L1 regularization factor (positive float).
        l2: L2 regularization factor (positive float).



keras.layers.Masking(mask_value=0.0)
    Masks a sequence by using a mask value to skip timesteps.
    For each timestep in the input tensor (dimension #1 in the tensor), 
    if all values in the input tensor at that timestep are equal to mask_value, 
    then the timestep will be masked (skipped) in all downstream layers 
    (as long as they support masking).
    If any downstream layer does not support masking yet receives such an input mask, an exception will be raised.
    #Example
    Consider a Numpy data array x of shape (samples, timesteps, features), to be fed to an LSTM layer. 
    You want to mask timestep #3 and #5 because you lack data for these timesteps. 
    set x[:, 3, :] = 0. and x[:, 5, :] = 0.
    insert a Masking layer with mask_value=0. before the LSTM layer:
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
    model.add(LSTM(32))





###Keras - Convolutional Layers


keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', 
            dilation_rate=1, activation=None, use_bias=True, 
            kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
    This layer creates a convolution kernel that is convolved 
    with the layer input over a single spatial (or temporal) dimension 
    to produce a tensor of outputs. 
    If use_bias is True, a bias vector is created and added to the outputs. 
    Finally, if activation is not None, it is applied to the outputs as well.
    When using this layer as the first layer in a model, 
    provide an input_shape argument (tuple of integers or None) 
    e.g. (10, 128) for sequences of 10 vectors of 128-dimensional vectors, 
    or (None, 128) for variable-length sequences of 128-dimensional vectors.
    Arguments
        filters: Integer, the dimensionality of the output space (i.e. the number output of filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer, specifying the stride length of the convolution. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
        padding: One of "valid", "causal" or "same" (case-insensitive). "valid" means "no padding". "same" results in padding the input such that the output has the same length as the original input. "causal" results in causal (dilated) convolutions, e.g. output[t] does not depend on input[t+1:]. Useful when modeling temporal data where the model should not violate the temporal order. See WaveNet: A Generative Model for Raw Audio, section 2.1.
        dilation_rate: an integer or tuple/list of a single integer, specifying the dilation rate to use for dilated convolution. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any strides value != 1.
        activation: Activation function to use (see activations). If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the kernel weights matrix (see initializers).
        bias_initializer: Initializer for the bias vector (see initializers).
        kernel_regularizer: Regularizer function applied to the kernel weights matrix (see regularizer).
        bias_regularizer: Regularizer function applied to the bias vector (see regularizer).
        activity_regularizer: Regularizer function applied to the output of the layer (its "activation"). (see regularizer).
        kernel_constraint: Constraint function applied to the kernel matrix (see constraints).
        bias_constraint: Constraint function applied to the bias vector (see constraints).
    Input shape
        3D tensor with shape: (batch_size, steps, input_dim)
    Output shape
        3D tensor with shape: (batch_size, new_steps, filters) 
        steps value might have changed due to padding or strides.


keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), 
            padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
    2D convolution layer (e.g. spatial convolution over images).
    When using this layer as the first layer in a model, 
    provide the keyword argument input_shape 
    (tuple of integers, does not include the sample axis), 
    e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".
    Input shape
        4D tensor with shape: (samples, channels, rows, cols) 
        if data_format='channels_first' 
        or 4D tensor with shape: (samples, rows, cols, channels) if data_format='channels_last'.
    Output shape
        4D tensor with shape: (samples, filters, new_rows, new_cols) 
        if data_format='channels_first' 
        or 4D tensor with shape: (samples, new_rows, new_cols, filters) 
        if data_format='channels_last'. 
        rows and cols values might have changed due to padding.



keras.layers.SeparableConv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None)
    Depthwise separable 2D convolution.
    Separable convolutions consist in first performing a depthwise spatial convolution 
    (which acts on each input channel separately) followed by a pointwise convolution 
    which mixes together the resulting output channels. 
    The depth_multiplier argument controls how many output channels are generated per input channel 
    in the depthwise step.
    Intuitively, separable convolutions can be understood 
    as a way to factorize a convolution kernel into two smaller kernels, or as an extreme version of an Inception block.
    Input shape
        4D tensor with shape: (batch, channels, rows, cols) if data_format='channels_first' or 4D tensor with shape: (batch, rows, cols, channels) if data_format='channels_last'.
    Output shape
        4D tensor with shape: (batch, filters, new_rows, new_cols) if data_format='channels_first' or 4D tensor with shape: (batch, new_rows, new_cols, filters) if data_format='channels_last'. rows and cols values might have changed due to padding.



keras.layers.Conv2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
    Transposed convolution layer (sometimes called Deconvolution).
    When using this layer as the first layer in a model, 
    provide the keyword argument input_shape (
    tuple of integers, does not include the sample axis), 
    e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".
    Input shape
        4D tensor with shape: (batch, channels, rows, cols) if data_format='channels_first' or 4D tensor with shape: (batch, rows, cols, channels) if data_format='channels_last'.
    Output shape
        4D tensor with shape: (batch, filters, new_rows, new_cols) if data_format='channels_first' or 4D tensor with shape: (batch, new_rows, new_cols, filters) if data_format='channels_last'. rows and cols values might have changed due to padding.



keras.layers.Conv3D(filters, kernel_size, strides=(1, 1, 1), 
            padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
    3D convolution layer (e.g. spatial convolution over volumes).
    When using this layer as the first layer in a model, 
    provide the keyword argument input_shape 
    (tuple of integers, does not include the sample axis), 
    e.g. input_shape=(128, 128, 128, 1) for 128x128x128 volumes with a single channel, 
    in data_format="channels_last".
    Input shape
        5D tensor with shape: (samples, channels, conv_dim1, conv_dim2, conv_dim3) if data_format='channels_first' or 5D tensor with shape: (samples, conv_dim1, conv_dim2, conv_dim3, channels) if data_format='channels_last'.
    Output shape
        5D tensor with shape: (samples, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3) if data_format='channels_first' or 5D tensor with shape: (samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters) if data_format='channels_last'. new_conv_dim1, new_conv_dim2 and new_conv_dim3 values might have changed due to padding.



keras.layers.Cropping1D(cropping=(1, 1))
    Cropping layer for 1D input (e.g. temporal sequence).
    It crops along the time dimension (axis 1).
    Arguments
        cropping: int or tuple of int (length 2) 
        How many units should be trimmed off at the beginning and end of the cropping dimension (axis 1). If a single int is provided, the same value will be used for both.
    Input shape
        3D tensor with shape (batch, axis_to_crop, features)
    Output shape
        3D tensor with shape (batch, cropped_axis, features)



keras.layers.Cropping2D(cropping=((0, 0), (0, 0)), data_format=None)
    Cropping layer for 2D input (e.g. picture).
    It crops along spatial dimensions, i.e. width and height.
    Arguments
        cropping: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
        If int: the same symmetric cropping is applied to width and height.
        If tuple of 2 ints: interpreted as two different symmetric cropping values for height and width: (symmetric_height_crop, symmetric_width_crop).
        If tuple of 2 tuples of 2 ints: interpreted as ((top_crop, bottom_crop), (left_crop, right_crop))
        data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
    Input shape
        4D tensor with shape: - If data_format is "channels_last": (batch, rows, cols, channels) - If data_format is "channels_first": (batch, channels, rows, cols)
    Output shape
        4D tensor with shape: - If data_format is "channels_last": (batch, cropped_rows, cropped_cols, channels) - If data_format is "channels_first": (batch, channels, cropped_rows, cropped_cols)
    Examples
    # Crop the input 2D images or feature maps
    model = Sequential()
    model.add(Cropping2D(cropping=((2, 2), (4, 4)),
                         input_shape=(28, 28, 3)))
    # now model.output_shape == (None, 24, 20, 3)
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Cropping2D(cropping=((2, 2), (2, 2))))
    # now model.output_shape == (None, 20, 16. 64)



keras.layers.Cropping3D(cropping=((1, 1), (1, 1), (1, 1)), data_format=None)
    Cropping layer for 3D data (e.g. spatial or spatio-temporal).
    Argument
        cropping: int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
        If int: the same symmetric cropping is applied to depth, height, and width.
        If tuple of 3 ints: interpreted as two different symmetric cropping values for depth, height, and width: (symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop).
        If tuple of 3 tuples of 2 ints: interpreted as ((left_dim1_crop, right_dim1_crop), (left_dim2_crop, right_dim2_crop), (left_dim3_crop, right_dim3_crop))
        data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, spatial_dim1, spatial_dim2, spatial_dim3, channels) while channels_first corresponds to inputs with shape (batch, channels, spatial_dim1, spatial_dim2, spatial_dim3). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
    Input shape
        5D tensor with shape: - If data_format is "channels_last": (batch, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop, depth) - If data_format is "channels_first": (batch, depth, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop)
    Output shape
        5D tensor with shape: - If data_format is "channels_last": (batch, first_cropped_axis, second_cropped_axis, third_cropped_axis, depth) - If data_format is "channels_first": (batch, depth, first_cropped_axis, second_cropped_axis, third_cropped_axis)



keras.layers.UpSampling1D(size=2)
    Upsampling layer for 1D inputs.
    Repeats each temporal step size times along the time axis.
    Arguments
        size: integer. Upsampling factor.
    Input shape
        3D tensor with shape: (batch, steps, features).
    Output shape
        3D tensor with shape: (batch, upsampled_steps, features).



keras.layers.UpSampling2D(size=(2, 2), data_format=None)
    Upsampling layer for 2D inputs.
    Repeats the rows and columns of the data by size[0] and size[1] respectively.
    Arguments
        size: int, or tuple of 2 integers. The upsampling factors for rows and columns.
        data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
    Input shape
        4D tensor with shape: - If data_format is "channels_last": (batch, rows, cols, channels) - If data_format is "channels_first": (batch, channels, rows, cols)
    Output shape
        4D tensor with shape: - If data_format is "channels_last": (batch, upsampled_rows, upsampled_cols, channels) - If data_format is "channels_first": (batch, channels, upsampled_rows, upsampled_cols)



keras.layers.UpSampling3D(size=(2, 2, 2), data_format=None)
    Upsampling layer for 3D inputs.
    Repeats the 1st, 2nd and 3rd dimensions of the data 
    by size[0], size[1] and size[2] respectively.
    Arguments
        size: int, or tuple of 3 integers. The upsampling factors for dim1, dim2 and dim3.
        data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, spatial_dim1, spatial_dim2, spatial_dim3, channels) while channels_first corresponds to inputs with shape (batch, channels, spatial_dim1, spatial_dim2, spatial_dim3). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
    Input shape
        5D tensor with shape: - If data_format is "channels_last": (batch, dim1, dim2, dim3, channels) - If data_format is "channels_first": (batch, channels, dim1, dim2, dim3)
    Output shape
        5D tensor with shape: - If data_format is "channels_last": (batch, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels) - If data_format is "channels_first": (batch, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3)



keras.layers.ZeroPadding1D(padding=1)
    Zero-padding layer for 1D input (e.g. temporal sequence).
    Arguments
        padding: int, or tuple of int (length 2), or dictionary.
        If int: How many zeros to add at the beginning and end of the padding dimension (axis 1).
        If tuple of int (length 2): How many zeros to add at the beginning and at the end of the padding dimension ((left_pad, right_pad)).
    Input shape
        3D tensor with shape (batch, axis_to_pad, features)
    Output shape
        3D tensor with shape (batch, padded_axis, features)



keras.layers.ZeroPadding2D(padding=(1, 1), data_format=None)
    Zero-padding layer for 2D input (e.g. picture).
    This layer can add rows and columns of zeros 
    at the top, bottom, left and right side of an image tensor.
    Arguments
        padding: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
        If int: the same symmetric padding is applied to width and height.
        If tuple of 2 ints: interpreted as two different symmetric padding values for height and width: (symmetric_height_pad, symmetric_width_pad).
        If tuple of 2 tuples of 2 ints: interpreted as ((top_pad, bottom_pad), (left_pad, right_pad))
        data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
    Input shape
        4D tensor with shape: - If data_format is "channels_last": (batch, rows, cols, channels) - If data_format is "channels_first": (batch, channels, rows, cols)
    Output shape
        4D tensor with shape: - If data_format is "channels_last": (batch, padded_rows, padded_cols, channels) - If data_format is "channels_first": (batch, channels, padded_rows, padded_cols)



keras.layers.ZeroPadding3D(padding=(1, 1, 1), data_format=None)
    Zero-padding layer for 3D data (spatial or spatio-temporal).
    Arguments
        padding: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
        If int: the same symmetric padding is applied to width and height.
        If tuple of 2 ints: interpreted as two different symmetric padding values for height and width: (symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad).
        If tuple of 2 tuples of 2 ints: interpreted as ((left_dim1_pad, right_dim1_pad), (left_dim2_pad, right_dim2_pad), (left_dim3_pad, right_dim3_pad))
        data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, spatial_dim1, spatial_dim2, spatial_dim3, channels) while channels_first corresponds to inputs with shape (batch, channels, spatial_dim1, spatial_dim2, spatial_dim3). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
    Input shape
        5D tensor with shape: - If data_format is "channels_last": (batch, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad, depth) - If data_format is "channels_first": (batch, depth, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad)
    Output shape
        5D tensor with shape: - If data_format is "channels_last": (batch, first_padded_axis, second_padded_axis, third_axis_to_pad, depth) - If data_format is "channels_first": (batch, depth, first_padded_axis, second_padded_axis, third_axis_to_pad)

        
        
        
        
        
###Keras -Pooling Layers


keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid')
    Max pooling operation for temporal data.
    Arguments
        pool_size: Integer, size of the max pooling windows.
        strides: Integer, or None. Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size.
        padding: One of "valid" or "same" (case-insensitive).
    Input shape
        3D tensor with shape: (batch_size, steps, features).
    Output shape
        3D tensor with shape: (batch_size, downsampled_steps, features).



keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    Max pooling operation for spatial data.
    Arguments
        pool_size: integer or tuple of 2 integers, factors by which to downscale (vertical, horizontal). (2, 2) will halve the input in both spatial dimension. If only one integer is specified, the same window length will be used for both dimensions.
        strides: Integer, tuple of 2 integers, or None. Strides values. If None, it will default to pool_size.
        padding: One of "valid" or "same" (case-insensitive).
        data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
    Input shape
        If data_format='channels_last': 4D tensor with shape: (batch_size, rows, cols, channels)
        If data_format='channels_first': 4D tensor with shape: (batch_size, channels, rows, cols)
    Output shape
        If data_format='channels_last': 4D tensor with shape: (batch_size, pooled_rows, pooled_cols, channels)
        If data_format='channels_first': 4D tensor with shape: (batch_size, channels, pooled_rows, pooled_cols)



keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
    Max pooling operation for 3D data (spatial or spatio-temporal).
    Arguments
        pool_size: tuple of 3 integers, factors by which to downscale (dim1, dim2, dim3). 
                  (2, 2, 2) will halve the size of the 3D input in each dimension.
        strides: tuple of 3 integers, or None. Strides values.
        padding: One of "valid" or "same" (case-insensitive).
        data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, spatial_dim1, spatial_dim2, spatial_dim3, channels) while channels_first corresponds to inputs with shape (batch, channels, spatial_dim1, spatial_dim2, spatial_dim3). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
    Input shape
        If data_format='channels_last': 5D tensor with shape: (batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)
        If data_format='channels_first': 5D tensor with shape: (batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)
    Output shape
        If data_format='channels_last': 5D tensor with shape: (batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)
        If data_format='channels_first': 5D tensor with shape: (batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)



keras.layers.AveragePooling1D(pool_size=2, strides=None, padding='valid')
    Average pooling for temporal data.
    Arguments
        pool_size: Integer, size of the max pooling windows.
        strides: Integer, or None. Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size.
        padding: One of "valid" or "same" (case-insensitive).
    Input shape
        3D tensor with shape: (batch_size, steps, features).
    Output shape
        3D tensor with shape: (batch_size, downsampled_steps, features).



keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    Average pooling operation for spatial data.
    Arguments
        pool_size: integer or tuple of 2 integers, factors by which to downscale (vertical, horizontal). (2, 2) will halve the input in both spatial dimension. If only one integer is specified, the same window length will be used for both dimensions.
        strides: Integer, tuple of 2 integers, or None. Strides values. If None, it will default to pool_size.
        padding: One of "valid" or "same" (case-insensitive).
        data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
    Input shape
        If data_format='channels_last': 4D tensor with shape: (batch_size, rows, cols, channels)
        If data_format='channels_first': 4D tensor with shape: (batch_size, channels, rows, cols)
    Output shape
        If data_format='channels_last': 4D tensor with shape: (batch_size, pooled_rows, pooled_cols, channels)
        If data_format='channels_first': 4D tensor with shape: (batch_size, channels, pooled_rows, pooled_cols)



keras.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
    Average pooling operation for 3D data (spatial or spatio-temporal).
    Arguments
        pool_size: tuple of 3 integers, factors by which to downscale (dim1, dim2, dim3). (2, 2, 2) will halve the size of the 3D input in each dimension.
        strides: tuple of 3 integers, or None. Strides values.
        padding: One of "valid" or "same" (case-insensitive).
        data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, spatial_dim1, spatial_dim2, spatial_dim3, channels) while channels_first corresponds to inputs with shape (batch, channels, spatial_dim1, spatial_dim2, spatial_dim3). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
    Input shape
        If data_format='channels_last': 5D tensor with shape: (batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)
        If data_format='channels_first': 5D tensor with shape: (batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)
    Output shape
        If data_format='channels_last': 5D tensor with shape: (batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)
        If data_format='channels_first': 5D tensor with shape: (batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)



keras.layers.GlobalMaxPooling1D()
    Global max pooling operation for temporal data.
    Input shape
        3D tensor with shape: (batch_size, steps, features).
    Output shape
        2D tensor with shape: (batch_size, features)




keras.layers.GlobalAveragePooling1D()
    Global average pooling operation for temporal data.
    Input shape
        3D tensor with shape: (batch_size, steps, features).
    Output shape
        2D tensor with shape: (batch_size, features)



keras.layers.GlobalMaxPooling2D(data_format=None)
    Global max pooling operation for spatial data.
    Arguments
        data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
    Input shape
        If data_format='channels_last': 4D tensor with shape: (batch_size, rows, cols, channels)
        If data_format='channels_first': 4D tensor with shape: (batch_size, channels, rows, cols)
    Output shape
        2D tensor with shape: (batch_size, channels)




keras.layers.GlobalAveragePooling2D(data_format=None)
    Global average pooling operation for spatial data.
    Arguments
        data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
    Input shape
        If data_format='channels_last': 4D tensor with shape: (batch_size, rows, cols, channels)
        If data_format='channels_first': 4D tensor with shape: (batch_size, channels, rows, cols)
    Output shape
        2D tensor with shape: (batch_size, channels)

        
        
        
###Keras -Locally-connected Layers

keras.layers.LocallyConnected1D(filters, kernel_size, strides=1, padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
    Locally-connected layer for 1D inputs.
    The LocallyConnected1D layer works similarly to the Conv1D layer, 
    except that weights are unshared, 
    that is, a different set of filters is applied at each different patch of the input.
    #Example
    # apply a unshared weight convolution 1d of length 3 to a sequence with
    # 10 timesteps, with 64 output filters
    model = Sequential()
    model.add(LocallyConnected1D(64, 3, input_shape=(10, 32)))
    # now model.output_shape == (None, 8, 64)
    # add a new conv1d on top
    model.add(LocallyConnected1D(32, 3))
    # now model.output_shape == (None, 6, 32)
    Arguments
        filters: Integer, the dimensionality of the output space (i.e. the number output of filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer, specifying the stride length of the convolution. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
        padding: Currently only supports "valid" (case-insensitive). "same" may be supported in the future.
        activation: Activation function to use (see activations). If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the kernel weights matrix (see initializers).
        bias_initializer: Initializer for the bias vector (see initializers).
        kernel_regularizer: Regularizer function applied to the kernel weights matrix (see regularizer).
        bias_regularizer: Regularizer function applied to the bias vector (see regularizer).
        activity_regularizer: Regularizer function applied to the output of the layer (its "activation"). (see regularizer).
        kernel_constraint: Constraint function applied to the kernel matrix (see constraints).
        bias_constraint: Constraint function applied to the bias vector (see constraints).
    Input shape
        3D tensor with shape: (batch_size, steps, input_dim)
    Output shape
        3D tensor with shape: (batch_size, new_steps, filters) steps value might have changed due to padding or strides.



keras.layers.LocallyConnected2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
    Locally-connected layer for 2D inputs.
    The LocallyConnected2D layer works similarly to the Conv2D layer, 
    except that weights are unshared, that is, a different set of filters is applied 
    at each different patch of the input.
    Examples
    # apply a 3x3 unshared weights convolution with 64 output filters on a 32x32 image
    # with `data_format="channels_last"`:
    model = Sequential()
    model.add(LocallyConnected2D(64, (3, 3), input_shape=(32, 32, 3)))
    # now model.output_shape == (None, 30, 30, 64)
    # notice that this layer will consume (30*30)*(3*3*3*64) + (30*30)*64 parameters
    # add a 3x3 unshared weights convolution on top, with 32 output filters:
    model.add(LocallyConnected2D(32, (3, 3)))
    # now model.output_shape == (None, 28, 28, 32)
    Arguments
        filters: Integer, the dimensionality of the output space (i.e. the number output of filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the width and height of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
        strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the width and height. Can be a single integer to specify the same value for all spatial dimensions.
        padding: Currently only support "valid" (case-insensitive). "same" will be supported in future.
        data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
        activation: Activation function to use (see activations). If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the kernel weights matrix (see initializers).
        bias_initializer: Initializer for the bias vector (see initializers).
        kernel_regularizer: Regularizer function applied to the kernel weights matrix (see regularizer).
        bias_regularizer: Regularizer function applied to the bias vector (see regularizer).
        activity_regularizer: Regularizer function applied to the output of the layer (its "activation"). (see regularizer).
        kernel_constraint: Constraint function applied to the kernel matrix (see constraints).
        bias_constraint: Constraint function applied to the bias vector (see constraints).
    Input shape
        4D tensor with shape: (samples, channels, rows, cols) if data_format='channels_first' or 4D tensor with shape: (samples, rows, cols, channels) if data_format='channels_last'.
    Output shape
        4D tensor with shape: (samples, filters, new_rows, new_cols) if data_format='channels_first' or 4D tensor with shape: (samples, new_rows, new_cols, filters) if data_format='channels_last'. rows and cols values might have changed due to padding.
        
    
    
###Keras -Recurrent Layers



keras.layers.SimpleRNN(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
    Fully-connected RNN where the output is to be fed back to input.
    Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use (see activations). If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the kernel weights matrix, used for the linear transformation of the inputs. (see initializers).
        recurrent_initializer: Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state. (see initializers).
        bias_initializer: Initializer for the bias vector (see initializers).
        kernel_regularizer: Regularizer function applied to the kernel weights matrix (see regularizer).
        recurrent_regularizer: Regularizer function applied to the recurrent_kernel weights matrix (see regularizer).
        bias_regularizer: Regularizer function applied to the bias vector (see regularizer).
        activity_regularizer: Regularizer function applied to the output of the layer (its "activation"). (see regularizer).
        kernel_constraint: Constraint function applied to the kernel weights matrix (see constraints).
        recurrent_constraint: Constraint function applied to the recurrent_kernel weights matrix (see constraints).
        bias_constraint: Constraint function applied to the bias vector (see constraints).
        dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.
        return_sequences: Boolean. Whether to return the last output. in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state in addition to the output.
        go_backwards: Boolean (default False). If True, process the input sequence backwards and return the reversed sequence.
        stateful: Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.
        unroll: Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.


keras.layers.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
    Gated Recurrent Unit - Cho et al. 2014.
    Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use (see activations). If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).
        recurrent_activation: Activation function to use for the recurrent step (see activations).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the kernel weights matrix, used for the linear transformation of the inputs. (see initializers).
        recurrent_initializer: Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state. (see initializers).
        bias_initializer: Initializer for the bias vector (see initializers).
        kernel_regularizer: Regularizer function applied to the kernel weights matrix (see regularizer).
        recurrent_regularizer: Regularizer function applied to the recurrent_kernel weights matrix (see regularizer).
        bias_regularizer: Regularizer function applied to the bias vector (see regularizer).
        activity_regularizer: Regularizer function applied to the output of the layer (its "activation"). (see regularizer).
        kernel_constraint: Constraint function applied to the kernel weights matrix (see constraints).
        recurrent_constraint: Constraint function applied to the recurrent_kernel weights matrix (see constraints).
        bias_constraint: Constraint function applied to the bias vector (see constraints).
        dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2. Mode 1 will structure its operations as a larger number of smaller dot products and additions, whereas mode 2 will batch them into fewer, larger operations. These modes will have different performance profiles on different hardware and for different applications.
        return_sequences: Boolean. Whether to return the last output. in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state in addition to the output.
        go_backwards: Boolean (default False). If True, process the input sequence backwards and return the reversed sequence.
        stateful: Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.
        unroll: Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.


keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
    Long-Short Term Memory layer - Hochreiter 1997.
    Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use (see activations). If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).
        recurrent_activation: Activation function to use for the recurrent step (see activations).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the kernel weights matrix, used for the linear transformation of the inputs. (see initializers).
        recurrent_initializer: Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state. (see initializers).
        bias_initializer: Initializer for the bias vector (see initializers).
        unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate at initialization. Setting it to true will also force bias_initializer="zeros". This is recommended in Jozefowicz et al.
        kernel_regularizer: Regularizer function applied to the kernel weights matrix (see regularizer).
        recurrent_regularizer: Regularizer function applied to the recurrent_kernel weights matrix (see regularizer).
        bias_regularizer: Regularizer function applied to the bias vector (see regularizer).
        activity_regularizer: Regularizer function applied to the output of the layer (its "activation"). (see regularizer).
        kernel_constraint: Constraint function applied to the kernel weights matrix (see constraints).
        recurrent_constraint: Constraint function applied to the recurrent_kernel weights matrix (see constraints).
        bias_constraint: Constraint function applied to the bias vector (see constraints).
        dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2. Mode 1 will structure its operations as a larger number of smaller dot products and additions, whereas mode 2 will batch them into fewer, larger operations. These modes will have different performance profiles on different hardware and for different applications.
        return_sequences: Boolean. Whether to return the last output. in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state in addition to the output.
        go_backwards: Boolean (default False). If True, process the input sequence backwards and return the reversed sequence.
        stateful: Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.
        unroll: Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.



keras.layers.ConvLSTM2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0)
    Convolutional LSTM.
    It is similar to an LSTM layer, but the input transformations and recurrent transformations are both convolutional.
    Arguments
        filters: Integer, the dimensionality of the output space (i.e. the number output of filters in the convolution).
        kernel_size: An integer or tuple/list of n integers, specifying the dimensions of the convolution window.
        strides: An integer or tuple/list of n integers, specifying the strides of the convolution. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
        padding: One of "valid" or "same" (case-insensitive).
        data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, time, ..., channels) while channels_first corresponds to inputs with shape (batch, time, channels, ...). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
        dilation_rate: An integer or tuple/list of n integers, specifying the dilation rate to use for dilated convolution. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any strides value != 1.
        activation: Activation function to use (see activations). If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
        recurrent_activation: Activation function to use for the recurrent step (see activations).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the kernel weights matrix, used for the linear transformation of the inputs. (see initializers).
        recurrent_initializer: Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state. (see initializers).
        bias_initializer: Initializer for the bias vector (see initializers).
        unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate at initialization. Use in combination with bias_initializer="zeros". This is recommended in Jozefowicz et al.
        kernel_regularizer: Regularizer function applied to the kernel weights matrix (see regularizer).
        recurrent_regularizer: Regularizer function applied to the recurrent_kernel weights matrix (see regularizer).
        bias_regularizer: Regularizer function applied to the bias vector (see regularizer).
        activity_regularizer: Regularizer function applied to the output of the layer (its "activation"). (see regularizer).
        kernel_constraint: Constraint function applied to the kernel weights matrix (see constraints).
        recurrent_constraint: Constraint function applied to the recurrent_kernel weights matrix (see constraints).
        bias_constraint: Constraint function applied to the bias vector (see constraints).
        return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
        go_backwards: Boolean (default False). If True, rocess the input sequence backwards.
        stateful: Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.
        dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.
    Input shape
        if data_format='channels_first' 5D tensor with shape: (samples,time, channels, rows, cols)
        if data_format='channels_last' 5D tensor with shape: (samples,time, rows, cols, channels)
    Output shape
        if return_sequences
        if data_format='channels_first' 5D tensor with shape: (samples, time, filters, output_row, output_col)
        if data_format='channels_last' 5D tensor with shape: (samples, time, output_row, output_col, filters)
        else
        if data_format ='channels_first' 4D tensor with shape: (samples, filters, output_row, output_col)
        if data_format='channels_last' 4D tensor with shape: (samples, output_row, output_col, filters) where o_row and o_col depend on the shape of the filter and the padding




keras.layers.SimpleRNNCell(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
    Cell class for SimpleRNN.
    Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use (see activations). If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the kernel weights matrix, used for the linear transformation of the inputs. (see initializers).
        recurrent_initializer: Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state. (see initializers).
        bias_initializer: Initializer for the bias vector (see initializers).
        kernel_regularizer: Regularizer function applied to the kernel weights matrix (see regularizer).
        recurrent_regularizer: Regularizer function applied to the recurrent_kernel weights matrix (see regularizer).
        bias_regularizer: Regularizer function applied to the bias vector (see regularizer).
        kernel_constraint: Constraint function applied to the kernel weights matrix (see constraints).
        recurrent_constraint: Constraint function applied to the recurrent_kernel weights matrix (see constraints).
        bias_constraint: Constraint function applied to the bias vector (see constraints).
        dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.



keras.layers.GRUCell(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1)
    Cell class for the GRU layer.
    Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use (see activations). If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).
        recurrent_activation: Activation function to use for the recurrent step (see activations).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the kernel weights matrix, used for the linear transformation of the inputs. (see initializers).
        recurrent_initializer: Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state. (see initializers).
        bias_initializer: Initializer for the bias vector (see initializers).
        kernel_regularizer: Regularizer function applied to the kernel weights matrix (see regularizer).
        recurrent_regularizer: Regularizer function applied to the recurrent_kernel weights matrix (see regularizer).
        bias_regularizer: Regularizer function applied to the bias vector (see regularizer).
        kernel_constraint: Constraint function applied to the kernel weights matrix (see constraints).
        recurrent_constraint: Constraint function applied to the recurrent_kernel weights matrix (see constraints).
        bias_constraint: Constraint function applied to the bias vector (see constraints).
        dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2. Mode 1 will structure its operations as a larger number of smaller dot products and additions, whereas mode 2 will batch them into fewer, larger operations. These modes will have different performance profiles on different hardware and for different applications.



keras.layers.LSTMCell(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1)
    Cell class for the LSTM layer.
    Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use (see activations). If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).
        recurrent_activation: Activation function to use for the recurrent step (see activations).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the kernel weights matrix, used for the linear transformation of the inputs. (see initializers).
        recurrent_initializer: Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state. (see initializers).
        bias_initializer: Initializer for the bias vector (see initializers).
        unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate at initialization. Setting it to true will also force bias_initializer="zeros". This is recommended in Jozefowicz et al.
        kernel_regularizer: Regularizer function applied to the kernel weights matrix (see regularizer).
        recurrent_regularizer: Regularizer function applied to the recurrent_kernel weights matrix (see regularizer).
        bias_regularizer: Regularizer function applied to the bias vector (see regularizer).
        kernel_constraint: Constraint function applied to the kernel weights matrix (see constraints).
        recurrent_constraint: Constraint function applied to the recurrent_kernel weights matrix (see constraints).
        bias_constraint: Constraint function applied to the bias vector (see constraints).
        dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2. Mode 1 will structure its operations as a larger number of smaller dot products and additions, whereas mode 2 will batch them into fewer, larger operations. These modes will have different performance profiles on different hardware and for different applications.



keras.layers.StackedRNNCells(cells)
    Wrapper allowing a stack of RNN cells to behave as a single cell.
    Used to implement efficient stacked RNNs.
    Arguments
        cells: List of RNN cell instances.
    Examples
    cells = [
        keras.layers.LSTMCell(output_dim),
        keras.layers.LSTMCell(output_dim),
        keras.layers.LSTMCell(output_dim),
    ]
    inputs = keras.Input((timesteps, input_dim))
    x = keras.layers.RNN(cells)(inputs)



keras.layers.CuDNNGRU(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)
    Fast GRU implementation backed by CuDNN.
    Can only be run on GPU, with the TensorFlow backend.
    Arguments
        units: Positive integer, dimensionality of the output space.
        kernel_initializer: Initializer for the kernel weights matrix, used for the linear transformation of the inputs. (see initializers).
        recurrent_initializer: Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state. (see initializers).
        bias_initializer: Initializer for the bias vector (see initializers).
        kernel_regularizer: Regularizer function applied to the kernel weights matrix (see regularizer).
        recurrent_regularizer: Regularizer function applied to the recurrent_kernel weights matrix (see regularizer).
        bias_regularizer: Regularizer function applied to the bias vector (see regularizer).
        activity_regularizer: Regularizer function applied to the output of the layer (its "activation"). (see regularizer).
        kernel_constraint: Constraint function applied to the kernel weights matrix (see constraints).
        recurrent_constraint: Constraint function applied to the recurrent_kernel weights matrix (see constraints).
        bias_constraint: Constraint function applied to the bias vector (see constraints).
        return_sequences: Boolean. Whether to return the last output. in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state in addition to the output.
        stateful: Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.



keras.layers.CuDNNLSTM(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)
    Fast LSTM implementation backed by CuDNN.
    Can only be run on GPU, with the TensorFlow backend.
    Arguments
        units: Positive integer, dimensionality of the output space.
        kernel_initializer: Initializer for the kernel weights matrix, used for the linear transformation of the inputs. (see initializers).
        unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate at initialization. Setting it to true will also force bias_initializer="zeros". This is recommended in Jozefowicz et al.
        recurrent_initializer: Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state. (see initializers).
        bias_initializer: Initializer for the bias vector (see initializers).
        kernel_regularizer: Regularizer function applied to the kernel weights matrix (see regularizer).
        recurrent_regularizer: Regularizer function applied to the recurrent_kernel weights matrix (see regularizer).
        bias_regularizer: Regularizer function applied to the bias vector (see regularizer).
        activity_regularizer: Regularizer function applied to the output of the layer (its "activation"). (see regularizer).
        kernel_constraint: Constraint function applied to the kernel weights matrix (see constraints).
        recurrent_constraint: Constraint function applied to the recurrent_kernel weights matrix (see constraints).
        bias_constraint: Constraint function applied to the bias vector (see constraints).
        return_sequences: Boolean. Whether to return the last output. in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state in addition to the output.
        stateful: Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.

     
     
     
###Keras - Embedding Layers


keras.layers.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)
    Turns positive integers (indexes) into dense vectors of fixed size. 
    eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
    This layer can only be used as the first layer in a model.
    #Example
    model = Sequential()
    model.add(Embedding(1000, 64, input_length=10))
    # the model will take as input an integer matrix of size (batch, input_length).
    # the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
    # now model.output_shape == (None, 10, 64), where None is the batch dimension.
    input_array = np.random.randint(1000, size=(32, 10))
    model.compile('rmsprop', 'mse')
    output_array = model.predict(input_array)
    assert output_array.shape == (32, 10, 64)
    Arguments
        input_dim: int > 0. Size of the vocabulary, i.e. maximum integer index + 1.
        output_dim: int >= 0. Dimension of the dense embedding.
        embeddings_initializer: Initializer for the embeddings matrix (see initializers).
        embeddings_regularizer: Regularizer function applied to the embeddings matrix (see regularizer).
        embeddings_constraint: Constraint function applied to the embeddings matrix (see constraints).
        mask_zero: Whether or not the input value 0 is a special "padding" value that should be masked out. This is useful when using recurrent layers which may take variable length input. If this is True then all subsequent layers in the model need to support masking or an exception will be raised. If mask_zero is set to True, as a consequence, index 0 cannot be used in the vocabulary (input_dim should equal size of vocabulary + 1).
        input_length: Length of input sequences, when it is constant. This argument is required if you are going to connect Flatten then Dense layers upstream (without it, the shape of the dense outputs cannot be computed).
    Input shape
        2D tensor with shape: (batch_size, sequence_length).
    Output shape
        3D tensor with shape: (batch_size, sequence_length, output_dim).

        
        
###Keras - Merge Layers
 

keras.layers.Add()
    Layer that adds a list of inputs.
    It takes as input a list of tensors, all of the same shape, 
    and returns a single tensor (also of the same shape).
    #Examples
    import keras
    input1 = keras.layers.Input(shape=(16,))
    x1 = keras.layers.Dense(8, activation='relu')(input1)
    input2 = keras.layers.Input(shape=(32,))
    x2 = keras.layers.Dense(8, activation='relu')(input2)
    added = keras.layers.Add()([x1, x2])  # equivalent to added = keras.layers.add([x1, x2])
    out = keras.layers.Dense(4)(added)
    model = keras.models.Model(inputs=[input1, input2], outputs=out)



keras.layers.Subtract()
    Layer that subtracts two inputs.
    It takes as input a list of tensors of size 2, 
    both of the same shape, and returns a single tensor, 
    (inputs[0] - inputs[1]), also of the same shape.
    #Examples
    import keras

    input1 = keras.layers.Input(shape=(16,))
    x1 = keras.layers.Dense(8, activation='relu')(input1)
    input2 = keras.layers.Input(shape=(32,))
    x2 = keras.layers.Dense(8, activation='relu')(input2)
    # Equivalent to subtracted = keras.layers.subtract([x1, x2])
    subtracted = keras.layers.Subtract()([x1, x2])
    out = keras.layers.Dense(4)(subtracted)
    model = keras.models.Model(inputs=[input1, input2], outputs=out)



keras.layers.Multiply()
    Layer that multiplies (element-wise) a list of inputs.
    It takes as input a list of tensors, all of the same shape, 
    and returns a single tensor (also of the same shape).



keras.layers.Average()
    Layer that averages a list of inputs.
    It takes as input a list of tensors, all of the same shape, 
    and returns a single tensor (also of the same shape).



keras.layers.Maximum()
    Layer that computes the maximum (element-wise) a list of inputs.
    It takes as input a list of tensors, all of the same shape, 
    and returns a single tensor (also of the same shape).



keras.layers.Concatenate(axis=-1)
    Layer that concatenates a list of inputs.
    It takes as input a list of tensors, all of the same shape 
    except for the concatenation axis, and returns a single tensor, 
    the concatenation of all inputs.
    Arguments
        axis: Axis along which to concatenate.
        **kwargs: standard layer keyword arguments.



keras.layers.Dot(axes, normalize=False)
    Layer that computes a dot product between samples in two tensors.
    E.g. if applied to two tensors a and b of shape (batch_size, n), 
    the output will be a tensor of shape (batch_size, 1) 
    where each entry i will be the dot product between a[i] and b[i].
    Arguments
        axes: Integer or tuple of integers, axis or axes along which to take the dot product.
        normalize: Whether to L2-normalize samples along the dot product axis before taking the dot product. If set to True, then the output of the dot product is the cosine proximity between the two samples.
        **kwargs: Standard layer keyword arguments.



keras.layers.add(inputs)
    Functional interface to the Add layer.
    Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.
    Returns
        A tensor, the sum of the inputs.
    #Examples
    import keras
    input1 = keras.layers.Input(shape=(16,))
    x1 = keras.layers.Dense(8, activation='relu')(input1)
    input2 = keras.layers.Input(shape=(32,))
    x2 = keras.layers.Dense(8, activation='relu')(input2)
    added = keras.layers.add([x1, x2])
    out = keras.layers.Dense(4)(added)
    model = keras.models.Model(inputs=[input1, input2], outputs=out)



keras.layers.subtract(inputs)
    Functional interface to the Subtract layer.
    Arguments
        inputs: A list of input tensors (exactly 2).
        **kwargs: Standard layer keyword arguments.
    Returns
    A tensor, the difference of the inputs.
    #Examples
    import keras
    input1 = keras.layers.Input(shape=(16,))
    x1 = keras.layers.Dense(8, activation='relu')(input1)
    input2 = keras.layers.Input(shape=(32,))
    x2 = keras.layers.Dense(8, activation='relu')(input2)
    subtracted = keras.layers.subtract([x1, x2])
    out = keras.layers.Dense(4)(subtracted)
    model = keras.models.Model(inputs=[input1, input2], outputs=out)



keras.layers.multiply(inputs)
    Functional interface to the Multiply layer.
    Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.
    Returns
        A tensor, the element-wise product of the inputs.


keras.layers.average(inputs)
    Functional interface to the Average layer.
    Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.
    Returns
    A tensor, the average of the inputs.


keras.layers.maximum(inputs)
    Functional interface to the Maximum layer.
    Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.
    Returns
    A tensor, the element-wise maximum of the inputs.


keras.layers.concatenate(inputs, axis=-1)
    Functional interface to the Concatenate layer.
    Arguments
        inputs: A list of input tensors (at least 2).
        axis: Concatenation axis.
        **kwargs: Standard layer keyword arguments.
    Returns
    A tensor, the concatenation of the inputs alongside axis axis.


keras.layers.dot(inputs, axes, normalize=False)
    Functional interface to the Dot layer.
    Arguments
        inputs: A list of input tensors (at least 2).
        axes: Integer or tuple of integers, axis or axes along which to take the dot product.
        normalize: Whether to L2-normalize samples along the dot product axis before taking the dot product. If set to True, then the output of the dot product is the cosine proximity between the two samples.
        **kwargs: Standard layer keyword arguments.
    Returns
        A tensor, the dot product of the samples from the inputs.
     
     
     
###Keras - Normalization Layers


keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
    Batch normalization layer (Ioffe and Szegedy, 2014).
    Normalize the activations of the previous layer at each batch, 
    i.e. applies a transformation that maintains the mean activation close to 0 
    and the activation standard deviation close to 1.
    Arguments
        axis: Integer, the axis that should be normalized (typically the features axis). For instance, after a Conv2D layer with data_format="channels_first", set axis=1 in BatchNormalization.
        momentum: Momentum for the moving average.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of beta to normalized tensor. If False, beta is ignored.
        scale: If True, multiply by gamma. If False, gamma is not used. When the next layer is linear (also e.g. nn.relu), this can be disabled since the scaling will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        moving_mean_initializer: Initializer for the moving mean.
        moving_variance_initializer: Initializer for the moving variance.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    Input shape
        Arbitrary. Use the keyword argument input_shape 
        (tuple of integers, does not include the samples axis) 
        when using this layer as the first layer in a model.
    Output shape
        Same shape as input.
        
        

###Keras - Noise layers


keras.layers.GaussianNoise(stddev)
    Apply additive zero-centered Gaussian noise.
    This is useful to mitigate overfitting 
    (you could see it as a form of random data augmentation). 
    Gaussian Noise (GS) is a natural choice as corruption process 
    for real valued inputs.
    As it is a regularization layer, it is only active at training time.
    Arguments
        stddev: float, standard deviation of the noise distribution.
    Input shape
        Arbitrary. Use the keyword argument input_shape 
        (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model.
    Output shape
        Same shape as input.



keras.layers.GaussianDropout(rate)
    Apply multiplicative 1-centered Gaussian noise.
    A Simple Way to Prevent Neural Networks from Overfitting 
    As it is a regularization layer, it is only active at training time.
    Arguments
        rate: float, drop probability (as with Dropout). 
        The multiplicative noise will have standard deviation sqrt(rate / (1 - rate)).
    Input shape
        Arbitrary. Use the keyword argument input_shape (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model.
    Output shape
        Same shape as input.



keras.layers.AlphaDropout(rate, noise_shape=None, seed=None)
    Applies Alpha Dropout to the input.
    Alpha Dropout is a Dropout that keeps mean and variance of inputs to their original values, in order to ensure the self-normalizing property even after this dropout. Alpha Dropout fits well to Scaled Exponential Linear Units by randomly setting activations to the negative saturation value.
    Arguments
        rate: float, drop probability (as with Dropout). The multiplicative noise will have standard deviation sqrt(rate / (1 - rate)).
        seed: A Python integer to use as random seed.
    Input shape
        Arbitrary. Use the keyword argument input_shape 
        (tuple of integers, does not include the samples axis) 
        when using this layer as the first layer in a model.
    Output shape
        Same shape as input.

        
        
###Keras - Layer wrappers


keras.layers.TimeDistributed(layer)
    This wrapper applies a layer to every temporal slice of an input.
    The input should be at least 3D, 
    and the dimension of index one will be considered to be the temporal dimension.
    Consider a batch of 32 samples,
    where each sample is a sequence of 10 vectors of 16 dimensions. 
    The batch input shape of the layer is then (32, 10, 16), 
    and the input_shape, not including the samples dimension, is (10, 16).

    You can then use TimeDistributed to apply a Dense layer 
    to each of the 10 timesteps, independently:
    # as the first layer in a model
    model = Sequential()
    model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
    # now model.output_shape == (None, 10, 8)
    #The output will then have shape (32, 10, 8).
    #In subsequent layers, there is no need for the input_shape:
    model.add(TimeDistributed(Dense(32)))
    # now model.output_shape == (None, 10, 32)
    #The output will then have shape (32, 10, 32).

    TimeDistributed can be used with arbitrary layers, 
    not just Dense, for instance with a Conv2D layer:

    model = Sequential()
    model.add(TimeDistributed(Conv2D(64, (3, 3)),
                              input_shape=(10, 299, 299, 3)))




keras.layers.Bidirectional(layer, merge_mode='concat', weights=None)
    Bidirectional wrapper for RNNs.
    Arguments
        layer: Recurrent instance.
        merge_mode: Mode by which outputs of the forward and backward RNNs will be combined. One of {'sum', 'mul', 'concat', 'ave', None}. If None, the outputs will not be combined, they will be returned as a list.
    #Examples
    model = Sequential()
    model.add(Bidirectional(LSTM(10, return_sequences=True),
                            input_shape=(5, 10)))
    model.add(Bidirectional(LSTM(10)))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    
    
    
    
###Keras - Sequence Preprocessing

keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32',
    padding='pre', truncating='pre', value=0.)
    Transform a list of num_samples sequences (lists of scalars) 
    into a 2D Numpy array of shape (num_samples, num_timesteps). 
    num_timesteps is either the maxlen argument if provided, 
    or the length of the longest sequence otherwise. 
    Sequences that are shorter than num_timesteps are padded with value at the end. 
    Sequences longer than num_timesteps are truncated 
    so that it fits the desired length. Position where padding or truncation happens is determined by padding or truncating, respectively.
        Return: 2D Numpy array of shape (num_samples, num_timesteps).
        Arguments:
            sequences: List of lists of int or float.
            maxlen: None or int. Maximum sequence length, longer sequences are truncated and shorter sequences are padded with zeros at the end.
            dtype: datatype of the Numpy array returned.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger than maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.



keras.preprocessing.sequence.skipgrams(sequence, vocabulary_size,
        window_size=4, negative_samples=1., shuffle=True,
        categorical=False, sampling_table=None)
    Transforms a sequence of word indexes (list of int) into couples of the form:
        (word, word in the same window), with label 1 (positive samples).
        (word, random word from the vocabulary), with label 0 (negative samples).
        Return: tuple (couples, labels).
            couples is a list of 2-elements lists of int: [word_index, other_word_index].
            labels is a list of 0 and 1, where 1 indicates that other_word_index was found in the same window as word_index, and 0 indicates that other_word_index was random.
            if categorical is set to True, the labels are categorical, ie. 1 becomes [0,1], and 0 becomes [1, 0].
        Arguments:
            sequence: list of int indexes. If using a sampling_table, the index of a word should be its the rank in the dataset (starting at 1).
            vocabulary_size: int.
            window_size: int. maximum distance between two words in a positive couple.
            negative_samples: float >= 0. 0 for no negative (=random) samples. 1 for same number as positive samples. etc.
            shuffle: boolean. Whether to shuffle the samples.
            categorical: boolean. Whether to make the returned labels categorical.
            sampling_table: Numpy array of shape (vocabulary_size,) where sampling_table[i] is the probability of sampling the word with index i (assumed to be i-th most common word in the dataset).



keras.preprocessing.sequence.make_sampling_table(size, sampling_factor=1e-5)
    Used for generating the sampling_table argument for skipgrams. 
    sampling_table[i] is the probability of sampling the word i-th most common word in a dataset 
    (more common words should be sampled less frequently, for balance).
        Return: Numpy array of shape (size,).
        Arguments:
            size: size of the vocabulary considered.
            sampling_factor: lower values result in a longer probability decay (common words will be sampled less frequently). If set to 1, no subsampling will be performed (all sampling probabilities will be 1).

            
            
            
###Keras - Text Preprocessing


keras.preprocessing.text.text_to_word_sequence(text,
                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                               lower=True,
                                               split=" ")
    Split a sentence into a list of words.
        Return: List of words (str).
        Arguments:
            text: str.
            filters: list (or concatenation) of characters to filter out, such as punctuation. Default: '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n' , includes basic punctuation, tabs, and newlines.
            lower: boolean. Whether to set the text to lowercase.
            split: str. Separator for word splitting.



keras.preprocessing.text.one_hot(text,
                                 n,
                                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                 lower=True,
                                 split=" ")
    One-hot encodes a text into a list of word indexes in a vocabulary of size n.
    This is a wrapper to the hashing_trick function 
    using hash as the hashing function.
        Return: List of integers in [1, n]. Each integer encodes a word (unicity non-guaranteed).
        Arguments:
            text: str.
            n: int. Size of vocabulary.
            filters: list (or concatenation) of characters to filter out, such as punctuation. Default: '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n' , includes basic punctuation, tabs, and newlines.
            lower: boolean. Whether to set the text to lowercase.
            split: str. Separator for word splitting.



keras.preprocessing.text.hashing_trick(text, 
                                       n,
                                       hash_function=None,
                                       filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                       lower=True,
                                       split=' ')
    Converts a text to a sequence of indices in a fixed-size hashing space
        Return: A list of integer word indices (unicity non-guaranteed).
        Arguments:
            text: str.
            n: Dimension of the hashing space.
            hash_function: defaults to python hash function, can be 'md5' or any function that takes in input a string and returns a int. Note that 'hash' is not a stable hashing function, so it is not consistent across different runs, while 'md5' is a stable hashing function.
            filters: list (or concatenation) of characters to filter out, such as punctuation. Default: '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n' , includes basic punctuation, tabs, and newlines.
            lower: boolean. Whether to set the text to lowercase.
            split: str. Separator for word splitting.



keras.preprocessing.text.Tokenizer(num_words=None,
                                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,
                                   split=" ",
                                   char_level=False)
    Class for vectorizing texts, or/and turning texts into sequences 
    (=list of word indexes, where the word of rank i in the dataset (starting at 1) has index i).
        Arguments: Same as text_to_word_sequence above.
            num_words: None or int. Maximum number of words to work with (if set, tokenization will be restricted to the top num_words most common words in the dataset).
            char_level: if True, every character will be treated as a token.
        Methods:
            fit_on_texts(texts):
                Arguments:
                    texts: list of texts to train on.
            texts_to_sequences(texts)
                Arguments:
                    texts: list of texts to turn to sequences.
                Return: list of sequences (one per text input).
            texts_to_sequences_generator(texts): generator version of the above.
                Return: yield one sequence per input text.
            texts_to_matrix(texts):
                Return: numpy array of shape (len(texts), num_words).
                Arguments:
                    texts: list of texts to vectorize.
                    mode: one of "binary", "count", "tfidf", "freq" (default: "binary").
            fit_on_sequences(sequences):
                Arguments:
                    sequences: list of sequences to train on.
            sequences_to_matrix(sequences):
                Return: numpy array of shape (len(sequences), num_words).
                Arguments:
                    sequences: list of sequences to vectorize.
                    mode: one of "binary", "count", "tfidf", "freq" (default: "binary").
        Attributes:
            word_counts: dictionary mapping words (str) to the number of times they appeared on during fit. Only set after fit_on_texts was called.
            word_docs: dictionary mapping words (str) to the number of documents/texts they appeared on during fit. Only set after fit_on_texts was called.
            word_index: dictionary mapping words (str) to their rank/index (int). Only set after fit_on_texts was called.
            document_count: int. Number of documents (texts/sequences) the tokenizer was trained on. Only set after fit_on_texts or fit_on_sequences was called.

            
###Keras - Image Preprocessing

keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-6,
        rotation_range=0.,
        width_shift_range=0.,
        height_shift_range=0.,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=K.image_data_format())
Generate batches of tensor image data with real-time data augmentation. 
The data will be looped over (in batches) indefinitely.
    Arguments:
        featurewise_center: Boolean. Set input mean to 0 over the dataset, feature-wise.
        samplewise_center: Boolean. Set each sample mean to 0.
        featurewise_std_normalization: Boolean. Divide inputs by std of the dataset, feature-wise.
        samplewise_std_normalization: Boolean. Divide each input by its std.
        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
        zca_whitening: Boolean. Apply ZCA whitening.
        rotation_range: Int. Degree range for random rotations.
        width_shift_range: Float (fraction of total width). Range for random horizontal shifts.
        height_shift_range: Float (fraction of total height). Range for random vertical shifts.
        shear_range: Float. Shear Intensity (Shear angle in counter-clockwise direction as radians)
        zoom_range: Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].
        channel_shift_range: Float. Range for random channel shifts.
        fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}. Points outside the boundaries of the input are filled according to the given mode:
            "constant": kkkkkkkk|abcd|kkkkkkkk (cval=k)
            "nearest": aaaaaaaa|abcd|dddddddd
            "reflect": abcddcba|abcd|dcbaabcd
            "wrap": abcdabcd|abcd|abcdabcd
        cval: Float or Int. Value used for points outside the boundaries when fill_mode = "constant".
        horizontal_flip: Boolean. Randomly flip inputs horizontally.
        vertical_flip: Boolean. Randomly flip inputs vertically.
        rescale: rescaling factor. Defaults to None. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (before applying any other transformation).
        preprocessing_function: function that will be implied on each input. The function will run before any other modification on it. The function should take one argument: one image (Numpy tensor with rank 3), and should output a Numpy tensor with the same shape.
        data_format: One of {"channels_first", "channels_last"}. "channels_last" mode means that the images should have shape (samples, height, width, channels), "channels_first" mode means that the images should have shape (samples, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
    Methods:
        fit(x): Compute the internal data stats related to the data-dependent transformations, based on an array of sample data. Only required if featurewise_center or featurewise_std_normalization or zca_whitening.
            Arguments:
                x: sample data. Should have rank 4. In case of grayscale data, the channels axis should have value 1, and in case of RGB data, it should have value 3.
                augment: Boolean (default: False). Whether to fit on randomly augmented samples.
                rounds: int (default: 1). If augment, how many augmentation passes over the data to use.
                seed: int (default: None). Random seed.
        flow(x, y): Takes numpy data & label arrays, and generates batches of augmented/normalized data. Yields batches indefinitely, in an infinite loop.
            Arguments:
                x: data. Should have rank 4. In case of grayscale data, the channels axis should have value 1, and in case of RGB data, it should have value 3.
                y: labels.
                batch_size: int (default: 32).
                shuffle: boolean (default: True).
                seed: int (default: None).
                save_to_dir: None or str (default: None). This allows you to optimally specify a directory to which to save the augmented pictures being generated (useful for visualizing what you are doing).
                save_prefix: str (default: ''). Prefix to use for filenames of saved pictures (only relevant if save_to_dir is set).
                save_format: one of "png", "jpeg" (only relevant if save_to_dir is set). Default: "png".
            yields: Tuples of (x, y) where x is a numpy array of image data and y is a numpy array of corresponding labels. The generator loops indefinitely.
        flow_from_directory(directory): Takes the path to a directory, and generates batches of augmented/normalized data. Yields batches indefinitely, in an infinite loop.
            Arguments:
                directory: path to the target directory. It should contain one subdirectory per class. Any PNG, JPG, BMP or PPM images inside each of the subdirectories directory tree will be included in the generator. See this script for more details.
                target_size: tuple of integers (height, width), default: (256, 256). The dimensions to which all images found will be resized.
                color_mode: one of "grayscale", "rbg". Default: "rgb". Whether the images will be converted to have 1 or 3 color channels.
                classes: optional list of class subdirectories (e.g. ['dogs', 'cats']). Default: None. If not provided, the list of classes will be automatically inferred from the subdirectory names/structure under directory, where each subdirectory will be treated as a different class (and the order of the classes, which will map to the label indices, will be alphanumeric). The dictionary containing the mapping from class names to class indices can be obtained via the attribute class_indices.
                class_mode: one of "categorical", "binary", "sparse" or None. Default: "categorical". Determines the type of label arrays that are returned: "categorical" will be 2D one-hot encoded labels, "binary" will be 1D binary labels, "sparse" will be 1D integer labels. If None, no labels are returned (the generator will only yield batches of image data, which is useful to use model.predict_generator(), model.evaluate_generator(), etc.). Please note that in case of class_mode None, the data still needs to reside in a subdirectory of directory for it to work correctly.
                batch_size: size of the batches of data (default: 32).
                shuffle: whether to shuffle the data (default: True)
                seed: optional random seed for shuffling and transformations.
                save_to_dir: None or str (default: None). This allows you to optimally specify a directory to which to save the augmented pictures being generated (useful for visualizing what you are doing).
                save_prefix: str. Prefix to use for filenames of saved pictures (only relevant if save_to_dir is set).
                save_format: one of "png", "jpeg" (only relevant if save_to_dir is set). Default: "png".
                follow_links: whether to follow symlinks inside class subdirectories (default: False).

#Examples:
#Example of using .flow(x, y):

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)
# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)
# here's a more "manual" example
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break

#Example of using .flow_from_directory(directory):
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)

#Example of transforming images and masks together.

# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_datagen.fit(images, augment=True, seed=seed)
mask_datagen.fit(masks, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
    'data/images',
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    'data/masks',
    class_mode=None,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)
    
    
    

###Keras - Losses

##Usage of loss functions

#A loss function (or objective function, or optimization score function) 
#is one of the two parameters required to compile a model:
model.compile(loss='mean_squared_error', optimizer='sgd')

from keras import losses
model.compile(loss=losses.mean_squared_error, optimizer='sgd')

#You can either pass the name of an existing loss function, 
#or pass a TensorFlow/Theano symbolic function 
#that returns a scalar for each data-point and takes the following two arguments:
    y_true: True labels. TensorFlow/Theano tensor.
    y_pred: Predictions. TensorFlow/Theano tensor of the same shape as y_true.


##Available loss functions
mean_squared_error(y_true, y_pred)
mean_absolute_error(y_true, y_pred)
mean_absolute_percentage_error(y_true, y_pred)
mean_squared_logarithmic_error(y_true, y_pred)
squared_hinge(y_true, y_pred)
hinge(y_true, y_pred)
categorical_hinge(y_true, y_pred)
logcosh(y_true, y_pred)
    Logarithm of the hyperbolic cosine of the prediction error.
categorical_crossentropy(y_true, y_pred)
sparse_categorical_crossentropy(y_true, y_pred)
binary_crossentropy(y_true, y_pred)
kullback_leibler_divergence(y_true, y_pred)
poisson(y_true, y_pred)
cosine_proximity(y_true, y_pred)


#when using the categorical_crossentropy loss, 
#your targets should be in categorical format 
#(e.g. if you have 10 classes, the target for each sample should be a 10-dimensional vector 
#that is all-zeros expect for a 1 at the index corresponding to the class of the sample). 

#In order to convert integer targets into categorical targets, 
#use the Keras utility to_categorical:

from keras.utils.np_utils import to_categorical
categorical_labels = to_categorical(int_labels, num_classes=None)



###Keras - Metrics

##Usage of metrics
#A metric is a function that is used to judge the performance of model. 




from keras import metrics

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=[metrics.mae, metrics.categorical_accuracy])

#A metric function is similar to an loss function, 
#except that the results from evaluating a metric 
#are not used when training the model.

#You can either pass the name of an existing metric, 
#or pass a Theano/TensorFlow symbolic function 
#Arguments
    y_true: True labels. Theano/TensorFlow tensor.
    y_pred: Predictions. Theano/TensorFlow tensor of the same shape as y_true.



##Available metrics
binary_accuracy(y_true, y_pred)
categorical_accuracy(y_true, y_pred)
sparse_categorical_accuracy(y_true, y_pred)
top_k_categorical_accuracy(y_true, y_pred, k=5)
sparse_top_k_categorical_accuracy(y_true, y_pred, k=5)


##Custom metrics

import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
              
              

###Keras - Optimizers
##Usage of optimizers

#An optimizer is one of the two arguments required for compiling a Keras model:

from keras import optimizers

model = Sequential()
model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(Activation('tanh'))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

#You can either instantiate an optimizer before passing it to model.compile() , 
#or you can call it by its name. 

# pass optimizer by name: default parameters will be used
model.compile(loss='mean_squared_error', optimizer='sgd')



##Parameters common to all Keras optimizers
#The parameters clipnorm and clipvalue can be used with all optimizers 
#to control gradient clipping:

from keras import optimizers

# All parameter gradients will be clipped to
# a maximum norm of 1.
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)

from keras import optimizers

# All parameter gradients will be clipped to
# a maximum value of 0.5 and
# a minimum value of -0.5.
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)






keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    Stochastic gradient descent optimizer.
    Includes support for momentum, learning rate decay, and Nesterov momentum.
    Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.



keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    RMSProp optimizer.
    It is recommended to leave the parameters of this optimizer 
    at their default values (except the learning rate, which can be freely tuned).
    This optimizer is usually a good choice for recurrent neural networks.
    Arguments
        lr: float >= 0. Learning rate.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.



keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    Adagrad optimizer.
    It is recommended to leave the parameters of this optimizer 
    at their default values.
    Arguments

        lr: float >= 0. Learning rate.
        epsilon: float >= 0.
        decay: float >= 0. Learning rate decay over each update.



keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    Adadelta optimizer.
    It is recommended to leave the parameters of this optimizer 
    at their default values.
    Arguments
        lr: float >= 0. Learning rate. It is recommended to leave it at the default value.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.



keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    Adam optimizer.
    Default parameters follow those provided in the original paper.
    Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.





keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    Adamax optimizer from Adam paper's Section 7.
    It is a variant of Adam based on the infinity norm. 
    Default parameters follow those provided in the paper.
    Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.



keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    Nesterov Adam optimizer.
    Much like Adam is essentially RMSprop with momentum,
    Nadam is Adam RMSprop with Nesterov momentum.
    Default parameters follow those provided in the paper. 
    It is recommended to leave the parameters of this optimizer at their default values.
    Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.



keras.optimizers.TFOptimizer(optimizer)
    Wrapper class for native TensorFlow optimizers.



###Keras - Callbacks


##Usage of callbacks
#A callback is a set of functions to be applied 
#at given stages of the training procedure. 

#You can use callbacks to get a view on internal states 
#and statistics of the model during training. 

#You can pass a list of callbacks (as the keyword argument callbacks) 
#to the .fit() method of the Sequential or Model classes. 
#The relevant methods of the callbacks will then be called 
#at each stage of the training.



keras.callbacks.Callback()
    Abstract base class used to build new callbacks.
    Properties
        params: dict. Training parameters 
            (eg. verbosity, batch size, number of epochs...).
        model: instance of keras.models.Model. Reference of the model being trained.
    The logs dictionary that callback methods take 
    as argument will contain keys for quantities relevant 
    to the current batch or epoch.
    Currently, the .fit() method of the Sequential model class will include 
    the following quantities in the logs that it passes to its callbacks:
        on_epoch_end: logs include acc and loss, and optionally include val_loss (if validation is enabled in fit), and val_acc (if validation and accuracy monitoring are enabled).
        on_batch_begin: logs include size, the number of samples in the current batch.
        on_batch_end: logs include loss, and optionally acc (if accuracy monitoring is enabled).



keras.callbacks.BaseLogger()
    Callback that accumulates epoch averages of metrics.
    This callback is automatically applied to every Keras model.



keras.callbacks.TerminateOnNaN()
    Callback that terminates training when a NaN loss is encountered.



keras.callbacks.ProgbarLogger(count_mode='samples')
    Callback that prints metrics to stdout.
    Arguments
        count_mode: One of "steps" or "samples". Whether the progress bar should count samples seen or steps (batches) seen.



keras.callbacks.History()
    Callback that records events into a History object.
    This callback is automatically applied to every Keras model. 
    The History object gets returned by the fit method of models.



keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    Save the model after every epoch.
    filepath can contain named formatting options, 
    which will be filled the value of epoch and keys in logs (passed in on_epoch_end).
    For example: if filepath is weights.{epoch:02d}-{val_loss:.2f}.hdf5, 
    then the model checkpoints will be saved with the epoch number 
    and the validation loss in the filename.
    Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if save_best_only=True, the latest best model according to the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity. For val_acc, this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be saved (model.save_weights(filepath)), else the full model is saved (model.save(filepath)).
        period: Interval (number of epochs) between checkpoints.



keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
    Stop training when a monitored quantity has stopped improving.
    Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.
        patience: number of epochs with no improvement after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In min mode, training will stop when the quantity monitored has stopped decreasing; in max mode it will stop when the quantity monitored has stopped increasing; in auto mode, the direction is automatically inferred from the name of the monitored quantity.



keras.callbacks.RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data', headers=None)
    Callback used to stream events to a server.
    Requires the requests library. 
    Events are sent to root + '/publish/epoch/end/' by default. 
    Calls are HTTP POST, with a data argument which is a JSON-encoded dictionary of event data.
    Arguments
        root: String; root url of the target server.
        path: String; path relative to root to which the events will be sent.
        field: String; JSON field under which the data will be stored.
        headers: Dictionary; optional custom HTTP headers.



keras.callbacks.LearningRateScheduler(schedule)
    Learning rate scheduler.
    Arguments
        schedule: a function that takes an epoch index as input (integer, indexed from 0) and returns a new learning rate as output (float).




keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    Tensorboard basic visualizations.
    TensorBoard is a visualization tool provided with TensorFlow.
    This callback writes a log for TensorBoard, 
    which allows you to visualize dynamic graphs of your training and test metrics, 
    as well as activation histograms for the different layers in your model.
    $ tensorboard --logdir=/full_path_to_your_logs
    Arguments
        log_dir: the path of the directory where to save the log files to be parsed by TensorBoard.
        histogram_freq: frequency (in epochs) at which to compute activation and weight histograms for the layers of the model. If set to 0, histograms won't be computed. Validation data (or split) must be specified for histogram visualizations.
        write_graph: whether to visualize the graph in TensorBoard. The log file can become quite large when write_graph is set to True.
        write_grads: whether to visualize gradient histograms in TensorBoard. histogram_freq must be greater than 0.
        batch_size: size of batch of inputs to feed to the network for histograms computation.
        write_images: whether to write model weights to visualize as image in TensorBoard.
        embeddings_freq: frequency (in epochs) at which selected embedding layers will be saved.
        embeddings_layer_names: a list of names of layers to keep eye on. If None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name in which metadata for this embedding layer is saved. See the details about metadata files format. In case if the same metadata file is used for all embedding layers, string can be passed.




keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor of 2-10 
    once learning stagnates. 
    This callback monitors a quantity 
    and if no improvement is seen for a 'patience' number of epochs, 
    the learning rate is reduced.
    #Example
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    model.fit(X_train, Y_train, callbacks=[reduce_lr])
    Arguments
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing; in auto mode, the direction is automatically inferred from the name of the monitored quantity.
        epsilon: threshold for measuring the new optimum, to only focus on significant changes.
        cooldown: number of epochs to wait before resuming normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.



keras.callbacks.CSVLogger(filename, separator=',', append=False)
    Callback that streams epoch results to a csv file.
    Supports all values that can be represented as a string, 
    including 1D iterables such as np.ndarray.
    #Example
    csv_logger = CSVLogger('training.log')
    model.fit(X_train, Y_train, callbacks=[csv_logger])
    Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing training). False: overwrite existing file,



keras.callbacks.LambdaCallback(on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None, on_train_begin=None, on_train_end=None)
    Callback for creating simple, custom callbacks on-the-fly.
    This callback is constructed with anonymous functions 
    that will be called at the appropriate time. 
    Note that the callbacks expects positional arguments, as:
        on_epoch_begin and on_epoch_end expect two positional arguments: epoch, logs
        on_batch_begin and on_batch_end expect two positional arguments: batch, logs
        on_train_begin and on_train_end expect one positional argument: logs
    Arguments
        on_epoch_begin: called at the beginning of every epoch.
        on_epoch_end: called at the end of every epoch.
        on_batch_begin: called at the beginning of every batch.
        on_batch_end: called at the end of every batch.
        on_train_begin: called at the beginning of model training.
        on_train_end: called at the end of model training.

    #Example
    # Print the batch number at the beginning of every batch.
    batch_print_callback = LambdaCallback(
        on_batch_begin=lambda batch,logs: print(batch))
    # Stream the epoch loss to a file in JSON format. The file content
    # is not well-formed JSON but rather has a JSON object per line.
    import json
    json_log = open('loss_log.json', mode='wt', buffering=1)
    json_logging_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: json_log.write(
            json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
        on_train_end=lambda logs: json_log.close()
    )
    # Terminate some processes after having finished model training.
    processes = ...
    cleanup_callback = LambdaCallback(
        on_train_end=lambda logs: [
            p.terminate() for p in processes if p.is_alive()])

    model.fit(...,
              callbacks=[batch_print_callback,
                         json_logging_callback,
                         cleanup_callback])

                         
                         
                         
                         
##Create a callback

#example saving a list of losses over each batch during training:

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

#Example: recording loss history

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

history = LossHistory()
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, callbacks=[history])

print(history.losses)
# outputs
'''
[0.66047596406559383, 0.3547245744908703, ..., 0.25953155204159617, 0.25901699725311789]
'''

#Example: model checkpoints

from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

'''
saves the model weights after each epoch if the validation loss decreased
'''
checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, validation_data=(X_test, Y_test), callbacks=[checkpointer])

    
    
    
###Keras - Datasets

##CIFAR10 small image classification
Dataset of 50,000 32x32 color training images, 
labeled over 10 categories, and 10,000 test images.
#Usage:
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#Returns:
    2 tuples:
        x_train, x_test: uint8 array of RGB image data with shape (num_samples, 3, 32, 32).
        y_train, y_test: uint8 array of category labels (integers in range 0-9) with shape (num_samples,).

            
##CIFAR100 small image classification
Dataset of 50,000 32x32 color training images, labeled over 100 categories, and 10,000 test images.
#Usage:
from keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

Returns:
    2 tuples:
        x_train, x_test: uint8 array of RGB image data with shape (num_samples, 3, 32, 32).
        y_train, y_test: uint8 array of category labels with shape (num_samples,).

Arguments:
    label_mode: "fine" or "coarse".

    
##IMDB Movie reviews sentiment classification
Dataset of 25,000 movies reviews from IMDB, 
labeled by sentiment (positive/negative). 
Reviews have been preprocessed, and each review is encoded as a sequence of word indexes (integers). 
For convenience, words are indexed by overall frequency in the dataset, 
so that for instance the integer "3" encodes the 3rd most frequent word in the data. 
This allows for quick filtering operations such as: 
"only consider the top 10,000 most common words, but eliminate the top 20 most common words".

As a convention, "0" does not stand for a specific word, but instead is used to encode any unknown word.
Usage:

from keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)

    Returns:
        2 tuples:
            x_train, x_test: list of sequences, which are lists of indexes (integers). If the num_words argument was specific, the maximum possible index value is num_words-1. If the maxlen argument was specified, the largest possible sequence length is maxlen.
            y_train, y_test: list of integer labels (1 or 0).
    Arguments:
        path: if you do not have the data locally (at '~/.keras/datasets/' + path), it will be downloaded to this location.
        num_words: integer or None. Top most frequent words to consider. Any less frequent word will appear as oov_char value in the sequence data.
        skip_top: integer. Top most frequent words to ignore (they will appear as oov_char value in the sequence data).
        maxlen: int. Maximum sequence length. Any longer sequence will be truncated.
        seed: int. Seed for reproducible data shuffling.
        start_char: int. The start of a sequence will be marked with this character. Set to 1 because 0 is usually the padding character.
        oov_char: int. words that were cut out because of the num_words or skip_top limit will be replaced with this character.
        index_from: int. Index actual words with this index and higher.

        
        
        
##Reuters newswire topics classification
Dataset of 11,228 newswires from Reuters, labeled over 46 topics. 
As with the IMDB dataset, each wire is encoded as a sequence of word indexes 
(same conventions).
#Usage:

from keras.datasets import reuters

(x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz",
                                                         num_words=None,
                                                         skip_top=0,
                                                         maxlen=None,
                                                         test_split=0.2,
                                                         seed=113,
                                                         start_char=1,
                                                         oov_char=2,
                                                         index_from=3)

#This dataset also makes available the word index used for encoding the sequences:
word_index = reuters.get_word_index(path="reuters_word_index.json")

    Returns: A dictionary where key are words (str) and values are indexes (integer). eg. word_index["giraffe"] might return 1234.
    Arguments:
        path: if you do not have the index file locally (at '~/.keras/datasets/' + path), it will be downloaded to this location.

        
        
        
##MNIST database of handwritten digits
Dataset of 60,000 28x28 grayscale images of the 10 digits, 
along with a test set of 10,000 images.

#Usage:
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
Returns:
    2 tuples:
        x_train, x_test: uint8 array of grayscale image data with shape (num_samples, 28, 28).
        y_train, y_test: uint8 array of digit labels (integers in range 0-9) with shape (num_samples,).

Arguments:
    path: if you do not have the index file locally (at '~/.keras/datasets/' + path), it will be downloaded to this location.

    
##Fashion-MNIST database of fashion articles

Dataset of 60,000 28x28 grayscale images of 10 fashion categories, 
along with a test set of 10,000 images. 
This dataset can be used as a drop-in replacement for MNIST. 
The class labels are:
    Label 	Description
    0 	T-shirt/top
    1 	Trouser
    2 	Pullover
    3 	Dress
    4 	Coat
    5 	Sandal
    6 	Shirt
    7 	Sneaker
    8 	Bag
    9 	Ankle boot
#Usage:
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

Returns:
    2 tuples:
        x_train, x_test: uint8 array of grayscale image data with shape (num_samples, 28, 28).
        y_train, y_test: uint8 array of labels (integers in range 0-9) with shape (num_samples,).


##Boston housing price regression dataset
Dataset taken from the StatLib library which is maintained at Carnegie Mellon University.
Samples contain 13 attributes of houses at different locations around the Boston suburbs in the late 1970s.
Targets are the median values of the houses at a location (in k$).
#Usage:

from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

Arguments:
    path: path where to cache the dataset locally (relative to ~/.keras/datasets).
    seed: Random seed for shuffling the data before computing the test split.
    test_split: fraction of the data to reserve as test set.

Returns: Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test).



###Keras - Applications

#Keras Applications are deep learning models 
#that are made available alongside pre-trained weights. 

#These models can be used for prediction, feature extraction, and fine-tuning.

#Weights are downloaded automatically when instantiating a model. 
#They are stored at ~/.keras/models/.


#Available models
#Models for image classification with weights trained on ImageNet:
    Xception
    VGG16
    VGG19
    ResNet50
    InceptionV3
    InceptionResNetV2
    MobileNet

#All of these architectures (except Xception and MobileNet) 
#are compatible with both TensorFlow and Theano, 
#and upon instantiation the models will be built according to the image data format 
#set in your Keras configuration file at ~/.keras/keras.json. 

#For instance, if you have set image_data_format=channels_last, 
#then any model loaded from this repository will get built according 
#to the TensorFlow data format convention, "Height-Width-Depth".

#The Xception model is only available for TensorFlow, 
#due to its reliance on SeparableConvolution layers. 

#The MobileNet model is only available for TensorFlow, 
#due to its reliance on DepthwiseConvolution layers.

##Usage examples for image classification models
#Classify ImageNet classes with ResNet50

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

Extract features with VGG16

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)

##Extract features from an arbitrary intermediate layer with VGG19

from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

block4_pool_features = model.predict(x)

##Fine-tune InceptionV3 on a new set of classes

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(200, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit_generator(...)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(...)

##Build InceptionV3 over a custom input tensor

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input

# this could also be the output a different Keras model or layer
input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'

model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)

Documentation for individual models
Model 	Size 	Top-1 Accuracy 	Top-5 Accuracy 	Parameters 	Depth
Xception 	88 MB 	0.790 	0.945 	22,910,480 	126
VGG16 	528 MB 	0.715 	0.901 	138,357,544 	23
VGG19 	549 MB 	0.727 	0.910 	143,667,240 	26
ResNet50 	99 MB 	0.759 	0.929 	25,636,712 	168
InceptionV3 	92 MB 	0.788 	0.944 	23,851,784 	159
InceptionResNetV2 	215 MB 	0.804 	0.953 	55,873,736 	572
MobileNet 	17 MB 	0.665 	0.871 	4,253,864 	88

#The top-1 and top-5 accuracy refers to the model's performance on the ImageNet validation dataset.

##List of models 

keras.applications.xception.Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    Xception V1 model, with weights pre-trained on ImageNet.
    On ImageNet, this model gets to a top-1 validation accuracy of 0.790 and a top-5 validation accuracy of 0.945.
    Note that this model is only available for the TensorFlow backend, due to its reliance on SeparableConvolution layers. Additionally it only supports the data format 'channels_last' (height, width, channels).
    The default input size for this model is 299x299.
    Arguments
        include_top: whether to include the fully-connected layer at the top of the network.
        weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of layers.Input()) to use as image input for the model.
        input_shape: optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (299, 299, 3). It should have exactly 3 inputs channels, and width and height should be no smaller than 71. E.g. (150, 150, 3) would be one valid value.
        pooling: Optional pooling mode for feature extraction when include_top is False.
            None means that the output of the model will be the 4D tensor output of the last convolutional layer.
            'avg' means that global average pooling will be applied to the output of the last convolutional layer, and thus the output of the model will be a 2D tensor.
            'max' means that global max pooling will be applied.
        classes: optional number of classes to classify images into, only to be specified if include_top is True, and if no weights argument is specified.
    Returns
        A Keras Model instance.

        


keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    VGG16 model, with weights pre-trained on ImageNet.
    This model is available for both the Theano and TensorFlow backend, and can be built both with 'channels_first' data format (channels, height, width) or 'channels_last' data format (height, width, channels).
    The default input size for this model is 224x224.
    Arguments
        include_top: whether to include the 3 fully-connected layers at the top of the network.
        weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of layers.Input()) to use as image input for the model.
        input_shape: optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (224, 224, 3) (with 'channels_last' data format) or (3, 224, 224) (with 'channels_first' data format). It should have exactly 3 inputs channels, and width and height should be no smaller than 48. E.g. (200, 200, 3) would be one valid value.
        pooling: Optional pooling mode for feature extraction when include_top is False.
            None means that the output of the model will be the 4D tensor output of the last convolutional layer.
            'avg' means that global average pooling will be applied to the output of the last convolutional layer, and thus the output of the model will be a 2D tensor.
            'max' means that global max pooling will be applied.
        classes: optional number of classes to classify images into, only to be specified if include_top is True, and if no weights argument is specified.
    Returns
        A Keras Model instance.


keras.applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    VGG19 model, with weights pre-trained on ImageNet.
    This model is available for both the Theano and TensorFlow backend, and can be built both with 'channels_first' data format (channels, height, width) or 'channels_last' data format (height, width, channels).
    The default input size for this model is 224x224.
    Arguments
        include_top: whether to include the 3 fully-connected layers at the top of the network.
        weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of layers.Input()) to use as image input for the model.
        input_shape: optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (224, 224, 3) (with 'channels_last' data format) or (3, 224, 224) (with 'channels_first' data format). It should have exactly 3 inputs channels, and width and height should be no smaller than 48. E.g. (200, 200, 3) would be one valid value.
        pooling: Optional pooling mode for feature extraction when include_top is False.
            None means that the output of the model will be the 4D tensor output of the last convolutional layer.
            'avg' means that global average pooling will be applied to the output of the last convolutional layer, and thus the output of the model will be a 2D tensor.
            'max' means that global max pooling will be applied.
        classes: optional number of classes to classify images into, only to be specified if include_top is True, and if no weights argument is specified.
    Returns
        A Keras Model instance.


keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    ResNet50 model, with weights pre-trained on ImageNet.
    This model is available for both the Theano and TensorFlow backend, and can be built both with 'channels_first' data format (channels, height, width) or 'channels_last' data format (height, width, channels).
    The default input size for this model is 224x224.
    Arguments
        include_top: whether to include the fully-connected layer at the top of the network.
        weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of layers.Input()) to use as image input for the model.
        input_shape: optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (224, 224, 3) (with 'channels_last' data format) or (3, 224, 224) (with 'channels_first' data format). It should have exactly 3 inputs channels, and width and height should be no smaller than 197. E.g. (200, 200, 3) would be one valid value.
        pooling: Optional pooling mode for feature extraction when include_top is False.
            None means that the output of the model will be the 4D tensor output of the last convolutional layer.
            'avg' means that global average pooling will be applied to the output of the last convolutional layer, and thus the output of the model will be a 2D tensor.
            'max' means that global max pooling will be applied.
        classes: optional number of classes to classify images into, only to be specified if include_top is True, and if no weights argument is specified.
    Returns
        A Keras Model instance.
    


keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    Inception V3 model, with weights pre-trained on ImageNet.
    This model is available for both the Theano and TensorFlow backend, and can be built both with 'channels_first' data format (channels, height, width) or 'channels_last' data format (height, width, channels).
    The default input size for this model is 299x299.
    Arguments
        include_top: whether to include the fully-connected layer at the top of the network.
        weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of layers.Input()) to use as image input for the model.
        input_shape: optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (299, 299, 3) (with 'channels_last' data format) or (3, 299, 299) (with 'channels_first' data format). It should have exactly 3 inputs channels, and width and height should be no smaller than 139. E.g. (150, 150, 3) would be one valid value.
        pooling: Optional pooling mode for feature extraction when include_top is False.
            None means that the output of the model will be the 4D tensor output of the last convolutional layer.
            'avg' means that global average pooling will be applied to the output of the last convolutional layer, and thus the output of the model will be a 2D tensor.
            'max' means that global max pooling will be applied.
        classes: optional number of classes to classify images into, only to be specified if include_top is True, and if no weights argument is specified.
    Returns
        A Keras Model instance.



keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    Inception-ResNet V2 model, with weights pre-trained on ImageNet.
    This model is available for Theano, TensorFlow and CNTK backends, and can be built both with 'channels_first' data format (channels, height, width) or 'channels_last' data format (height, width, channels).
    The default input size for this model is 299x299.
    Arguments
        include_top: whether to include the fully-connected layer at the top of the network.
        weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of layers.Input()) to use as image input for the model.
        input_shape: optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (299, 299, 3) (with 'channels_last' data format) or (3, 299, 299) (with 'channels_first' data format). It should have exactly 3 inputs channels, and width and height should be no smaller than 139. E.g. (150, 150, 3) would be one valid value.
        pooling: Optional pooling mode for feature extraction when include_top is False.
            None means that the output of the model will be the 4D tensor output of the last convolutional layer.
            'avg' means that global average pooling will be applied to the output of the last convolutional layer, and thus the output of the model will be a 2D tensor.
            'max' means that global max pooling will be applied.
        classes: optional number of classes to classify images into, only to be specified if include_top is True, and if no weights argument is specified.
    Returns
        A Keras Model instance.


keras.applications.mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
    MobileNet model, with weights pre-trained on ImageNet.
    Note that only TensorFlow is supported for now, therefore it only works with the data format image_data_format='channels_last' in your Keras config at ~/.keras/keras.json. To load a MobileNet model via load_model, import the custom objects relu6 and DepthwiseConv2D and pass them to the custom_objects parameter.
    E.g.
    model = load_model('mobilenet.h5', custom_objects={
                       'relu6': mobilenet.relu6,
                       'DepthwiseConv2D': mobilenet.DepthwiseConv2D})
    The default input size for this model is 224x224.
    Arguments
        input_shape: optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (224, 224, 3) (with 'channels_last' data format) or (3, 224, 224) (with 'channels_first' data format). It should have exactly 3 inputs channels, and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value.
        alpha: controls the width of the network.
            If alpha < 1.0, proportionally decreases the number of filters in each layer.
            If alpha > 1.0, proportionally increases the number of filters in each layer.
            If alpha = 1, default number of filters from the paper are used at each layer.
        depth_multiplier: depth multiplier for depthwise convolution (also called the resolution multiplier)
        dropout: dropout rate
        include_top: whether to include the fully-connected layer at the top of the network.
        weights: None (random initialization) or 'imagenet' (ImageNet weights)
        input_tensor: optional Keras tensor (i.e. output of layers.Input()) to use as image input for the model.
        pooling: Optional pooling mode for feature extraction when include_top is False.
            None means that the output of the model will be the 4D tensor output of the last convolutional layer.
            'avg' means that global average pooling will be applied to the output of the last convolutional layer, and thus the output of the model will be a 2D tensor.
            'max' means that global max pooling will be applied.
        classes: optional number of classes to classify images into, only to be specified if include_top is True, and if no weights argument is specified.
    Returns
        A Keras Model instance.

        
        
###Keras - Usage of initializers

#Initializations define the way to set the initial random weights of Keras layers.

model.add(Dense(64,
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
#OR 
from keras import initializers

model.add(Dense(64, kernel_initializer=initializers.random_normal(stddev=0.01)))

# also works; will use the default parameters.
model.add(Dense(64, kernel_initializer='random_normal'))


##Available initializers
keras.initializers.Zeros()
    Initializer that generates tensors initialized to 0.
keras.initializers.Ones()
    Initializer that generates tensors initialized to 1.
keras.initializers.Constant(value=0)
    Arguments
        value: float; the value of the generator tensors.
keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    Initializer that generates tensors with a normal distribution.
keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
    Initializer that generates tensors with a uniform distribution.
keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
    Initializer that generates a truncated normal distribution.
keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
    Initializer capable of adapting its scale to the shape of weights.
    With distribution="normal", samples are drawn from a truncated normal distribution centered on zero, with stddev = sqrt(scale / n) where n is:
        number of input units in the weight tensor, if mode = "fan_in"
        number of output units, if mode = "fan_out"
        average of the numbers of input and output units, if mode = "fan_avg"
    With distribution="uniform", samples are drawn from a uniform distribution within [-limit, limit], with limit = sqrt(3 * scale / n).
    Arguments
        scale: Scaling factor (positive float).
        mode: One of "fan_in", "fan_out", "fan_avg".
        distribution: Random distribution to use. One of "normal", "uniform".
        seed: A Python integer. Used to seed the random generator.
keras.initializers.Orthogonal(gain=1.0, seed=None)
    Initializer that generates a random orthogonal matrix.
    Arguments
        gain: Multiplicative factor to apply to the orthogonal matrix.
        seed: A Python integer. Used to seed the random generator.

keras.initializers.Identity(gain=1.0)
    Initializer that generates the identity matrix.
    Only use for square 2D matrices.
    Arguments
        gain: Multiplicative factor to apply to the identity matrix.

lecun_uniform(seed=None)
    LeCun uniform initializer.
    It draws samples from a uniform distribution within [-limit, limit] where limit is sqrt(3 / fan_in) where fan_in is the number of input units in the weight tensor.
    Arguments
        seed: A Python integer. Used to seed the random generator.

glorot_normal(seed=None)
    Glorot normal initializer, also called Xavier normal initializer.
    It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / (fan_in + fan_out)) where fan_in is the number of input units in the weight tensor and fan_out is the number of output units in the weight tensor.
    Arguments
        seed: A Python integer. Used to seed the random generator.

glorot_uniform(seed=None)
    Glorot uniform initializer, also called Xavier uniform initializer.
    It draws samples from a uniform distribution within [-limit, limit] where limit is sqrt(6 / (fan_in + fan_out)) where fan_in is the number of input units in the weight tensor and fan_out is the number of output units in the weight tensor.
    Arguments
        seed: A Python integer. Used to seed the random generator.

he_normal(seed=None)
    He normal initializer.
    It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / fan_in) where fan_in is the number of input units in the weight tensor.
    Arguments
        seed: A Python integer. Used to seed the random generator.

lecun_normal(seed=None)
    LeCun normal initializer.
    It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(1 / fan_in) where fan_in is the number of input units in the weight tensor.
    Arguments
        seed: A Python integer. Used to seed the random generator.

he_uniform(seed=None)
    He uniform variance scaling initializer.
    It draws samples from a uniform distribution within [-limit, limit] where limit is sqrt(6 / fan_in) where fan_in is the number of input units in the weight tensor.
    Arguments
        seed: A Python integer. Used to seed the random generator.



##Using custom initializers
#If passing a custom callable, then it must take the argument shape 
#(shape of the variable to initialize) and dtype (dtype of generated values):

from keras import backend as K

def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

model.add(Dense(64, kernel_initializer=my_init))





###Keras - Regularizers
#Usage of regularizers

#Regularizers allow to apply penalties on layer parameters 
#or layer activity during optimization. 
#These penalties are incorporated in the loss function that the network optimizes.

#The penalties are applied on a per-layer basis. 
#The exact API will depend on the layer, 
#but the layers Dense, Conv1D, Conv2D and Conv3D have a unified API.
#These layers expose 3 keyword arguments:
    kernel_regularizer: instance of keras.regularizers.Regularizer
    bias_regularizer: instance of keras.regularizers.Regularizer
    activity_regularizer: instance of keras.regularizers.Regularizer

#Example

from keras import regularizers
model.add(Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))

##Available penalties
keras.regularizers.l1(0.)
keras.regularizers.l2(0.)
keras.regularizers.l1_l2(0.)


##Developing new regularizers
#Any function that takes in a weight matrix 
#and returns a loss contribution tensor can be used as a regularizer, e.g.:

from keras import backend as K

def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))

model.add(Dense(64, input_dim=64,
                kernel_regularizer=l1_reg))
                
                

###Keras - Usage of constraints

#Functions from the constraints module allow setting constraints 
#(eg. non-negativity) on network parameters during optimization.

#The exact API will depend on the layer, 
#but the layers Dense, Conv1D, Conv2D and Conv3D have a unified API.
#These layers expose 2 keyword arguments:
    kernel_constraint for the main weights matrix
    bias_constraint for the bias.

from keras.constraints import max_norm
model.add(Dense(64, kernel_constraint=max_norm(2.)))

##Available constraints
max_norm(max_value=2, axis=0): maximum-norm constraint
non_neg(): non-negativity constraint
unit_norm(axis=0): unit-norm constraint
min_max_norm(min_value=0.0, max_value=1.0, rate=1.0, axis=0): minimum/maximum-norm constraint






###Keras - Visualization
    

##Model visualization

from keras.utils import plot_model
plot_model(model, to_file='model.png')

#plot_model takes two optional arguments:
    show_shapes (defaults to False) controls whether output shapes are shown in the graph.
    show_layer_names (defaults to True) controls whether layer names are shown in the graph.

#You can also directly obtain the pydot.Graph object 
#and render it yourself, for example to show it in an ipython notebook :

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))






###Keras - Utils


keras.utils.CustomObjectScope()
    Provides a scope that changes to _GLOBAL_CUSTOM_OBJECTS cannot escape.
    Code within a with statement will be able to access custom objects by name. 
    Changes to global custom objects persist within the enclosing with statement. 
    At end of the with statement, global custom objects are reverted to state at beginning of the with statement.
    #Example
    with CustomObjectScope({'MyObject':MyObject}):
        layer = Dense(..., kernel_regularizer='MyObject')
        # save, load, etc. will recognize custom object by name



keras.utils.HDF5Matrix(datapath, dataset, start=0, end=None, normalizer=None)
    Representation of HDF5 dataset to be used instead of a Numpy array.
    #Example
        x_data = HDF5Matrix('input/file.hdf5', 'data')
        model.predict(x_data)
    Providing start and end allows use of a slice of the dataset.
    Optionally, a normalizer function (or lambda) can be given. 
    This will be called on every slice of data retrieved.
    Arguments
        datapath: string, path to a HDF5 file
        dataset: string, name of the HDF5 dataset in the file specified in datapath
        start: int, start of desired slice of the specified dataset
        end: int, end of desired slice of the specified dataset
        normalizer: function to be called on data when retrieved
    Returns
        An array-like HDF5 dataset.



keras.utils.Sequence()
    Base object for fitting to a sequence of data, such as a dataset.
    Every Sequence must implements the __getitem__ and the __len__ methods. 
    If you want to modify your dataset between epochs 
    you may implement on_epoch_end. 
    The method __getitem__ should return a complete batch.
    Sequence are a safer way to do multiprocessing. 
    This structure guarantees that the network will only train once on each sample per epoch 
    which is not the case with generators.
    #Examples
    from skimage.io import imread
    from skimage.transform import resize
    import numpy as np
    import math

    # Here, `x_set` is list of path to the images
    # and `y_set` are the associated classes.

    class CIFAR10Sequence(Sequence):

        def __init__(self, x_set, y_set, batch_size):
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size

        def __len__(self):
            return math.ceil(len(self.x) / self.batch_size)

        def __getitem__(self, idx):
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

            return np.array([
                resize(imread(file_name), (200, 200))
                   for file_name in batch_x]), np.array(batch_y)



keras.utils.to_categorical(y, num_classes=None)
    Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    Arguments
        y: class vector to be converted into a matrix (integers from 0 to num_classes).
        num_classes: total number of classes.
    Returns
        A binary matrix representation of the input.


keras.utils.normalize(x, axis=-1, order=2)
    Normalizes a Numpy array.
    Arguments
        x: Numpy array to normalize.
        axis: axis along which to normalize.
        order: Normalization order (e.g. 2 for L2 norm).
    Returns
        A normalized copy of the array.


keras.utils.get_file(fname, origin, untar=False, md5_hash=None, file_hash=None, cache_subdir='datasets', hash_algorithm='auto', extract=False, archive_format='auto', cache_dir=None)
    Downloads a file from a URL if it not already in the cache.
    By default the file at the url origin is downloaded to the cache_dir ~/.keras, 
    placed in the cache_subdir datasets, and given the filename fname. 
    The final location of a file example.txt would therefore be ~/.keras/datasets/example.txt.
    Files in tar, tar.gz, tar.bz, and zip formats can also be extracted. 
    Passing a hash will verify the file after download. 
    The command line programs shasum and sha256sum can compute the hash.
    Arguments
        fname: Name of the file. If an absolute path /path/to/file.txt is specified the file will be saved at that location.
        origin: Original URL of the file.
        untar: Deprecated in favor of 'extract'. boolean, whether the file should be decompressed
        md5_hash: Deprecated in favor of 'file_hash'. md5 hash of the file for verification
        file_hash: The expected hash string of the file after download. The sha256 and md5 hash algorithms are both supported.
        cache_subdir: Subdirectory under the Keras cache dir where the file is saved. If an absolute path /path/to/folder is specified the file will be saved at that location.
        hash_algorithm: Select the hash algorithm to verify the file. options are 'md5', 'sha256', and 'auto'. The default 'auto' detects the hash algorithm in use.
        extract: True tries extracting the file as an Archive, like tar or zip.
        archive_format: Archive format to try for extracting the file. Options are 'auto', 'tar', 'zip', and None. 'tar' includes tar, tar.gz, and tar.bz files. The default 'auto' is ['tar', 'zip']. None or an empty list will return no matches found.
        cache_dir: Location to store cached files, when None it defaults to the Keras Directory.
    Returns
    Path to the downloaded file


keras.utils.print_summary(model, line_length=None, positions=None, print_fn=<built-in function print>)
    Prints a summary of a model.
    Arguments
        model: Keras model instance.
        line_length: Total length of printed lines (e.g. set this to adapt the display to different terminal window sizes).
        positions: Relative or absolute positions of log elements in each line. If not provided, defaults to [.33, .55, .67, 1.].
        print_fn: Print function to use. It will be called on each line of the summary. You can set it to a custom function in order to capture the string summary.



keras.utils.plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB')
    Converts a Keras model to dot format and save to a file.
    Arguments
        model: A Keras model instance
        to_file: File name of the plot image.
        show_shapes: whether to display shape information.
        show_layer_names: whether to display layer names.
        rankdir: rankdir argument passed to PyDot, a string specifying the format of the plot: 'TB' creates a vertical plot; 'LR' creates a horizontal plot.


###Keras- multi-gpu model
keras.utils.multi_gpu_model(model, gpus)
    Replicates a model on different GPUs.
    Specifically, this function implements single-machine multi-GPU data parallelism. It works in the following way:
        Divide the model's input(s) into multiple sub-batches.
        Apply a model copy on each sub-batch. Every model copy is executed on a dedicated GPU.
        Concatenate the results (on CPU) into one big batch.
    E.g. if your batch_size is 64 and you use gpus=2, then we will divide the input into 2 sub-batches of 32 samples, process each sub-batch on one GPU, then return the full batch of 64 processed samples.
    This induces quasi-linear speedup on up to 8 GPUs.
    This function is only available with the TensorFlow backend for the time being.
    Arguments
        model: A Keras model instance. To avoid OOM errors, this model could have been built on CPU, for instance (see usage example below).
        gpus: Integer >= 2 or list of integers, number of GPUs or list of GPU IDs on which to create model replicas.
    Returns
        A Keras Model instance which can be used just like the initial model argument, but which distributes its workload on multiple GPUs.
#Example
import tensorflow as tf
from keras.applications import Xception
from keras.utils import multi_gpu_model
import numpy as np

num_samples = 1000
height = 224
width = 224
num_classes = 1000

# Instantiate the base model (or "template" model).
# We recommend doing this with under a CPU device scope,
# so that the model's weights are hosted on CPU memory.
# Otherwise they may end up hosted on a GPU, which would
# complicate weight sharing.
with tf.device('/cpu:0'):
    model = Xception(weights=None,
                     input_shape=(height, width, 3),
                     classes=num_classes)

# Replicates the model on 8 GPUs.
# This assumes that your machine has 8 available GPUs.
parallel_model = multi_gpu_model(model, gpus=8)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')

# Generate dummy data.
x = np.random.random((num_samples, height, width, 3))
y = np.random.random((num_samples, num_classes))

# This `fit` call will be distributed on 8 GPUs.
# Since the batch size is 256, each GPU will process 32 samples.
parallel_model.fit(x, y, epochs=20, batch_size=256)

# Save model via the template model (which shares the same weights):
model.save('my_model.h5')

##On model saving
#To save the multi-gpu model, use .save(fname) 
#or .save_weights(fname) with the template model 
#(the argument you passed to multi_gpu_model), 
#rather than the model returned by multi_gpu_model.




