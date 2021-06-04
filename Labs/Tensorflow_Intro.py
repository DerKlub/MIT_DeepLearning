from re import X
import tensorflow as tf
import mitdeeplearning as mdl
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_logging_ops import histogram_summary


'''
Called Tensorflow because it hands the flow(node/mathematical operation) of tensors,
which are data structures that can be thought of as multi-dimensional arrays.
Tensors are represented as n-dimensional arrays of base datatypes such as string or integer.
They provide a way to generalize vectors and matrices to higher dimensions.

The shape of a tensor defines its number of dimensions and the size of each dimension.
The rank of a tensor provides the number of dimensions (n-dimensions)... think of as order/degree
'''

#here is a 0-d tensor, of which a scalar is an example:

sport = tf.constant("Tennis", tf.string)
number = tf.constant(1.41421356237, tf.float64)
print('\n')
print("`sport` is a {}-d Tensor".format(tf.rank(sport).numpy()))
print("`number` is a {}-d Tensor".format(tf.rank(number).numpy()))

print('\n')


#vectors and lists can be used to create 1D tensors:

sports = tf.constant(["Tennis", "Basketball"], tf.string)   #dimension = num lists= 1D, shape = num elements = 2
number = tf.constant([3.141592, 1.414213, 2.71821], tf.float64) # 1D, shape = 3

print("`sports` is a {}-d Tensor with shape: {}".format(tf.rank(sports).numpy(), tf.shape(sports)))
print("'number' is a {}-d Tensor with shape {}".format(tf.rank(number).numpy(), tf.shape(number)))

print('\n')



'''
Next we consider creating 2D (i.e., matrices) and higher-rank tensors.
In a future lab using a 4D tensor for computer vision, the dimensions correspond to the number 
of example images in our batch, image height, image width, and number of color channels
'''

#Defining higher-order tensors

matrix = tf.constant([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

assert isinstance(matrix, tf.Tensor), "matrix must be a tf tensor object"
assert tf.rank(matrix).numpy() == 2


#Define a 4D tensor
#Use tf.zeros to initialize a 4-d Tensor of zeros with size 10 x 256 x 256 x 3. 
#You can think of this as 10 images where each image is RGB 256 x 256.

images = tf.zeros([10, 256, 256, 3])

assert isinstance(images, tf.Tensor), "matrix must be a tf Tensor object"
assert tf.rank(images).numpy() == 4, "matrix must be of rank 4"
assert tf.shape(images).numpy().tolist() == [10, 256, 256, 3], "matrix is correct shape"


#the shape of a tensor provides the number of elements in each tensor dimension
#you can also use slicing to access subtensors within a higher-rank tensor:

row_vector = matrix[1]
column_vector = matrix[:,2]
scalar = matrix[1, 2]

print("`row_vector`: {}".format(row_vector.numpy()))
print("`column_vector`: {}".format(column_vector.numpy()))
print("`scalar`: {}".format(scalar.numpy()))

print('\n')









'''
COMPUTATIONS ON TENSORS

Convenient way to think about/visualize computations in Tensorflow is in terms of graphs.
We can define this graph in terms of tensors, which hold data, and the mathematical operations
that act on these tensors in the same order... going to have to look at google colab :(
'''

#Create the nodes in the graph and initialize the values
a = tf.constant(15)
b = tf.constant(61)

#Add them
c1 = tf.add(a,b)
c2 = a + b #Tensorflow overrides the + operation so that it is able to act on tensors

print(f"{a} + {b} =")
print(f"c1: {c1}")
print(f"cs: {c2}")

print('\n')

def func(a,b):
    c = tf.add(a,b)
    d = tf.subtract(a,b)
    e = tf.multiply(c,d)
    return e

a, b = 1.5, 2.5
e_out = func(a,b)
print(e_out)
print('''
Notice how our output is a tensor with value defined by output of 
the computation, and that the output has no shape as it is a single
scalar value''')


print('\n')






'''
NEURAL NETWORKS IN TENSORFLOW

TensorFlow uses high-level API called Keras

First consider simple perceptron defined by on dense layer: 
y=σ(Wx+b), where W represents a matrix of weights, b is a bias, x is the
input, σ is the sigmoid activation function, and  y  is the output. 

Tensors can flow through abstract types called layers.
Layers implement common neural net operations, and are used to update
weights, compute losses, and define inter-layer connectivity.
'''


####Defining a network layer

# n_output_nodes = number of output nodes
# input shape = shape of the input
# x = input to the layer

class OurDenseLayer(tf.keras.layers.Layer):
    def __init__(self, n_output_nodes):
        super(OurDenseLayer, self).__init__()
        self.n_output_nodes = n_output_nodes
    
    def build(self, input_shape):
        d = int(input_shape[-1])
        #define and initialize parameters: weight and bias matrices
        #note that parameter initialization is random
        self.W = self.add_weight("weight", shape=[d, self.n_output_nodes]) #note the dimensionality
        self.b = self.add_weight("bias", shape=[1, self.n_output_nodes])  #note the dimensionality

    def call(self, x):
        z = tf.matmul(x, self.W) + self.b
        y = tf.sigmoid(z)
        return y

#since layer parameters are initialized randomly, we will set a random seed for reproducibility
tf.random.set_seed(1)
layer = OurDenseLayer(3)
layer.build((1,2))
x_input = tf.constant([[1, 2.]], shape =(1,2))
y = layer.call(x_input)

#test the output
print(y.numpy())
mdl.lab1.test_custom_dense_layer_output(y)


print('\n')
'''
Conveniently, TensorFlow has defined a number of layers that are commonly used in
neural nets, for example, a "Dense." Now, instead of using a single Layer to define
our simple neural net, we'll use the "Sequential" model from Keras and a single 
"Dense" layer to define our net. With the "Sequential" API, you can readily create
neural nets by stacking together layers like building blocks
'''


### Defining a neural net using the Sequential API:


#import relevant packages:
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense #no reason to do this if you just import tensorflow lol

n_output_nodes = 3  #define number of outputs

model = Sequential()  #first define the model

dense_layer = tf.keras.layers.Dense(n_output_nodes, activation ='sigmoid') #https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense?version=stable

model.add(dense_layer)  #just adds this layer to the model


#Test model with example input:
x_input = tf.constant([[1,2]], shape=(1,2))

model_output = model(x_input).numpy()
print(model_output)



print('\n')
'''
In addition to defining models using the Sequential API, we can also define neural
nets by directly subclassing the "Model" class, which groups layers together to enable
model training and inference. The "Model" class captures what we refer to as a model
or as a network. Using Subclassing, we can create a class for our model, and then define
the forward pass through the net using the "call" function. Subclassing affords the 
flexibility to define custom layers, custom training loops, custom activation functions,
and custom models. 

Let's define the same neural net as above now using Subclassing rather than the Sequential model.
'''

### Defining a model using subclassing ###

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class SubclassModel(tf.keras.Model):
    def __init__(self, n_output_nodes):  #define model's layers in __init__()
        super(SubclassModel, self).__init__()

        self.dense_layer = tf.keras.layers.Dense(n_output_nodes, activation='sigmoid')
    
    def call(self, inputs):  #in the call function, we define the Model's forward pass
        return self.dense_layer(inputs)

#test this net 

n_output_nodes = 3
model = SubclassModel(n_output_nodes)

x_input = tf.constant([[1,2.]], shape=(1,2))

print(model.call(x_input))



print('\n')
'''
Importantly, Subclassing affors us a lot of flexibility to define custom models. 
For example, we can use the boolean arguments in the call function to specify different
network behaviors, for example, different behaviors during training and inference.

Let's suppose under some instances we want our network to simply output the input, without
any pertubation. We define a boolean argument "isidentity" to conduct this behavior:
'''

### Defining a model using subclassing and specifying custom behavior ###

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class IdentityModel(tf.keras.Model):

    #As before, in __init__() we define the Model's layers
    #Since our desired behavior involves the forward pass, this part is unchanged
    def __init__(self, n_output_nodes):
        super(IdentityModel, self).__init__()
        self.dense_layer = tf.keras.layers.Dense(n_output_nodes, activation='sigmoid')

    def call(self, inputs, isidentity=False): #implements behavior where net outputs the input, unchanged, under control of isidentity argument
        x = self.dense_layer(inputs)
        if isidentity:
            return inputs
        return x

#testing the net
n_output_nodes = 3
model = IdentityModel(n_output_nodes)

x_input = tf.constant([[1,2.]], shape=(1,2))

out_activate = model(x_input)
out_identity = model(x_input, isidentity=True)

print("Network output with activation: {}; network identity output: {}".format(out_activate.numpy(), out_identity.numpy()))







print('\n')





'''
AUTOMATIC DIFFERENTIATION IN TENSORFLOW

Automatic differentiation is one of the most important parts of TensorFlow and is the backbone
of training with backpropagation. We will use the TensorFlow GradientTape "tf.GradientTape" to
trace operations for computing gradients later.

When a forward pass is made through the net, all forward-pass operations get
recorded to a "tape"; then, to compute the gradient, the tape is played backwards.
By default, the tape is discarded after it is played backwards; this means
that a particular "tf.GradientTape" can only compute one gradient, and 
subsequent calls throw a runtime error. However, we can compute multiple
gradients over the same computation by creating a "persistent" gradient tape.

First, we will look at how we can compute gradients using GradientTape and access
them for computation.

We define the simple function y=x**2  and compute the gradient:
'''


### Gradient computation with GradientTape ###

# y = x^2
# Example: x = 3.0
x = tf.Variable(3.0)

# Initiate the gradient tape
with tf.GradientTape() as tape:
  # Define the function
  y = x * x
# Access the gradient -- derivative of y with respect to x
dy_dx = tape.gradient(y, x)

assert dy_dx.numpy() == 6.0
print(f"Gradient of y = x^2 when x = 3 is: {dy_dx}")



print('\n')
'''
In training neural nets, we use differentiation and stochastic gradient
descent (SGD) to optimize a loss function. Now that we have a sense of how
GradientTape can be used to compute and access derivatives, we will look at
an example where we use automatic differentiation and SGD to find the 
minimum of L=(x−x_f)^2 . Here, x_f is a variable for a desired value
we are trying to optimize for; L represents a loss that we are trying 
to minimize. While we can clearly solve this problem analytically
(x_min = x_f), considering how we can compute this using GradientTape
sets us up nicely for future labs where we use gradient descnet to 
optimize entire neural net losses
'''

### Function minimization with automatic differentiation and SGD ###

#initialize a random value for our initial x
x = tf.Variable(tf.random.normal([1]))
print("Initializing x = {}".format(x.numpy()))

learning_rate = 1e-2  #learning rate for SGD
history = []
x_f = 4  #define the target value

#We will run SGD for a number of iterations. At each iteration, we compute the loss,
#compute the derivative of the loss with respect to x, and perform the SGD update
for i in range(500):
    with tf.GradientTape() as tape:
        loss = (x - x_f)**2
    
    #loss minimization using gradient tape
    grad = tape.gradient(loss, x) #compute derivative of loss with respect to x
    new_x = x - learning_rate * grad  #SGD update
    x.assign(new_x) #update the value of x
    history.append(x.numpy()[0])

#plot the evolution of x as we optimize towards x_f
plt.plot(history)
plt.plot([0, 500], [x_f, x_f])
plt.legend(('Predicted', 'True'))
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.show()