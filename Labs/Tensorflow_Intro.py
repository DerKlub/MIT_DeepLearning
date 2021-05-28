import tensorflow as tf
import mitdeeplearning as mdl
import numpy as np
import matplotlib.pyplot as plt


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

sports = tf.constant(["Tennis", "Basketball"], tf.string)
number = tf.constant([3.141592, 1.414213, 2.71821], tf.float64)

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