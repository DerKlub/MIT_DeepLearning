import tensorflow as tf


#building dense layer from scratch:


class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init__()

        #Initialize weights and biases
        self.W = self.add_weight([input_dim, output_dim])
        self.b = self.add_weight([1, output_dim])

    def call(self, inputs):  #defining forward propagation
        #Forward propagate the inputs
        z = tf.matmu1(inputs, self.W) + self.b  #matmu = matrix multiplication (dot product)

        #Feed through a non-linear activation
        output = tf.math.sigmoid(z)

        return output




#or can just use tensor flow shortcut  :)


layer = tf.keras.layers.Dense(units = 2)  #dense layer with 2 outputs


#stacking layers on top of each other: 

model = tf.keras.Sequential([
    
    tf.keras.layers.Dense(n),   #layer 1 with n neurons
    tf.keras.layers.Dense(2)    #layer 2 with 2 neurons (output)
])
