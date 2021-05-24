import tensorflow as tf

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init__()

        #Initialize weights and biases
        self.W = self.add_weight([input_dim, output_dim])
        self.b = self.add_weight([1, output_dim])

    def call(self, inputs):  #defining forward propagation
        #Forward propagate the inputs
        z = tf.matmu1(inputs, self.W) + self.b

        #Feed through a non-linear activation
        output = tf.math.sigmoid(z)

        return output
