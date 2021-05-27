import tensorflow as tf


#RNN Intuition  (pseudocode)
def Rnn():
    print('this is just here so the errors go away')

my_rnn = Rnn()
hidden_state = [0, 0, 0, 0]

sentence = ["I", "love", "recurrent", "neural"]

for word in sentence:
    prediction, hidden_state = my_rnn(word, hidden_state)
    #feeding current word and previous hidden state into RNN
    #this generates prediction and updates hidden state

next_word_prediction = prediction





#RNNs from Scratch

class MyRNNCell(tf.keras.layers.Layer):    
    def __init__(self, rnn_units, input_dim, output_dim):
        super(MyRNNCell, self).__init__()

        #initialize weight matrices
        self.W_xh = self.add_weight([rnn_units, input_dim])
        self.W_hh = self.add_weight([rnn_units, rnn_units])
        self.W_hy = self.add_weight([output_dim, rnn_units])

        #initialize hidden state to zeros
        self.h = tf.zeros([rnn_units, 1])
    
    def call(self, x):  #describes how to make forward pass
        #update the hidden state
        self.h = tf.math.tanh(self.W_hh * self.h + self.W_xh * x)

        #compute the output
        output = self.W_hy * self.h
        
        #retuen the current output and hidden state
        return output, self.h


#tensorflow RNN shortcut:

tf.keras.layers.SimpleRNN(rnn_units)





#Long Short Term Memory

tf.keras.layers.LSTM(num_units)