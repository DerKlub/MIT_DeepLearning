import tensorflow as tf
import mitdeeplearning as mdl
import numpy as np
import os
import time
import functools
from IPython import display as ipythondisplay
from tensorflow.python.keras.backend import rnn
from tqdm import tqdm

import regex as re
import subprocess
import urllib
import matplotlib.pyplot as plt
#import timidity
#pip install timidity




#Download the dataset
songs = mdl.lab1.load_training_data()

#print one of the songs to inspect in greater detail
example_song = songs[0]
print("\nExample song: ")
print(example_song)

#convert the ABC notation to audio file and listen
#mdl.lab1.play_song(example_song)        #doesn't work, but will include cool things... check out the Google Colab



print('\n')
'''
Notation of music does not simply contain info on the notes being played, 
but additionally there is meta info such as the song title, key, tempo, etc.
How does the number of different characters that are present in the text
file impact the complexity of the learning problem? This will become
important soon, when we generate a numerical representation for the text data.
'''
# Join our list of song strings into a single string containing all songs
songs_joined = "\n\n".join(songs) 

# Find all unique characters in the joined string
vocab = sorted(set(songs_joined))
print("There are", len(vocab), "unique characters in the dataset\n")



'''
Goal of RNN:

Given a character, or a sequence of characters, what is the most probable
next character. To achieve this, we will inout a sequence of characters to 
the model, and train the model to predict the output, that is, the following
character at each time step. RNNs maintain an internal state that depends on
previously seen elements, so info about all characts seen up to a given moment
will be taken into account in generating the prediction


Vectorize the text:

Before we begin training our RNN model, we'll need to create a numerical
representation of our text-based dataset. To do this, we'll generate two
lookup tables: one that maps characters to numbers, and a second that
maps numbers back to characters. Recall that we just identified the
unique characters present in the text above.
'''
### Define numerical representations of the text ###

#Create a mapping from character to unique index
#For example, to get index of character "d",
#   we can evaluate 'char2idx["d"]'
char2idx = {u:i for i, u in enumerate(vocab)}


#Create a mapping from indices to characters. This is the inverse
#   of char2idx and allows us to convert back from a unique index
#   to the character in our vocabulary
idx2char = np.array(vocab)


'''
This gives us an integer representation for each character. Observe that
the unique characters (vocab) in the text are mappped as indices from 0
to len(unique). Check it out
'''
print('{')
for char, _ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}\n')





### Vectorize the songs string

'''TODO: Write a function to convert the all songs string to a vectorized
    (i.e., numeric) representation. Use the appropriate mapping
    above to convert from vocab characters to the corresponding indices.

  NOTE: the output of the `vectorize_string` function 
  should be a np.array with `N` elements, where `N` is
  the number of characters in the input string
'''

def vectorize_string(string):
  vectorized_output = np.array([char2idx[char] for char in string])
  return vectorized_output

vectorized_songs = vectorize_string(songs_joined)


print ('{} ---- characters mapped to int ----> {}\n'.format(repr(songs_joined[:10]), vectorized_songs[:10]))
# check that vectorized_songs is a numpy array
assert isinstance(vectorized_songs, np.ndarray), "returned result should be a numpy array"



'''
Creating training examples and targets

Each input sequence that we feed into our RNN will contain "seq_length" characters
from the text. We'll also need to define a target sequence for each input
sequence, which will be used in training the RNN to predict the next character.
For each input, the corresponding target will contain the same length of text, 
except shifted one character to the right.

To do this, we'll break the text into chunks of "seq_length+1". Suppose 
"seq_length" is 4 and our text is "Hello". Then, our input is "Hell" and
the target sequence is "ello"

The batch method will then let us convert this stream of character 
indices to sequences of the desired size
'''
### Batch definition to create training examples ###

def get_batch(vectorized_songs, seq_length, batch_size):
    #the length of the vectorized songs string
    n = vectorized_songs.shape[0] - 1
    #randomly choose the starting indices for the examples in the training batch
    idx = np.random.choice(n-seq_length, batch_size)

    #list of input sequences for the training batch
    input_batch = [vectorized_songs[i : i+seq_length] for i in idx]
    #list of output sequences for the training batch
    output_batch = [vectorized_songs[i+1 : i+seq_length+1] for i in idx]

    #x_batch, y_batch provide the true inputs and targets for net training
    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])
    return x_batch, y_batch

# Perform some simple tests to make sure your batch function is working properly! 
test_args = (vectorized_songs, 10, 2)
if not mdl.lab1.test_batch_func_types(get_batch, test_args) or \
   not mdl.lab1.test_batch_func_shapes(get_batch, test_args) or \
   not mdl.lab1.test_batch_func_next_step(get_batch, test_args): 
   print("======\n[FAIL] could not pass tests\n")
else: 
   print("======\n[PASS] passed all tests!\n")




'''
For each of these vectors, each index is processed at a single time step.
So, for the input at time step 0, the model receives the index for the
first character in the sequence, and tries to predict the index of the 
next character. At the next timestep, it does the same thing, but the RNN
considers the info from the previous step (its updated state), in addition
to the current input

Check it out:

'''

x_batch, y_batch = get_batch(vectorized_songs, seq_length=5, batch_size=1)

for i, (input_idx, target_idx) in enumerate(zip(np.squeeze(x_batch), np.squeeze(y_batch))):
    print("Step {:3d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))







print('\n\n')
'''
THE RNN MODEL

The model is based off the LSTM architecture, where we use a state vector
to maintain info about the temporal relationships between consecutive
characters. The final output of the LSTM is then fed into a fully connected
Dense layer where we'll output a softmax over each character in the vocabulary,
and then sample from this distribution to predict the next character

Will be using Keras API:  tf.keras.Sequential  to define model and three layers:
    1. tf.keras.layers.Embedding
        this is the input layer, consisting of a trainable lookup table 
        that maps the numbers of each character to a vector with embedding_dim dimension
    2. tf.keras.layers.LSTM
        our LSTM net, with size units = rnn_units
    3. tf.keras.layers.Dense
        the output layer, with vocab_size output
'''


### Defining the RNN Model ###

def LSTM(rnn_units):
    return tf.keras.layers.LSTM(
        rnn_units, 
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        recurrent_activation='sigmoid',
        stateful=True
    )

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        #Layer 1: Embedding layer to transform indices into dense vectors of a fixed embedding size
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape= [batch_size, None]),

        #Layer 2: LSTM with 'rnn_units' number of units
        LSTM(rnn_units),

        #Layer 3: Dense layer that transforms LSTM output into vocabulary size
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


#Build a simple model with default hyperparameters. You will get the chance to change these later:

model = build_model(len(vocab), embedding_dim=256, rnn_units=1024, batch_size=32)


'''
Test out the RNN model
use Model.summary() to print out a summary of model's internal workings
'''

model.summary()

print('\n')
#we can also check dimensionality of output, using sequence length 100 (can use any length)

x, y = get_batch(vectorized_songs, seq_length=100, batch_size=32)
pred = model(x)
print("Input shape:      ", x.shape, " # (batch_size, sequence_length)")
print("Prediction shape: ", pred.shape, "# (batch_size, sequence_length, vocab_size)")




print('\n')
'''
Predictions from the untrained model

To get actual predictions from the model, we sample the output distribution,
which is defined by a softmax over our character vocabulary. This will give
us actual character indices. This means we are using a categorical distribution
to sample over the example prediction. This gives a prediction of the next
character (specifically its index) at each timestamp.

Note here that we sample from this probability distribution, as opposed to simply
taking the "argmax", which can cause the model to get stuck in a loop ---same character always used becuase it has highest value

Let's try this sampling out for the first example in the batch:
'''

sampled_indices = tf.random.categorical(pred[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=1).numpy()
print(sampled_indices)

print('\n')
#we can now decode these to see the text predicted by the untrained model

print("Input: \n", repr("".join(idx2char[x[0]])))
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))
        
#outputs pretty nonsensical characters because net is untrained





print('\n')
'''
Training the model: loss and training operations

At this point, we can think of our next character prediction problem as
a standard classification problem. Given the previous state of the RNN, 
as well as the input at a given timestep, we want to predict the next
class of the next character-- that is, to actually predict the next character.

To train our model on this classification task, we can use a form of the
"crossentropy" loss (negative log likelihood loss). Specifically, we will use
the "sparse_categorical_crossentropy" loss, as it utilizes integer targets
for categorical classification tasks. We will want to compute the loss
using the true targets--- the "labels"--- and the predicted targets----the "logits"
'''

### Defining the loss function ###

def compute_loss(labels, logits):   #(true, predicted)
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return loss

#compute loss using true next characters from example batch and predictions from untrained model
example_batch_loss = compute_loss(y, pred)

print("Prediction shape: ", pred.shape, " # (batch_size, sequence_length, vocab_size)") 
print("scalar_loss:      ", example_batch_loss.numpy().mean())


print('\n')
'''
Let's start by defining some hyperparameters for the training model. To start,
we have provided some reasonable values for some of the parameters. It is up
to you to use what we've learned in the class to help optimize the parameter
selection here:    :(
'''

### Hyperparameter setting and optimization ###

#Optimization parameters:
num_training_iterations = 2000 #increase this to train longer
batch_size = 4  #experiment between 1 and 64
seq_length = 100  #experiment between 50 and 500
learning_rate = 5e-3  #experiment between 1e-5 and 1e-1

#Model parameters:
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024  #experiments between 1 and 2048

#Checkpoint location:
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")


'''
Now we are ready to define our training operation---- the optimizer and 
duration of training--- and use this function to train the model. You will
experiment with the choice of optimizer and the duration for which your train
your models, and see how these changes affect the net's output. Some optimizers
you may like to try are "Adam" and "Adagrad"

First, we will instantiate a new model and an optimizer. Then, we will 
use the "tf.GradientTape" method to preform the backpropagation operations.

We will also generate a print-out of the model's progress through training, 
which will help us easily visualoze whether or not we are minimizing loss.
'''

### Define optimizer and training operation ###

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size) #can change these in section above, or just put arg = x

optimizer = tf.keras.optimizers.Adam(learning_rate)  #can change this

#tf.function
def train_step(x, y):
    #use tf.GradientTape)()
    with tf.GradientTape() as tape:
        y_hat = model(x)   #x = input

        loss = compute_loss(y, y_hat)    #(labels, logits) --> (true, predicted)


    #now compute the gradients
    #model.trainable_variables gives list of all parameters
    grads = tape.gradient(loss, model.trainable_variables)