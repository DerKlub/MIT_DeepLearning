#RNN Intuition  (pseudocode)

my_rnn = Rnn()
hidden_state = [0, 0, 0, 0]

sentence = ["I", "love", "recurrent", "neural"]

for word in sentence:
    prediction, hidden_state = my_rnn(word, hidden_state)
    #feeding current word and previous hidden state into RNN
    #this generates prediction and updates hidden state

next_word_prediction = prediction
