import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# rb stand for read binary
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

# select some part of the text 
# text = text[300000:800000]


# Prepare feature and target data

# goign to user 40 characters in order to predict the next character
# and we're going to define a step size 
SEQ_LENGTH = 40
# Step size is how many characters are we going to shift to the next sentence
STEP_SIZE = 3

sentences = [] #features
next_characters = [] #target
# example: 'sentences = ['how are yo'] -> next_characters = ['u']
# feed a bunch of characters (ex. 40) into neural network 
# and then as result we want to get the next character

for i in range(0,len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i+SEQ_LENGTH]) # <0:4>
    next_characters.append(text[i+SEQ_LENGTH]) # <5>
    # print(sentences)


# Convert text into numerical format - because neural network will not be able to deal with sentences

#character set - contains all the possible characters that occur in the text
characters = sorted(set(text))
# print("characters", characters)

# dictionaries to convert characters into numerical format and back
char_to_index = dict((c, i) for i, c in enumerate(characters))
# print("char_to_index",char_to_index)

index_to_char = dict((i, c) for i, c in enumerate(characters))
# print("index_to_char",index_to_char)

x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=bool)
# how many senteces * sequence length * amount of possible characters
# boolean for each possbillity?

#whenever in a specific sentence at a specific postion a certain character 
# occurs we're going that to true and oll other valyes remian 0


y = np.zeros((len(sentences), len(characters)), dtype=bool)
# which would be the enxt character for which sentence
# at sentence 5 the next character is character with enumeration 8

for i, sentence in enumerate(sentences):
    for n, character in enumerate(sentence):
        x[i, n, char_to_index[character]] = 1
        # każdy znak ma tablicę booleanów o wielkości wszystkich możliwych znaków
        # ustawiamy w tej tabeli znak który reprezentuje jako 1
        # załóżmy, że mamy w słowniku [abcdefgh]
        # a dany znak to 'c' to ustawimy [i][n][00100000]

        y[i, char_to_index[next_characters[i]]] = 1
        # tutaj analogicznie każda sekwencja ma odpowiedź
        # w postaci tablicy wszystkich możliwych znaków
        # i jedynę ustawiamy na znaku któy go reprezentuje




# building the neural network 

model = Sequential()
# LTSM - memory of our network it will remember the past couple of characters 
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

# batch_size - how many example to put in network at once
# epochs - how many times network is going to see the same data
model.fit(x,y, batch_size=256, epochs = 4)

model.save('textgenerator.model')
