from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors, Word2Vec
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import datetime

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Merge
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint

token_train = pd.read_pickle("./data/token_train.pkl")
token_test = pd.read_pickle("./data/token_test.pkl")

googlenews_embedding = './data/GoogleNews-vectors-negative300.bin.gz'
word2vec = KeyedVectors.load_word2vec_format(googlenews_embedding, binary=True)

vocabulary = dict()
inverse_vocabulary = ['<unk>']
stops = set(stopwords.words('english'))
questions = ['tokenq1', 'tokenq2']

# Iterate over the questions of both training and test datasets
for dataset in [token_train, token_test]:
    for index, row in dataset.iterrows():

        # Iterate through the text of both questions of the row
        for question in questions:

            question_to_number = []  # q2n -> question numbers representation
            for word in row[question]:

                # Check for unwanted words
                if word in stops and word not in word2vec.vocab:
                    continue

                if word not in vocabulary:
                    vocabulary[word] = len(inverse_vocabulary)
                    question_to_number.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(word)
                else:
                    question_to_number.append(vocabulary[word])

            # Replace questions as word to question as number representation
            dataset.set_value(index, question, question_to_number)
            
# Build the embedding matrix
embedding_dim = 300
embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
embeddings[0] = 0  # So that the padding will be ignored

for word, index in vocabulary.items():
    if word in word2vec.vocab:
        embeddings[index] = word2vec.word_vec(word)
        
# split to train validation
validation_size = 40000
training_size = len(token_train) - validation_size

question_cols = token_train[['tokenq1', 'tokenq2']]
duplicate = token_train['is_duplicate']

X_train, X_validation, Y_train, Y_validation = train_test_split(question_cols, duplicate, test_size=validation_size)

# Split to dicts
X_train = {'left': X_train.tokenq1, 'right': X_train.tokenq2}
X_validation = {'left': X_validation.tokenq1, 'right': X_validation.tokenq2}
X_test = {'left': token_test.tokenq1, 'right': token_test.tokenq1}

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

max_seq_length = 30

# Zero padding
for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)
    
# TEST
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

n_hidden = 10
#n_hidden = 1
gradient_clipping_norm = 1.25
batch_size = 50
#batch_size = 1
n_epoch = 25
#n_epoch = 10

# MaLSTM similarity function
def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))
    
# The visible layer
left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')

embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length, trainable=False)

# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

# Since this is a siamese network, both sides share the same LSTM
shared_lstm = LSTM(n_hidden)

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

# Calculates the distance as defined by the MaLSTM model
malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

# Pack it all up into a model
malstm = Model([left_input, right_input], [malstm_distance])

# Adadelta optimizer, with gradient clipping by norm
optimizer = Adadelta(clipnorm=gradient_clipping_norm)

malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

# Start training
training_start_time = time()

malstm_trained = malstm.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, epochs=n_epoch,
                            validation_data=([X_validation['left'], X_validation['right']], Y_validation), validation_split = 0.2)
 
print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))


# Plot accuracy
plt.plot(malstm_trained.history['acc'])
plt.plot(malstm_trained.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot loss
plt.plot(malstm_trained.history['loss'])
plt.plot(malstm_trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

