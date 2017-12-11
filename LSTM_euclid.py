import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import itertools

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Merge
import keras.backend as K
from keras.optimizers import Adadelta, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping


# READ DATA
default = 'D:/GitHub/Quora'
token_train = pd.read_pickle(default + "/data/token_train_n.pkl")
token_test = pd.read_pickle(default + "/data/token_test_n.pkl")

# load the embeddings and vocab:        
embeddings = np.load(file = default + "/data/embeddings.npy")
vocabulary = np.load(file = default + "/data/vocabulary.npy")
inverse_vocabulary = np.load(file = default + "/data/inverse_vocabulary.npy")


# SPECIFICATIONS
TRAIN_SIZE = 0.8
MAX_SEQ_LEN = 30
EMBEDDING_DIM = embeddings.shape[1]
NB_WORDS = embeddings.shape[0]
       
# Split to train validation
data_all = token_train[['tokenq1', 'tokenq2','is_duplicate']]

train, val = train_test_split(data_all, train_size = TRAIN_SIZE)

# Split data to dicts for easy ref
X_train = {'left': train.tokenq1, 'right': train.tokenq2}
X_val = {'left': val.tokenq1, 'right': val.tokenq2}
X_test = {'left': token_test.tokenq1, 'right': token_test.tokenq2}

# Convert labels to their numpy representations
Y_train = train['is_duplicate'].values
Y_val = val['is_duplicate'].values

# Zero padding
for dataset, side in itertools.product([X_train, X_val], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=MAX_SEQ_LEN)
    
# MODELLING SPECIFICATION
N_LAYERS = 10
#GRAD_CLIPPING_NORM = 1.25
BATCH_SIZE = 50
N_EPOCH = 25
DROP_OUT = 0.5


# Euclidean function
def exponent_neg_euclid_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs
        FROM https://medium.com/mlreview/implementing-eulstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07
    '''
    return K.exp(-K.sum(K.sqrt(K.square(left-right), axis=1, keepdims=True)))
    
# The visible layer
left_input = Input(shape=(MAX_SEQ_LEN,), dtype='int32')
right_input = Input(shape=(MAX_SEQ_LEN,), dtype='int32')

embedding_layer = Embedding(NB_WORDS, 
                           EMBEDDING_DIM, 
                           weights=[embeddings], 
                           input_length=MAX_SEQ_LEN, 
                           trainable=False)

# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

lstm_layer = LSTM(N_LAYERS)

left_output = lstm_layer(encoded_left)
right_output = lstm_layer(encoded_right)

# Calculates the distance as defined by the eulstm model
eulstm_distance = Merge(mode = lambda x: exponent_neg_euclid_distance(x[0], x[1]), 
                        output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

# Define model
eulstm = Model([left_input, right_input], [eulstm_distance])

# Adadelta optimizer, with gradient clipping by norm
#optimizer = Adadelta(clipnorm=gradient_clipping_norm)

# Compile model
learning_rate = 0.01
decay_rate = learning_rate / N_EPOCH

sgd = SGD(lr=learning_rate, 
          decay=decay_rate, 
          nesterov=False)

eulstm.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

# =============================================================================
 # Start training
STAMP = 'eulstm_%d'%(N_LAYERS)
early_stopping =EarlyStopping(monitor='val_loss', patience=3)
model_path = STAMP + '.h5'

model_checkpoint = ModelCheckpoint(default + "/" + model_path, 
                                   save_best_only=True, save_weights_only=True)


eulstm_trained = eulstm.fit([X_train['left'], X_train['right']], Y_train, 
                            batch_size=BATCH_SIZE, 
                            epochs=N_EPOCH,
                            validation_data=([X_val['left'], X_val['right']], Y_val), 
                            validation_split = 0.2)


# Save history of trainning:
np.save(file = default+'/'+ STAMP+'_trainhist.npy', arr =eulstm_trained.history)

# Save model weights
eulstm.save_weights(default+'/'+ STAMP + 'eulstm_weights.npy')

# Save val_prediction for f_score
predict_val = eulstm.predict([X_val['left'], X_val['right']], batch_size = None)
val_score = pd.DataFrame()
val_score['labels'] = val['is_duplicate']
val_score['predict'] = predict_val
val_score.to_csv(default + '/' + STAMP+ 'val_score.csv', index = False) 

# Plot accuracy
plt.plot(eulstm_trained.history['acc'])
plt.plot(eulstm_trained.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot loss
plt.plot(eulstm_trained.history['loss'])
plt.plot(eulstm_trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

########################################
# TRAIN AGAIN WITH THE WHOLE DATASET####
X_train_all =  {'left': data_all.tokenq1, 'right': data_all.tokenq2}
Y_train_all = data_all['is_duplicate'].values

for side in ['left', 'right']:
    X_train_all[side] = pad_sequences(X_train_all[side], 
               maxlen=MAX_SEQ_LEN)

eulstm_trained = eulstm.fit([X_train_all['left'], X_train_all['right']], 
                            Y_train_all, 
                            batch_size=BATCH_SIZE, 
                            epochs=N_EPOCH - 10)
 
# Save history of trainning:
np.save(file = default+'/eulstm_history_train_all2.npy', arr =eulstm_trained.history)

#Save model weights
eulstm.save_weights(default+ '/eulstm_all2.npy')


# Make prediction
for side in ['left', 'right']:
    X_test[side] = pad_sequences(X_test[side], maxlen=MAX_SEQ_LEN)
     
 
predict = eulstm.predict([X_test['left'], X_test['right']], batch_size = None)
submission = pd.DataFrame()

submission['test_id'] = token_test['test_id']
submission['is_duplicate'] = predict.round(0)

submission.to_csv(default + '/submission2.csv', index = False) 
