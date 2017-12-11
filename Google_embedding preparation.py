import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from tqdm import tqdm


default = 'D:/GitHub/Quora'
token_train = pd.read_pickle(default + "/data/token_train.pkl")
token_test = pd.read_pickle(default + "/data/token_test.pkl")

# Download from https://code.google.com/archive/p/word2vec/
word2vec = KeyedVectors.load_word2vec_format(default + '/data/GoogleNews-vectors-negative300.bin.gz', 
                                             binary=True)
# SPECIFICATION
EMBEDDING_DIM = word2vec.vector_size

# Embeddings preparation
# From https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb
vocabulary = dict()
inverse_vocabulary = ['<unk>']
stops = set(stopwords.words('english'))
questions = ['tokenq1', 'tokenq2']

for dataset in [token_train, token_test]:
    for index, row in tqdm(dataset.iterrows()):
        for question in questions:
            question_to_number = []           
            for word in row[question]:
                if word in stops and word not in word2vec.vocab:
                    continue
                if word not in vocabulary:
                    vocabulary[word] = len(inverse_vocabulary)
                    question_to_number.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(word)
                else:
                    question_to_number.append(vocabulary[word])
            dataset.set_value(index, question, question_to_number)
            
# Build the embedding matrix
embeddings = np.zeros((len(vocabulary), EMBEDDING_DIM)) 
for word, index in tqdm(vocabulary.items()):
    if word in word2vec.vocab:
        embeddings[index] = word2vec.word_vec(word)

# Save embeddings and vocabulary to file
np.save(file = default + "/data/embeddings.npy", arr = embeddings,allow_pickle = False)
np.save(file = default + "/data/vocabulary.npy",arr = vocabulary, allow_pickle = True )
np.save(file = default + "/data/inverse_vocabulary.npy",arr = inverse_vocabulary, allow_pickle = True )