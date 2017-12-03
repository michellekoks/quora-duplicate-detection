import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
import itertools

import re
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def combine(df_train, df_test):
    df_all = pd.concat((df_train, df_test))
    df_all = df_all[['question1', 'qid1', 'question2', 'qid2']]
    df_all['question1'].fillna('', inplace=True)
    df_all['question2'].fillna('', inplace=True)
    df_all.head()
    return df_all

def vectorize(df_train, df_test):
    df_all = combine(df_train, df_test)
    counts_vectorizer = CountVectorizer(max_features=10000-1).fit(itertools.chain(df_all['question1'], df_all['question2']))
    other_index = len(counts_vectorizer.vocabulary_)
    print(counts_vectorizer)
    return counts_vectorizer, other_index

def tokenize(df_train, df_test):
    counts_vectorizer, other_index = vectorizer(df_train, df_test)
    words_tokenizer = re.compile(counts_vectorizer.token_pattern)
    print(words_tokenizer)
    return words_tokenizer

def padding_sequences(texts, max_len=10):
    for w in words_tokenizer.findall(s.lower()):
        if w in counts_vectorizer.vocabulary_:
            seqs = counts_vectorizer.vocabulary_[w]
        else:
            seqs = other_index
    return pad_sequences(seqs, maxlen=max_len)
    
    
def create_padded_seqs(texts, max_len=10):
    seqs = texts.apply(lambda s: [counts_vectorizer.vocabulary_[w] if w in counts_vectorizer.vocabulary_ else other_index
         for w in words_tokenizer.findall(s.lower())])
    return pad_sequences(seqs, maxlen=max_len)
















