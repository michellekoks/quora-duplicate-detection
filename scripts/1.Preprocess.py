import pandas as pd
import numpy as np
from util import *

print('Loading the csv data...')
x_train = pd.read_csv('./data/train_data.csv')
y_train = pd.read_csv('./data/train_labels.csv')
x_test = pd.read_csv('./data/test_data.csv')
print('Done')

x_train, x_test = unique_question_map(x_train, x_test)

df_train = x_train.drop('is_duplicate',1)
df_train = df_train.merge(y_train,on="id")

print('Preprocess the data, remove the punctuation, spell correcting, convert to lower-case....')
df_train = df_train[['id','question1', 'qid1', 'question2', 'qid2','is_duplicate']].dropna()
df_train['question1'] = df_train['question1'].apply(preprocess) 
df_train['question2'] = df_train['question2'].apply(preprocess)

df_test = x_test
df_test = df_test[['test_id','question1', 'qid1', 'question2', 'qid2']].dropna()
df_test['question1'] = df_test['question1'].apply(preprocess) 
df_test['question2'] = df_test['question2'].apply(preprocess)
print('Done')

print('Tokenize data...')
token_train = df_train.assign(tokenq1 = df_train['question1'].apply(tokenize),
                              tokenq2 = df_train['question2'].apply(tokenize))
token_train = token_train[['id','qid1','tokenq1','qid2','tokenq2','is_duplicate']]



token_test = df_test.assign(tokenq1 = df_test['question1'].apply(tokenize),
                              tokenq2 = df_test['question2'].apply(tokenize))
token_test = token_test[['test_id','qid1','tokenq1','qid2','tokenq2']]
print('Done')
# Save to file
#df_train.to_pickle("./data/df_train.pkl")
#df_test.to_pickle("./data/df_test.pkl")

print('Save file to pickle files...')
token_train.to_pickle("./data/token_train.pkl")
token_test.to_pickle("./data/token_test.pkl")
print('Done')


