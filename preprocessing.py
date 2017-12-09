import pandas as pd
import nltk
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
import string
from pygoose import *
from gensim.models import KeyedVectors, Word2Vec

#####################################################
# LOAD READY-MADE STOPWORDS AND SPELL CHECK         #
# https://github.com/YuriyGuts/                     #
# kaggle-quora-question-pairs                       #
#####################################################

stopwords = set(kg.io.load_lines('./data/rm/stopwords.vocab'))
spelling_corrections = kg.io.load_json('./data/rm/spelling_corrections.json')

#####################################################
#CREATE ID FOR QUESTION WITH EXACTLY SIMILAR STRINGS#
#####################################################


def unique_question_dict(x_train, x_test):
    """
    :param: a list of datasets
    :return: dictionary of unique ids for unique questions

    :purpose: easy ref for questions, instead of comparing
    strings for each questions
    """
    qs = pd.concat([x_train['question1'], x_train['question2'],
                   x_test['question1'], x_test['question2']])

    ids = {}
    for question in qs:
        ids.setdefault(question, len(ids))
    return ids


def unique_question_map(x_train, x_test):
    """
    :param: list of datasets that need to have id on questions
    :return: datasets with unique id for unique questions
    """
    ids = unique_question_dict(x_train, x_test)

    x_train = x_train.assign(qid1 = x_train['question1'].map(ids),
                            qid2 = x_train['question2'].map(ids))


    x_test = x_test.assign(qid1 = x_test['question1'].map(ids),
                            qid2 = x_test['question2'].map(ids))

    return (x_train, x_test)


#####################################################
# TEXT PRE-PROCESSING                               #
#####################################################


def preprocess(text):
    """
    :param text: text string
    :return: processed text (dependent on the used functions)
    TO BE UPDATED
    """
    text = spell_digits(text)
    text = expand_negations(text)
    text = remove_punctuation(text)
    text = correct_spelling(text)
    text = text.lower()

    return text

def tokenize(text):
    """
    :param text: (supposedly) cleaned text sentence
    :return: list of word tokens (using nltk package)

    :required: stopwords
    """
    tokens = tokenizer.tokenize(text)
    #tokens = [t for t in tokens if t not in stopwords]

    return tokens


def translate(text, translation):
    """
    :param text: text stringfgas
    :param translation: a dictionary mapping the characters need to be replaced
        and the replacing characters.
    :example:
        trans =  {'?':"!"}
        translate("a?b", trans)

        return "a ! b"
    """
    for token, replacement in translation.items():
        text = text.replace(token, ' ' + replacement + ' ')
    text = text.replace('  ', ' ')
    return text


def spell_digits(text):
    """
    :param text: text string with digit 0-9
    :return:text translated to words from digit
    https://github.com/YuriyGuts/kaggle-quora-question-pairs/blob/master/notebooks/preproc-tokenize-spellcheck.ipynb
    """
    translation = {
        '0': 'zero',
        '1': 'one',
        '2': 'two',
        '3': 'three',
        '4': 'four',
        '5': 'five',
        '6': 'six',
        '7': 'seven',
        '8': 'eight',
        '9': 'nine',
    }
    return translate(text, translation)


def expand_negations(text):
    """
    :param text:
    :return: full text without contraction
    'https://github.com/YuriyGuts/kaggle-quora-question-pairs/blob/master/notebooks/preproc-tokenize-spellcheck.ipynb
    """
    translation = {
        "can't": 'can not',
        "won't": 'would not',
        "shan't": 'shall not',
    }
    text = translate(text, translation)
    return text.replace("n't", " not")


def remove_punctuation(s):
    """
    :param s: text string
    :return: remove '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    """
    result = "".join(i for i in s if i not in string.punctuation)
    return result

def correct_spelling(text):
    """
    :param text: text string
    :return: spell corrected text
    'https://github.com/YuriyGuts/kaggle-quora-question-pairs/blob/master/notebooks/preproc-tokenize-spellcheck.ipynb
    """
    return ' '.join(
        spelling_corrections.get(token, token)
        for token in tokenizer.tokenize(text)
    )
#####################################################
# LOADING GLOVE VECTOR                              #
#####################################################

def loadGloveFile(gloveFilePath):
    """
    load glove file from txt
    """
    print ("Loading Glove Model from " + gloveFilePath + " ....")
    w2v ={}
    with open(gloveFilePath, "rb") as lines:
        for line in lines:
            if (len(w2v) % 50000 == 0):
                print(len(w2v))
            w2v[line.split()[0].decode('utf-8')] = [float(val) for val in line.split()[1:]]
    print ("Done.",len(w2v)," words loaded!")
    return w2v

def token_to_vec(tokenized_questions, glove_vec):
    """
    :param tokenized_questions: list/array of tokenized questions
    :return: 100-D vector for word representation
    """
    print('Look up words in Glove wiki...')
    missing_word = 0
    result = []
    for i, words in enumerate(tokenized_questions):
        if i % 100000 == 0:
            print(i)

        vector = []
        for word in words:
            try:
                vector += [glove_vec[word]]
            except KeyError:
                # print('the word '+word+' is not in the dictionary')
                vector += [[0] * 100]
                missing_word += 1

        result += [vector]
    print("There are {} missing words".format(missing_word))
    print("Those words are replaced with vectors of 100 zeros")
    return result

def word2vec_embedding(token_train, token_test, embedding_dim):
    from gensim.models import KeyedVectors, Word2Vec
    from nltk.corpus import stopwords

    # Load Google News vectors
    googlenews_embedding = './data/GoogleNews-vectors-negative300.bin.gz'
    word2vec = KeyedVectors.load_word2vec_format(googlenews_embedding, binary=True)
    print('google news vectors loaded')
    
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
                print('iteration completed')
                
    # Build the embedding matrix
    embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  
                                            # This will be the embedding matrix
    embeddings[0] = 0  # So that the padding will be ignored

    for word, index in vocabulary.items():
        if word in word2vec.vocab:
            embeddings[index] = word2vec.word_vec(word)
            print('embedding matrix created')
            
    return embeddings


#################################

