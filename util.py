import pandas as pd
import nltk
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
import string
from pygoose import *

#####################################################
# LOAD READY-MADE STOPWORDS AND SPELL CHECK         #
# https://github.com/YuriyGuts/                     #                       #
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
# LOADING GLOVE VECTOR                             #
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
    :return: vector for word representation
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
#################################

