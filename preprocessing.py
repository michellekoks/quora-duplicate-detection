import pandas as pd
import nltk
nltk.download('stopwords')
import string

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

    return text


def translate(text, translation):
    """
    :param text: text string
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

