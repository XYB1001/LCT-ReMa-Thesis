'''
Different LinearSVC-based systems to try on EFCamdat data
'''
# import argparse
import pickle
import statistics as stats
import numpy as np
import feats
import spacy

from scipy.sparse import hstack
from preprocess import strip_cl_chars
# from feats import SentLen

# from nltk.tag import pos_tag_sents
# from nltk.tokenize import word_tokenize, sent_tokenize

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


def shuffle_prepare_data(data, clean_char=True):
    '''
    Preparing data for classification. Data will be a list of tuples with the format
    (text, id, label, user, time)
    Shuffle, then extract paired lists of samples and labels
    if clean_char is True (default), remove control characters and replace all emojis and emoticons with '<emo> '
    '''
    np.random.shuffle(data)

    if clean_char:
        samples = [ strip_cl_chars(tup[0]) for tup in data ]
    else:
        samples = [ tup[0] for tup in data]

    # samples = [ tup[0] for tup in data]
    labels = [ tup[1] for tup in data]

    return samples, labels


def evaluate(Ygold, Yguess):
    '''Evaluating model performance and printing out scores in readable way'''

    print('-'*50)
    print("Accuracy:", accuracy_score(Ygold, Yguess))
    print('-'*50)
    print("Precision, recall and F-score per class:")

    # get all labels in sorted way
    # Ygold is a regular list while Yguess is a numpy array
    labs = sorted(set(Ygold + Yguess.tolist()))

    # printing out precision, recall, f-score for each class in easily readable way
    PRFS = precision_recall_fscore_support(Ygold, Yguess, labels=labs)
    print('{:10s} {:>10s} {:>10s} {:>10s}'.format("", "Precision", "Recall", "F-score"))
    for idx, label in enumerate(labs):
        print("{0:10s} {1:10f} {2:10f} {3:10f}".format(label, PRFS[0][idx],PRFS[1][idx],PRFS[2][idx]))

    print('-'*50)
    print("Average (macro) F-score:", stats.mean(PRFS[2]))
    print('-'*50)
    print('Confusion matrix:')
    print('Labels:', labs)
    print(confusion_matrix(Ygold, Yguess, labels=labs))
    print()

def identity(x):
    return x


if __name__ == '__main__':

    print('Fetching and preparing data...')
    # Fetching pickled ef data
    f = open('../Data/efcamdat-len_norm.pickle', 'rb')
    data = pickle.load(f)
    f.close()

    # Preparing + shuffling data
    X, Y = shuffle_prepare_data(data)
    # X = X[:1000]
    # Y = Y[:1000]
    print('len(X):', len(X))
    print('len(Y):', len(Y))

    # Using 25% of the data as validation data
    split_point = int(0.75*len(X))
    Xtrain = X[:split_point]
    Ytrain = Y[:split_point]
    Xtest = X[split_point:]
    Ytest = Y[split_point:]

    '''
    Actual modelling
    '''
    # Setting up SVM baseline with word (and char) ngrams)
    print('Starting SVM modeling (Linear SVC)...')

    # Transforming input, 1: postags + freqs as word uni, bi, trigrams
    print('Performing abstraction of training data to POS-tags (with freq)...')
    model = spacy.load('en')
    print('Spacy default English model loaded')

    Xtrain_pos = [ feats.noun_freq_abstraction(sample, model=model) for sample in Xtrain ] # list of POS or POS/word -sequences
    # print(Xtrain_pos[:5])

    vec_POS = CountVectorizer(ngram_range=(1,3))
    Xtrain_pos = vec_POS.fit_transform(Xtrain_pos)
    print('Shape of Xtrain_pos:', Xtrain_pos.shape)

    # Transforming input, 2: char bi + trigrams
    vec_char = CountVectorizer(analyzer='char', ngram_range=(2,3))
    Xtrain_char = vec_char.fit_transform(Xtrain)
    print('Shape of Xtrain_char:', Xtrain_char.shape)

    # Hstack them
    Xtrain_feats = hstack((Xtrain_pos, Xtrain_char)) # They ought to have the same dim along axis 1, viz. num of samples
    print('Shape of full features space for Xtrain:', Xtrain_feats.shape)

    # word_gram = CountVectorizer(ngram_range=(1,3))
    #
    # # char_gram = CountVectorizer(analyzer='char', ngram_range=(3,6))
    # # vectorizer = FeatureUnion([('word', word_gram),
    # #                             ('sent_len', SentLen())])
    # vectorizer = word_gram
    #
    # # print('Vectorizing postag-transformed data...')
    # Xtrain = vectorizer.fit_transform(Xtrain_pos)
    # print('Shape of Xtrain:', Xtrain.shape)


    estimator = Pipeline([('clf', LinearSVC(random_state=0))])

    print('Fitting SVM ...')
    estimator.fit(Xtrain_feats, Ytrain)

    print('Vectorizing Xtest...')
    Xtest_pos = [ feats.noun_freq_abstraction(sample, model=model) for sample in Xtest ]
    Xtest_pos = vec_POS.transform(Xtest_pos)
    Xtest_char = vec_char.transform(Xtest)
    Xtest_feats = hstack((Xtest_pos, Xtest_char))
    print('Shape of full feat space for Xtest:', Xtest_feats.shape)

    # Predicting
    print('Predicting...')
    Yguess = estimator.predict(Xtest_feats)

    # Evaluating
    print()
    print('*'*50)
    print('Results for SVM model:')
    evaluate(Ytest, Yguess)
    print('*'*50)

    ########################################################
    '''
    print()
    print('BELOW: POS uni + bi + trigrams')
    print()


    # Setting up SVM baseline with word (and char) ngrams)
    print('Setting up SVM model (Linear SVC)...')
    print('Getting word ngrams...')
    word_gram = CountVectorizer(preprocessor = identity,
                                  tokenizer = identity,
                                  ngram_range=(1,3))
    # print('Getting char ngrams...')
    # char_gram = CountVectorizer(analyzer='char', ngram_range=(3,6))
    # vectorizer = FeatureUnion([('word', word_gram),
    #                            ('char', char_gram)])

    print('Vectorizing postag-transformed data...')
    Xtrain = word_gram.fit_transform(Xtrain_pos)
    print('Shape of Xtrain(pos):', Xtrain.shape)

    #print('Numerifying labels...')
    #le = LabelEncoder()
    # Fit label encoder on Y for now, not Ytrain, to ensure we really 'know' all labels in the val set (This should usually be guaranteed by task)
    #le.fit(Y)
    #Ytrain = le.transform(Ytrain)
    print('Shape of Ytrain:', Ytrain.shape)

    print('Fitting SVM ...')
    clf = LinearSVC(random_state=0)
    clf.fit(Xtrain, Ytrain)

    print('Predicting...')
    Yguess_svm = clf.predict(word_gram.transform(Xtest_pos))

    # Transform Yguess back to nominal labels
    Yguess_svm = le.inverse_transform(Yguess_svm)
    # Evaluate on val set
    print()
    print('*'*50)
    print('Results for SVM postag-transformed (uni + bi + tri):')
    evaluate(Ytest, Yguess_svm)
    print('*'*50)
    '''
