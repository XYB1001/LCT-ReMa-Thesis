'''
Systems using the combined set of sm + efcamdat data
'''
# import argparse
import pickle
import statistics as stats
import numpy as np

from preprocess import strip_emoticons, strip_cl_chars, nouns_to_POSTAG
from collections import Counter
# from imblearn.under_sampling import RandomUnderSampler

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
        samples = [ strip_emoticons(strip_cl_chars(tup[0])) for tup in data ]
    else:
        samples = [ tup[0] for tup in data]

    # samples = [ tup[0] for tup in data]
    labels = []
    for tup in data:
        if len(tup) > 2:
            labels.append(tup[2]) # Twitter + Reddit data
        else:
            labels.append(tup[1]) # Efcamdat data
    # labels = [ tup[2] for tup in data]
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
    '''Dummy function'''
    return x


if __name__ == '__main__':

    print('Fetching and preparing data...')
    # Fetching pickled data
    tw_path = '../Data/twitter-data.pickle'
    red_path = '../Data/reddit-data.pickle'
    efcamdat_path = '../Data/efcamdat-len_norm.pickle'

    ftw = open(tw_path,'rb')
    twitter = pickle.load(ftw)
    ftw.close()
    fred = open(red_path,'rb')
    reddit = pickle.load(fred)
    fred.close()
    fef = open(efcamdat_path,'rb')
    efcamdat = pickle.load(fef)
    fef.close()

    # Preparing + shuffling data
    full_data = twitter + reddit + efcamdat
    X, Y = shuffle_prepare_data(full_data)
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
    SVM baseline
    '''

    # Changing samples to POS
    print('Performing abstraction of data to POS-tags...')
    Xtrain_pos = [ nouns_to_POSTAG(sample) for sample in Xtrain ] # list of POS or POS/word -sequences
    Xtest_pos = [ nouns_to_POSTAG(sample) for sample in Xtest ]
    # print(Xtrain_pos[:5])
    # print(Ytrain[:5])
    # print(Xtest_pos[:5])
    # print(Ytest[:5])

    # Setting up SVM baseline with word and char ngrams
    print('Setting up SVM baseline (Linear SVC)...')
    word_gram = CountVectorizer(preprocessor = identity,
                                  tokenizer = identity,
                                  ngram_range=(1,3))
    # char_gram = CountVectorizer(analyzer='char', ngram_range=(3,6))
    # vectorizer = FeatureUnion([('word', word_gram),
    #                             ('char', char_gram)])
    vectorizer = word_gram

    print('Vectorizing raw data...')
    Xtrain = vectorizer.fit_transform(Xtrain_pos)
    print('Shape of Xtrain:', Xtrain.shape)

    print('Numerifying labels...')
    le = LabelEncoder()
    # Fit label encoder on Y for now, not Ytrain, to ensure we really 'know' all labels in the val set (This should usually be guaranteed by task)
    le.fit(Y)
    Ytrain = le.transform(Ytrain)
    print('Shape of Ytrain:', Ytrain.shape)

    # # Over/Under_sampling to smallest class
    # print('Over/Undersampling on training data:')
    # print('Current Y distribution:', sorted(Counter(Ytrain).items()))
    # smote_enn = SMOTEENN(random_state=0)
    # Xtrain, Ytrain = smote_enn.fit_sample(Xtrain, Ytrain)
    # print('Oversampled Y distribution:', sorted(Counter(Ytrain).items()))
    # print('New shape of Xtrain:', Xtrain.shape)

    print('Fitting SVM ...')
    clf = LinearSVC(random_state=0)
    clf.fit(Xtrain, Ytrain)

    print('Predicting...')
    Yguess_svm = clf.predict(vectorizer.transform(Xtest_pos))

    # Transform Yguess back to nominal labels
    Yguess_svm = le.inverse_transform(Yguess_svm)
    # Evaluate on val set
    print()
    print('*'*50)
    print('Results for SVM baseline:')
    evaluate(Ytest, Yguess_svm)
    print('*'*50)
