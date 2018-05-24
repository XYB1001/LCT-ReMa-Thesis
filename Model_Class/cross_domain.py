'''
Baseline LinearSVC system for cross_domain

'''
# import argparse
import pickle
import statistics as stats
import numpy as np
import spacy
import feats

from scipy.sparse import hstack
from preprocess import strip_emoticons, strip_cl_chars
from collections import Counter

# from sklearn.dummy import DummyClassifier
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

    # '''
    # TRAIN ON SM, TEST ON EFCAMDAT
    # '''
    #
    # # Preparing + shuffling data
    # training = twitter + reddit
    # Xtrain, Ytrain = shuffle_prepare_data(training)
    # # Xtrain = Xtrain[:1000]
    # # Ytrain = Ytrain[:1000]
    # print('len(Xtrain):', len(Xtrain))
    # print('len(Ytrain):', len(Ytrain))
    #
    # efcamdatX, efcamdatY = shuffle_prepare_data(efcamdat)
    # Xtest = efcamdatX[:5000]
    # Ytest = efcamdatY[:5000]
    #
    # # Setting up SVM baseline with word and char ngrams
    # print('Setting up SVM baseline (Linear SVC)...')
    # print('Getting word ngrams...')
    # word_gram = CountVectorizer()
    # print('Getting char ngrams...')
    # char_gram = CountVectorizer(analyzer='char', ngram_range=(3,6))
    # vectorizer = FeatureUnion([('word', word_gram),
    #                             ('char', char_gram)])
    #
    # print('Vectorizing raw data...')
    # Xtrain = vectorizer.fit_transform(Xtrain)
    # print('Shape of Xtrain:', Xtrain.shape)
    #
    # print('Numerifying labels...')
    # le = LabelEncoder()
    # le.fit(Ytrain)
    # Ytrain = le.transform(Ytrain)
    # print('Shape of Ytrain:', Ytrain.shape)
    #
    # print('Fitting SVM ...')
    # clf = LinearSVC(random_state=0)
    # clf.fit(Xtrain, Ytrain)
    #
    # print('Predicting...')
    # Yguess_svm = clf.predict(vectorizer.transform(Xtest))
    #
    # # Transform Yguess back to nominal labels
    # Yguess_svm = le.inverse_transform(Yguess_svm)
    # # Evaluate on val set
    # print()
    # print('*'*50)
    # print('Results TRAIN ON SM, TEST ON EFCAMDAT:')
    # evaluate(Ytest, Yguess_svm)
    # print('*'*50)



    '''
    TRAIN ON EFCAMDAT, TEST ON SM
    '''
    # print()
    # print('Vice Versa')

    # Preparing + shuffling data
    Xtrain, Ytrain = shuffle_prepare_data(efcamdat)
    # Xtrain = Xtrain[:100]
    # Ytrain = Ytrain[:100]
    print('len(Xtrain):', len(Xtrain))
    print('len(Ytrain):', len(Ytrain))
    smX, smY = shuffle_prepare_data(twitter + reddit)
    Xtest = smX[:5000]
    Ytest = smY[:5000]

    # Transforming training data to Spacy postags
    model = spacy.load('en')
    print('Pos-tagging with Spacy ...')
    Xtrain_pos = [ feats.noun_freq_abstraction(sample, model) for sample in Xtrain]
    # print(Xtrain_pos[:5])

    print('Saving Spacy_postagged EF data')
    pos_n_data = []
    for i in range(len(Xtrain_pos)):
        pos_n_data.append((Xtrain_pos[i], Ytrain[i]))

    fout = open('../Data/efcamdat_spacy_nounfreq.pickle', 'wb')
    pickle.dump(pos_n_data, fout)
    fout.close()

    # Transforming input 1: uni + bi + trigrams on posfreq-abstracted data
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


    # Setting up SVM
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




## sapce ##
