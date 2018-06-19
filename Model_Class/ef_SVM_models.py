'''
Different LinearSVC systems to try on SM data
'''
# import argparse
import pickle
import statistics as stats
import numpy as np

from preprocess import strip_emoticons, strip_cl_chars
from feats import SentLen
from collections import Counter
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.combine import SMOTEENN

# from sklearn.dummy import DummyClassifier
# from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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

def split_long_samps(XOld, YOld, len_thresh):
    '''
    Takes in original dataset, splits those samples in half which are above a given threshold
    '''
    old_data = list(zip(XOld, YOld))
    print('{} samples originally'.format(len(old_data)))

    long_samples = { tup for tup in old_data if len(tup[0]) > len_thresh}
    print('{} samples to be split'.format(len(long_samples)))
    intermediate_data = list(set(old_data) - long_samples)
    print('{} samples after removing long samples'.format(len(intermediate_data)))

    split_samps = []
    for tup in long_samples:
        content = list(tup)
        # make 2 copies of sample to split. The text of the sample is to be replaced with split text
        samp1 = copy.deepcopy(content)
        samp2 = copy.deepcopy(content)
        # content[0] is the text
        sents = sent_tokenize(content[0])
        split_point = int(len(sents)*0.5)

        # In case sent_tokenize does not perform well and len(sents) is 1, don't split and attach an empty string as sample
        if len(sents[:split_point]) > 0:
            text1 = ' '.join(sents[:split_point])
            # substitute text part of samp1 with text1
            samp1[0] = text1
            split_samps.append(tuple(samp1)) # change back to tuple for sake of conformity
        if len(sents[split_point:]) > 0:
            text2 = ' '.join(sents[split_point:])
            # substitute text part of samp2 with text2
            samp2[0] = text2
            split_samps.append(tuple(samp1))

    # Put split samples into dataset
    new_data = intermediate_data + split_samps
    print('{} samples after splitting'.format(len(new_data)))

    # Unzip data
    XNew = list(list(zip(*new_data))[0])
    YNew = list(list(zip(*new_data))[1])

    return XNew, YNew


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
    # Fetching pickled ef data
    f = open('../Data/efcamdat-data.pickle', 'rb')
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

    # Splitting
    # Xtrain, Ytrain = split_long_samps(Xtrain, Ytrain, 800)


    '''
    Actual modelling
    '''
    # Hyper-parameters for SVM:
    # C_val = 1
    # Kernel = 'linear'

    # Setting up SVM baseline with word and char ngrams and PCA dim reduction
    print('Setting up SVM (Linear SVC)...')
    word_gram = CountVectorizer()
    char_gram = CountVectorizer(analyzer='char', ngram_range=(3,6))

    vectorizer = FeatureUnion([('word', word_gram),
                                ('char', char_gram)])
    #                            ('sent_len', SentLen())])

    # vectorizer = Pipeline([('feat', FeatureUnion([('word', word_gram),
    #                                             ('char', char_gram)])),
    #                         ('dim_red', tSD)])
    # vectorizer = SentLen()

    print('Vectorizing raw data...')
    Xtrain = vectorizer.fit_transform(Xtrain)
    print('Shape of Xtrain:', Xtrain.shape)

    estimator = Pipeline([('clf', LinearSVC(random_state=0))])

    print('Fitting SVM ...')

    estimator.fit(Xtrain, Ytrain)

    print('Predicting...')
    Yguess_svm = estimator.predict(vectorizer.transform(Xtest))

    # Evaluate on val set
    print()
    print('*'*50)
    print('Results for SVM model:')
    evaluate(Ytest, Yguess_svm)
    print('*'*50)
    print()

    '''
    Printing out most predictive features
    '''

    # Get all classes:
    classes = sorted(set(Y))
    print('{} classes'.format(len(classes)))

    # Get all features:
    features = vectorizer.get_feature_names()
    print('{} features'.format(len(features)))

    # Get coefficients / weights
    # coefs is a matrix of shape [n_classes, n_features]. Thus, axis 0 should correspond to classes, axis 1 to features
    coefs = estimator.named_steps['clf'].coef_

    # Get top n predictive features for each class
    n = 10
    print('-'*30)
    print('Top {} predictive features by class:'.format(n))
    for cidx, c in enumerate(classes):
        weights = coefs[cidx] # weights of each feature for class c. array of shape(1, n_features)
        # sort and get indices of features with highest weights
        # np.argsort returns indices in Ascending order, 'reverse=True' not available here.
        # Using [::-1] to read in the indices backwards to get indices in DEscending order
        top_feats_id = np.argsort(weights)[::-1][:n]
        top_feats = [ features[ID] for ID in top_feats_id ] # list of feature names
        top_feats = " ".join(top_feats)
        print('{}:\t{}'.format(c, top_feats))



    # print(all_feats[-100:])
    #
    # print()
    # print('WEIGHTS:')
    # coefs = estimator.named_steps['clf'].coef_
    # print(len(coefs))
    # print(len(coefs[0]))






## space ##
