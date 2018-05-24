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

from sklearn.dummy import DummyClassifier
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


def shuffle_prepare_data(data, clean_char=True, num_labs=6):
    '''
    Preparing data for classification. Data will be a list of tuples with the format
    (text, id, label, user, time)
    Shuffle, then extract paired lists of samples and labels
    if clean_char is True (default), remove control characters and replace all emojis and emoticons with '<emo> '
    num_labs is number of labels / classes to use. Default is 6, i.e. unmapped from data. Alternatives: 3 and 4
    '''
    lab3_mapping = {'a1':'A', 'a2':'A','b1':'B','b2':'B', 'c1':'C','c2':'C'}
    lab4_mapping = {'a1':'a', 'a2':'a','b1':'b','b2':'b', 'c1':'c1','c2':'c2'}

    # Shuffling data
    np.random.shuffle(data)

    # # Discriminating solely between C1 and C2
    # samples, labels = [],[]
    # for tup in data:
    #     if tup[2] in {'c1','c2'}:
    #         samples.append(strip_emoticons(strip_cl_chars(tup[0])))
    #         labels.append(tup[2])

    if clean_char:
        samples = [ strip_emoticons(strip_cl_chars(tup[0])) for tup in data ]
    else:
        samples = [ tup[0] for tup in data]

    if num_labs == 6:
        labels = [ tup[2] for tup in data]
    elif num_labs == 4:
        labels = [ lab4_mapping[tup[2]] for tup in data]
    elif num_labs == 3:
        labels = [ lab3_mapping[tup[2]] for tup in data]


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

    ftw = open(tw_path,'rb')
    twitter = pickle.load(ftw)
    ftw.close()
    fred = open(red_path,'rb')
    reddit = pickle.load(fred)
    fred.close()

    # Preparing + shuffling data
    full_data = twitter + reddit
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

    # '''
    # MFC baseline
    # '''
    # # Setting up most frequent class (mfc) baseline
    # print('Working on MFC baseline...')
    # classifer_mfc = Pipeline([('vec', CountVectorizer()),
    #                             ('classify', DummyClassifier(strategy='most_frequent', random_state=0))])
    #
    # # Fitting model
    # classifer_mfc.fit(Xtrain,Ytrain)
    # # Predicting
    # Yguess_mfc = classifer_mfc.predict(Xtest)
    # # Evaluating on val set
    # print()
    # print('*'*50)
    # print('Results for most frequent class baseline:')
    # evaluate(Ytest, Yguess_mfc)
    # print()

    '''
    SVM model
    '''
    # Hyper-parameters for SVM:
    # C_val = 1
    # Kernel = 'linear'

    # Setting up SVM baseline with word and char ngrams and PCA dim reduction
    print('Setting up SVM (Linear SVC)...')
    word_gram = CountVectorizer()
    char_gram = CountVectorizer(analyzer='char', ngram_range=(3,6))
    #tSD = TruncatedSVD(n_components=5000)

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


    #### print('Numerifying labels...')
    #### le = LabelEncoder()
    #### # Fit label encoder on Y for now, not Ytrain, to ensure we really 'know' all labels in the val set (This should usually be guaranteed by task)
    #### le.fit(Y)
    #### Ytrain = le.transform(Ytrain)
    #### print('Shape of Ytrain:', Ytrain.shape)


    # # Over/Under_sampling to smallest class
    # print('Over/Undersampling on training data:')
    # print('Current Y distribution:', sorted(Counter(Ytrain).items()))
    # smote_enn = SMOTEENN(random_state=0)
    # Xtrain, Ytrain = smote_enn.fit_sample(Xtrain, Ytrain)
    # print('Oversampled Y distribution:', sorted(Counter(Ytrain).items()))
    # print('New shape of Xtrain:', Xtrain.shape)

    # tSD = TruncatedSVD(n_components=100)
    estimator = Pipeline([('clf', LinearSVC(random_state=0))])

    print('Fitting SVM ...')
    '''
    clf = LinearSVC(random_state=0)
    clf.fit(Xtrain, Ytrain)
    '''
    estimator.fit(Xtrain, Ytrain)

    print('Predicting...')
    Yguess_svm = estimator.predict(vectorizer.transform(Xtest))

    #### Transform Yguess back to nominal labels
    #### Yguess_svm = le.inverse_transform(Yguess_svm)
    # Evaluate on val set
    print()
    print('*'*50)
    print('Results for SVM model:')
    evaluate(Ytest, Yguess_svm)
    print('*'*50)





## space ##
