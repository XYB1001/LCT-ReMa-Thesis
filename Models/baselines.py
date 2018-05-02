'''
Baseline systems for proficiency prediction on SM data
1) Most freq. class (MFC)
2) Unweighted word unigram + char ngram (3,6) linear SVM
'''
# import argparse
import pickle
import statistics as stats
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


def shuffle_prepare_data(data):
    '''
    Preparing data for classification. Data will be a list of tuples with the format
    (text, id, label, user, time)
    Shuffle, then extract paired lists of samples and labels
    '''
    np.random.shuffle(data)

    samples = [ tup[0] for tup in data]
    labels = [ tup[2] for tup in data]

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

    # Parse arguments
    # parser = argparse.ArgumentParser(description='Run models for either binary or multi-class task')
    # parser.add_argument('file', metavar='f', type=str, help='Path to data file')
    # parser.add_argument('--task', metavar='t', type=str, default='binary', help="'binary' for binary and 'multi' for multi-class task")
    # args = parser.parse_args()

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
#    X = X[:2000]
#    Y = Y[:2000]
    print('len(X):', len(X))
    print('len(Y):', len(Y))

    # Using 25% of the data as validation data
    split_point = int(0.75*len(X))
    Xtrain = X[:split_point]
    Ytrain = Y[:split_point]
    Xtest = X[split_point:]
    Ytest = Y[split_point:]

    '''
    MFC baseline
    '''
    # Setting up most frequent class (mfc) baseline
    print('Working on MFC baseline...')
    classifer_mfc = Pipeline([('vec', CountVectorizer()),
                                ('classify', DummyClassifier(strategy='most_frequent', random_state=0))])

    # Fitting model
    classifer_mfc.fit(Xtrain,Ytrain)
    # Predicting
    Yguess_mfc = classifer_mfc.predict(Xtest)
    # Evaluating on val set
    print()
    print('*'*50)
    print('Results for most frequent class baseline:')
    evaluate(Ytest, Yguess_mfc)
    print()

    '''
    SVM baseline
    '''
    # Hyper-parameters for SVM:
    C_val = 1
    Kernel = 'linear'

    # Setting up SVM baseline with word and char ngrams
    print('Setting up SVM baseline...')
    # print('Getting word ngrams...')
    word_gram = CountVectorizer()
    # print('Getting char ngrams...')
    char_gram = CountVectorizer(analyzer='char', ngram_range=(3,6))
    vectorizer = FeatureUnion([('word', word_gram),
                                ('char', char_gram)])

    print('Vectorizing raw data...')
    Xtrain = vectorizer.fit_transform(Xtrain)
    print('Shape of Xtrain:', Xtrain.shape)

    print('Numerifying labels...')
    le = LabelEncoder()
    # Fit label encoder on Y for now, not Ytrain, to ensure we really 'know' all labels in the val set (This should usually be guaranteed by task)
    le.fit(Y)
    Ytrain = le.transform(Ytrain)
    print('Shape of Ytrain:', Ytrain.shape)

    print('Fitting SVM ...')
    clf = SVC(kernel=Kernel, C=C_val)
    clf.fit(Xtrain, Ytrain)

    print('Predicting...')
    Yguess_svm = clf.predict(vectorizer.transform(Xtest))

    # Transform Yguess back to nominal labels
    Yguess_svm = le.inverse_transform(Yguess_svm)
    # Evaluate on val set
    print()
    print('*'*50)
    print('Results for SVM baseline:')
    evaluate(Ytest, Yguess_svm)
    print('*'*50)












    '''
    # classifier_svm = Pipeline([('vec', vectorizer),
    #                             ('classify', SVC(kernel=Kernel, C=C_val))])
    X_mat = vectorizer.fit_transform(X)
    print('shape of X_mat:', X_mat.shape)



    print('Fitting svm baseline...')
    classifier_svm.fit(Xtrain,Ytrain)


    Yguess_svm = classifier_svm.predict(Xtest)


    print()

    print('Results for svm baseline:')
    evaluate(Ytest, Yguess_svm)
    '''
