'''
This is an example CNN implementation using Dynet, written by Graham Neubig.
Original source: https://github.com/neubig/nn4nlp-code/blob/master/05-cnn/cnn-class.py
'''

from collections import defaultdict
from nltk.tokenize import word_tokenize

import time
import random
import dynet as dy
import numpy as np
import pickle

from preprocess import strip_emoticons, strip_cl_chars


np.random.seed(2018)

# Functions to read in the corpus
# GLOBAL variables (called from functions)

w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
l2i = defaultdict(lambda: len(l2i))
UNK = w2i["<unk>"]

def prepare_data(data, clean_char=True):
    '''
    Preparing data for classification. SM Data will be a list of tuples with the format
    (text, id, label, user, time), EF data is simply (text, label)
    Shuffle, then extract paired lists of samples and labels
    if clean_char is True, remove control characters and replace all emojis and emoticons with ' E'
    '''

    # Each tup is one sample paired with label/tag, no matter length
    for item in data:
        # Get text, as list of word tokens
        if clean_char:
            sample = strip_emoticons(strip_cl_chars(item[0]))
        else:
            sample = item[0]
        words = word_tokenize(sample)

        # Get tag (level)
        if len(item) > 2:
            tag = item[2] # Twitter + Reddit data
        else:
            tag = item[1] # EFcamdat data
        # Get label (source)
        slab = item[-1]

        # Always map words, tag and source label to corresponding indices
        # Format of input we need is:
        # sample: [word_idx, word_idx, word_idx...]
        # tag: tag_idx
        # (source label): lab_idx

        yield ([ w2i[word] for word in words ], t2i[tag], l2i[slab])

def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])

'''
Data preparation part
'''


print('Fetching and preparing data from files...')
# Fetching pickled data
tw_path = '../Data/twitter-data.pickle'
red_path = '../Data/reddit-data-30more.pickle'
efcamdat_path = '../Data/efcamdat-data.pickle'

ftw = open(tw_path,'rb')
twitter = pickle.load(ftw)
ftw.close()
fred = open(red_path,'rb')
reddit = pickle.load(fred)
fred.close()
fef = open(efcamdat_path,'rb')
efcamdat = pickle.load(fef)
fef.close()

# Add source label for each dataset
nt, nr, nef = [],[],[]
for tup in twitter:
    new = list(tup)
    new.append('twitter')
    nt.append(new)
for tup in reddit:
    new = list(tup)
    new.append('reddit')
    nr.append(new)
for tup in efcamdat:
    new = list(tup)
    new.append('efcamdat')
    nef.append(new)
twitter, reddit, efcamdat = nt, nr, nef

full_data = twitter + reddit + efcamdat # list of tuples where tuples have different len. Depending on if EF or SM
np.random.shuffle(full_data)

# Using 25% of the data as test data
split_point = int(0.75*len(full_data))
non_test = full_data[:split_point]
test = full_data[split_point:]
# Using 80% of non_test data as train, the other 20% as dev set, for param tuning
np.random.shuffle(non_test)
split_point = int(0.8*len(non_test))
train = non_test[:split_point]
dev = non_test[split_point:]

# Prepare the train, test and dev data
train = list(prepare_data(train))
w2i = defaultdict(lambda: UNK, w2i)
test = list(prepare_data(test))
dev = list(prepare_data(dev))
print('Data sizes\nTrain:{}, Test:{}, Dev:{}'.format(len(train), len(test), len(dev)))

nwords = len(w2i)
ntags = len(t2i)
nlabs = len(l2i)


# Start DyNet and define trainer
model = dy.Model()
trainer = dy.AdamTrainer(model)

# Define the model
EMB_SIZE = 64
W_emb = model.add_lookup_parameters((nwords, 1, 1, EMB_SIZE)) # Word embeddings
WIN_SIZE = 3
FILTER_SIZE = 64
W_cnn = model.add_parameters((1, WIN_SIZE, EMB_SIZE, FILTER_SIZE)) # cnn weights
b_cnn = model.add_parameters((FILTER_SIZE)) # cnn bias

W_sm = model.add_parameters((ntags, FILTER_SIZE))          # Softmax weights for main task
b_sm = model.add_parameters((ntags))                      # Softmax bias for main task

Waux_sm = model.add_parameters((nlabs, FILTER_SIZE))          # Softmax weights for aux task
baux_sm = model.add_parameters((nlabs))                      # Softmax bias for aux task


def calc_scores(words):
    dy.renew_cg()
    W_cnn_express = dy.parameter(W_cnn)
    b_cnn_express = dy.parameter(b_cnn)
    W_sm_express = dy.parameter(W_sm)
    b_sm_express = dy.parameter(b_sm)
    Waux_sm_express = dy.parameter(Waux_sm)
    baux_sm_express = dy.parameter(baux_sm)
    # basically, win size tells you how many words/chars/pixels (?) we're 'looking at' at each step.
    # Here, 1 unit is 1 word. If a sample has fewer words than win size, then we probably do need some padding.
    # Padd with index 0. (so we're treating the pad words as UNK (?))
    if len(words) < WIN_SIZE:
        words += [0] * (WIN_SIZE-len(words))

    # Convolution + pooling layer
    cnn_in = dy.concatenate([W_emb[x] for x in words], d=1) # concat repr of all words
    cnn_out = dy.conv2d_bias(cnn_in, W_cnn_express, b_cnn_express, stride=(1, 1), is_valid=False)
    pool_out = dy.max_dim(cnn_out, d=1) # Is this max pooling?
    pool_out = dy.reshape(pool_out, (FILTER_SIZE,))
    pool_out = dy.rectify(pool_out) # Is this ReLU activation?

    # get scores for either task
    scores_main = W_sm_express * pool_out + b_sm_express
    scores_aux = Waux_sm_express * pool_out + baux_sm_express
    return scores_main, scores_aux

print('Starting training...')

# for ITER in range(100):
for ITER in range(2):
    # Perform training
    random.shuffle(train)
    train_loss_main = 0.0
    train_loss_aux = 0.0
    train_correct_main = 0.0
    train_correct_aux = 0.0
    start = time.time()
    for words, tag, slab in train:
        scores_main, scores_aux = calc_scores(words)
        predict_main = np.argmax(scores_main.npvalue())
        predict_aux = np.argmax(scores_aux.npvalue())
        if predict_main == tag:
            train_correct_main += 1
        if predict_aux == slab:
            train_correct_aux += 1

        loss_main = dy.pickneglogsoftmax(scores_main, tag)
        loss_aux = dy.pickneglogsoftmax(scores_aux, slab)
        my_loss = dy.esum([1.7 * loss_main, 0.3 * loss_aux])
        train_loss_main += loss_main.value()
        train_loss_aux += loss_aux.value()
        my_loss.backward() # backprop
        trainer.update() # update params
    print("iter %r: MAIN -- train loss/sent=%.4f, acc=%.4f; AUX -- train loss/sent=%.4f, acc=%.4f; time=%.2fs" % (ITER, train_loss_main/len(train), train_correct_main/len(train), train_loss_aux/len(train), train_correct_aux/len(train), time.time()-start))

    # Perform testing on dev
    dev_correct_main = 0.0
    dev_correct_aux = 0.0
    for words, tag, slab in dev:
        scores_main, scores_aux = calc_scores(words)
        predict_main = np.argmax(scores_main.npvalue())
        predict_aux = np.argmax(scores_aux.npvalue())
        if predict_main == tag:
            dev_correct_main += 1
        if predict_aux == slab:
            dev_correct_aux += 1
    print("iter %r: dev MAIN acc=%.4f; dev AUX acc=%.4f" % (ITER, dev_correct_main/len(dev), dev_correct_aux/len(dev)))
