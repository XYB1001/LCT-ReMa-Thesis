"""
A simple BiLSTM architecture
"""

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
UNK = w2i["<unk>"]

def prepare_data(data, clean_char=True):
    '''
    Preparing data for classification. SM Data will be a list of tuples with the format
    (text, id, label, user, time), EF data is simply (text, label)
    Shuffle, then extract paired lists of samples and labels
    if clean_char is True, remove control characters and replace all emojis and emoticons with ' E'
    '''

    # Each tup is one sample paired with label/tag, no matter length
    for tup in data:
        # Get text, as list of word tokens
        if clean_char:
            sample = strip_emoticons(strip_cl_chars(tup[0]))
        else:
            sample = tup[0]
        words = word_tokenize(sample)

        # Get tag (label)
        if len(tup) > 2:
            tag = tup[2] # Twitter + Reddit data
        else:
            tag = tup[1] # EFcamdat data

        # Always map words and tag to corresponding indices
        # Format of input we need is:
        # sample: [word_idx, word_idx, word_idx...]
        # tag/lable: label_idx

        yield ([ w2i[word] for word in words ], t2i[tag])


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

full_data = twitter + reddit + efcamdat # list of tuples where tuples have different len. Depending on if EF or SM
np.random.shuffle(full_data)

# Using 25% of the data as validation data
split_point = int(0.75*len(full_data))
train = full_data[:split_point]
test = full_data[split_point:]

# Prepare the train and test data
train = list(prepare_data(train))
w2i = defaultdict(lambda: UNK, w2i)
test = list(prepare_data(test))
nwords = len(w2i)
ntags = len(t2i)

'''
Dynet model training part
'''

# Start DyNet and define trainer
model = dy.Model()
trainer = dy.AdamTrainer(model)

# Define the model and parameters
EMB_SIZE = 64
HID_SIZE = 64
W_emb = model.add_lookup_parameters((nwords, EMB_SIZE))
fwdLSTM = dy.LSTMBuilder(1, EMB_SIZE, HID_SIZE, model)  # Forward LSTM
bwdLSTM = dy.LSTMBuilder(1, EMB_SIZE, HID_SIZE, model)  # Backward LSTM
W_sm = model.add_parameters((ntags, 2 * HID_SIZE))  # Softmax weights    # XB: Why 2*HID_SIZE? Because this already has the BI-LSTM in mind?
b_sm = model.add_parameters((ntags))  # Softmax bias


# A function to calculate scores for one value
# XB: And this is the computation graph (?) -> Yes! Must be!
def calc_scores(words):
    dy.renew_cg()
    word_embs = [W_emb[x] for x in words]
    # initialise and run forward LSTM
    fwd_init = fwdLSTM.initial_state()
    fwd_outs = [ state.output() for state in fwd_init.add_inputs(word_embs) ]
    fwd = fwd_outs[-1] # take last hidden state of (forward) LSTM

    # initialise and run backward LSTM
    bwd_init = bwdLSTM.initial_state()
    bwd_outs = [ state.output() for state in bwd_init.add_inputs(reversed(word_embs)) ]
    bwd = bwd_outs[-1] # again, take last hidden state of (backward) LSTM

    fwd_bwd = dy.concatenate([fwd, bwd])
    W_sm_exp = dy.parameter(W_sm)
    b_sm_exp = dy.parameter(b_sm)
    return W_sm_exp * fwd_bwd + b_sm_exp

print('Starting training...')

# for ITER in range(100):
for ITER in range(5):
    # Perform training
    random.shuffle(train)
    train_loss = 0.0
    start = time.time()
    for words, tag in train:
        my_loss = dy.pickneglogsoftmax(calc_scores(words), tag)
        train_loss += my_loss.value()
        my_loss.backward()
        trainer.update()
    print("iter %r: train loss/sample=%.4f, time=%.2fs" % (ITER, train_loss / len(train), time.time() - start))
    # Perform testing
    test_correct = 0.0
    for words, tag in test:
        scores = calc_scores(words).npvalue()
        predict = np.argmax(scores)
        if predict == tag:
            test_correct += 1
    print("iter %r: test acc=%.4f" % (ITER, test_correct / len(test)))
