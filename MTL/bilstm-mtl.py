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

full_data = twitter + reddit + efcamdat # list of lists where lists have different len. Depending on if EF or SM
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
nlabs = len(l2i)

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
bwdLSTM = dy.LSTMBuilder(1, EMB_SIZE, HID_SIZE, model)  # Backward LSTM(
# softmax layer for main task (tag prediction)
W_sm = model.add_parameters((ntags, 2 * HID_SIZE))  # Softmax weights
b_sm = model.add_parameters((ntags))  # Softmax bias
# softmax layer for aux task (slab prediction)
Waux_sm = model.add_parameters((nlabs, 2 * HID_SIZE))  # Softmax weights
baux_sm = model.add_parameters((nlabs))


# (part of) computation graph to be used for each sample
# A function to calculate the prediction scores in both tasks for one sample
def calc_scores(words):
    dy.renew_cg()
    word_embs = [W_emb[x] for x in words]
    # initialise and run forward LSTM
    fwd_init = fwdLSTM.initial_state()
    fwd_outs = fwd_init.transduce(word_embs)
    fwd = fwd_outs[-1] # take last hidden state of (forward) LSTM

    # initialise and run backward LSTM
    bwd_init = bwdLSTM.initial_state()
    bwd_outs = bwd_init.transduce(reversed(word_embs))
    bwd = bwd_outs[-1] # again, take last hidden state of (backward) LSTM

    # feed concatenated forward_backward representation to softmax layer for either task
    fwd_bwd = dy.concatenate([fwd, bwd])
    W_sm_exp = dy.parameter(W_sm)
    b_sm_exp = dy.parameter(b_sm)
    Waux_sm_exp = dy.parameter(Waux_sm)
    baux_sm_exp = dy.parameter(baux_sm)

    # get scores for either task
    score_main = W_sm_exp * fwd_bwd + b_sm_exp
    score_aux = Waux_sm_exp * fwd_bwd + baux_sm_exp
    return score_main, score_aux


print('Starting training...')

# for ITER in range(100):
for ITER in range(1):
    # Perform training
    random.shuffle(train)
    train_loss_main = 0.0
    train_loss_aux = 0.0
    start = time.time()
    for words, tag, slab in train:
        tag_guess, slab_guess = calc_scores(words)
        loss_main = dy.pickneglogsoftmax(tag_guess, tag)
        loss_aux = dy.pickneglogsoftmax(slab_guess, slab)
        my_loss = dy.esum([1.5*loss_main, 0.5*loss_aux])
        train_loss_main += loss_main.value()
        train_loss_aux += loss_aux.value()
        my_loss.backward()
        trainer.update()
    print("iter %r: train loss/sample (main)=%.4f, train loss/sample (aux)=%.4f, time=%.2fs" % (ITER, train_loss_main / len(train), train_loss_aux / len(train), time.time() - start))
    # Perform testing
    test_correct_main = 0.0
    test_correct_aux = 0.0
    for words, tag, slab in test:
        scores_tag, scores_slab = calc_scores(words)
        predict_tag = np.argmax(scores_tag)
        predict_slab = np.argmax(scores_slab)
        if predict_tag == tag:
            test_correct_main += 1
        if predict_slab == slab:
            test_correct_aux += 1
    print("iter %r: test acc (main)=%.4f, test acc (aux)=%.4f" % (ITER, test_correct_main / len(test), test_correct_aux / len(test)))
