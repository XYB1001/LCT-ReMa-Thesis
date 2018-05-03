'''
Getting some basic statistics concerning SM dataset
'''

import pickle
import statistics as stats
from collections import Counter
from nltk.tokenize import word_tokenize

f1 = open('twitter-data.pickle', 'rb')
twitter = pickle.load(f1)
f1.close()

f2 = open('reddit-data.pickle', 'rb')
reddit = pickle.load(f2)
f2.close()

data = twitter + reddit
print('len(data):', len(data))

# Checking distribution of labels. Expected: highly skewed towards higher classes
labels = [ tup[2] for tup in data]
print('len(labels):', len(labels))

lab_counts = dict(Counter(labels))
for l, c in lab_counts.items():
    print(l + '\t' + str(c))

# Checking average length (with sd) of sample. Doing this separately for Twitter and Reddit
# in either case tup[0] is the text
len_twitter = [ len(word_tokenize(tup[0])) for tup in twitter]
tw_mean = stats.mean(len_twitter)
tw_sd = stats.stdev(len_twitter)
print('Twitter data')
print('Mean length:', tw_mean)
print('St. deviation:', tw_sd)

len_reddit = [ len(word_tokenize(tup[0])) for tup in reddit]
red_mean = stats.mean(len_reddit)
red_sd = stats.stdev(len_reddit)
print('Reddit data')
print('Mean length:', red_mean)
print('St. deviation:', red_sd)

print('Overall lengths:')
print('Mean length:', stats.mean(len_twitter + len_reddit))
print('St. deviation:', stats.stdev(len_twitter + len_reddit))















###############################################
