'''
File containing some classes + functions to extract features
'''
from sklearn.base import TransformerMixin
from scipy.sparse import csr_matrix

import statistics as stats
import numpy as np
import wordfreq as wfreq

from nltk.tag import pos_tag_sents
from nltk.tokenize import word_tokenize, sent_tokenize
# import spacy



class SentLen(TransformerMixin):
    """ Transformer that transforms sample into a single value that is the average number of char per sentence of char in it """

    def __init__(self):
        super().__init__()

    def transform(self, X, **transform_params):
        ''' Transforming X: X is list of samples, type: list(str) '''
        spmat_values = csr_matrix([self.get_sent_len(sample) for sample in X])
        # need to be transposed
        return csr_matrix.transpose(spmat_values)

    def get_sent_len(self, sample):
        ''' Get average sentence length within each sample '''
        sent_lens = [ len(sent) for sent in sent_tokenize(sample)]
        return stats.mean(sent_lens)

    def fit(self, X, y=None, **fit_params):
        return self

def to_POSTAG(sample):
    '''
    Turns each sample into a long string of postags, abstracting away from the actual words
    :param sample: full text sample
    :type sample: str
    Output: str
    '''
    # Convert samplt into input form needed by pos_tag_sents: list(list(str))
    # Needed: sentence-tokenize, then word-tokenize
    sents_list = [ word_tokenize(sent) for sent in sent_tokenize(sample) ]
    # pos-tagging
    POS = pos_tag_sents(sents_list)
    # extracting the postags and put into one str to replace original sample
    POS_sents = []
    for sent_list in POS:
        pos_seq = [ tag for word,tag in sent_list]
        pos_seq = ' '.join(pos_seq) # This will be a string like 'NNP VB DET NNP .'
        POS_sents.append(pos_seq)
    POS_sample = ' '.join(POS_sents)

    return POS_sample

def nouns_to_POSTAG(sample):
    '''
    Same principle as to_POSTAG but only applies transformation to nouns
    :param sample: full text sample
    :type sample: str
    Output: str
    '''
    # Convert sample into input form needed by pos_tag_sents: list(list(str))
    # Needed: sentence-tokenize, then word-tokenize
    sents_list = [ word_tokenize(sent) for sent in sent_tokenize(sample) ]
    # pos-tagging
    POS = pos_tag_sents(sents_list)
    # extracting the postags and put into one str to replace original sample
    # list(list((word, tag),(word, tag),(word, tag)))
    POS_sents = []
    for sent_list in POS:
        word_pos_seq = []
        for word,tag in sent_list:
            if tag.startswith('NN'):
                word_pos_seq.append(tag)
            else:
                word_pos_seq.append(word)

        word_pos_seq = ' '.join(word_pos_seq) # This will be a string like 'NNP sees the NN .'
        POS_sents.append(word_pos_seq)
    POS_sample = ' '.join(POS_sents)

    return POS_sample

def nouns_verbs_to_POSTAG(sample):
    '''
    Same principle as to_POSTAG but only applies transformation to nouns and verbs
    :param sample: full text sample
    :type sample: str
    Output: str
    '''
    # Convert sample into input form needed by pos_tag_sents: list(list(str))
    # Needed: sentence-tokenize, then word-tokenize
    sents_list = [ word_tokenize(sent) for sent in sent_tokenize(sample) ]
    # pos-tagging
    POS = pos_tag_sents(sents_list)
    # extracting the postags and put into one str to replace original sample
    # list(list((word, tag),(word, tag),(word, tag)))
    POS_sents = []
    for sent_list in POS:
        word_pos_seq = []
        for word,tag in sent_list:
            if tag.startswith('NN') or tag.startswith('VB'):
                word_pos_seq.append(tag)
            else:
                word_pos_seq.append(word)

        word_pos_seq = ' '.join(word_pos_seq) # This will be a string like 'NNP sees the NN .'
        POS_sents.append(word_pos_seq)
    POS_sample = ' '.join(POS_sents)

    return POS_sample

def spacy_NV_postag(sample, model):
    '''
    Same as nouns_verbs_to_POSTAG, but using Spacy module for pos-tagging
    :param sample: full text sample
    :type sample: str
    :param model: spacy model of choice
    :type model: spacy.lang.[MODEL] object
    Output: str
    '''

    doc = model(sample)
    tokens = []
    for token in doc:
        if token.tag_.startswith('V') or token.tag_.startswith('N'):
            tokens.append(token.tag_)
        else:
            tokens.append(token.text)

    return ' '.join(tokens)

def spacy_N_postag(sample, model):
    '''
    Same as nouns_verbs_to_POSTAG, but using Spacy module for pos-tagging
    :param sample: full text sample
    :type sample: str
    :param model: spacy model of choice
    :type model: spacy.lang.[MODEL] object
    Output: str
    '''
    doc = model(sample)
    tokens = []
    for token in doc:
        if token.tag_.startswith('N'):
            tokens.append(token.tag_)
        else:
            tokens.append(token.text)

    return ' '.join(tokens)

def get_word_freq(word):
    '''
    Get a freqeuncy indication for word, using library:
    @misc{robert_speer_2017_998161,
      author       = {Robert Speer and
                      Joshua Chin and
                      Andrew Lin and
                      Sara Jewett and
                      Lance Nathan},
      title        = {LuminosoInsight/wordfreq: v1.7},
      month        = sep,
      year         = 2017,
      doi          = {10.5281/zenodo.998161},
      url          = {https://doi.org/10.5281/zenodo.998161}
    }
    We use the 'zipf_frequency' function and round the value to one after the decimal
    param word: str
    output: float
    '''
    return round(wfreq.zipf_frequency(word, 'en'),1)

def noun_freq_abstraction(sample, model=None):
    '''
    Combination of above noun abstraction methods and get_word_freq
    '''
    # if using default tagger (i.e. nltk tagger)
    if model == None:
        sents_list = [ word_tokenize(sent) for sent in sent_tokenize(sample) ]
        # pos-tagging
        POS = pos_tag_sents(sents_list)
        # extracting the postags and put into one str to replace original sample
        # list(list((word, tag),(word, tag),(word, tag)))
        POS_sents = []
        for sent_list in POS:
            word_pos_seq = []
            for word,tag in sent_list:
                if tag.startswith('NN'):
                    freq = str(get_word_freq(word))
                    info = tag + '_' + freq
                    word_pos_seq.append(info)
                else:
                    word_pos_seq.append(word)
            word_pos_seq = ' '.join(word_pos_seq) # This will be a string like 'NNP_5.3 sees the NN_4.8 .'
            POS_sents.append(word_pos_seq)
        POS_sample = ' '.join(POS_sents)
    # if using, e.g. spacy
    else:
        doc = model(sample)
        tokens = []
        for token in doc:
            if token.tag_.startswith('N'):
                freq = str(get_word_freq(token.text))
                info = token.tag_ + '_' + freq
                tokens.append(info)
            else:
                tokens.append(token.text)
        POS_sample = ' '.join(tokens)

    return POS_sample

if __name__ == '__main__':

    import spacy

    model = spacy.load('en')
    sample = '''Hello little cat! The orange cat is  a bit smaller, I believe! Haha!
    Western diplomatic delegations have been instructed to continue negotiations with the Plantagnets. Why do they
    do that? What a wonder!'''

    print(noun_freq_abstraction(sample))
    print()
    print(noun_freq_abstraction(sample, model))
















### space ########
