import pickle
import re
import unicodedata


def strip_emoticons(text):
    '''
    adapted from http://stackoverflow.com/a/13752628/6762004)
    strips (almost) all emoticons
    '''
    re_emoticon = re.compile(r'[\U00010000-\U0010ffff]|:-?[\)\(9D/P3]|[oO]\.[oO]')
    return re_emoticon.sub(r' <emo>', text)

def strip_cl_chars(text):
    '''
    Only keep those characters whose unicode category does not start with 'C'
    This will strip control characters like \t, \n etc.
    '''
    newtext = ''.join([char for char in text if unicodedata.category(char)[0]!='C'])
    return newtext


if __name__ == '__main__':
    f1 = open('reddit-data.pickle', 'rb')
    data = pickle.load(f1)
    f1.close()

    print(len(data))
    #print(strip_emoticons('This video is beyond epic 😎Go watch it if you love bookish discussions and chatty videos with a cozy feel to them (how topical) ☕🍰'))
    # print(data[0])
    for tw in data:
        text = strip_cl_chars(tw[0])
        text = strip_emoticons(text)
        #print(text)
        #print(tw[3])
        #print('-'*100)

# Both datasets came through without errors
