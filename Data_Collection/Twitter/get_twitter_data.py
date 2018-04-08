import pickle

import tweepy
from tweepy import OAuthHandler

consumer_key = '3rCmW4F20eYWxah4wTtVewncS'
consumer_secret = 'QBQFFsAHCTzPCDnhJelDAO810hbLFZ2CTTTWOKLQoWoLoRSDSt'
access_token = '971808281277665281-ZfV6plmz24OyKJCCWzjF5srcwJDDrq2'
access_secret = '5L4mVIbEIdSQfbRS4gXRQFjRO3PkLPYmAKuoA3n6vRSJm'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

def log(message, outfile):
    with open(outfile, 'a', encoding='utf-8') as fo:
        print(message, file=fo)
        print(message)

def get_labeled_user_tweets(API, user, label, error_file):
    ''' Get all accesible tweets by same user '''

    try:
        tweets = tweepy.Cursor(API.user_timeline, id=user).items()
        for tw in tweets:
            yield (tw.text, tw.id_str, label, user, str(tw.created_at))
            #yield (tw.text, user, str(tw.created_at))
    except tweepy.error.TweepError as e:
        log('Error: user ' + user, error_file)
        log(e, error_file)


##########################################################################################

error_log = 'user_errors.log'

# Getting list of users
with open('users_labels.txt', 'r', encoding='utf-8') as fi:
    users = []
    for line in fi:
        # ignore possible empty lines
        if line.strip() != '':
            user = line.strip().split('\t')[0]
            label = line.strip().split('\t')[1]
            users.append((user, label))
print(len(users), 'users')

# Get all tweets for each user, store in pickle.
# Output file showing how many tweets found per user.

# Set error log file to blank doc
f_error = open(error_log, 'w')
f_error.close()

data = []
with open('count_file.txt', 'w', encoding='utf-8') as fo:
    count_user = 1
    for user in users:
        print('Working on user', count_user)
        count = 0
        for tw_data in get_labeled_user_tweets(api, user[0], user[1], error_log):
            data.append(tw_data)
            count += 1
        fo.write(str(user) + '\t' + str(count) + '\n')
        count_user += 1


print('Done')
print('Tweets found:', len(data))

# Pickle currently saved data
save_file = open('data-07.04.pickle', 'wb')
pickle.dump(data, save_file)
save_file.close()













###############
