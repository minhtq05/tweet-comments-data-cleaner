import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import pandas as pd
import ssl
import re
import sys

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def get_data(head=None):
    f = open('data.txt')

    data = f.read()

    data = data.split('\n')

    data = [x.split(',', maxsplit=2) for x in data]

    return data[:head]


def export_data_to_excel(data, split_data='true'):
    data = pd.DataFrame(
        data, columns=['sentence_id', 'sentence_label', 'sentence_text'])

    data_0, data_1 = data[data['sentence_label']
                          == 0], data[data['sentence_label'] == 1]

    data_0.to_csv('cleaned_0.csv', index=False)
    data_1.to_csv('cleaned_1.csv', index=False)


def remove_internet_contents(text):

    text = re.sub(r"http\S+", "", text)  # remove urls
    text = re.sub("www.[A-Za-z0-9-?[-`{-~]", "", text)  # remove urls
    text = re.sub("@[A-Za-z0-9!-?[-`{-~]+", "", text)  # remove usernames
    text = text.lower()
    return text


def lemmatize(word, wn):
    for i in ['n', 'v', 'a', 'r']:
        lm = wn.lemmatize(word, i)
        if (lm != word):
            return lm
    return wn.lemmatize(word)


def clean_data(data):
    ten_percent = len(data) / 10

    words = dict.fromkeys(nltk.corpus.words.words(), 1)

    wn = WordNetLemmatizer()

    stopwords = nltk.corpus.stopwords.words('english')
    
    tweet = TweetTokenizer()

    for i in range(len(data)):
        id, label, text = data[i]
        id, label = int(id), int(label)

        tokens = tweet.tokenize(remove_internet_contents(text))
        tokens = [lemmatize(word, wn) for word in tokens]
        tokens = [word for word in tokens if word in words]
        data[i] = id, label, tokens

        if (id % ten_percent == 0):
            print("Done cleaning - {}% ({}/{}) of items".format(id /
                  len(data) * 100, id, len(data)))

    export_data_to_excel(data)


if __name__ == '__main__':
    args = sys.argv[1:]
    try:
        head_id = args.index('-h') + 1
    except ValueError:
        head_id = -1
    if (head_id == -1 or len(args) <= head_id):
        clean_data(get_data())
    else:
        head = args[head_id]
        clean_data(get_data(int(head)))
