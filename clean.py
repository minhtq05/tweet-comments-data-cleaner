from nltk.tokenize import TreebankWordTokenizer, WordPunctTokenizer, sent_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
# from spellchecker import SpellChecker
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


def get_data(head=None, specific=None):
    f = open('data.txt')

    data = f.read()

    data = data.split('\n')

    data = [x.split(',', maxsplit=2) for x in data]

    return data[:head] if (specific == None) else [data[specific - 1]]


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


def clean_data(data):
    length = len(data)
    jump = length / 10

    # tk = TweetTokenizer()
    tk = TreebankWordTokenizer()
    english_stops = set(stopwords.words('english'))

    for i in range(len(data)):
        id, label, text = data[i]
        id, label = int(id), int(label)

        text = remove_internet_contents(text)
        sentences = sent_tokenize(text)
        result = []

        for sentence in sentences:
            tokens = tk.tokenize(sentence)
            tokens = [
                word for word in tokens if word not in english_stops]
            tokens = [word for word in tokens if len(wn.synsets(word)) > 0]
            if (len(tokens) > 0):
                result.append(tokens)

        data[i] = id, label, result

        if (id % jump == 0):
            print(
                f"Done cleaning - {id / length * 100}% ({id}/{length}) of items")
    if (len(data) == 1):
        print(f"Done cleaning - line {data[0][0]}")
    export_data_to_excel(data)


if __name__ == '__main__':
    args = sys.argv[1:]
    head, specific = None, None

    # head argument: only clean the first 'head' lines
    try:
        head_id = args.index('-h') + 1
        try:
            head = int(args[head_id])
        except IndexError:
            head = None
            print('Error: Missing head argument. -h for head or -s for specific')
            exit()
    except ValueError:
        head = None

    # specific argument for cleaning a single 'specific' line of text
    try:
        specific_id = args.index('-s') + 1
        try:
            specific = int(args[specific_id])
        except IndexError:
            specific = None
            print('Error: Missing specific argument. -h for head or -s for specific')
            exit()
    except ValueError:
        specific = None

    if (head != None and specific != None):
        print('Error: Can use one type of argument only. -h for head or -s for specific')
        exit()
    clean_data(get_data(head, specific))
