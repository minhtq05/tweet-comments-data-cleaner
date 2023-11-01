# from transformers import BertTokenizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import ssl
import re
import pandas as pd
import sys

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def export_data_to_excel(data):
    data = pd.DataFrame(
        data, columns=['sentence_id', 'sentence_label', 'sentence_text'])
    data = data[data['sentence_text'] != '']
    data_0, data_1 = data[data['sentence_label']
                          == 0], data[data['sentence_label'] == 1]

    data_0.to_csv('../data/cleaned_0.csv', index=False)
    data_1.to_csv('../data/cleaned_1.csv', index=False)


def get_data(start=None, stop=None, specific=None):
    f = open('../data/data.txt')

    data = f.read()

    data = data.split('\n')

    data = [x.split(',', maxsplit=2) for x in data]

    return data[start:stop] if (specific == None) else [data[specific - 1]]


def remove_internet_contents(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove urls
    text = re.sub("www.[A-Za-z0-9-?[-`{-~]", "", text)  # remove urls
    text = re.sub("@[A-Za-z0-9!-?[-`{-~]+", "", text)  # remove usernames
    text = re.sub("#[A-Za-z0-9!-?[-`{-~]+", "", text)  # remove hashtags
    text = re.sub("[~`!@#$%^&*()-_+=[\]{};':\"\\|,.<>/?]",
                  "", text)  # remove special characters
    return text


def clean_data(data):
    length = len(data)
    jump = length / 10

    # tk = BertTokenizer.from_pretrained('bert-base-uncased')
    tk = TweetTokenizer()
    english_stops = set(stopwords.words('english'))

    for i in range(len(data)):
        id, label, text = data[i]
        id, label = int(id), int(label)

        text = remove_internet_contents(text)
        # sentences = sent_tokenize(text)
        tokens = tk.tokenize(text)

        # for sentence in sentences:
        # tokens = tk.tokenize(sentence)
        tokens = [word for word in tokens if word not in english_stops]
        # tokens = [word for word in tokens if len(wn.synsets(word)) > 0]
        # if (len(tokens) > 0):
        # result.append(tokens)

        data[i] = id, label, ' '.join(tokens)

        if (id % jump == 0):
            print(
                f"Done cleaning - {id / length * 100}% ({id}/{length}) of items")
    if (len(data) == 1):
        print(f"Done cleaning - line {data[0][0]}")
    export_data_to_excel(data)
    if (__name__ != "__main__"):
        return data


if __name__ == '__main__':
    args = sys.argv[1:]
    range_id, specific_id = None, None
    # range arguments: only clean lines in range(start, stop)
    try:
        range_id = args.index('-r')
        try:
            try:
                start = int(args[range_id + 1])
                stop = int(args[range_id + 2])
            except:
                stop = int(args[range_id + 1])
        except:
            print(
                'Error: Missing range argument. -r for range or -s for specific (all count from 1)')
            exit()
    except:
        start, stop = None, None

    # specific argument for cleaning a single 'specific' line of text

    try:
        specific_id = args.index('-s')
        try:
            specific = int(args[specific_id + 1])
        except:
            print(
                'Error: Missing specific argument. -r for range or -s for specific (all count from 1)')
            exit()
    except:
        specific = None

    if (range_id != None and specific_id != None):
        print('Error: Can use one type of argument only. -r for range or -s for specific (all count from 1)')
        exit()

    clean_data(get_data(start, stop, specific))
