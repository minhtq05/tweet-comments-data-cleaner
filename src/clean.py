from torchtext.data.utils import get_tokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import ssl
import re
import pandas as pd
import sys
import time


start_time = time.time()


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def export_data_to_excel(data):
    df = pd.DataFrame(
        data, columns=['id', 'label', 'text'])
    df = df[df['text'] != '']
    df_0, df_1 = df[df['label']
                    == 0], df[df['label'] == 1]

    df_0.to_csv('../data/cleaned_0.csv', index=False)
    df_1.to_csv('../data/cleaned_1.csv', index=False)


def get_data(start=None, stop=None, specific=None):
    f = open('../data/data.txt')

    data = f.read()

    data = data.split('\n')

    data = [x.split(',', maxsplit=2) for x in data]

    return data[start:stop] if (specific == None) else [data[specific - 1]]


def tokenize(text):
    """tokenize the text"""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove urls
    text = re.sub("@[A-Za-z0-9!-?[-`{-~]+", "", text)  # remove usernames
    text = re.sub("#[A-Za-z0-9!-?[-`{-~]+", "", text)  # remove hashtags
    text = re.sub("[~`!@#$%^&*()-_+=[\]{};:\"\\|,.<>/?]",
                  "", text)  # remove special characters
    text = re.findall("[a-z0-9]+", text)
    return text


# def remove_internet_contents(text):
#     text = text.lower()
#     text = re.sub(r"http\S+", "", text)  # remove urls
#     # text = re.sub("www.[A-Za-z0-9-?[-`{-~]", "", text)  # also remove urls
#     # text = re.sub("@[A-Za-z0-9!-?[-`{-~]+", "", text)  # remove usernames
#     # text = re.sub("#[A-Za-z0-9!-?[-`{-~]+", "", text)  # remove hashtags
#     text = re.sub()
#     text = re.sub("[~`!@#$%^&*()-_+=[\]{};:\"\\|,.<>/?]",
#                   "", text)  # remove special characters
#     return text


def clean_data(data):
    """ clean the dataset"""
    length = len(data)
    jump = length / 10

    # tk = BertTokenizer.from_pretrained('bert-base-uncased')
    english_stops = set(stopwords.words('english'))

    for i, (id, label, text) in enumerate(data):
        id, label = int(id), int(label)

        # text = remove_internet_contents(text)
        tokens = tokenize(text)

        tokens = [word for word in tokens if word not in english_stops]

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

    print(f"Execution time: {time.time() - start_time:.4f}(s)")
