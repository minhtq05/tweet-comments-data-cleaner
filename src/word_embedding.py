import re
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import time
from nltk.corpus import stopwords

start_time = time.time()

np.random.seed(42)


# read data from pre-cleaned datasets

def read_data():
    df = pd.concat([pd.read_csv('../data/cleaned_0.csv'),
                    pd.read_csv('../data/cleaned_1.csv')], ignore_index=True)
    return df


df = read_data()

# get the corpus text, only take 50 first lines to process

text = list(df['text'])[:50]

stop_words = stopwords.words('english')

# re-implement a tokenizer, also remove stop words


def tokenize(text):
    text = text.lower()
    text = re.findall(r"[A-Za-z0-9]+", text)
    text = [w for w in text if w not in stop_words]
    return text

# generate a dictionary to convert words to ids and vice versa


def generate_dict(tokens):
    word_to_id, id_to_word = {}, {}

    for i, word in enumerate(set(tokens)):
        word_to_id[word] = i
        id_to_word[i] = word

    # print(word_to_id)
    # print(id_to_word)

    return word_to_id, id_to_word

# list of all tokens


tokens = []

for sentence in text:
    tokens += tokenize(sentence)

# get the dictionary

word_to_id, id_to_word = generate_dict(tokens)

# one hot encoding the word ids


def one_hot_encode(id, len_ids):
    r = [0] * len_ids
    r[id] = 1
    return r


# generate data from corpus using n-gram with n = 5, window size = 2

def generate_data(text, word_to_id, window_size=1):
    X, y = [], []

    n_words = len(word_to_id)

    for sentence in text:
        tokens = tokenize(sentence)
        for i in range(len(tokens)):
            n_tokens = len(tokens)
            for j in range(max(0, i - window_size), min(i + window_size + 1, n_tokens)):
                if i == j:
                    continue
                X.append(one_hot_encode(word_to_id[tokens[i]], n_words))
                y.append(one_hot_encode(word_to_id[tokens[j]], n_words))

    return np.asarray(X), np.asarray(y)

# X: input date
# y: label of the inputs


X, y = generate_data(text, word_to_id, 2)

# get the size of the dataset's inputs

print(f"Size of dataset: {X.shape}")

n_embedding = 30

# number of unique words, also the size of the word dictionary

vocab_size = len(word_to_id)

print(f"Number of unique words in the corpus: {vocab_size} words")

# define the model using torch,


class NN(nn.Module):
    def __init__(self):
        super().__init__()

        # use three layers,

        # the first layer can be the embedding layer, which means convert the words into
        # a vector with the size of n_embedding ( = 10 in this case)
        self.fc1 = nn.Linear(vocab_size, n_embedding)

        # the second layer can be the layer that converts the vector containing all the meanings
        # to a vector of size vocab_size, which is the distribution of all the words in the corpus
        self.fc2 = nn.Linear(n_embedding, vocab_size)

        # the last layer is the softmax layer, which redistributes the distribution to probabilities
        # to be able to calculaute the loss function
        self.fc3 = nn.Softmax()

        # => therefore, the first layer is the embedding layer that we can use to vectorize words

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# train the model


def train(model, X, y, n_epochs=30, learning_rate=0.01):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9)
    losses = []
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(n_epochs):
        n_loss = 0
        for i in range(len(X)):
            optimizer.zero_grad()
            output = model.forward(torch.tensor(X[i], dtype=torch.float32))
            loss = loss_fn(output, torch.tensor(y[i], dtype=torch.float32))
            loss.backward()
            optimizer.step()
            n_loss += loss.item()
        losses.append(n_loss / len(X))

    return losses


model = NN()

loss = train(model, X, y)

# plt.plot(range(len(loss)), loss, color="skyblue")


# testing
# test = one_hot_encode(word_to_id["early"], vocab_size)

# result = model.forward(torch.tensor(test, dtype=torch.float32))
# print(result)

# result_dict = {}

# for i, prob in enumerate(list(result)):
# result_dict[float(prob)] = id_to_word[i]

# print(dict(sorted(result_dict.items(), reverse=True)))


# vectorizing the words

def vectorize(word):
    one_hot = torch.tensor(one_hot_encode(
        word_to_id[word], vocab_size), dtype=torch.float32)
    return model.fc1(one_hot)


print(vectorize("early"))


print(f"Execution time: {time.time() - start_time}(s)")
