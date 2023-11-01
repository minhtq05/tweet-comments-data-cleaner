import torch
from transformers import BertTokenizer, BertModel
from nltk.tokenize import TweetTokenizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import time

start_time = time.time()


clf = svm.SVC()
tk = BertTokenizer.from_pretrained('bert-base-uncased')
tk2 = TweetTokenizer()
model = BertModel.from_pretrained('bert-base-uncased')


def read_data():
    data = pd.concat([pd.read_csv('../data/cleaned_0.csv'),
                      pd.read_csv('../data/cleaned_1.csv')], ignore_index=True)

    return data['sentence_text'].to_list(), data['sentence_label'].to_list()


def train():
    X, y = read_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # id, label, tokens = data[0]
    # print(tokens)
    # tokens_ids = tk.convert_tokens_to_ids(tokens)
    # print(tokens_ids)
    # embeddings = model.embeddings.word_embeddings(torch.tensor(tokens_ids))
    # print(embeddings.shape)
    tokens_ids = tk.convert_tokens_to_ids(X_train)
    embeddings = model.embeddings.word_embeddings(
        torch.tensor(tokens_ids)).detach().numpy()

    clf.fit(embeddings, y_train)

    test_tokens_ids = tk.convert_tokens_to_ids(X_test)
    test_embeddings = model.embeddings.word_embeddings(
        torch.tensor(test_tokens_ids)).detach().numpy()

    y_pred = clf.predict(test_embeddings)

    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    print("Accuracy: ", accuracy)
    print("Time: ", round(time.time() - start_time, 2), '(s)')


if (__name__ == "__main__"):
    train()
