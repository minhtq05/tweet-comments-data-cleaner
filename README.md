# Tweet Comments Data Cleaner

This is my in-progress Tweets Comments Data Processor for a project I'm working on.

# Usage

To clean the data, execute this command:

```bash
python clean.py
```

We're using a SVM to predict the binary state (positive / negative) of comments.

To train the model with the cleaned dataset, execute this command:

```bash
python train.py
```


Arguments available:

```bash
-h head: only clean the first 'head' number of items in the dataset
-s specific: clean a specific line of text in the dataset, count from 1
```
